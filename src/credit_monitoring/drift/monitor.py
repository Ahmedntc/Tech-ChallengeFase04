"""Monitoramento mensal do modelo em produção.

Para cada mês simulado:

1. **Gate de ingestão:** todos os lotes recebidos passam pelo contrato de dados;
   os reprovados vão para a quarentena e geram alerta.
2. **Predição:** o modelo pontua os lotes aprovados.
3. **Data drift:** cada variável é comparada com a referência por PSI (magnitude,
   faixas por decis) e KS do Evidently (significância).
4. **Prediction drift:** a distribuição do score também é comparada.
5. **Desempenho:** com os rótulos (que em produção chegam meses depois), AUC,
   recall e calibração são comparados com o teste da Referência.
6. **Alertas e relatórios:** relatório do Evidently por mês, tabelas e alertas.

A base de comparação é o **conjunto de teste da Referência**: mesma distribuição do
treino, mas fora do ajuste do modelo, então seus scores não têm o viés otimista de
dados de treino.

Uso: ``python -m credit_monitoring.drift.monitor``
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from credit_monitoring.config import load_settings, resolve_path
from credit_monitoring.contracts.gate import ingest_batch
from credit_monitoring.data.columns import FEATURES, ID_COLUMN, TARGET
from credit_monitoring.drift.stats import drift_severity, psi
from credit_monitoring.models.artifacts import ModelBundle
from credit_monitoring.models.metrics import classification_metrics
from credit_monitoring.models.train import split_reference

logger = logging.getLogger(__name__)

PREDICTION = "prediction"
PREDICTION_LABEL = "prediction_label"
SEVERITY_ORDER = ["estável", "leve", "moderado", "significativo"]


def _br(value: float, spec: str) -> str:
    """Formata números no padrão brasileiro (vírgula decimal) para as mensagens."""
    return format(value, spec).replace(".", ",")


@dataclass
class Alert:
    month: str
    category: str
    severity: str
    subject: str
    message: str
    value: float | None = None
    threshold: float | None = None


@dataclass
class MonthResult:
    month: str
    title: str
    narrative: str
    batches: list[dict[str, Any]]
    n_rows: int
    feature_drift: pd.DataFrame = field(repr=False)
    n_drifted: int
    drift_share: float
    prediction_psi: float
    prediction_ks_pvalue: float
    mean_score: float
    approval_rate: float
    performance: dict[str, float]
    alerts: list[Alert]
    evidently_report: str

    def summary_row(self) -> dict[str, Any]:
        severities = self.feature_drift["severity"].value_counts()
        top = self.feature_drift.sort_values("psi", ascending=False).head(3)
        return {
            "month": self.month,
            "title": self.title,
            "n_rows": self.n_rows,
            "blocked_batches": sum(b["status"] == "bloqueado" for b in self.batches),
            "n_drifted": self.n_drifted,
            "drift_share": self.drift_share,
            **{f"n_{sev}": int(severities.get(sev, 0)) for sev in SEVERITY_ORDER[1:]},
            "psi_max": float(self.feature_drift["psi"].max()),
            "top_psi_features": ", ".join(top["feature"]),
            "prediction_psi": self.prediction_psi,
            "mean_score": self.mean_score,
            "actual_default_rate": self.performance["actual_default_rate"],
            "approval_rate": self.approval_rate,
            "auc": self.performance["auc"],
            "ks": self.performance["ks"],
            "recall": self.performance["recall"],
            "precision": self.performance["precision"],
            "n_alerts": len(self.alerts),
            "n_critical_alerts": sum(a.severity == "crítico" for a in self.alerts),
        }


@dataclass
class Baseline:
    """Conjunto de comparação: teste da Referência com score e rótulo."""

    data: pd.DataFrame
    metrics: dict[str, float]


def build_baseline(settings: dict[str, Any], bundle: ModelBundle) -> Baseline:
    reference = pd.read_parquet(resolve_path(settings["paths"]["reference_file"]))
    _, _, test = split_reference(reference, settings)
    test = test.reset_index(drop=True)
    scores = bundle.predict_proba(test)
    data = test.assign(
        **{PREDICTION: scores, PREDICTION_LABEL: (scores >= bundle.threshold).astype(int)}
    )
    return Baseline(data, classification_metrics(test[TARGET], scores, bundle.threshold))


def run_evidently(
    current: pd.DataFrame, baseline: pd.DataFrame, ks_alpha: float, html_path: Path
) -> dict[str, float]:
    """Relatório do Evidently (drift por KS + desempenho) e p-valores do KS por coluna."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from evidently import BinaryClassification, DataDefinition, Dataset, Report
        from evidently.presets import ClassificationPreset, DataDriftPreset

        columns = [*FEATURES, PREDICTION]
        definition = DataDefinition(
            numerical_columns=columns,
            classification=[
                BinaryClassification(
                    target=TARGET,
                    prediction_probas=PREDICTION,
                    prediction_labels=PREDICTION_LABEL,
                )
            ],
        )
        report = Report(
            [
                DataDriftPreset(columns=columns, method="ks", threshold=ks_alpha),
                ClassificationPreset(),
            ]
        )
        snapshot = report.run(
            Dataset.from_pandas(current[[*columns, PREDICTION_LABEL, TARGET]], definition),
            Dataset.from_pandas(baseline[[*columns, PREDICTION_LABEL, TARGET]], definition),
        )
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
        metrics = snapshot.dict()["metrics"]

    return {
        m["config"]["column"]: float(m["value"])
        for m in metrics
        if m.get("config", {}).get("type", "").endswith("ValueDrift")
    }


def feature_drift_table(
    current: pd.DataFrame,
    baseline: pd.DataFrame,
    ks_pvalues: dict[str, float],
    model_features: list[str],
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        value = psi(baseline[feature], current[feature], cfg["psi_bins"])
        pvalue = ks_pvalues[feature]
        rows.append(
            {
                "feature": feature,
                "in_model": feature in model_features,
                "psi": value,
                "ks_pvalue": pvalue,
                "severity": drift_severity(
                    value, pvalue, cfg["psi_moderate"], cfg["psi_significant"], cfg["ks_alpha"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_alerts(
    month: str,
    batches: list[dict[str, Any]],
    drift: pd.DataFrame,
    drift_share: float,
    prediction_psi: float,
    performance: dict[str, float],
    mean_score: float,
    baseline_metrics: dict[str, float],
    cfg: dict[str, Any],
) -> list[Alert]:
    alerts: list[Alert] = []
    for batch in batches:
        if batch["status"] == "bloqueado":
            alerts.append(
                Alert(
                    month,
                    "qualidade de dados",
                    "crítico",
                    batch["file"],
                    f"Lote bloqueado pelo contrato (regras {', '.join(batch['violated_rules'])}) "
                    "e enviado para a quarentena.",
                )
            )

    for row in drift[drift["severity"].isin(["moderado", "significativo"])].itertuples():
        limit = cfg["psi_significant"] if row.severity == "significativo" else cfg["psi_moderate"]
        alerts.append(
            Alert(
                month,
                "data drift",
                "crítico" if row.severity == "significativo" else "atenção",
                row.feature,
                f"Drift {row.severity} em {row.feature} (PSI {_br(row.psi, '.3f')}).",
                round(row.psi, 4),
                limit,
            )
        )
    mild = drift.loc[drift["severity"] == "leve", "feature"].tolist()
    if mild:
        alerts.append(
            Alert(
                month,
                "data drift",
                "info",
                ", ".join(mild),
                f"Drift leve (detectado pelo KS, PSI < {_br(cfg['psi_moderate'], '.2f')}) "
                f"em {len(mild)} variável(is): mudança estatisticamente significativa, "
                "ainda pequena.",
            )
        )
    if drift_share >= cfg["drift_share_alert"]:
        alerts.append(
            Alert(
                month,
                "data drift",
                "atenção",
                "dataset",
                f"{_br(drift_share, '.0%')} das variáveis com drift.",
                round(drift_share, 4),
                cfg["drift_share_alert"],
            )
        )
    if prediction_psi >= cfg["prediction_psi_alert"]:
        alerts.append(
            Alert(
                month,
                "prediction drift",
                "crítico" if prediction_psi >= cfg["psi_significant"] else "atenção",
                PREDICTION,
                f"Distribuição do score mudou (PSI {_br(prediction_psi, '.3f')}).",
                round(prediction_psi, 4),
                cfg["prediction_psi_alert"],
            )
        )

    perf = cfg["performance"]
    auc_drop = baseline_metrics["auc"] - performance["auc"]
    if auc_drop >= perf["auc_drop"]:
        alerts.append(
            Alert(
                month,
                "desempenho",
                "crítico",
                "auc",
                f"AUC caiu {_br(auc_drop, '.3f')} em relação à referência "
                f"({_br(baseline_metrics['auc'], '.3f')} → {_br(performance['auc'], '.3f')}).",
                round(auc_drop, 4),
                perf["auc_drop"],
            )
        )
    recall_drop = baseline_metrics["recall"] - performance["recall"]
    if recall_drop >= perf["recall_drop"]:
        alerts.append(
            Alert(
                month,
                "desempenho",
                "atenção",
                "recall",
                f"Recall caiu {_br(recall_drop, '.3f')}: o modelo deixa passar mais inadimplentes.",
                round(recall_drop, 4),
                perf["recall_drop"],
            )
        )
    gap = performance["actual_default_rate"] - mean_score
    if abs(gap) >= perf["calibration_gap"]:
        direction = "subestima" if gap > 0 else "superestima"
        alerts.append(
            Alert(
                month,
                "desempenho",
                "crítico" if gap > 0 else "atenção",
                "calibração",
                f"O modelo {direction} o risco: inadimplência real "
                f"{_br(performance['actual_default_rate'], '.1%')} x "
                f"score médio {_br(mean_score, '.1%')}.",
                round(gap, 4),
                perf["calibration_gap"],
            )
        )
    return alerts


def monitor_month(
    month_info: dict[str, Any],
    bundle: ModelBundle,
    baseline: Baseline,
    settings: dict[str, Any],
) -> MonthResult:
    paths = settings["paths"]
    cfg = settings["drift"]
    month = month_info["month"]
    incoming = resolve_path(paths["incoming_dir"])

    batches, accepted = [], []
    for name in month_info["files"]:
        result = ingest_batch(incoming / name, settings)
        batches.append(
            {
                "file": name,
                "status": "aprovado" if result.passed else "bloqueado",
                "n_rows": result.n_rows,
                "violated_rules": result.violated_rules,
            }
        )
        if result.passed:
            accepted.append(result.data)
    if not accepted:
        raise RuntimeError(f"Nenhum lote aprovado em {month}.")

    current = pd.concat(accepted, ignore_index=True)
    scores = bundle.predict_proba(current)
    current = current.assign(
        **{PREDICTION: scores, PREDICTION_LABEL: (scores >= bundle.threshold).astype(int)}
    )
    predictions_dir = resolve_path(paths["predictions_dir"])
    predictions_dir.mkdir(parents=True, exist_ok=True)
    current[[ID_COLUMN, PREDICTION, PREDICTION_LABEL]].assign(
        month=month, model_version=bundle.version
    ).to_parquet(predictions_dir / f"{month}.parquet", index=False)

    # Os rótulos chegam depois; o desempenho é medido nos clientes já rotulados.
    labels = pd.read_parquet(resolve_path(paths["labels_dir"]) / f"{month}.parquet")
    current = current.merge(labels, on=ID_COLUMN, how="inner")

    reports_dir = resolve_path(paths["drift_reports_dir"])
    html_path = reports_dir / f"evidently_{month}.html"
    ks_pvalues = run_evidently(current, baseline.data, cfg["ks_alpha"], html_path)

    drift = feature_drift_table(current, baseline.data, ks_pvalues, bundle.features, cfg)
    n_drifted = int((drift["severity"] != "estável").sum())
    drift_share = n_drifted / len(drift)
    prediction_psi = psi(baseline.data[PREDICTION], current[PREDICTION], cfg["psi_bins"])
    performance = classification_metrics(current[TARGET], current[PREDICTION], bundle.threshold)
    mean_score = float(current[PREDICTION].mean())
    alerts = build_alerts(
        month,
        batches,
        drift,
        drift_share,
        prediction_psi,
        performance,
        mean_score,
        baseline.metrics,
        cfg,
    )
    logger.info(
        "%s: %d variáveis com drift, AUC %.3f, %d alertas",
        month,
        n_drifted,
        performance["auc"],
        len(alerts),
    )
    return MonthResult(
        month=month,
        title=month_info["title"],
        narrative=month_info.get("narrative", ""),
        batches=batches,
        n_rows=int(sum(len(df) for df in accepted)),
        feature_drift=drift,
        n_drifted=n_drifted,
        drift_share=drift_share,
        prediction_psi=prediction_psi,
        prediction_ks_pvalue=ks_pvalues[PREDICTION],
        mean_score=mean_score,
        approval_rate=float(1 - current[PREDICTION_LABEL].mean()),
        performance=performance,
        alerts=alerts,
        evidently_report=html_path.name,
    )


def save_outputs(results: list[MonthResult], baseline: Baseline, settings: dict[str, Any]) -> Path:
    """Grava as tabelas, os alertas e o relatório consolidado."""
    from credit_monitoring.drift.report import render_drift_report

    reports_dir = resolve_path(settings["paths"]["drift_reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    feature_drift = pd.concat(
        [r.feature_drift.assign(month=r.month) for r in results], ignore_index=True
    )
    feature_drift[["month", "feature", "in_model", "psi", "ks_pvalue", "severity"]].to_csv(
        reports_dir / "feature_drift.csv", index=False, float_format="%.6g"
    )
    summary = pd.DataFrame([r.summary_row() for r in results])
    summary.to_csv(reports_dir / "monthly_summary.csv", index=False, float_format="%.6g")
    alerts = [asdict(a) for r in results for a in r.alerts]
    (reports_dir / "alerts.json").write_text(
        json.dumps(alerts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = reports_dir / "drift_report.html"
    report_path.write_text(
        render_drift_report(results, baseline.metrics, settings), encoding="utf-8"
    )
    return report_path


def run_monitoring(settings: dict[str, Any] | None = None) -> list[MonthResult]:
    settings = settings or load_settings()
    paths = settings["paths"]
    bundle = ModelBundle.load(resolve_path(paths["model_file"]))
    baseline = build_baseline(settings, bundle)
    manifest = json.loads(resolve_path(paths["simulation_manifest"]).read_text(encoding="utf-8"))
    results = [monitor_month(m, bundle, baseline, settings) for m in manifest["months"]]
    save_outputs(results, baseline, settings)
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    results = run_monitoring(settings)

    print(
        f"\n{'mês':<5}{'cenário':<27}{'drift':>7}{'PSI máx':>9}{'PSI score':>11}"
        f"{'AUC':>7}{'score':>8}{'real':>7}{'alertas':>9}"
    )
    for r in results:
        print(
            f"{r.month:<5}{r.title:<27}{r.n_drifted:>4}/23{r.feature_drift['psi'].max():>9.3f}"
            f"{r.prediction_psi:>11.3f}{r.performance['auc']:>7.3f}{r.mean_score:>8.1%}"
            f"{r.performance['actual_default_rate']:>7.1%}{len(r.alerts):>9}"
        )
    reports = settings["paths"]["drift_reports_dir"]
    print(f"\nRelatório consolidado: {reports}/drift_report.html")
    print(f"Relatórios do Evidently: {reports}/evidently_M*.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
