import copy
import json

import pandas as pd
import pytest

from credit_monitoring.data.columns import FEATURES
from credit_monitoring.data.prepare import split_reference_production
from credit_monitoring.drift.monitor import build_alerts, run_monitoring
from credit_monitoring.drift.simulate import simulate
from credit_monitoring.models.train import train

BASELINE = {"auc": 0.78, "recall": 0.56}
HEALTHY = {"auc": 0.77, "recall": 0.55, "actual_default_rate": 0.22}


def _drift(**severity_by_feature) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature": f,
                "psi": {"significativo": 0.3, "moderado": 0.15}.get(
                    severity_by_feature.get(f), 0.01
                ),
                "severity": severity_by_feature.get(f, "estável"),
            }
            for f in FEATURES
        ]
    )


def _alerts(base_settings, **overrides):
    args = {
        "month": "M1",
        "batches": [{"file": "M1.csv", "status": "aprovado", "violated_rules": []}],
        "drift": _drift(),
        "drift_share": 0.0,
        "prediction_psi": 0.01,
        "performance": HEALTHY,
        "mean_score": 0.22,
        "baseline_metrics": BASELINE,
        "cfg": base_settings["drift"],
    }
    args.update(overrides)
    return build_alerts(**args)


def test_healthy_month_raises_no_alerts(base_settings):
    assert _alerts(base_settings) == []


def test_blocked_batch_raises_critical_data_quality_alert(base_settings):
    batches = [{"file": "M4_parceiro.csv", "status": "bloqueado", "violated_rules": ["R1", "R3"]}]
    (alert,) = _alerts(base_settings, batches=batches)
    assert (alert.category, alert.severity) == ("qualidade de dados", "crítico")
    assert "R1, R3" in alert.message


def test_feature_drift_alerts_follow_severity(base_settings):
    drift = _drift(pay_status_1="significativo", age="moderado", bill_amt_1="leve")
    alerts = _alerts(base_settings, drift=drift, drift_share=3 / 23)
    by_subject = {a.subject: a.severity for a in alerts}
    assert by_subject["pay_status_1"] == "crítico"
    assert by_subject["age"] == "atenção"
    assert by_subject["bill_amt_1"] == "info"


def test_performance_drop_and_underestimated_risk_are_critical(base_settings):
    performance = {"auc": 0.70, "recall": 0.40, "actual_default_rate": 0.40}
    alerts = _alerts(base_settings, performance=performance, mean_score=0.28)
    by_subject = {a.subject: a for a in alerts}
    assert by_subject["auc"].severity == "crítico"
    assert by_subject["recall"].severity == "atenção"
    assert by_subject["calibração"].severity == "crítico"
    assert "subestima" in by_subject["calibração"].message


@pytest.fixture
def small_world(settings, prepared_df, tmp_path):
    """Pipeline completo em escala reduzida: referência, pool, modelo e 2 meses."""
    cfg = copy.deepcopy(settings)
    cfg["paths"].update(
        labels_dir=str(tmp_path / "labels"),
        predictions_dir=str(tmp_path / "predictions"),
        simulation_manifest=str(tmp_path / "manifest.json"),
        drift_reports_dir=str(tmp_path / "drift"),
    )
    cfg["model"]["xgboost"]["n_estimators"] = 50
    cfg["simulation"]["batch_size"] = 800
    months = cfg["simulation"]["months"]
    cfg["simulation"]["months"] = [months[0], months[-1]]

    reference, pool = split_reference_production(prepared_df, 0.3, seed=42)
    reference.to_parquet(cfg["paths"]["reference_file"], index=False)
    pool.to_parquet(cfg["paths"]["production_pool_file"], index=False)
    train(cfg, track=False)
    simulate(cfg)
    return cfg, tmp_path


def test_monitoring_end_to_end_flags_the_crisis_and_not_the_control(small_world):
    cfg, tmp_path = small_world
    control, crisis = run_monitoring(cfg)

    assert control.n_drifted <= 1
    assert not any(a.severity == "crítico" for a in control.alerts)
    assert crisis.n_drifted > 5
    assert crisis.performance["actual_default_rate"] > crisis.mean_score + 0.05
    assert any(a.category == "desempenho" for a in crisis.alerts)

    drift_dir = tmp_path / "drift"
    for name in ("drift_report.html", "feature_drift.csv", "monthly_summary.csv", "alerts.json"):
        assert (drift_dir / name).exists()
    assert (drift_dir / f"evidently_{control.month}.html").exists()
    alerts = json.loads((drift_dir / "alerts.json").read_text(encoding="utf-8"))
    assert len(alerts) == len(control.alerts) + len(crisis.alerts)
    assert (tmp_path / "predictions" / f"{crisis.month}.parquet").exists()
