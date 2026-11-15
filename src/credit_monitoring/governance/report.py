"""Relatório de governança: causalidade, equidade, explicabilidade e privacidade.

Gera ``reports/governance/governance_report.html`` (autocontido) e as tabelas em CSV
que sustentam a documentação de governança do README.

Uso: ``python -m credit_monitoring.governance.report``
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from credit_monitoring.config import load_settings, resolve_path
from credit_monitoring.data.columns import ID_COLUMN, TARGET
from credit_monitoring.governance.causal import (
    METRICS,
    attribute_degradation,
    retraining_experiment,
)
from credit_monitoring.governance.explain import feature_label, global_importance, reason_codes
from credit_monitoring.governance.fairness import (
    ATTRIBUTE_LABELS,
    compare_variants,
    evaluate,
    production_fairness,
    sampling_noise,
)
from credit_monitoring.governance.privacy import expired_files, reidentification_risk
from credit_monitoring.models.artifacts import ModelBundle
from credit_monitoring.models.train import split_reference
from credit_monitoring.monitoring.logger import configure_logging, log_event
from credit_monitoring.reporting import (
    MUTED,
    SERIES,
    base_layout,
    figure_html,
    num,
    page,
    pct,
)

logger = logging.getLogger(__name__)

SHORT_LABELS = {
    "inflacao": "Inflação",
    "pagamentos": "Pagamentos",
    "limites": "Limites",
    "perfil_jovem": "Clientes jovens",
    "desemprego": "Desemprego",
    "concept_drift": "Concept drift",
}

EXTRA_STYLE = """
.verdict { font-weight:600; }
.two { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }
@media (max-width: 520px) { .two { grid-template-columns:1fr; } }
ol.reasons { margin:0; padding-left:18px; }
.callout { background:var(--card); border:1px solid var(--line); border-left:6px solid #2a78d6; border-radius:10px;
           padding:14px 16px; margin:16px 0 0; }
.callout p { margin:0; }
"""


def _pp(value: float) -> str:
    """Diferença em pontos percentuais."""
    return f"{num(100 * value, 1)} p.p."


def _signed(value: float, digits: int = 3) -> str:
    return ("+" if value >= 0 else "−") + num(abs(value), digits)


def _check(ok: bool, good: str = "dentro do limite", bad: str = "acima do limite") -> str:
    return (
        f"<span class='pill ok'>✔ {good}</span>"
        if ok
        else f"<span class='pill warn'>▲ {bad}</span>"
    )


def _table(
    header: list[str], rows: list[list[str]], numeric: set[int], min_width: int = 760
) -> str:
    head = "".join(
        f"<th class='{'num' if i in numeric else ''}'>{h}</th>" for i, h in enumerate(header)
    )
    body = "".join(
        "<tr>"
        + "".join(
            f"<td class='{'num' if i in numeric else ''}'>{c}</td>" for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return (
        f"<div class='card table-wrap'><table style='min-width:{min_width}px'>"
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _card(title: str, note: str, chart: str) -> str:
    return f"<div class='card chart'><h3>{title}</h3><p>{note}</p>{chart}</div>"


# --------------------------------------------------------------------------- gráficos


def effect_chart(effects: pd.DataFrame, metric: str, title: str, div_id: str) -> str:
    """Efeito de cada mecanismo (barras horizontais, com erro padrão)."""
    data = effects.iloc[::-1]
    values = data[metric].to_numpy()
    fig = go.Figure(
        go.Bar(
            x=values,
            y=[SHORT_LABELS[m] for m in data.index],
            orientation="h",
            marker={"color": SERIES[0]},
            error_x={
                "type": "data",
                "array": 1.96 * data[f"{metric}_se"].to_numpy(),
                "color": MUTED,
            },
            text=[_signed(v) for v in values],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"%{{y}}: %{{x:+.4f}} de {title}<extra></extra>",
        )
    )
    layout = base_layout("", ".2f")
    layout.update(bargap=0.45, barcornerradius=4, hovermode="closest", height=340)
    layout["margin"].update(l=120, r=24)
    span = max(abs(values).max() * 1.6, 0.01)
    layout["xaxis"].update(
        showgrid=True,
        gridcolor="#e1e0d9",
        range=[-span, span],
        tickformat=".2f",
        zeroline=True,
        zerolinecolor="#898781",
        title={"text": f"Variação de {title}", "font": {"size": 12, "color": MUTED}},
    )
    layout["yaxis"].update(showgrid=False, tickformat="")
    fig.update_layout(**layout)
    return figure_html(fig, div_id)


def importance_chart(importance: pd.Series, top: int = 10) -> str:
    data = importance.head(top).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=data.to_numpy(),
            y=[feature_label(f) for f in data.index],
            orientation="h",
            marker={"color": SERIES[0]},
            text=[num(v, 2) for v in data.to_numpy()],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    layout = base_layout("", ".2f")
    layout.update(bargap=0.45, barcornerradius=4, hovermode="closest", height=380)
    layout["margin"].update(l=260, r=50)
    layout["xaxis"].update(
        showgrid=True,
        gridcolor="#e1e0d9",
        rangemode="tozero",
        tickformat=".1f",
        title={
            "text": "Contribuição média absoluta (log-odds)",
            "font": {"size": 12, "color": MUTED},
        },
    )
    layout["yaxis"].update(showgrid=False, tickformat="")
    fig.update_layout(**layout)
    return figure_html(fig, "chart-importance")


# --------------------------------------------------------------------------- cálculo


def load_production(settings: dict[str, Any], bundle: ModelBundle) -> dict[str, pd.DataFrame]:
    """Lotes mensais com features, score e rótulo, para a equidade em produção."""
    paths = settings["paths"]
    manifest = json.loads(resolve_path(paths["simulation_manifest"]).read_text(encoding="utf-8"))
    batches = {}
    for month in [m["month"] for m in manifest["months"]]:
        features = pd.read_csv(
            resolve_path(paths["incoming_dir"]) / f"{month}.csv", dtype={ID_COLUMN: str}
        )
        labels = pd.read_parquet(resolve_path(paths["labels_dir"]) / f"{month}.parquet")
        batch = features.merge(labels, on=ID_COLUMN)
        batches[month] = batch.assign(score=bundle.predict_proba(batch))
    return batches


def build(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Calcula todas as análises de governança e grava os resultados."""
    settings = settings or load_settings()
    paths = settings["paths"]
    gov = settings["governance"]
    bundle = ModelBundle.load(resolve_path(paths["model_file"]))
    reference = pd.read_parquet(resolve_path(paths["reference_file"]))
    pool = pd.read_parquet(resolve_path(paths["production_pool_file"]))
    rates = reference.groupby("pay_status_1")[TARGET].mean().reindex([1, 2, 3]).to_dict()
    _, valid, test = split_reference(reference, settings)

    causal = attribute_degradation(settings, bundle, pool, rates)
    retraining = retraining_experiment(settings, bundle, reference, pool, rates)

    scored_test = test.assign(score=bundle.predict_proba(test))
    groups, indicators = evaluate(scored_test, bundle.threshold, gov["fairness"])
    refusal_rate = float((bundle.predict_proba(valid) >= bundle.threshold).mean())
    variants = compare_variants(settings, reference, refusal_rate)
    monthly = load_production(settings, bundle)
    windows = {
        name: pd.concat([monthly[m] for m in months], ignore_index=True)
        for name, months in gov["fairness"]["production_windows"].items()
    }
    production = production_fairness(windows, bundle.threshold, gov["fairness"])
    size = settings["simulation"]["batch_size"]
    noise = pd.DataFrame(
        {
            "month": sampling_noise(
                pool, bundle.predict_proba, bundle.threshold, size, gov["fairness"]
            ),
            "window": sampling_noise(
                pool, bundle.predict_proba, bundle.threshold, 3 * size, gov["fairness"]
            ),
        }
    )

    crisis = monthly[causal["month"]]
    importance = global_importance(bundle, crisis)
    refused = crisis[crisis["score"] >= bundle.threshold].sort_values("score", ascending=False)
    examples = refused.iloc[:: max(len(refused) // 5, 1)].head(5)
    reasons = reason_codes(bundle, examples)

    privacy = reidentification_risk(reference, gov["privacy"])
    expired = expired_files(settings)

    return {
        "settings": settings,
        "bundle": bundle,
        "causal": causal,
        "retraining": retraining,
        "groups": groups,
        "indicators": indicators,
        "variants": variants,
        "production": production,
        "noise": noise,
        "importance": importance,
        "examples": examples,
        "reasons": reasons,
        "privacy": privacy,
        "expired": expired,
    }


# --------------------------------------------------------------------------- página


def render(result: dict[str, Any]) -> str:
    settings, bundle = result["settings"], result["bundle"]
    gov = settings["governance"]
    causal, retraining = result["causal"], result["retraining"]
    by_mech, by_group = causal["by_mechanism"], causal["by_group"]
    month = causal["month"]
    fair_cfg = gov["fairness"]

    gap_total = causal["full"]["calibration_gap"] - causal["baseline"]["calibration_gap"]
    concept_share = by_group.loc["concept_drift", "calibration_gap"] / gap_total
    ret = retraining.set_index("response")
    auc_gain = ret.loc["retreinar", "auc"] - ret.loc["manter", "auc"]
    worst = result["indicators"].sort_values("equal_opportunity_gap").iloc[-1]
    risk = result["privacy"].iloc[-1]

    stats = [
        (pct(concept_share, 0), f"da descalibração em {month} vem do concept drift"),
        (
            _signed(by_group.loc["concept_drift", "auc"]),
            f"de AUC em {month} causados pelo concept drift",
        ),
        (_signed(auc_gain), "de AUC recuperados com retreino"),
        (
            _pp(worst["equal_opportunity_gap"]),
            f"maior diferença de recusa indevida ({ATTRIBUTE_LABELS[worst['attribute']].lower()})",
        ),
        (
            pct(risk["share_unique"]),
            "dos clientes únicos por idade, sexo, escolaridade, estado civil e limite",
        ),
    ]
    stats_html = "".join(
        f"<div class='stat'><b>{escape(v)}</b><span>{escape(k)}</span></div>" for v, k in stats
    )

    # Causalidade -----------------------------------------------------------
    mech_rows = [
        [
            f"<b>{escape(row.label)}</b>",
            _signed(row.auc),
            _signed(row.calibration_gap),
            _signed(row.mean_score),
            _signed(row.actual_default_rate),
            _signed(row.approval_rate),
        ]
        for row in by_mech.itertuples()
    ]
    total = {k: causal["full"][k] - causal["baseline"][k] for k in METRICS}
    mech_rows.append(
        ["<b>Total (soma dos efeitos)</b>"]
        + [
            f"<b>{_signed(total[k])}</b>"
            for k in (
                "auc",
                "calibration_gap",
                "mean_score",
                "actual_default_rate",
                "approval_rate",
            )
        ]
    )
    mech_table = _table(
        ["Mecanismo", "AUC", "Real − score", "Score médio", "Inadimplência real", "Aprovação"],
        mech_rows,
        {1, 2, 3, 4, 5},
        820,
    )
    group_rows = [
        [
            "<b>Data drift</b> (inflação, pagamentos, limites, perfil, desemprego)"
            if name == "data_drift"
            else "<b>Concept drift</b>",
            f"{_signed(row['auc'])} ± {num(1.96 * row['auc_se'])}",
            f"{_signed(row['calibration_gap'])} ± {num(1.96 * row['calibration_gap_se'])}",
            pct(row["calibration_gap"] / gap_total, 0),
        ]
        for name, row in by_group.iterrows()
    ]
    group_table = _table(
        ["Bloco", "Efeito no AUC", "Efeito em real − score", "Parcela da descalibração"],
        group_rows,
        {1, 2, 3},
    )

    names = {
        "manter": "Manter o modelo",
        "recalibrar": "Recalibrar o score",
        "retreinar": "Retreinar com dados recentes",
    }
    retrain_rows = [
        [
            f"<b>{names[row.response]}</b>",
            f"{num(row.auc)} ± {num(1.96 * row.auc_se)}",
            num(row.ks),
            _pp(row.calibration_gap),
            pct(row.recall),
            pct(row.precision),
            pct(row.approval_rate),
        ]
        for row in retraining.itertuples()
    ]
    retrain_table = _table(
        ["Resposta", "AUC", "KS", "Real − score", "Recall", "Precisão", "Aprovação"],
        retrain_rows,
        {1, 2, 3, 4, 5, 6},
    )

    # Equidade ---------------------------------------------------------------
    ind_rows = [
        [
            f"<b>{ATTRIBUTE_LABELS[row.attribute]}</b>",
            num(row.disparate_impact),
            _check(row.passes_disparate_impact),
            _pp(row.equal_opportunity_gap),
            _check(row.passes_equal_opportunity),
            num(row.auc_gap),
            escape(row.most_refused_group),
        ]
        for row in result["indicators"].itertuples()
    ]
    ind_table = _table(
        [
            "Atributo",
            "Impacto desproporcional",
            "Regra dos 4/5",
            "Diferença de recusa indevida",
            f"Limite {_pp(fair_cfg['equal_opportunity_max_gap'])}",
            "Diferença de AUC",
            "Grupo mais recusado",
        ],
        ind_rows,
        {1, 3, 5},
        900,
    )
    group_rows = [
        [
            ATTRIBUTE_LABELS[row.attribute],
            escape(row.group)
            + (
                ""
                if row.n >= fair_cfg["min_group_size"]
                else " <span class='tag'>amostra pequena</span>"
            ),
            num(row.n, 0),
            pct(row.default_rate),
            pct(row.approval_rate),
            pct(row.false_refusal_rate),
            "—" if pd.isna(row.auc) else num(row.auc),
        ]
        for row in result["groups"].itertuples()
    ]
    groups_table = _table(
        [
            "Atributo",
            "Grupo",
            "Clientes",
            "Inadimplência real",
            "Aprovação",
            "Recusa indevida",
            "AUC",
        ],
        group_rows,
        {2, 3, 4, 5, 6},
        820,
    )
    attrs = ["sex", "age_band", "education"]
    var_rows = []
    for row in result["variants"].itertuples():
        values = row._asdict()
        var_rows.append(
            [
                f"<b>{escape(row.variant)}</b>",
                escape(row.excluded),
                f"{num(row.auc, 4)} ± {num(row.auc_std, 4)}",
                pct(row.approval_rate),
            ]
            + [_pp(values[f"eo_{a}"]) for a in attrs]
        )
    var_table = _table(
        ["Variante", "Atributos fora", "AUC", "Aprovação"]
        + [f"Recusa indevida: {ATTRIBUTE_LABELS[a].lower()}" for a in attrs],
        var_rows,
        {2, 3, 4, 5, 6},
        980,
    )

    prod = result["production"].pivot(
        index="attribute", columns="window", values="equal_opportunity_gap"
    )
    reference_gap = result["indicators"].set_index("attribute")["equal_opportunity_gap"]
    windows = list(fair_cfg["production_windows"])
    prod_rows = [
        [f"<b>{ATTRIBUTE_LABELS[a]}</b>", _pp(reference_gap[a])]
        + [_pp(prod.loc[a, w]) for w in windows]
        + [
            f"± {_pp(result['noise'].loc[a, 'window'])}",
            f"± {_pp(result['noise'].loc[a, 'month'])}",
        ]
        for a in reference_gap.index
    ]
    prod_table = _table(
        [
            "Atributo",
            "Referência (teste)",
            *windows,
            "Oscilação por acaso (3 meses)",
            "Oscilação por acaso (1 mês)",
        ],
        prod_rows,
        {1, 2, 3, 4, 5},
        860,
    )

    # Explicabilidade -------------------------------------------------------
    reason_rows = []
    for (_, row), reasons in zip(result["examples"].iterrows(), result["reasons"], strict=True):
        items = "".join(
            f"<li>{escape(r['motivo'])}: <b>{escape(r['valor'])}</b></li>" for r in reasons
        )
        reason_rows.append(
            [
                f"<code>{escape(row[ID_COLUMN])}</code>",
                pct(row["score"]),
                f"<ol class='reasons'>{items}</ol>",
            ]
        )
    reason_table = _table(
        ["Cliente (pseudonimizado)", "Score", "Principais motivos da recusa"], reason_rows, {1}, 760
    )

    # Privacidade -----------------------------------------------------------
    priv_rows = [
        [
            ", ".join(f"<code>{q}</code>" for q in row.quasi_identifiers),
            num(row.classes, 0),
            pct(row.share_below_k),
            pct(row.share_unique),
        ]
        for row in result["privacy"].itertuples()
    ]
    priv_table = _table(
        [
            "Combinação de atributos",
            "Combinações distintas",
            f"Registros com k < {gov['privacy']['k_min']}",
            "Registros únicos (k = 1)",
        ],
        priv_rows,
        {1, 2, 3},
    )
    retention_rows = []
    for folder, days in gov["retention"].items():
        root = resolve_path(folder)
        files = (
            [p for p in root.rglob("*") if p.is_file() and p.name != ".gitkeep"]
            if root.exists()
            else []
        )
        n_expired = sum(1 for e in result["expired"] if root in e.path.parents)
        retention_rows.append(
            [
                f"<code>{folder}</code>",
                f"{num(days, 0)} dias",
                num(len(files), 0),
                num(n_expired, 0),
            ]
        )
    retention_table = _table(
        ["Pasta", "Prazo", "Arquivos hoje", "Vencidos"], retention_rows, {1, 2, 3}, 600
    )

    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    excluded = ", ".join(settings["model"]["excluded_features"])
    body = f"""  <p class="eyebrow">Governança do modelo de crédito</p>
  <h1>Relatório de governança</h1>
  <p class="lead">Por que o modelo degradou, se ele trata grupos de clientes de forma equilibrada, como explicar
  cada decisão e qual o risco de privacidade dos dados. Modelo <code>{escape(bundle.version)}</code>.</p>
  <section class="stats">{stats_html}</section>

  <h2>Análise causal da degradação em {month}</h2>
  <p class="section-note">Cada mecanismo do cenário foi ligado e desligado em todas as {2 ** len(by_mech)} combinações,
  com {causal["seeds"]} sementes cada ({num(causal["simulations"], 0)} simulações). O efeito de cada um é o seu valor de
  Shapley: a soma dos efeitos é exatamente a mudança total. Sem nenhum mecanismo, o AUC é
  {num(causal["baseline"]["auc"])}; com todos, {num(causal["full"]["auc"])}. As barras de erro são intervalos de 95%.</p>
  <div class="two">
    {_card("Efeito no AUC", "Negativo = o modelo passa a ordenar pior o risco.", effect_chart(by_mech, "auc", "AUC", "chart-causal-auc"))}
    {_card("Efeito na descalibração", "Positivo = a inadimplência real fica acima do que o modelo prevê.", effect_chart(by_mech, "calibration_gap", "real − score", "chart-causal-gap"))}
  </div>
  <div class="block" style="margin-top:16px">{mech_table}</div>
  <div class="block" style="margin-top:16px">{group_table}</div>
  <div class="callout"><p><b>Conclusão:</b> o data drift muda bastante os dados, mas não é ele que derruba o modelo:
  somado, até melhora o AUC, porque clientes com atrasos novos são fáceis de identificar. A degradação vem do
  concept drift: clientes em dia e muito endividados passam a inadimplir sem que nenhuma variável mude. O modelo
  não tem como ver esse risco, e só os rótulos revelam o problema.</p></div>

  <h3 style="margin:28px 0 8px">Prescrição: o que fazer com o modelo</h3>
  <p class="section-note">Metade do Pool de Produção simula a crise já observada (com rótulos) e a outra metade, a
  crise que continua. Média de {gov["retraining"]["repetitions"]} repetições.</p>
  {retrain_table}

  <h2>Equidade entre grupos</h2>
  <p class="section-note">Conjunto de teste da Referência, com o limiar do modelo em produção. <b>Impacto
  desproporcional</b>: menor taxa de aprovação ÷ maior (regra dos 4/5: ≥ {num(fair_cfg["disparate_impact_min"], 2)}).
  <b>Recusa indevida</b>: bons pagadores recusados; a diferença entre grupos mede a igualdade de oportunidade.
  Grupos com menos de {fair_cfg["min_group_size"]} clientes ficam fora dos indicadores. Atributos fora do modelo:
  <code>{escape(excluded)}</code>.</p>
  {ind_table}
  <div class="block" style="margin-top:16px">{groups_table}</div>

  <h3 style="margin:28px 0 8px">Mitigação: o que muda ao retirar atributos do modelo</h3>
  <p class="section-note">Cada variante foi treinada com {len(fair_cfg["seeds"])} sementes e comparada com a <b>mesma taxa
  de aprovação</b> do modelo em produção, para que nenhuma pareça mais justa só por recusar menos.</p>
  {var_table}

  <h3 style="margin:28px 0 8px">Equidade em produção</h3>
  <p class="section-note">Diferença de recusa indevida entre grupos, com as decisões tomadas em produção. As duas
  últimas colunas mostram quanto essa diferença oscila só por acaso, sem nenhum drift (intervalo central de 90% em
  30 sorteios do Pool). Num mês de {num(settings["simulation"]["batch_size"], 0)} clientes os grupos são pequenos e a
  oscilação é grande, então a equidade deve ser acompanhada em janelas de pelo menos 3 meses.</p>
  {prod_table}

  <h2>Explicabilidade das decisões</h2>
  <p class="section-note">Valores SHAP calculados pelo próprio XGBoost: a contribuição de cada variável soma
  exatamente o score de cada cliente. Base para atender ao pedido de revisão de decisão automatizada (LGPD, art. 20).
  Clientes de {month}.</p>
  <div class="two">
    {_card("Variáveis que mais pesam nas decisões", "Média do valor absoluto da contribuição, em " + month + ".", importance_chart(result["importance"]))}
  </div>
  <h3 style="margin:28px 0 8px">Exemplos de motivos de recusa</h3>
  <p class="section-note">Clientes recusados em {month}, com as três variáveis que mais aumentaram o risco de cada
  um. Para explicar qualquer cliente: <code>make explain BATCH=data/incoming/{month}.csv CLIENT=&lt;client_id&gt;</code>.</p>
  {reason_table}

  <h2>Privacidade</h2>
  <p class="section-note">O <code>client_id</code> é pseudonimizado, mas a combinação de atributos comuns pode identificar
  alguém. Um registro é k-anônimo quando pelo menos k clientes têm a mesma combinação. Dataset de Referência.</p>
  {priv_table}
  <h3 style="margin:28px 0 8px">Retenção</h3>
  <p class="section-note">Prazos em <code>config/settings.yaml</code>; <code>make retention</code> lista os vencidos e
  <code>--apply</code> apaga.</p>
  {retention_table}

  <footer>Gerado em {generated} · tabelas completas em CSV nesta pasta.</footer>"""
    return page("Relatório de Governança", body, EXTRA_STYLE)


def save(result: dict[str, Any]) -> Path:
    out = resolve_path(result["settings"]["paths"]["governance_reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    causal = result["causal"]
    causal["by_mechanism"].to_csv(out / "causal_by_mechanism.csv", float_format="%.6g")
    causal["by_group"].to_csv(out / "causal_by_group.csv", float_format="%.6g")
    result["retraining"].to_csv(out / "retraining.csv", index=False, float_format="%.6g")
    result["groups"].to_csv(out / "fairness_groups.csv", index=False, float_format="%.6g")
    result["indicators"].to_csv(out / "fairness_indicators.csv", index=False, float_format="%.6g")
    result["variants"].to_csv(out / "fairness_variants.csv", index=False, float_format="%.6g")
    result["production"].to_csv(out / "fairness_production.csv", index=False, float_format="%.6g")
    result["noise"].to_csv(out / "fairness_sampling_noise.csv", float_format="%.6g")
    result["importance"].rename("mean_abs_contribution").to_csv(
        out / "feature_importance.csv", float_format="%.6g"
    )
    reasons = [
        {"client_id": row[ID_COLUMN], "score": float(row["score"]), "rank": i + 1, **reason}
        for (_, row), items in zip(result["examples"].iterrows(), result["reasons"], strict=True)
        for i, reason in enumerate(items)
    ]
    pd.DataFrame(reasons).to_csv(
        out / "reason_codes_examples.csv", index=False, float_format="%.6g"
    )
    result["privacy"].to_csv(out / "reidentification_risk.csv", index=False, float_format="%.6g")
    (out / "causal_summary.json").write_text(
        json.dumps(
            {k: causal[k] for k in ("month", "seeds", "simulations", "baseline", "full")},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    path = out / "governance_report.html"
    path.write_text(render(result), encoding="utf-8")
    return path


def main() -> int:
    settings = load_settings()
    configure_logging(settings)
    result = build(settings)
    path = save(result)
    causal = result["causal"]
    by_group = causal["by_group"]
    log_event(
        logger,
        logging.INFO,
        "governanca",
        f"Relatório de governança gerado: {path}",
        month=causal["month"],
        auc_concept_drift=float(by_group.loc["concept_drift", "auc"]),
        auc_data_drift=float(by_group.loc["data_drift", "auc"]),
        fairness_attention=int((~result["indicators"]["passes_equal_opportunity"]).sum()),
    )

    print(f"\nAnálise causal ({causal['month']}, {causal['simulations']} simulações):")
    for row in causal["by_mechanism"].itertuples():
        print(f"  {row.label:<44} AUC {row.auc:+.4f}  real−score {row.calibration_gap:+.4f}")
    print("\nPrescrição:")
    for row in result["retraining"].itertuples():
        print(
            f"  {row.response:<11} AUC {row.auc:.4f}  real−score {row.calibration_gap:+.4f}  aprovação {row.approval_rate:.1%}"
        )
    print("\nEquidade (teste da Referência):")
    for row in result["indicators"].itertuples():
        print(
            f"  {ATTRIBUTE_LABELS[row.attribute]:<14} DI {row.disparate_impact:.3f}  "
            f"recusa indevida {row.equal_opportunity_gap:.1%}  mais recusado: {row.most_refused_group}"
        )
    print(f"\nRelatório: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
