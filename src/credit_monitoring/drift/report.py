"""Relatório consolidado de drift (HTML autocontido)."""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from typing import TYPE_CHECKING, Any

from credit_monitoring.data.columns import FEATURES
from credit_monitoring.reporting import (
    ALERT_ORDER,
    SEVERITY_STYLE,
    alert_pill,
    auc_chart,
    calibration_chart,
    num,
    page,
    pct,
)

if TYPE_CHECKING:
    from credit_monitoring.drift.monitor import MonthResult

EXTRA_STYLE = """
.heat { min-width:760px; }
.heat td.cell { text-align:center; font-variant-numeric:tabular-nums; font-size:13px; border:2px solid var(--card); }
.heat td.feat { white-space:nowrap; }
"""

BATCH_ISSUES = {"bloqueado": "Lote bloqueado pelo contrato", "falha": "Lote ilegível"}


def _heatmap(results: list[MonthResult], excluded: list[str]) -> str:
    head = "".join(f"<th class='num'>{r.month}</th>" for r in results)
    rows = []
    for feature in FEATURES:
        tag = "<span class='tag'>fora do modelo</span>" if feature in excluded else ""
        cells = []
        for r in results:
            row = r.feature_drift.set_index("feature").loc[feature]
            bg, fg = SEVERITY_STYLE[row["severity"]]
            tip = (
                f"{feature} em {r.month}: {row['severity']} · PSI {num(row['psi'])} · "
                f"KS p-valor {row['ks_pvalue']:.2g}"
            )
            cells.append(
                f"<td class='cell' style='background:{bg};color:{fg}' title='{escape(tip)}'>"
                f"{num(row['psi'], 2)}</td>"
            )
        rows.append(f"<tr><td class='feat'><code>{feature}</code>{tag}</td>{''.join(cells)}</tr>")
    legend = "".join(
        f"<span><i style='background:{bg};border:1px solid #c3c2b7'></i>{label}</span>"
        for label, (bg, _) in SEVERITY_STYLE.items()
    )
    return (
        "<div class='card table-wrap'><table class='heat'>"
        f"<thead><tr><th>Variável</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody>"
        f"</table></div><div class='legend'>{legend}</div>"
    )


def _overview(results: list[MonthResult]) -> str:
    rows = []
    for r in results:
        sev = r.feature_drift["severity"].value_counts()
        detail = " · ".join(
            f"{int(sev[s])} {s}" for s in ("significativo", "moderado", "leve") if s in sev
        )
        issues_html = "".join(
            f"<div class='desc'>{BATCH_ISSUES[b['status']]}: <code>{escape(b['file'])}</code></div>"
            for b in r.batches
            if b["status"] in BATCH_ISSUES
        )
        n_crit = sum(a.severity == "crítico" for a in r.alerts)
        rows.append(
            "<tr>"
            f"<td><b>{r.month}</b></td>"
            f"<td><b>{escape(r.title)}</b><div class='desc'>{escape(r.narrative)}</div>{issues_html}</td>"
            f"<td class='num'>{r.n_drifted}/23<div class='desc'>{detail or 'nenhuma'}</div></td>"
            f"<td class='num'>{num(r.prediction_psi)}</td>"
            f"<td class='num'>{num(r.performance['auc'])}</td>"
            f"<td class='num'>{pct(r.mean_score)}</td>"
            f"<td class='num'>{pct(r.performance['actual_default_rate'])}</td>"
            f"<td class='num'>{len(r.alerts)}<div class='desc'>{n_crit} crítico(s)</div></td>"
            "</tr>"
        )
    return (
        "<div class='card table-wrap'><table style='min-width:900px'><thead><tr>"
        "<th>Mês</th><th>Cenário</th><th class='num'>Variáveis com drift</th>"
        "<th class='num'>PSI do score</th><th class='num'>AUC</th><th class='num'>Score médio</th>"
        "<th class='num'>Inadimplência real</th><th class='num'>Alertas</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _alerts_table(results: list[MonthResult]) -> str:
    rows = []
    for r in results:
        for alert in sorted(r.alerts, key=lambda a: ALERT_ORDER[a.severity]):
            rows.append(
                "<tr>"
                f"<td><b>{alert.month}</b></td>"
                f"<td>{alert_pill(alert.severity)}</td>"
                f"<td>{escape(alert.category)}</td>"
                f"<td>{escape(alert.message)}</td>"
                "</tr>"
            )
    if not rows:
        return "<p>Nenhum alerta.</p>"
    return (
        "<div class='card table-wrap'><table style='min-width:720px'><thead><tr>"
        "<th>Mês</th><th>Severidade</th><th>Categoria</th><th>Alerta</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_drift_report(
    results: list[MonthResult], baseline: dict[str, float], settings: dict[str, Any]
) -> str:
    cfg = settings["drift"]
    excluded = settings["model"]["excluded_features"]
    all_alerts = [a for r in results for a in r.alerts]
    all_batches = [b for r in results for b in r.batches]
    worst = max(
        ((r.month, row.feature, row.psi) for r in results for row in r.feature_drift.itertuples()),
        key=lambda item: item[2],
    )
    stats = [
        (str(len(results)), "meses monitorados"),
        (
            str(sum(b["status"] != "aprovado" for b in all_batches)),
            "lotes barrados (bloqueados ou ilegíveis)",
        ),
        (str(sum(a.severity == "crítico" for a in all_alerts)), "alertas críticos"),
        (num(worst[2]), f"maior PSI ({worst[1]}, {worst[0]})"),
        (
            num(min(r.performance["auc"] for r in results)),
            f"menor AUC (referência {num(baseline['auc'])})",
        ),
    ]
    stats_html = "".join(
        f"<div class='stat'><b>{escape(v)}</b><span>{escape(k)}</span></div>" for v, k in stats
    )
    links = "".join(
        f"<li><a href='{escape(r.evidently_report)}'>{r.month} — {escape(r.title)}</a></li>"
        for r in results
    )
    months = [r.month for r in results]
    auc_html = auc_chart(
        months,
        [r.performance["auc"] for r in results],
        baseline["auc"],
        cfg["performance"]["auc_drop"],
    )
    calibration_html = calibration_chart(
        months,
        [r.mean_score for r in results],
        [r.performance["actual_default_rate"] for r in results],
    )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    body = f"""  <p class="eyebrow">Monitoramento do modelo de crédito em produção</p>
  <h1>Relatório de drift: {len(results)} meses simulados</h1>
  <p class="lead">Cada mês, o lote de produção é comparado com o conjunto de teste da Referência
  (mesma distribuição do treino, fora do ajuste do modelo). O PSI mede o tamanho da mudança e o
  teste KS do Evidently, se ela é estatisticamente significativa.</p>
  <section class="stats">{stats_html}</section>

  <h2>Linha do tempo</h2>
  <p class="section-note">Cenário econômico de cada mês e o que o monitoramento detectou.</p>
  {_overview(results)}

  <h2>Quais variáveis derivaram (PSI por mês)</h2>
  <p class="section-note">Cor pela severidade: <b>significativo</b> PSI ≥ {num(cfg["psi_significant"], 2)},
  <b>moderado</b> PSI ≥ {num(cfg["psi_moderate"], 2)}, <b>leve</b> quando só o KS detecta a mudança
  (p-valor &lt; {num(cfg["ks_alpha"], 2)}). Passe o mouse sobre uma célula para ver o p-valor do KS.</p>
  {_heatmap(results, excluded)}

  <h2>Desempenho do modelo</h2>
  <p class="section-note">Medido quando os rótulos chegam. Data drift nem sempre derruba o modelo; o que
  derruba é a mudança na relação entre as variáveis e a inadimplência (concept drift).</p>
  <div class="charts">
    <div class="card chart"><h3>AUC por mês</h3>
      <p>Capacidade de separar bons e maus pagadores.</p>{auc_html}</div>
    <div class="card chart"><h3>Score médio x inadimplência real</h3>
      <p>Quando as linhas se separam, o modelo está descalibrado.</p>{calibration_html}</div>
  </div>

  <h2>Alertas gerados</h2>
  <p class="section-note">Também disponíveis em <code>alerts.json</code>. As tabelas completas estão em
  <code>feature_drift.csv</code> e <code>monthly_summary.csv</code>. A saúde operacional do pipeline está no
  <a href="../monitoring/dashboard.html">dashboard de monitoramento</a>.</p>
  {_alerts_table(results)}

  <h2>Relatórios do Evidently por mês</h2>
  <p class="section-note">Drift por variável (KS) e desempenho de classificação, comparados com a referência.</p>
  <div class="card" style="padding:14px 16px"><ul class="links">{links}</ul></div>

  <footer>Gerado em {generated} · limiares em <code>config/settings.yaml</code> (seção <code>drift</code>).</footer>"""
    return page("Relatório de Drift", body, EXTRA_STYLE)
