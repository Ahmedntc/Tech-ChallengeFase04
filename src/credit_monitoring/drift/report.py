"""Relatório consolidado de drift (HTML autocontido, com o Plotly embutido para abrir offline)."""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from credit_monitoring.data.columns import FEATURES

if TYPE_CHECKING:
    from credit_monitoring.drift.monitor import MonthResult

# Paleta (referência de dataviz): série 1 azul, série 2 laranja, tinta e grade neutras.
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
CRITICAL = "#d03b3b"

# Escala ordinal (um matiz, claro → escuro) para a severidade do drift.
SEVERITY_STYLE = {
    "estável": ("#f0efec", MUTED),
    "leve": ("#86b6ef", INK),
    "moderado": ("#2a78d6", "#ffffff"),
    "significativo": ("#104281", "#ffffff"),
}
ALERT_STYLE = {
    "crítico": ("✖", "crit"),
    "atenção": ("▲", "warn"),
    "info": ("●", "info"),
}

STYLE = """
:root { color-scheme: light; --bg:#f9f9f7; --card:#fcfcfb; --text:#0b0b0b; --muted:#52514e; --faint:#898781;
        --line:#e1e0d9; --crit:#b42318; --crit-bg:#fde8e7; --warn:#7a4f00; --warn-bg:#fff1cc;
        --info:#1c5cab; --info-bg:#e6f0fc; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:32px 16px 56px; }
.eyebrow { margin:0; color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.06em; }
h1 { margin:4px 0 6px; font-size:28px; line-height:1.2; }
h2 { margin:40px 0 6px; font-size:20px; }
.lead { margin:0; color:var(--muted); max-width:820px; }
.section-note { margin:0 0 14px; color:var(--muted); font-size:14px; max-width:860px; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:24px 0 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.stat b { display:block; font-size:26px; font-variant-numeric:tabular-nums; line-height:1.2; }
.stat span { color:var(--muted); font-size:13px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; }
.table-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; }
th, td { padding:9px 12px; text-align:left; vertical-align:top; border-bottom:1px solid var(--line); }
th { font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; font-weight:600; }
tr:last-child td { border-bottom:none; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.desc { color:var(--muted); font-size:13px; }
.heat { min-width:760px; }
.heat td.cell { text-align:center; font-variant-numeric:tabular-nums; font-size:13px; border:2px solid var(--card); }
.heat td.feat { white-space:nowrap; }
.tag { display:inline-block; margin-left:6px; padding:1px 6px; border-radius:6px; font-size:11px;
       background:var(--line); color:var(--muted); }
.legend { display:flex; flex-wrap:wrap; gap:16px; margin:12px 0 0; font-size:13px; color:var(--muted); }
.legend i { display:inline-block; width:14px; height:14px; border-radius:4px; vertical-align:-2px; margin-right:6px; }
.charts { display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:16px; }
.chart { padding:12px 8px 4px; min-width:0; }
.chart h3 { margin:4px 12px 0; font-size:15px; }
.chart p { margin:2px 12px 0; color:var(--muted); font-size:13px; }
.pill { display:inline-flex; gap:6px; align-items:center; white-space:nowrap; padding:2px 8px; border-radius:6px;
        font-size:12px; font-weight:600; }
.pill.crit { background:var(--crit-bg); color:var(--crit); }
.pill.warn { background:var(--warn-bg); color:var(--warn); }
.pill.info { background:var(--info-bg); color:var(--info); }
.pill.ok { background:#e3f4ea; color:#1f7a4d; }
ul.links { margin:0; padding-left:18px; }
a { color:#1c5cab; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
footer { margin-top:32px; color:var(--muted); font-size:13px; }
@media (max-width: 520px) { .charts { grid-template-columns:1fr; } }
"""


def _pct(value: float) -> str:
    return f"{value:.1%}".replace(".", ",")


def _num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _base_layout(y_title: str, y_format: str) -> dict[str, Any]:
    return {
        "template": "none",
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {
            "family": "system-ui, -apple-system, Segoe UI, sans-serif",
            "color": INK_2,
            "size": 13,
        },
        "margin": {"l": 56, "r": 24, "t": 16, "b": 40},
        "height": 320,
        "separators": ",.",
        "hovermode": "x unified",
        "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": GRID, "font": {"color": INK}},
        "xaxis": {"showgrid": False, "linecolor": AXIS, "ticks": "", "fixedrange": True},
        "yaxis": {
            "title": {"text": y_title, "font": {"size": 12, "color": MUTED}},
            "gridcolor": GRID,
            "zeroline": False,
            "tickformat": y_format,
            "fixedrange": True,
        },
        "showlegend": False,
    }


def _figure_html(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )


def auc_chart(results: list[MonthResult], baseline: dict[str, float], auc_drop: float) -> str:
    months = [r.month for r in results]
    aucs = [r.performance["auc"] for r in results]
    limit = baseline["auc"] - auc_drop
    fig = go.Figure(
        go.Scatter(
            x=months,
            y=aucs,
            mode="lines+markers",
            name="AUC",
            line={"color": SERIES_1, "width": 2},
            marker={"size": 8, "color": SERIES_1, "line": {"color": SURFACE, "width": 2}},
            hovertemplate="AUC %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=baseline["auc"],
        line={"color": MUTED, "width": 1, "dash": "dash"},
        annotation={
            "text": f"Referência {_num(baseline['auc'])}",
            "font": {"color": INK_2, "size": 12},
        },
        annotation_position="top left",
    )
    fig.add_hline(
        y=limit,
        line={"color": CRITICAL, "width": 1, "dash": "dot"},
        annotation={
            "text": f"Limite de alerta {_num(limit)}",
            "font": {"color": CRITICAL, "size": 12},
        },
        annotation_position="bottom left",
    )
    layout = _base_layout("AUC", ".2f")
    layout["yaxis"]["range"] = [
        min(min(aucs), limit) - 0.04,
        max(max(aucs), baseline["auc"]) + 0.03,
    ]
    fig.update_layout(**layout)
    return _figure_html(fig, "chart-auc")


def calibration_chart(results: list[MonthResult]) -> str:
    months = [r.month for r in results]
    series = [
        ("Score médio (previsto)", [r.mean_score for r in results], SERIES_1),
        ("Inadimplência real", [r.performance["actual_default_rate"] for r in results], SERIES_2),
    ]
    fig = go.Figure()
    for name, values, color in series:
        fig.add_trace(
            go.Scatter(
                x=months,
                y=values,
                mode="lines+markers",
                name=name,
                line={"color": color, "width": 2},
                marker={"size": 8, "color": color, "line": {"color": SURFACE, "width": 2}},
                hovertemplate=f"{name}: %{{y:.1%}}<extra></extra>",
            )
        )
        # Rótulo direto no último ponto.
        fig.add_annotation(
            x=months[-1],
            y=values[-1],
            text=_pct(values[-1]),
            showarrow=False,
            xanchor="left",
            xshift=10,
            font={"color": INK_2, "size": 12},
        )
    layout = _base_layout("Taxa", ".0%")
    layout["showlegend"] = True
    layout["legend"] = {"orientation": "h", "x": 0, "y": 1.12, "font": {"size": 12, "color": INK_2}}
    layout["margin"]["t"] = 36
    layout["margin"]["r"] = 56
    layout["yaxis"]["rangemode"] = "tozero"
    fig.update_layout(**layout)
    return _figure_html(fig, "chart-calibration")


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
                f"{feature} em {r.month}: {row['severity']} · PSI {_num(row['psi'])} · "
                f"KS p-valor {row['ks_pvalue']:.2g}"
            )
            cells.append(
                f"<td class='cell' style='background:{bg};color:{fg}' title='{escape(tip)}'>"
                f"{_num(row['psi'], 2)}</td>"
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
        blocked = [b["file"] for b in r.batches if b["status"] == "bloqueado"]
        blocked_html = (
            f"<div class='desc'>Lote bloqueado: <code>{escape(', '.join(blocked))}</code></div>"
            if blocked
            else ""
        )
        n_crit = sum(a.severity == "crítico" for a in r.alerts)
        rows.append(
            "<tr>"
            f"<td><b>{r.month}</b></td>"
            f"<td><b>{escape(r.title)}</b><div class='desc'>{escape(r.narrative)}</div>{blocked_html}</td>"
            f"<td class='num'>{r.n_drifted}/23<div class='desc'>{detail or 'nenhuma'}</div></td>"
            f"<td class='num'>{_num(r.prediction_psi)}</td>"
            f"<td class='num'>{_num(r.performance['auc'])}</td>"
            f"<td class='num'>{_pct(r.mean_score)}</td>"
            f"<td class='num'>{_pct(r.performance['actual_default_rate'])}</td>"
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
    order = {"crítico": 0, "atenção": 1, "info": 2}
    rows = []
    for r in results:
        for alert in sorted(r.alerts, key=lambda a: order[a.severity]):
            icon, css = ALERT_STYLE[alert.severity]
            rows.append(
                "<tr>"
                f"<td><b>{alert.month}</b></td>"
                f"<td><span class='pill {css}'>{icon} {alert.severity}</span></td>"
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
    worst = max(
        ((r.month, row.feature, row.psi) for r in results for row in r.feature_drift.itertuples()),
        key=lambda item: item[2],
    )
    stats = [
        (str(len(results)), "meses monitorados"),
        (
            str(sum(b["status"] == "bloqueado" for r in results for b in r.batches)),
            "lotes bloqueados",
        ),
        (str(sum(a.severity == "crítico" for a in all_alerts)), "alertas críticos"),
        (_num(worst[2]), f"maior PSI ({worst[1]}, {worst[0]})"),
        (
            _num(min(r.performance["auc"] for r in results)),
            f"menor AUC (referência {_num(baseline['auc'])})",
        ),
    ]
    stats_html = "".join(
        f"<div class='stat'><b>{escape(v)}</b><span>{escape(k)}</span></div>" for v, k in stats
    )
    links = "".join(
        f"<li><a href='{escape(r.evidently_report)}'>{r.month} — {escape(r.title)}</a></li>"
        for r in results
    )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Relatório de Drift</title>
<script>{get_plotlyjs()}</script>
<style>{STYLE}</style>
</head>
<body>
<main>
  <p class="eyebrow">Monitoramento do modelo de crédito em produção</p>
  <h1>Relatório de drift: 6 meses simulados</h1>
  <p class="lead">Cada mês, o lote de produção é comparado com o conjunto de teste da Referência
  (mesma distribuição do treino, fora do ajuste do modelo). O PSI mede o tamanho da mudança e o
  teste KS do Evidently, se ela é estatisticamente significativa.</p>
  <section class="stats">{stats_html}</section>

  <h2>Linha do tempo</h2>
  <p class="section-note">Cenário econômico de cada mês e o que o monitoramento detectou.</p>
  {_overview(results)}

  <h2>Quais variáveis derivaram (PSI por mês)</h2>
  <p class="section-note">Cor pela severidade: <b>significativo</b> PSI ≥ {_num(cfg["psi_significant"], 2)},
  <b>moderado</b> PSI ≥ {_num(cfg["psi_moderate"], 2)}, <b>leve</b> quando só o KS detecta a mudança
  (p-valor &lt; {_num(cfg["ks_alpha"], 2)}). Passe o mouse sobre uma célula para ver o p-valor do KS.</p>
  {_heatmap(results, excluded)}

  <h2>Desempenho do modelo</h2>
  <p class="section-note">Medido quando os rótulos chegam. Data drift nem sempre derruba o modelo; o que
  derruba é a mudança na relação entre as variáveis e a inadimplência (concept drift).</p>
  <div class="charts">
    <div class="card chart"><h3>AUC por mês</h3>
      <p>Capacidade de separar bons e maus pagadores.</p>{auc_chart(results, baseline, cfg["performance"]["auc_drop"])}</div>
    <div class="card chart"><h3>Score médio x inadimplência real</h3>
      <p>Quando as linhas se separam, o modelo está descalibrado.</p>{calibration_chart(results)}</div>
  </div>

  <h2>Alertas gerados</h2>
  <p class="section-note">Também disponíveis em <code>alerts.json</code>. As tabelas completas estão em
  <code>feature_drift.csv</code> e <code>monthly_summary.csv</code>.</p>
  {_alerts_table(results)}

  <h2>Relatórios do Evidently por mês</h2>
  <p class="section-note">Drift por variável (KS) e desempenho de classificação, comparados com a referência.</p>
  <div class="card" style="padding:14px 16px"><ul class="links">{links}</ul></div>

  <footer>Gerado em {generated} · limiares em <code>config/settings.yaml</code> (seção <code>drift</code>).</footer>
</main>
<script>
  // Os gráficos são desenhados enquanto a página carrega, antes de a grade definir a largura
  // final dos cards; ajusta o tamanho de cada um quando o layout termina.
  window.addEventListener("load", () => {{
    document.querySelectorAll(".plotly-graph-div").forEach((div) => Plotly.Plots.resize(div));
  }});
</script>
</body>
</html>
"""
