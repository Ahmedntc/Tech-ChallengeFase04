"""Peças compartilhadas pelos relatórios HTML: paleta, estilo base e gráficos Plotly.

Os relatórios são arquivos únicos e autocontidos: o Plotly vai embutido para que
abram offline, direto do repositório.
"""

from __future__ import annotations

from html import escape
from typing import Any

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

# Paleta (referência de dataviz): séries categóricas em ordem fixa, tinta e grade neutras.
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SERIES_1, SERIES_2 = SERIES[0], SERIES[1]
CRITICAL = "#d03b3b"
WARNING = "#fab219"

# Escala ordinal (um matiz, claro → escuro) para a severidade do drift: (fundo, texto).
SEVERITY_STYLE = {
    "estável": ("#f0efec", MUTED),
    "leve": ("#86b6ef", INK),
    "moderado": ("#2a78d6", "#ffffff"),
    "significativo": ("#104281", "#ffffff"),
}
# Severidade do alerta: (ícone, classe CSS). O ícone garante que a cor nunca vá sozinha.
ALERT_STYLE = {
    "crítico": ("✖", "crit"),
    "atenção": ("▲", "warn"),
    "info": ("●", "info"),
}
ALERT_ORDER = {"crítico": 0, "atenção": 1, "info": 2}

BASE_STYLE = """
:root { color-scheme: light; --bg:#f9f9f7; --card:#fcfcfb; --text:#0b0b0b; --muted:#52514e; --faint:#898781;
        --line:#e1e0d9; --crit:#b42318; --crit-bg:#fde8e7; --warn:#7a4f00; --warn-bg:#fff1cc;
        --info:#1c5cab; --info-bg:#e6f0fc; --ok:#1f7a4d; --ok-bg:#e3f4ea; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text);
       font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
main { max-width:1180px; margin:0 auto; padding:32px 16px 56px; }
.eyebrow { margin:0; color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.06em; }
h1 { margin:4px 0 6px; font-size:28px; line-height:1.2; }
h2 { margin:40px 0 6px; font-size:20px; }
.lead { margin:0; color:var(--muted); max-width:820px; }
.section-note { margin:0 0 14px; color:var(--muted); font-size:14px; max-width:860px; }
.stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin:24px 0 0; }
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
.pill.ok { background:var(--ok-bg); color:var(--ok); }
ul.links { margin:0; padding-left:18px; }
a { color:#1c5cab; }
code { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
footer { margin-top:32px; color:var(--muted); font-size:13px; }
@media (max-width: 520px) { .charts { grid-template-columns:1fr; } }
"""

# Os gráficos são desenhados enquanto a página carrega, antes de a grade definir a largura
# final dos cards; este script ajusta o tamanho de cada um quando o layout termina.
RESIZE_SCRIPT = """<script>
  window.addEventListener("load", () => {
    document.querySelectorAll(".plotly-graph-div").forEach((div) => Plotly.Plots.resize(div));
  });
</script>"""


def pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}%}".replace(".", ",")


def num(value: float, digits: int = 3) -> str:
    return f"{value:,.{digits}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def alert_pill(severity: str) -> str:
    icon, css = ALERT_STYLE[severity]
    return f"<span class='pill {css}'>{icon} {escape(severity)}</span>"


def page(title: str, body: str, extra_style: str = "") -> str:
    """Documento HTML completo, com o Plotly embutido."""
    return f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<script>{get_plotlyjs()}</script>
<style>{BASE_STYLE}{extra_style}</style>
</head>
<body>
<main>
{body}
</main>
{RESIZE_SCRIPT}
</body>
</html>
"""


def base_layout(y_title: str, y_format: str) -> dict[str, Any]:
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


def with_legend(layout: dict[str, Any]) -> dict[str, Any]:
    """Legenda horizontal acima do gráfico (obrigatória com duas ou mais séries)."""
    layout["showlegend"] = True
    layout["legend"] = {
        "orientation": "h",
        "x": 0,
        "y": 1.12,
        "font": {"size": 12, "color": INK_2},
        "traceorder": "normal",
    }
    layout["margin"]["t"] = 36
    return layout


def figure_html(fig: go.Figure, div_id: str) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config={"displayModeBar": False, "responsive": True},
    )


def line_trace(name: str, x: list[Any], y: list[float], color: str, hover: str) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=name,
        line={"color": color, "width": 2},
        marker={"size": 8, "color": color, "line": {"color": SURFACE, "width": 2}},
        hovertemplate=f"{hover}<extra></extra>",
    )


def threshold_line(
    fig: go.Figure, y: float, text: str, color: str, dash: str, position: str
) -> None:
    fig.add_hline(
        y=y,
        line={"color": color, "width": 1, "dash": dash},
        annotation={"text": text, "font": {"color": color, "size": 12}},
        annotation_position=position,
    )


def auc_chart(
    months: list[str],
    aucs: list[float],
    baseline_auc: float,
    auc_drop: float,
    div_id: str = "chart-auc",
) -> str:
    """AUC por mês, com a referência e o limite de alerta."""
    limit = baseline_auc - auc_drop
    fig = go.Figure(line_trace("AUC", months, aucs, SERIES_1, "AUC %{y:.3f}"))
    threshold_line(fig, baseline_auc, f"Referência {num(baseline_auc)}", INK_2, "dash", "top left")
    threshold_line(fig, limit, f"Limite de alerta {num(limit)}", CRITICAL, "dot", "bottom left")
    layout = base_layout("AUC", ".2f")
    layout["yaxis"]["range"] = [min(min(aucs), limit) - 0.04, max(max(aucs), baseline_auc) + 0.03]
    fig.update_layout(**layout)
    return figure_html(fig, div_id)


def calibration_chart(
    months: list[str],
    mean_scores: list[float],
    actual_rates: list[float],
    div_id: str = "chart-calibration",
) -> str:
    """Score médio previsto x inadimplência real: quando se separam, há descalibração."""
    series = [
        ("Score médio (previsto)", mean_scores, SERIES_1),
        ("Inadimplência real", actual_rates, SERIES_2),
    ]
    fig = go.Figure()
    for name, values, color in series:
        fig.add_trace(line_trace(name, months, values, color, f"{name}: %{{y:.1%}}"))
        # Rótulo direto no último ponto.
        fig.add_annotation(
            x=months[-1],
            y=values[-1],
            text=pct(values[-1]),
            showarrow=False,
            xanchor="left",
            xshift=10,
            font={"color": INK_2, "size": 12},
        )
    layout = with_legend(base_layout("Taxa", ".0%"))
    layout["margin"]["r"] = 56
    layout["yaxis"]["rangemode"] = "tozero"
    fig.update_layout(**layout)
    return figure_html(fig, div_id)
