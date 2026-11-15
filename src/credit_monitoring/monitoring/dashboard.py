"""Dashboard de monitoramento (HTML autocontido), montado a partir do log centralizado.

Lê ``logs/pipeline.jsonl``, seleciona uma execução do pipeline (a mais recente, por
padrão) e mostra em uma página só:

- **Saúde do pipeline:** status de cada tarefa por mês, volume recebido e rejeitado,
  incidentes de ingestão e duração por etapa.
- **Infraestrutura:** latência, CPU, memória e vazão por tarefa.
- **Saúde do modelo:** AUC, calibração, drift das variáveis e do score ao longo dos meses.
- **Alertas** e **histórico de execuções**.

Uso: ``python -m credit_monitoring.monitoring.dashboard [--run-id ID]``
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from credit_monitoring.config import load_settings, resolve_path
from credit_monitoring.reporting import (
    ALERT_ORDER,
    CRITICAL,
    MUTED,
    SERIES,
    SEVERITY_STYLE,
    SURFACE,
    WARNING,
    alert_pill,
    auc_chart,
    base_layout,
    calibration_chart,
    figure_html,
    line_trace,
    num,
    page,
    pct,
    threshold_line,
    with_legend,
)

TASK_LABELS = {
    "verificar_dataset": "Verificar dataset",
    "preparar_dados": "Preparar dados (transformação)",
    "treinar_modelo": "Treinar modelo",
    "simular_producao": "Simular produção",
    "ingestao": "Ingestão (contrato de dados)",
    "predicao": "Predição",
    "avaliacao_desempenho": "Avaliação de desempenho",
    "deteccao_drift": "Detecção de drift",
    "gerar_relatorios": "Gerar relatórios",
    "registrar_mlflow": "Registrar no MLflow",
}
# Tarefas que rodam a cada mês, na ordem (fixa) em que recebem as cores das séries.
MONTHLY_TASKS = {
    "deteccao_drift": "Drift",
    "ingestao": "Ingestão",
    "predicao": "Predição",
    "avaliacao_desempenho": "Desempenho",
}
GRID_TASKS = ["ingestao", "predicao", "avaliacao_desempenho", "deteccao_drift"]
STATUS_PILL = {
    "sucesso": ("ok", "✔"),
    "bloqueado": ("warn", "▲"),
    "falha": ("crit", "✖"),
}
HEALTH_LABEL = {"ok": ("✔", "Saudável"), "warn": ("▲", "Atenção"), "crit": ("✖", "Crítico")}

EXTRA_STYLE = """
.health { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; margin:24px 0 0; }
.health .tile { background:var(--card); border:1px solid var(--line); border-left-width:6px; border-radius:10px; padding:14px 16px; }
.health .tile.ok { border-left-color:#1f7a4d; } .health .tile.warn { border-left-color:#fab219; }
.health .tile.crit { border-left-color:#d03b3b; }
.health h3 { margin:0 0 6px; font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }
.health .state { font-size:20px; font-weight:700; }
.health .ok .state { color:var(--ok); } .health .warn .state { color:var(--warn); } .health .crit .state { color:var(--crit); }
.health p { margin:6px 0 0; color:var(--muted); font-size:14px; }
.stat small { display:block; color:var(--faint); font-size:12px; margin-top:2px; }
.grid td { vertical-align:middle; }
.grid .pill { margin:1px 0; }
.grid .dur { color:var(--faint); font-size:12px; margin-left:6px; white-space:nowrap; }
h2 + .section-note { margin-top:0; }
.block { margin-top:16px; }
.block > h3 { margin:0 0 8px; font-size:15px; }
"""


@dataclass
class RunData:
    """Tudo o que o log registrou sobre uma execução do pipeline."""

    run_id: str
    started: dict[str, Any]
    finished: dict[str, Any] | None
    baseline: dict[str, Any] | None
    tasks: pd.DataFrame
    months: pd.DataFrame
    alerts: list[dict[str, Any]]
    events: list[dict[str, Any]]

    @property
    def status(self) -> str:
        return (self.finished or {}).get("status", "interrompida")


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Lê o log centralizado. Linhas truncadas (escrita interrompida) são ignoradas."""
    path = Path(path)
    events = []
    if not path.exists():
        return events
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def list_runs(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Resumo de cada execução do pipeline registrada no log, da mais recente para a mais antiga."""
    runs: dict[str, dict[str, Any]] = {}
    for event in events:
        run_id = event.get("run_id")
        kind = event.get("event")
        if kind == "run_started":
            runs[run_id] = {
                "run_id": run_id,
                "started_at": event["ts"],
                "status": "interrompida",
                "duration_seconds": None,
                "tasks": 0,
                "tasks_failed": 0,
                "tasks_blocked": 0,
                "critical_alerts": 0,
            }
        elif run_id in runs:
            if kind == "run_finished":
                runs[run_id].update(
                    {
                        key: event[key]
                        for key in (
                            "status",
                            "duration_seconds",
                            "tasks",
                            "tasks_failed",
                            "tasks_blocked",
                        )
                    }
                )
            elif kind == "alerta" and event.get("severity") == "crítico":
                runs[run_id]["critical_alerts"] += 1
    return sorted(runs.values(), key=lambda run: run["started_at"], reverse=True)


def collect_run(events: list[dict[str, Any]], run_id: str | None = None) -> RunData:
    """Reúne os eventos de uma execução (a mais recente, se ``run_id`` não for informado)."""
    runs = list_runs(events)
    if not runs:
        raise ValueError("Nenhuma execução do pipeline encontrada no log. Rode `make pipeline`.")
    run_id = run_id or runs[0]["run_id"]
    own = [e for e in events if e.get("run_id") == run_id]
    if not any(e.get("event") == "run_started" for e in own):
        raise ValueError(f"Execução {run_id} não encontrada no log.")

    def of(kind: str) -> list[dict[str, Any]]:
        return [e for e in own if e.get("event") == kind]

    task_columns = [
        "task",
        "stage",
        "month",
        "batch",
        "status",
        "started_at",
        "duration_seconds",
        "cpu_seconds",
        "memory_rss_mb",
        "rows_in",
        "rows_out",
        "rows_rejected",
        "detail",
        "error",
    ]
    return RunData(
        run_id=run_id,
        started=of("run_started")[0],
        finished=(of("run_finished") or [None])[-1],
        baseline=(of("baseline_referencia") or [None])[-1],
        tasks=pd.DataFrame(of("task_finished"), columns=task_columns),
        months=pd.DataFrame(of("resumo_mensal")),
        alerts=of("alerta"),
        events=own,
    )


def plural(count: int, singular: str, plural_form: str) -> str:
    return f"{count} {singular if count == 1 else plural_form}"


def health(run: RunData, limits: dict[str, Any]) -> list[dict[str, str]]:
    """Semáforo de saúde em três frentes: pipeline, infraestrutura e modelo."""
    tasks = run.tasks
    n_failed = int((tasks["status"] == "falha").sum())
    n_blocked = int((tasks["status"] == "bloqueado").sum())
    if run.status != "sucesso":
        error = (run.finished or {}).get("error") or "sem evento de encerramento no log"
        pipeline = ("crit", f"A execução terminou como {run.status}: {error}")
    elif n_failed or n_blocked:
        pipeline = (
            "warn",
            f"{plural(n_blocked, 'lote bloqueado', 'lotes bloqueados')} pelo contrato e "
            f"{plural(n_failed, 'falha', 'falhas')} de leitura, com os arquivos em quarentena. "
            "Nenhum dado inválido chegou ao modelo.",
        )
    else:
        pipeline = ("ok", "Todas as tarefas concluídas com sucesso.")

    infra_alerts = [a for a in run.alerts if a["category"] == "infraestrutura"]
    slowest = float(tasks["duration_seconds"].max()) if len(tasks) else 0.0
    peak = float(tasks["memory_rss_mb"].max()) if len(tasks) else 0.0
    usage = (
        f"Tarefa mais lenta: {num(slowest, 1)} s (limite {limits['max_task_seconds']} s). "
        f"Pico de memória: {num(peak, 0)} MB (limite {num(limits['max_memory_mb'], 0)} MB)."
    )
    infra = (
        ("warn", f"{plural(len(infra_alerts), 'alerta', 'alertas')} de limite excedido. {usage}")
        if infra_alerts
        else ("ok", usage)
    )

    if run.months.empty:
        model = ("warn", "Nenhum mês monitorado nesta execução.")
    else:
        last = run.months.iloc[-1]
        last_alerts = [a for a in run.alerts if a["month"] == last["month"]]
        reference = f" (referência {num(run.baseline['auc'])})" if run.baseline else ""
        facts = (
            f"{last['month']}: AUC {num(last['auc'])}{reference}, inadimplência real de "
            f"{pct(last['actual_default_rate'])} contra score médio de {pct(last['mean_score'])}."
        )
        degraded = [
            a for a in last_alerts if a["category"] == "desempenho" and a["severity"] == "crítico"
        ]
        drifting = [a for a in last_alerts if "drift" in a["category"] and a["severity"] != "info"]
        if degraded:
            model = ("crit", f"{facts} Modelo degradado: retreino recomendado.")
        elif drifting:
            model = ("warn", f"{facts} Há drift nos dados, mas o desempenho segue no limite.")
        else:
            model = ("ok", facts)

    return [
        {"name": name, "level": level, "reason": reason}
        for name, (level, reason) in (
            ("Pipeline de dados", pipeline),
            ("Infraestrutura", infra),
            ("Modelo", model),
        )
    ]


# --------------------------------------------------------------------------- gráficos


def _stacked_bars(
    x: list[str],
    series: list[tuple[str, list[float], str]],
    y_title: str,
    y_format: str,
    hover: str,
    div_id: str,
) -> str:
    fig = go.Figure()
    for name, values, color in series:
        fig.add_trace(
            go.Bar(
                x=x,
                y=values,
                name=name,
                marker={"color": color, "line": {"color": SURFACE, "width": 2}},
                hovertemplate=f"{name}: {hover}<extra></extra>",
            )
        )
    layout = with_legend(base_layout(y_title, y_format))
    layout.update(barmode="stack", bargap=0.5, barcornerradius=4)
    layout["yaxis"]["rangemode"] = "tozero"
    fig.update_layout(**layout)
    return figure_html(fig, div_id)


def volume_chart(tasks: pd.DataFrame, months: list[str]) -> str:
    ingestion = tasks[tasks["task"] == "ingestao"]
    by_month = ingestion.groupby("month")[["rows_out", "rows_rejected"]].sum()
    approved = [float(by_month["rows_out"].get(m, 0)) for m in months]
    rejected = [float(by_month["rows_rejected"].get(m, 0)) for m in months]
    return _stacked_bars(
        months,
        [("Aprovados", approved, SERIES[0]), ("Rejeitados", rejected, SERIES[1])],
        "Registros",
        ",d",
        "%{y:,d}",
        "chart-volume",
    )


def duration_chart(tasks: pd.DataFrame, months: list[str]) -> str:
    by = tasks.groupby(["month", "task"])["duration_seconds"].sum()
    series = [
        (label, [float(by.get((m, task), 0)) for m in months], SERIES[i])
        for i, (task, label) in enumerate(MONTHLY_TASKS.items())
    ]
    return _stacked_bars(months, series, "Segundos", ".1f", "%{y:.2f} s", "chart-duration")


def memory_chart(tasks: pd.DataFrame, months: list[str]) -> str:
    peak = tasks.groupby("month")["memory_rss_mb"].max()
    values = [float(peak.get(m, 0)) for m in months]
    fig = go.Figure(line_trace("Memória", months, values, SERIES[0], "%{y:.0f} MB"))
    layout = base_layout("MB", ",d")
    layout["yaxis"]["rangemode"] = "tozero"
    fig.update_layout(**layout)
    return figure_html(fig, "chart-memory")


def prediction_latency_chart(tasks: pd.DataFrame, months: list[str]) -> str:
    """Latência da predição em milissegundos por 1.000 registros."""
    prediction = tasks[tasks["task"] == "predicao"].set_index("month")
    values = [
        float(prediction.loc[m, "duration_seconds"] / prediction.loc[m, "rows_in"] * 1e6)
        if m in prediction.index
        else 0.0
        for m in months
    ]
    fig = go.Figure(line_trace("Latência", months, values, SERIES[0], "%{y:.1f} ms"))
    layout = base_layout("ms por 1.000 registros", ".0f")
    layout["yaxis"]["rangemode"] = "tozero"
    fig.update_layout(**layout)
    return figure_html(fig, "chart-latency")


def severity_chart(months: pd.DataFrame) -> str:
    series = [
        (label, months[f"n_{label}"].astype(float).tolist(), SEVERITY_STYLE[label][0])
        for label in ("leve", "moderado", "significativo")
    ]
    return _stacked_bars(
        months["month"].tolist(), series, "Variáveis", "d", "%{y:d}", "chart-severity"
    )


def prediction_psi_chart(months: pd.DataFrame, limit: float) -> str:
    values = months["prediction_psi"].astype(float).tolist()
    fig = go.Figure(
        line_trace("PSI do score", months["month"].tolist(), values, SERIES[0], "PSI %{y:.3f}")
    )
    threshold_line(fig, limit, f"Limite de alerta {num(limit, 2)}", CRITICAL, "dot", "top left")
    layout = base_layout("PSI", ".2f")
    layout["yaxis"]["range"] = [0, max(max(values), limit) * 1.25]
    fig.update_layout(**layout)
    return figure_html(fig, "chart-prediction-psi")


def alerts_chart(alerts: list[dict[str, Any]], months: list[str]) -> str:
    frame = pd.DataFrame(alerts, columns=["month", "severity"])
    counts = frame.groupby(["month", "severity"]).size()
    colors = {"crítico": CRITICAL, "atenção": WARNING, "info": MUTED}
    series = [
        (severity, [float(counts.get((m, severity), 0)) for m in months], colors[severity])
        for severity in ("crítico", "atenção", "info")
    ]
    return _stacked_bars(months, series, "Alertas", "d", "%{y:d}", "chart-alerts")


def alerts_by_category_chart(alerts: list[dict[str, Any]]) -> str:
    """Quantidade de alertas por categoria (barras horizontais, uma série)."""
    counts = pd.Series([a["category"] for a in alerts]).value_counts().sort_values()
    fig = go.Figure(
        go.Bar(
            x=counts.to_numpy(),
            y=counts.index.tolist(),
            orientation="h",
            marker={"color": SERIES[0]},
            text=counts.to_numpy(),
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x} alerta(s)<extra></extra>",
        )
    )
    layout = base_layout("", "d")
    layout.update(bargap=0.5, barcornerradius=4, hovermode="closest")
    layout["margin"]["l"] = 150
    layout["xaxis"].update(showgrid=True, gridcolor="#e1e0d9", rangemode="tozero", tickformat="d")
    layout["yaxis"].update(showgrid=False)
    fig.update_layout(**layout)
    return figure_html(fig, "chart-alert-categories")


# --------------------------------------------------------------------------- tabelas


def _pill(status: str, text: str) -> str:
    css, icon = STATUS_PILL[status]
    return f"<span class='pill {css}'>{icon} {escape(text)}</span>"


def _seconds(value: float) -> str:
    return f"{num(value, 2)} s"


def _rows(value: Any) -> str:
    return "—" if pd.isna(value) else num(float(value), 0)


def status_grid(tasks: pd.DataFrame, months: list[str]) -> str:
    """Status de cada tarefa mensal: uma linha por tarefa, uma coluna por mês."""
    head = "".join(f"<th>{m}</th>" for m in months)
    rows = []
    for task in GRID_TASKS:
        cells = []
        for month in months:
            records = tasks[(tasks["task"] == task) & (tasks["month"] == month)]
            if records.empty:
                cells.append("<td><span class='desc'>não executada</span></td>")
            elif task == "ingestao":
                names = {"sucesso": "aprovado", "bloqueado": "bloqueado", "falha": "falha"}
                counts = records["status"].value_counts()
                pills = "<br>".join(
                    _pill(status, f"{int(counts[status])} {names[status]}")
                    for status in ("sucesso", "bloqueado", "falha")
                    if status in counts
                )
                cells.append(f"<td>{pills}</td>")
            else:
                record = records.iloc[0]
                label = "ok" if record["status"] == "sucesso" else record["status"]
                duration = f"<span class='dur'>{_seconds(record['duration_seconds'])}</span>"
                cells.append(f"<td>{_pill(record['status'], label)}{duration}</td>")
        rows.append(f"<tr><td><b>{TASK_LABELS[task]}</b></td>{''.join(cells)}</tr>")
    return (
        "<div class='card table-wrap'><table class='grid' style='min-width:860px'>"
        f"<thead><tr><th>Tarefa</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def incidents_table(tasks: pd.DataFrame) -> str:
    incidents = tasks[tasks["status"] != "sucesso"]
    if incidents.empty:
        return "<p class='desc'>Nenhum incidente nesta execução.</p>"
    rows = []
    for record in incidents.itertuples():
        what = record.error if record.status == "falha" else record.detail
        rows.append(
            "<tr>"
            f"<td><b>{escape(record.month or '—')}</b></td>"
            f"<td>{TASK_LABELS.get(record.task, record.task)}</td>"
            f"<td><code>{escape(record.batch or '—')}</code></td>"
            f"<td>{_pill(record.status, record.status)}</td>"
            f"<td class='num'>{_rows(record.rows_in)}</td>"
            f"<td>{escape(what or '')}</td>"
            "</tr>"
        )
    return (
        "<div class='card table-wrap'><table style='min-width:820px'><thead><tr>"
        "<th>Mês</th><th>Tarefa</th><th>Lote</th><th>Status</th>"
        "<th class='num'>Registros</th><th>O que aconteceu</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def run_tasks_table(tasks: pd.DataFrame) -> str:
    """Tarefas que rodam uma vez por execução (preparação e fechamento)."""
    once = tasks[tasks["month"].isna()]
    if once.empty:
        return "<p class='desc'>Nenhuma tarefa de preparação nesta execução.</p>"
    rows = []
    for record in once.itertuples():
        rows.append(
            "<tr>"
            f"<td><b>{TASK_LABELS.get(record.task, record.task)}</b></td>"
            f"<td>{_pill(record.status, 'ok' if record.status == 'sucesso' else record.status)}</td>"
            f"<td class='num'>{_seconds(record.duration_seconds)}</td>"
            f"<td class='num'>{_rows(record.rows_in)}</td>"
            f"<td class='num'>{_rows(record.rows_out)}</td>"
            f"<td>{escape(record.error or record.detail or '')}</td>"
            "</tr>"
        )
    return (
        "<div class='card table-wrap'><table style='min-width:760px'><thead><tr>"
        "<th>Tarefa</th><th>Status</th><th class='num'>Duração</th>"
        "<th class='num'>Registros de entrada</th><th class='num'>Registros de saída</th>"
        f"<th>Detalhe</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def resources_table(tasks: pd.DataFrame) -> str:
    """Latência, CPU, memória e vazão agregadas por tipo de tarefa."""
    rows = []
    for task, group in tasks.groupby("task", sort=False):
        total_rows = group["rows_in"].fillna(0).sum()
        total_time = group["duration_seconds"].sum()
        throughput = (
            f"{num(total_rows / total_time, 0)}/s" if total_rows and total_time > 0 else "—"
        )
        rows.append(
            "<tr>"
            f"<td><b>{TASK_LABELS.get(task, task)}</b></td>"
            f"<td class='num'>{len(group)}</td>"
            f"<td class='num'>{_seconds(group['duration_seconds'].mean())}</td>"
            f"<td class='num'>{_seconds(group['duration_seconds'].max())}</td>"
            f"<td class='num'>{_seconds(group['cpu_seconds'].sum())}</td>"
            f"<td class='num'>{num(group['memory_rss_mb'].max(), 0)} MB</td>"
            f"<td class='num'>{throughput}</td>"
            "</tr>"
        )
    return (
        "<div class='card table-wrap'><table style='min-width:820px'><thead><tr>"
        "<th>Tarefa</th><th class='num'>Execuções</th><th class='num'>Duração média</th>"
        "<th class='num'>Duração máxima</th><th class='num'>CPU total</th>"
        "<th class='num'>Memória (pico)</th><th class='num'>Vazão (registros)</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def critical_alerts_table(alerts: list[dict[str, Any]]) -> str:
    shown = sorted(
        (a for a in alerts if a["severity"] != "info"),
        key=lambda a: (a["month"], ALERT_ORDER[a["severity"]]),
    )
    if not shown:
        return "<p class='desc'>Nenhum alerta crítico ou de atenção nesta execução.</p>"
    rows = "".join(
        "<tr>"
        f"<td><b>{escape(a['month'])}</b></td>"
        f"<td>{alert_pill(a['severity'])}</td>"
        f"<td>{escape(a['category'])}</td>"
        f"<td>{escape(a['message'])}</td>"
        "</tr>"
        for a in shown
    )
    return (
        "<div class='card table-wrap'><table style='min-width:720px'><thead><tr>"
        "<th>Mês</th><th>Severidade</th><th>Categoria</th><th>Alerta</th>"
        f"</tr></thead><tbody>{rows}</tbody></table></div>"
    )


def _timestamp(value: str) -> str:
    return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S UTC")


def history_table(runs: list[dict[str, Any]], current: str) -> str:
    rows = []
    for run in runs:
        status = run["status"]
        pill = _pill("sucesso", "sucesso") if status == "sucesso" else _pill("falha", status)
        duration = (
            "—" if run["duration_seconds"] is None else f"{num(run['duration_seconds'], 1)} s"
        )
        tag = "<span class='tag'>esta execução</span>" if run["run_id"] == current else ""
        rows.append(
            "<tr>"
            f"<td><code>{escape(run['run_id'])}</code>{tag}</td>"
            f"<td>{_timestamp(run['started_at'])}</td>"
            f"<td>{pill}</td>"
            f"<td class='num'>{duration}</td>"
            f"<td class='num'>{run['tasks']}</td>"
            f"<td class='num'>{run['tasks_failed']}</td>"
            f"<td class='num'>{run['tasks_blocked']}</td>"
            f"<td class='num'>{run['critical_alerts']}</td>"
            "</tr>"
        )
    return (
        "<div class='card table-wrap'><table style='min-width:860px'><thead><tr>"
        "<th>Execução</th><th>Início</th><th>Status</th><th class='num'>Duração</th>"
        "<th class='num'>Tarefas</th><th class='num'>Falhas</th><th class='num'>Bloqueios</th>"
        f"<th class='num'>Alertas críticos</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


# --------------------------------------------------------------------------- página


def _chart_card(title: str, note: str, chart: str) -> str:
    return f"<div class='card chart'><h3>{title}</h3><p>{note}</p>{chart}</div>"


def render_dashboard(run: RunData, runs: list[dict[str, Any]], settings: dict[str, Any]) -> str:
    limits = settings["observability"]
    drift_cfg = settings["drift"]
    tasks = run.tasks
    months = [m for m in tasks["month"].dropna().unique().tolist()]
    ingestion = tasks[tasks["task"] == "ingestao"]
    n_alerts = len(run.alerts)
    n_critical = sum(a["severity"] == "crítico" for a in run.alerts)
    levels = pd.Series([e["level"] for e in run.events]).value_counts()

    tiles = "".join(
        f"<div class='tile {h['level']}'><h3>{h['name']}</h3>"
        f"<div class='state'>{HEALTH_LABEL[h['level']][0]} {HEALTH_LABEL[h['level']][1]}</div>"
        f"<p>{escape(h['reason'])}</p></div>"
        for h in health(run, limits)
    )
    duration = (run.finished or {}).get("duration_seconds")
    stats = [
        (
            str(len(tasks)),
            "tarefas executadas",
            f"{plural(int((tasks['status'] == 'falha').sum()), 'falha', 'falhas')} · "
            f"{plural(int((tasks['status'] == 'bloqueado').sum()), 'bloqueio', 'bloqueios')}",
        ),
        (
            str(len(ingestion)),
            "lotes recebidos",
            plural(int((ingestion["status"] == "sucesso").sum()), "aprovado", "aprovados"),
        ),
        (
            num(float(ingestion["rows_out"].fillna(0).sum()), 0),
            "registros processados",
            f"{num(float(ingestion['rows_rejected'].sum()), 0)} rejeitados",
        ),
        ("—" if duration is None else f"{num(duration, 1)} s", "duração da execução", ""),
        (
            f"{num(float(tasks['memory_rss_mb'].max()), 0)} MB" if len(tasks) else "—",
            "pico de memória",
            f"limite {num(limits['max_memory_mb'], 0)} MB",
        ),
        (str(n_alerts), "alertas gerados", plural(n_critical, "crítico", "críticos")),
    ]
    stats_html = "".join(
        f"<div class='stat'><b>{escape(value)}</b><span>{escape(label)}</span>"
        f"<small>{escape(sub)}</small></div>"
        for value, label, sub in stats
    )

    model_version = run.baseline["model_version"] if run.baseline else "—"
    host = (
        f"{run.started.get('cpu_count', '?')} vCPU · "
        f"{num(run.started.get('memory_total_mb', 0) / 1024, 1)} GB de RAM · "
        f"Python {run.started.get('python', '?')}"
    )

    if run.months.empty:
        model_section = "<p class='desc'>Nenhum mês monitorado nesta execução.</p>"
    else:
        m = run.months
        month_names = m["month"].tolist()
        charts = [
            _chart_card(
                "AUC por mês",
                "Capacidade de separar bons e maus pagadores.",
                auc_chart(
                    month_names,
                    m["auc"].astype(float).tolist(),
                    run.baseline["auc"],
                    drift_cfg["performance"]["auc_drop"],
                ),
            ),
            _chart_card(
                "Score médio x inadimplência real",
                "Quando as linhas se separam, o modelo está descalibrado.",
                calibration_chart(
                    month_names,
                    m["mean_score"].astype(float).tolist(),
                    m["actual_default_rate"].astype(float).tolist(),
                ),
            ),
            _chart_card(
                "Variáveis com drift, por severidade",
                "De 23 variáveis monitoradas (PSI e teste KS).",
                severity_chart(m),
            ),
            _chart_card(
                "Drift das predições (PSI do score)",
                "Mudança na distribuição do score, visível antes de os rótulos chegarem.",
                prediction_psi_chart(m, drift_cfg["prediction_psi_alert"]),
            ),
        ]
        model_section = f"<div class='charts'>{''.join(charts)}</div>"

    body = f"""  <p class="eyebrow">Observabilidade do pipeline de crédito</p>
  <h1>Dashboard de monitoramento</h1>
  <p class="lead">Execução <code>{escape(run.run_id)}</code>, iniciada em {_timestamp(run.started["ts"])}.
  Modelo <code>{escape(model_version)}</code> · {escape(host)}.</p>
  <section class="health">{tiles}</section>
  <section class="stats">{stats_html}</section>

  <h2>Saúde do pipeline</h2>
  <p class="section-note">Status de cada tarefa em cada mês. Um lote <b>bloqueado</b> foi barrado pelo
  contrato de dados; uma <b>falha</b> é um erro técnico (por exemplo, arquivo ilegível). Nos dois casos o
  lote vai para a quarentena e o mês segue com os lotes aprovados.</p>
  {status_grid(tasks, months)}
  <div class="block"><h3>Incidentes</h3>{incidents_table(tasks)}</div>
  <div class="charts block">
    {_chart_card("Volume de registros por mês", "Aprovados pelo contrato x rejeitados.", volume_chart(tasks, months))}
    {_chart_card("Duração do processamento por mês", "Tempo de cada etapa; a detecção de drift (Evidently) domina.", duration_chart(tasks, months))}
  </div>
  <div class="block"><h3>Tarefas de preparação e fechamento</h3>{run_tasks_table(tasks)}</div>

  <h2>Infraestrutura</h2>
  <p class="section-note">Uso de recursos do processo, medido em cada tarefa com <code>psutil</code>. Limites
  configurados: {limits["max_task_seconds"]} s por tarefa e {num(limits["max_memory_mb"], 0)} MB de memória.</p>
  {resources_table(tasks)}
  <div class="charts block">
    {_chart_card("Memória do processo por mês", "Maior valor observado entre as tarefas do mês.", memory_chart(tasks, months))}
    {_chart_card("Latência da predição", "Milissegundos para pontuar 1.000 registros.", prediction_latency_chart(tasks, months))}
  </div>

  <h2>Saúde do modelo</h2>
  <p class="section-note">Resultados do monitoramento de drift e de desempenho. O detalhe por variável está
  no <a href="../drift/drift_report.html">relatório de drift</a>.</p>
  {model_section}

  <h2>Alertas</h2>
  <p class="section-note">{n_alerts} alertas nesta execução. A tabela lista os críticos e os de atenção;
  os informativos (drift leve) aparecem só no gráfico.</p>
  <div class="charts">
    {_chart_card("Alertas por mês e severidade", "A quantidade e a gravidade crescem com a degradação.", alerts_chart(run.alerts, months))}
    {_chart_card("Alertas por categoria", "Total da execução em cada frente de monitoramento.", alerts_by_category_chart(run.alerts))}
  </div>
  <div class="block">{critical_alerts_table(run.alerts)}</div>

  <h2>Histórico de execuções</h2>
  <p class="section-note">Últimas execuções do pipeline registradas no log centralizado. Durações estáveis
  e ausência de falhas indicam infraestrutura estável.</p>
  {history_table(runs[: limits["history_runs"]], run.run_id)}

  <h2>De onde vêm os dados</h2>
  <div class="card" style="padding:14px 16px"><ul class="links">
    <li>Log centralizado em <code>{escape(settings["paths"]["logs_file"])}</code> (JSON Lines): esta execução gerou
    {len(run.events)} eventos ({int(levels.get("INFO", 0))} INFO, {int(levels.get("WARNING", 0))} WARNING,
    {int(levels.get("ERROR", 0))} ERROR). Cópia desta execução: <a href="run_log.jsonl">run_log.jsonl</a>.</li>
    <li>Métricas por tarefa: <a href="tasks.csv">tasks.csv</a>.</li>
    <li>Relatórios: <a href="../drift/drift_report.html">drift consolidado</a>, Evidently por mês em
    <code>reports/drift/</code> e validação do contrato em <code>reports/validation/</code>.</li>
    <li>MLflow: experimento <code>{escape(settings["mlflow"]["monitoring_experiment"])}</code>
    (<code>make mlflow-ui</code>), com as mesmas métricas por mês e os relatórios como artefatos.</li>
  </ul></div>

  <footer>Limites em <code>config/settings.yaml</code> (seções <code>observability</code> e <code>drift</code>).</footer>"""
    return page("Dashboard de Monitoramento", body, EXTRA_STYLE)


def build_dashboard(settings: dict[str, Any] | None = None, run_id: str | None = None) -> Path:
    """Gera o dashboard e exporta as métricas por tarefa e o log da execução."""
    settings = settings or load_settings()
    events = load_events(resolve_path(settings["paths"]["logs_file"]))
    run = collect_run(events, run_id)
    output_dir = resolve_path(settings["paths"]["monitoring_reports_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run.tasks.to_csv(output_dir / "tasks.csv", index=False)
    with (output_dir / "run_log.jsonl").open("w", encoding="utf-8") as fh:
        for event in run.events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    dashboard_path = output_dir / "dashboard.html"
    dashboard_path.write_text(render_dashboard(run, list_runs(events), settings), encoding="utf-8")
    return dashboard_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera o dashboard de monitoramento.")
    parser.add_argument("--run-id", help="Execução a exibir (padrão: a mais recente)")
    args = parser.parse_args(argv)
    try:
        path = build_dashboard(run_id=args.run_id)
    except ValueError as exc:
        print(exc)
        return 1
    print(f"Dashboard: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
