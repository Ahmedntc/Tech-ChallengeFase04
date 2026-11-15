import json

import pandas as pd
import pytest

from credit_monitoring.monitoring.dashboard import (
    RunData,
    build_dashboard,
    collect_run,
    health,
    list_runs,
    load_events,
)

LIMITS = {"max_task_seconds": 60, "max_memory_mb": 2048, "history_runs": 10}
TASK_COLUMNS = ["task", "month", "status", "duration_seconds", "memory_rss_mb"]


def _run(
    tasks: list[tuple], alerts: list[dict], status: str = "sucesso", last_month=None
) -> RunData:
    months = pd.DataFrame(
        [
            last_month
            or {"month": "M1", "auc": 0.77, "actual_default_rate": 0.22, "mean_score": 0.22}
        ]
    )
    return RunData(
        run_id="teste",
        started={"ts": "2026-01-01T00:00:00+00:00"},
        finished={"status": status, "error": "falhou" if status != "sucesso" else None},
        baseline={"auc": 0.78, "model_version": "v1"},
        tasks=pd.DataFrame(tasks, columns=TASK_COLUMNS),
        months=months,
        alerts=alerts,
        events=[],
    )


def _levels(run: RunData) -> list[str]:
    return [h["level"] for h in health(run, LIMITS)]


def _alert(category: str, severity: str, month: str = "M1") -> dict:
    return {"month": month, "category": category, "severity": severity, "message": "x"}


OK_TASK = ("predicao", "M1", "sucesso", 1.0, 500.0)


def test_health_is_green_when_everything_is_fine():
    assert _levels(_run([OK_TASK], [])) == ["ok", "ok", "ok"]


def test_handled_ingestion_incidents_put_the_pipeline_on_attention():
    blocked = ("ingestao", "M1", "bloqueado", 0.1, 500.0)
    failed = ("ingestao", "M1", "falha", 0.0, 500.0)
    tiles = health(_run([OK_TASK, blocked, failed], []), LIMITS)
    assert tiles[0]["level"] == "warn"
    assert "1 lote bloqueado" in tiles[0]["reason"] and "1 falha" in tiles[0]["reason"]


def test_crashed_run_makes_the_pipeline_critical():
    assert _levels(_run([OK_TASK], [], status="falha"))[0] == "crit"


def test_infrastructure_alert_puts_infrastructure_on_attention():
    run = _run([OK_TASK], [_alert("infraestrutura", "atenção", "execução")])
    assert _levels(run) == ["ok", "warn", "ok"]


def test_model_health_follows_the_last_month_alerts():
    drift = _run([OK_TASK], [_alert("data drift", "atenção")])
    degraded = _run([OK_TASK], [_alert("desempenho", "crítico")])
    old = _run([OK_TASK], [_alert("desempenho", "crítico", month="M0")])
    assert _levels(drift)[2] == "warn"
    assert _levels(degraded)[2] == "crit"
    assert _levels(old)[2] == "ok"


def test_load_events_skips_truncated_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"event": "run_started", "run_id": "a", "ts": "t"}\n{"event": "trunc', "utf-8")
    assert [e["run_id"] for e in load_events(path)] == ["a"]
    assert load_events(tmp_path / "inexistente.jsonl") == []


def test_runs_are_listed_newest_first_and_unfinished_ones_are_flagged():
    events = [
        {"event": "run_started", "run_id": "a", "ts": "2026-01-01T00:00:00+00:00"},
        {
            "event": "run_finished",
            "run_id": "a",
            "status": "sucesso",
            "duration_seconds": 10,
            "tasks": 3,
            "tasks_failed": 0,
            "tasks_blocked": 1,
        },
        {"event": "alerta", "run_id": "a", "severity": "crítico"},
        {"event": "run_started", "run_id": "b", "ts": "2026-01-02T00:00:00+00:00"},
        {"event": "log", "run_id": "avulso", "ts": "2026-01-03T00:00:00+00:00"},
    ]
    runs = list_runs(events)
    assert [(r["run_id"], r["status"]) for r in runs] == [("b", "interrompida"), ("a", "sucesso")]
    assert runs[1]["critical_alerts"] == 1
    assert collect_run(events).run_id == "b"
    with pytest.raises(ValueError, match="não encontrada"):
        collect_run(events, "avulso")


def test_collect_run_requires_a_pipeline_run():
    with pytest.raises(ValueError, match="Nenhuma execução"):
        collect_run([])


def test_dashboard_shows_the_pipeline_run(pipeline_run):
    cfg, tmp, result, records = pipeline_run
    html = result.dashboard.read_text(encoding="utf-8")

    for section in (
        "Saúde do pipeline",
        "Infraestrutura",
        "Saúde do modelo",
        "Alertas",
        "Histórico",
    ):
        assert section in html
    assert result.run_id in html
    assert "M4_parceiro.csv" in html and "M5_parceiro.csv" in html
    assert "arquivo vazio (0 byte)" in html
    for chart in ("chart-volume", "chart-duration", "chart-memory", "chart-auc", "chart-alerts"):
        assert f'id="{chart}"' in html

    run = collect_run(load_events(cfg["paths"]["logs_file"]), result.run_id)
    assert len(run.tasks) == len(records)
    assert run.months["month"].tolist() == ["M1", "M4", "M5"]
    assert health(run, cfg["observability"])[0]["level"] == "warn"


def test_dashboard_can_be_rebuilt_for_a_given_run(pipeline_run):
    cfg, tmp, result, _ = pipeline_run
    path = build_dashboard(cfg, result.run_id)
    assert path.exists()
    exported = (tmp / "reports" / "monitoring" / "run_log.jsonl").read_text("utf-8").splitlines()
    assert all(json.loads(line)["run_id"] == result.run_id for line in exported)
