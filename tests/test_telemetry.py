import json

import pytest

from credit_monitoring.monitoring.telemetry import (
    TaskRecord,
    finish_run,
    infrastructure_alerts,
    start_run,
    task_records,
    track_task,
)

LIMITS = {"max_task_seconds": 60, "max_memory_mb": 2048}


def _events(settings, kind: str) -> list[dict]:
    with open(settings["paths"]["logs_file"], encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh]
    return [e for e in events if e["event"] == kind]


def test_successful_task_records_duration_resources_and_volume(settings):
    run_id = start_run(settings)
    with track_task("predicao", stage="monitoramento", month="M1") as task:
        sum(range(200_000))
        task.rows_in = task.rows_out = 1400

    (record,) = task_records()
    assert (record.task, record.month, record.status) == ("predicao", "M1", "sucesso")
    assert record.duration_seconds > 0
    assert record.memory_rss_mb > 0
    assert record.rows_out == 1400

    (event,) = _events(settings, "task_finished")
    assert event["run_id"] == run_id
    assert (event["level"], event["task"], event["rows_in"]) == ("INFO", "predicao", 1400)


def test_failing_task_is_recorded_as_failure_and_the_error_propagates(settings):
    start_run(settings)
    with (
        pytest.raises(ZeroDivisionError),
        track_task("preparar_dados", stage="preparacao"),
    ):
        1 / 0  # noqa: B018

    (record,) = task_records()
    assert record.status == "falha"
    assert record.error.startswith("ZeroDivisionError")
    (event,) = _events(settings, "task_finished")
    assert event["level"] == "ERROR"


def test_blocked_task_is_logged_as_warning(settings):
    start_run(settings)
    with track_task("ingestao", stage="monitoramento", month="M4", batch="M4_parceiro.csv") as task:
        task.status = "bloqueado"
        task.rows_in = task.rows_rejected = 303
    (event,) = _events(settings, "task_finished")
    assert (event["level"], event["status"], event["batch"]) == (
        "WARNING",
        "bloqueado",
        "M4_parceiro.csv",
    )


def test_start_run_resets_records_and_finish_run_summarizes(settings):
    start_run(settings)
    with track_task("ingestao", stage="monitoramento", month="M1") as task:
        task.rows_in = task.rows_out = 100
    start_run(settings)
    assert task_records() == []

    with track_task("ingestao", stage="monitoramento", month="M4", batch="a.csv") as task:
        task.rows_in = task.rows_out = 1400
    with track_task("ingestao", stage="monitoramento", month="M4", batch="b.csv") as task:
        task.status = "bloqueado"
        task.rows_in = task.rows_rejected = 303
    summary = finish_run("sucesso", 12.3)

    assert summary["tasks"] == 2
    assert (summary["tasks_blocked"], summary["tasks_failed"]) == (1, 0)
    assert (summary["rows_processed"], summary["rows_rejected"]) == (1400, 303)
    (event,) = _events(settings, "run_finished")
    assert (event["status"], event["duration_seconds"]) == ("sucesso", 12.3)
    assert len(_events(settings, "run_started")) == 2


def test_infrastructure_alerts_fire_only_above_the_limits():
    ok = TaskRecord("predicao", "monitoramento", "M1", duration_seconds=1.0, memory_rss_mb=500)
    slow = TaskRecord("deteccao_drift", "monitoramento", "M2", duration_seconds=75.0)
    heavy = TaskRecord("treinar_modelo", "preparacao", memory_rss_mb=4096)

    assert infrastructure_alerts([ok], LIMITS) == []
    alerts = infrastructure_alerts([ok, slow, heavy], LIMITS)
    assert [(a.month, a.subject, a.category) for a in alerts] == [
        ("M2", "deteccao_drift", "infraestrutura"),
        ("execução", "treinar_modelo", "infraestrutura"),
    ]
    assert "75,0s" in alerts[0].message
    assert "4096 MB" in alerts[1].message
