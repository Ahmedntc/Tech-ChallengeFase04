import json

import pytest

from credit_monitoring import pipeline
from credit_monitoring.monitoring.dashboard import collect_run, health, load_events
from credit_monitoring.pipeline import run_pipeline


def test_pipeline_run_succeeds_and_handles_ingestion_incidents(pipeline_run):
    _, tmp, result, records = pipeline_run

    assert result.status == "sucesso"
    assert [m.month for m in result.months] == ["M1", "M4", "M5"]
    assert (result.summary["tasks_blocked"], result.summary["tasks_failed"]) == (1, 1)
    assert result.summary["rows_processed"] == 3 * 800
    assert result.summary["rows_rejected"] == 303

    by_status = {r.batch: r.status for r in records if r.task == "ingestao"}
    assert by_status["M4_parceiro.csv"] == "bloqueado"
    assert by_status["M5_parceiro.csv"] == "falha"
    assert len(list((tmp / "quarantine").iterdir())) == 2


def test_every_stage_is_tracked_once_per_run_or_per_month(pipeline_run):
    *_, records = pipeline_run
    once = [r.task for r in records if r.month is None]
    assert once == [
        "verificar_dataset",
        "preparar_dados",
        "treinar_modelo",
        "simular_producao",
        "gerar_relatorios",
    ]
    for task in ("predicao", "avaliacao_desempenho", "deteccao_drift"):
        assert [r.month for r in records if r.task == task] == ["M1", "M4", "M5"]
    assert all(r.duration_seconds >= 0 and r.memory_rss_mb > 0 for r in records)


def test_central_log_holds_pipeline_drift_and_alert_events(pipeline_run):
    cfg, _, result, records = pipeline_run
    events = [e for e in load_events(cfg["paths"]["logs_file"]) if e["run_id"] == result.run_id]
    kinds = {e["event"] for e in events}

    assert {"run_started", "task_finished", "baseline_referencia", "resumo_mensal"} <= kinds
    assert {"alerta", "run_finished"} <= kinds
    assert sum(e["event"] == "task_finished" for e in events) == len(records)
    assert sum(e["event"] == "alerta" for e in events) == sum(len(m.alerts) for m in result.months)
    pipeline_alert = next(
        e for e in events if e["event"] == "alerta" and e["category"] == "pipeline"
    )
    assert (pipeline_alert["level"], pipeline_alert["month"]) == ("ERROR", "M5")


def test_log_has_no_personal_data(pipeline_run, prepared_df):
    """Só contagens, métricas e nomes de arquivos: nenhum identificador de cliente."""
    cfg, *_ = pipeline_run
    with open(cfg["paths"]["logs_file"], encoding="utf-8") as fh:
        content = fh.read()
    assert not any(client_id in content for client_id in prepared_df["client_id"].head(500))


def test_outputs_are_written_for_the_run(pipeline_run):
    _, tmp, result, records = pipeline_run
    monitoring = tmp / "reports" / "monitoring"
    assert result.dashboard == monitoring / "dashboard.html"
    assert (tmp / "reports" / "drift" / "drift_report.html").exists()
    tasks_csv = (monitoring / "tasks.csv").read_text(encoding="utf-8").splitlines()
    assert len(tasks_csv) == len(records) + 1
    run_log = (monitoring / "run_log.jsonl").read_text(encoding="utf-8").splitlines()
    assert {json.loads(line)["run_id"] for line in run_log} == {result.run_id}


def test_failed_run_is_logged_and_still_produces_a_dashboard(settings, monkeypatch, tmp_path):
    settings["model"]["xgboost"]["n_estimators"] = 20

    def broken_simulation(_settings):
        raise OSError("disco cheio")

    monkeypatch.setattr(pipeline, "simulate", broken_simulation)

    with pytest.raises(OSError, match="disco cheio"):
        run_pipeline(settings, track=False)

    run = collect_run(load_events(settings["paths"]["logs_file"]))
    assert run.status == "falha"
    assert "disco cheio" in run.finished["error"]
    failed = run.tasks[run.tasks["status"] == "falha"]
    assert failed["task"].tolist() == ["simular_producao"]
    assert health(run, settings["observability"])[0]["level"] == "crit"
    dashboard = (tmp_path / "reports" / "monitoring" / "dashboard.html").read_text(encoding="utf-8")
    assert "disco cheio" in dashboard
