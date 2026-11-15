from credit_monitoring.monitoring.tracking import log_monitoring_run, mlflow, month_metrics


def test_month_metrics_combine_model_pipeline_and_infrastructure(pipeline_run):
    _, _, result, records = pipeline_run
    m4 = month_metrics(result.months[1], records)

    assert (m4["rows_received"], m4["rows_processed"], m4["rows_rejected"]) == (1103, 800, 303)
    assert (m4["blocked_batches"], m4["failed_batches"]) == (1, 0)
    assert 0.5 < m4["auc"] < 1
    assert m4["duration_seconds"] > 0 and m4["memory_rss_mb"] > 0


def test_monitoring_run_is_logged_with_one_step_and_one_child_run_per_month(pipeline_run):
    cfg, _, result, records = pipeline_run
    parent_id = log_monitoring_run(cfg, result.run_id, "v-teste", result.months, records)

    client = mlflow.MlflowClient()
    history = client.get_metric_history(parent_id, "auc")
    assert [m.step for m in history] == [1, 2, 3]
    assert client.get_run(parent_id).data.tags["pipeline_run_id"] == result.run_id

    children = mlflow.search_runs(
        experiment_names=[cfg["mlflow"]["monitoring_experiment"]],
        filter_string=f"tags.mlflow.parentRunId = '{parent_id}'",
    )
    assert sorted(children["tags.month"]) == ["M1", "M4", "M5"]
    assert children["metrics.psi_age"].notna().all()
    m5 = children.set_index("tags.month").loc["M5"]
    assert m5["metrics.failed_batches"] == 1
    artifacts = {a.path for a in client.list_artifacts(m5["run_id"])}
    assert artifacts == {"alerts.json", "evidently_M5.html"}
