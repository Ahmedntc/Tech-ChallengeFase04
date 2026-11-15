"""Configuração do MLflow (tracking local em SQLite + pasta de artefatos)."""

from __future__ import annotations

import os
from typing import Any

from credit_monitoring.config import resolve_path

# Silencia uma dica de log do MLflow irrelevante para este projeto.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
# O MLflow 3 envia telemetria de uso por padrão. Fica desligada: nada sai do ambiente.
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")

import mlflow  # noqa: E402


def tracking_uri(settings: dict[str, Any]) -> str:
    return f"sqlite:///{resolve_path(settings['mlflow']['tracking_db'])}"


def use_experiment(settings: dict[str, Any], experiment_key: str) -> str:
    """Aponta o MLflow para o banco local e ativa (ou cria) o experimento indicado."""
    mlflow.set_tracking_uri(tracking_uri(settings))
    name = settings["mlflow"][experiment_key]
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        artifact_root = resolve_path(settings["mlflow"]["artifact_dir"])
        experiment_id = mlflow.create_experiment(name, artifact_location=artifact_root.as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return experiment_id


def month_metrics(result: Any, records: list[Any]) -> dict[str, float]:
    """Métricas de um mês para o MLflow: saúde do modelo, do pipeline e da infraestrutura."""
    tasks = [r for r in records if r.month == result.month]
    ingestion = [r for r in tasks if r.task == "ingestao"]
    prediction = [r for r in tasks if r.task == "predicao"]
    summary = result.summary_row()
    metrics = {
        key: float(summary[key])
        for key in (
            "n_drifted",
            "drift_share",
            "psi_max",
            "prediction_psi",
            "auc",
            "ks",
            "recall",
            "precision",
            "mean_score",
            "actual_default_rate",
            "approval_rate",
            "n_alerts",
            "n_critical_alerts",
            "blocked_batches",
            "failed_batches",
        )
    }
    metrics.update(
        calibration_gap=summary["actual_default_rate"] - summary["mean_score"],
        rows_received=float(sum(r.rows_in or 0 for r in ingestion)),
        rows_processed=float(sum(r.rows_out or 0 for r in ingestion)),
        rows_rejected=float(sum(r.rows_rejected for r in ingestion)),
        duration_seconds=float(sum(r.duration_seconds for r in tasks)),
        prediction_seconds=float(sum(r.duration_seconds for r in prediction)),
        memory_rss_mb=float(max((r.memory_rss_mb for r in tasks), default=0.0)),
    )
    return metrics


def log_monitoring_run(
    settings: dict[str, Any],
    run_id: str,
    model_version: str,
    results: list[Any],
    records: list[Any],
) -> str:
    """Registra a execução de monitoramento no MLflow.

    - Run pai (``pipeline-<run_id>``): cada métrica é registrada com ``step`` = número do
      mês, o que gera os gráficos de evolução na interface do MLflow.
    - Runs filhos (um por mês): métricas do mês, PSI por variável, relatório do Evidently
      e alertas como artefatos.
    """
    use_experiment(settings, "monitoring_experiment")
    reports_dir = resolve_path(settings["paths"]["drift_reports_dir"])
    with mlflow.start_run(run_name=f"pipeline-{run_id}") as parent:
        mlflow.set_tags({"pipeline_run_id": run_id, "model_version": model_version})
        mlflow.log_params(
            {
                "months": len(results),
                "batch_size": settings["simulation"]["batch_size"],
                **{f"drift_{k}": v for k, v in settings["drift"].items() if k != "performance"},
                **{f"perf_{k}": v for k, v in settings["drift"]["performance"].items()},
            }
        )
        for step, result in enumerate(results, start=1):
            metrics = month_metrics(result, records)
            mlflow.log_metrics(metrics, step=step)
            with mlflow.start_run(run_name=result.month, nested=True):
                mlflow.set_tags(
                    {
                        "pipeline_run_id": run_id,
                        "model_version": model_version,
                        "month": result.month,
                        "scenario": result.title,
                    }
                )
                psi_by_feature = result.feature_drift.set_index("feature")["psi"]
                mlflow.log_metrics(
                    metrics | {f"psi_{name}": float(v) for name, v in psi_by_feature.items()}
                )
                mlflow.log_artifact(str(reports_dir / result.evidently_report))
                mlflow.log_dict([alert.__dict__ for alert in result.alerts], "alerts.json")
        return parent.info.run_id
