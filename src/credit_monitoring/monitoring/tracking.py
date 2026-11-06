"""Configuração do MLflow (tracking local em SQLite + pasta de artefatos)."""

from __future__ import annotations

import os
from typing import Any

from credit_monitoring.config import resolve_path

# Silencia uma dica de log do MLflow irrelevante para este projeto.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

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
