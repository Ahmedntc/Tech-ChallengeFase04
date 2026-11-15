import copy
from pathlib import Path

import pandas as pd
import pytest

from credit_monitoring.config import load_settings
from credit_monitoring.data.columns import TARGET
from credit_monitoring.data.download import ensure_raw_dataset
from credit_monitoring.data.prepare import apply_pseudonymization, clean, standardize_columns
from credit_monitoring.monitoring.logger import reset_logging
from credit_monitoring.monitoring.telemetry import task_records
from credit_monitoring.pipeline import run_pipeline

TEST_KEY = b"chave-apenas-para-testes"


def isolated_settings(base: dict, tmp: Path) -> dict:
    """Configuração com todas as saídas redirecionadas para uma pasta temporária."""
    cfg = copy.deepcopy(base)
    cfg["paths"].update(
        reference_file=str(tmp / "reference.parquet"),
        production_pool_file=str(tmp / "production_pool.parquet"),
        incoming_dir=str(tmp / "incoming"),
        quarantine_dir=str(tmp / "quarantine"),
        validation_reports_dir=str(tmp / "reports" / "validation"),
        model_file=str(tmp / "model.joblib"),
        model_metadata_file=str(tmp / "model_metadata.json"),
        labels_dir=str(tmp / "labels"),
        predictions_dir=str(tmp / "predictions"),
        simulation_manifest=str(tmp / "manifest.json"),
        drift_reports_dir=str(tmp / "reports" / "drift"),
        logs_file=str(tmp / "logs" / "pipeline.jsonl"),
        monitoring_reports_dir=str(tmp / "reports" / "monitoring"),
        governance_reports_dir=str(tmp / "reports" / "governance"),
    )
    cfg["governance"]["retention"] = {str(tmp / "quarantine"): 90}
    cfg["pseudonymization"]["key_file"] = str(tmp / ".secrets" / "key")
    cfg["mlflow"].update(tracking_db=str(tmp / "mlflow.db"), artifact_dir=str(tmp / "mlruns"))
    return cfg


@pytest.fixture(scope="session")
def base_settings() -> dict:
    return load_settings()


@pytest.fixture
def settings(base_settings, tmp_path) -> dict:
    return isolated_settings(base_settings, tmp_path)


@pytest.fixture(autouse=True)
def _no_leftover_log_handlers():
    """Nenhum teste herda (nem deixa) handlers de log apontando para arquivos de outro teste."""
    yield
    reset_logging()


@pytest.fixture(scope="session")
def prepared_df(base_settings) -> pd.DataFrame:
    """Dataset completo, limpo e pseudonimizado (com alvo)."""
    df, _ = clean(standardize_columns(ensure_raw_dataset(base_settings)))
    return apply_pseudonymization(df, TEST_KEY)


@pytest.fixture
def valid_batch(prepared_df) -> pd.DataFrame:
    """Lote de produção válido: 200 registros, sem o alvo."""
    return prepared_df.drop(columns=[TARGET]).sample(200, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="session")
def pipeline_run(base_settings, tmp_path_factory):
    """Uma execução completa do pipeline, em escala reduzida, compartilhada pelos testes.

    Três meses: controle (M1), lote corrompido de parceiro (M4) e arquivo vazio (M5).
    """
    tmp = tmp_path_factory.mktemp("pipeline")
    cfg = isolated_settings(base_settings, tmp)
    cfg["model"]["xgboost"]["n_estimators"] = 50
    cfg["simulation"]["batch_size"] = 800
    months = cfg["simulation"]["months"]
    cfg["simulation"]["months"] = [months[0], months[3], months[4]]

    result = run_pipeline(cfg, track=False)
    records = task_records()
    reset_logging()
    return cfg, tmp, result, records
