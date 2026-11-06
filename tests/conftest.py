import copy

import pandas as pd
import pytest

from credit_monitoring.config import load_settings
from credit_monitoring.data.columns import TARGET
from credit_monitoring.data.download import ensure_raw_dataset
from credit_monitoring.data.prepare import apply_pseudonymization, clean, standardize_columns

TEST_KEY = b"chave-apenas-para-testes"


@pytest.fixture(scope="session")
def base_settings() -> dict:
    return load_settings()


@pytest.fixture
def settings(base_settings, tmp_path) -> dict:
    """Configuração com todas as saídas redirecionadas para uma pasta temporária."""
    cfg = copy.deepcopy(base_settings)
    cfg["paths"].update(
        reference_file=str(tmp_path / "reference.parquet"),
        production_pool_file=str(tmp_path / "production_pool.parquet"),
        incoming_dir=str(tmp_path / "incoming"),
        quarantine_dir=str(tmp_path / "quarantine"),
        validation_reports_dir=str(tmp_path / "reports"),
        model_file=str(tmp_path / "model.joblib"),
        model_metadata_file=str(tmp_path / "model_metadata.json"),
    )
    cfg["pseudonymization"]["key_file"] = str(tmp_path / ".secrets" / "key")
    return cfg


@pytest.fixture(scope="session")
def prepared_df(base_settings) -> pd.DataFrame:
    """Dataset completo, limpo e pseudonimizado (com alvo)."""
    df, _ = clean(standardize_columns(ensure_raw_dataset(base_settings)))
    return apply_pseudonymization(df, TEST_KEY)


@pytest.fixture
def valid_batch(prepared_df) -> pd.DataFrame:
    """Lote de produção válido: 200 registros, sem o alvo."""
    return prepared_df.drop(columns=[TARGET]).sample(200, random_state=0).reset_index(drop=True)
