import pandas as pd

from credit_monitoring.data.columns import FEATURES, ID_COLUMN, MONETARY, TARGET
from credit_monitoring.data.download import ensure_raw_dataset
from credit_monitoring.data.prepare import (
    clean,
    load_pseudonymization_key,
    prepare,
    pseudonymize_ids,
    split_reference_production,
    standardize_columns,
)


def test_standardize_columns_maps_every_raw_column(base_settings):
    df = standardize_columns(ensure_raw_dataset(base_settings))
    assert set(df.columns) == {"id", *FEATURES, TARGET}


def test_clean_fixes_codes_and_removes_duplicates(base_settings):
    df, report = clean(standardize_columns(ensure_raw_dataset(base_settings)))

    assert report.education_remapped == 345
    assert report.marriage_remapped == 54
    assert report.duplicates_removed == 35
    assert report.rows_out == len(df) == 29_965
    assert set(df["education"]) <= {1, 2, 3, 4}
    assert set(df["marriage"]) <= {1, 2, 3}
    assert not df.drop(columns="id").duplicated().any()
    assert all(df[col].dtype == "float64" for col in MONETARY)


def test_pseudonymization_is_deterministic_and_depends_on_key():
    ids = pd.Series([1, 2, 3])
    first = pseudonymize_ids(ids, b"chave-a")
    assert first.equals(pseudonymize_ids(ids, b"chave-a"))
    assert not first.equals(pseudonymize_ids(ids, b"chave-b"))
    assert first.str.len().eq(16).all()
    assert first.is_unique


def test_prepared_data_has_no_original_identifier(prepared_df):
    assert "id" not in prepared_df.columns
    assert prepared_df[ID_COLUMN].is_unique


def test_key_from_environment_takes_precedence(settings, monkeypatch):
    monkeypatch.setenv(settings["pseudonymization"]["key_env_var"], "chave-do-ambiente")
    assert load_pseudonymization_key(settings) == b"chave-do-ambiente"


def test_key_file_is_generated_once_with_restricted_permissions(settings, monkeypatch, tmp_path):
    monkeypatch.delenv(settings["pseudonymization"]["key_env_var"], raising=False)
    first = load_pseudonymization_key(settings)
    key_file = tmp_path / ".secrets" / "key"
    assert key_file.stat().st_mode & 0o777 == 0o600
    assert load_pseudonymization_key(settings) == first


def test_split_is_disjoint_stratified_and_reproducible(prepared_df):
    reference, production = split_reference_production(prepared_df, 0.3, seed=42)
    again, _ = split_reference_production(prepared_df, 0.3, seed=42)

    assert set(reference[ID_COLUMN]).isdisjoint(production[ID_COLUMN])
    assert len(reference) + len(production) == len(prepared_df)
    assert abs(reference[TARGET].mean() - production[TARGET].mean()) < 0.005
    assert reference[ID_COLUMN].equals(again[ID_COLUMN])


def test_prepare_writes_reference_and_production_layers(settings, monkeypatch):
    monkeypatch.delenv(settings["pseudonymization"]["key_env_var"], raising=False)
    summary = prepare(settings)

    reference = pd.read_parquet(settings["paths"]["reference_file"])
    production = pd.read_parquet(settings["paths"]["production_pool_file"])
    assert len(reference) == summary["reference_rows"]
    assert len(production) == summary["production_pool_rows"]
    assert len(reference) + len(production) == 29_965
