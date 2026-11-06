import copy

import pandas as pd
import pytest

from credit_monitoring.config import load_settings
from credit_monitoring.data import download
from credit_monitoring.data.download import (
    DatasetIntegrityError,
    content_fingerprint,
    ensure_raw_dataset,
    verify_dataset,
)


@pytest.fixture
def small_df() -> pd.DataFrame:
    return pd.DataFrame({"AGE": [25, 40], "LIMIT_BAL": [20000, 50000]})


def test_fingerprint_is_deterministic(small_df):
    assert content_fingerprint(small_df) == content_fingerprint(small_df.copy())


def test_fingerprint_detects_any_value_change(small_df):
    changed = small_df.copy()
    changed.loc[0, "AGE"] = 26
    assert content_fingerprint(changed) != content_fingerprint(small_df)


def test_verify_rejects_wrong_shape(small_df):
    with pytest.raises(DatasetIntegrityError, match="Formato"):
        verify_dataset(small_df, content_fingerprint(small_df), [3, 2])


def test_verify_rejects_tampered_content(small_df):
    with pytest.raises(DatasetIntegrityError, match="Impressão digital"):
        verify_dataset(small_df, "0" * 64, list(small_df.shape))


def test_ensure_downloads_when_file_is_missing(tmp_path, monkeypatch, small_df):
    settings = copy.deepcopy(load_settings())
    settings["dataset"].update(
        raw_file=str(tmp_path / "missing.xls"),
        expected_shape=list(small_df.shape),
        content_sha256=content_fingerprint(small_df),
    )
    calls = []
    monkeypatch.setattr(download, "download_raw_dataset", lambda url, dest: calls.append(url))
    monkeypatch.setattr(download, "read_raw_dataset", lambda path, header: small_df)

    ensure_raw_dataset(settings)

    assert calls == [settings["dataset"]["url"]]


def test_versioned_raw_dataset_is_intact():
    """O arquivo versionado em data/raw deve bater com a impressão digital registrada."""
    df = ensure_raw_dataset()
    assert df.shape == (30000, 25)
