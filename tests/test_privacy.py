import json
import os
import time

import pandas as pd
import pytest

from credit_monitoring.governance.privacy import (
    apply_retention,
    expired_files,
    k_anonymity,
    reidentification_risk,
)
from credit_monitoring.monitoring.logger import configure_logging

DAY = 86_400


def test_k_anonymity_counts_small_and_unique_classes():
    df = pd.DataFrame({"age": [30, 30, 30, 40, 50], "sex": [1, 1, 1, 2, 2]})
    result = k_anonymity(df, ["age", "sex"], k_min=2)
    assert (result["classes"], result["min_k"]) == (3, 1)
    assert result["share_below_k"] == pytest.approx(2 / 5)
    assert result["share_unique"] == pytest.approx(2 / 5)


def test_adding_identifiers_increases_reidentification_risk(prepared_df, base_settings):
    risk = reidentification_risk(prepared_df, base_settings["governance"]["privacy"])
    assert len(risk) == 2
    assert risk["share_unique"].is_monotonic_increasing
    assert risk["share_unique"].iloc[-1] > 0.1


@pytest.fixture
def retention_settings(settings, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    settings["governance"]["retention"] = {
        str(tmp_path / "quarantine"): 90,
        str(tmp_path / "incoming"): 30,
    }
    now = time.time()
    for folder, name, age_days in [
        ("quarantine", "velho.csv", 120),
        ("quarantine", "novo.csv", 10),
        ("quarantine", ".gitkeep", 999),
        ("incoming", "M1.csv", 31),
    ]:
        path = tmp_path / folder / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("x")
        os.utime(path, (now - age_days * DAY, now - age_days * DAY))
    return settings


def test_only_files_older_than_their_folder_limit_expire(retention_settings):
    expired = expired_files(retention_settings)
    assert sorted(item.path.name for item in expired) == ["M1.csv", "velho.csv"]


def test_retention_is_a_dry_run_unless_applied(retention_settings, tmp_path):
    configure_logging(retention_settings)
    apply_retention(retention_settings)
    assert (tmp_path / "quarantine" / "velho.csv").exists()

    apply_retention(retention_settings, apply=True)
    remaining = sorted(p.name for p in (tmp_path / "quarantine").iterdir())
    assert remaining == [".gitkeep", "novo.csv"]
    assert not (tmp_path / "incoming" / "M1.csv").exists()

    with open(retention_settings["paths"]["logs_file"], encoding="utf-8") as fh:
        events = [json.loads(line) for line in fh if '"retencao"' in line]
    assert [(e["expired"], e["applied"]) for e in events] == [(2, False), (2, True)]
