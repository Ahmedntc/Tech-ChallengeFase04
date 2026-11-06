from pathlib import Path

from credit_monitoring.config import PROJECT_ROOT, load_settings, resolve_path


def test_load_settings_has_required_sections():
    settings = load_settings()
    assert {"project", "dataset", "paths"} <= settings.keys()
    assert settings["dataset"]["raw_file"].endswith(".xls")


def test_resolve_path_relative_is_under_project_root():
    assert resolve_path("data/raw") == PROJECT_ROOT / "data" / "raw"


def test_resolve_path_keeps_absolute_paths(tmp_path: Path):
    assert resolve_path(tmp_path) == tmp_path
