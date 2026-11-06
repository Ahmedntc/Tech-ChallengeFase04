"""Carregamento da configuração central (config/settings.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def load_settings(path: Path | None = None) -> dict[str, Any]:
    """Lê o arquivo de configuração YAML e devolve um dicionário."""
    settings_path = path or DEFAULT_SETTINGS_PATH
    with settings_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_path(path: str | Path) -> Path:
    """Converte um caminho relativo à raiz do repositório em caminho absoluto."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
