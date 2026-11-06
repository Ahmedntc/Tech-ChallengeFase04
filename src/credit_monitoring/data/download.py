"""Obtenção e verificação de integridade do dataset bruto.

A planilha original da UCI fica versionada em ``data/raw/``. Se estiver ausente,
ela é baixada da fonte oficial. Em ambos os casos o conteúdo é conferido contra
uma impressão digital SHA-256 registrada em ``config/settings.yaml``, garantindo
que todos os experimentos partem exatamente dos mesmos dados.

Uso: ``python -m credit_monitoring.data.download``
"""

from __future__ import annotations

import hashlib
import io
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from credit_monitoring.config import load_settings, resolve_path

logger = logging.getLogger(__name__)


class DatasetIntegrityError(RuntimeError):
    """O conteúdo do dataset não corresponde ao esperado."""


def read_raw_dataset(path: Path, header_row: int) -> pd.DataFrame:
    """Lê a planilha original da UCI."""
    return pd.read_excel(path, header=header_row, engine="calamine")


def content_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 do conteúdo tabular serializado como CSV canônico.

    Independe do formato do arquivo (xls, csv...), então detecta qualquer
    alteração nos dados sem depender de metadados da planilha.
    """
    canonical = df.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_dataset(df: pd.DataFrame, expected_sha256: str, expected_shape: list[int]) -> None:
    """Garante que o dataset tem o formato e o conteúdo esperados."""
    if list(df.shape) != list(expected_shape):
        raise DatasetIntegrityError(
            f"Formato inesperado: {df.shape}, esperado {tuple(expected_shape)}."
        )
    actual = content_fingerprint(df)
    if actual != expected_sha256:
        raise DatasetIntegrityError(
            f"Impressão digital divergente: {actual}, esperado {expected_sha256}."
        )


def download_raw_dataset(url: str, destination: Path) -> Path:
    """Baixa o zip oficial da UCI e grava a planilha em ``destination``."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        spreadsheets = [name for name in archive.namelist() if name.lower().endswith(".xls")]
        if not spreadsheets:
            raise DatasetIntegrityError(f"Nenhuma planilha .xls encontrada em {url}.")
        destination.write_bytes(archive.read(spreadsheets[0]))
    return destination


def ensure_raw_dataset(settings: dict[str, Any] | None = None) -> pd.DataFrame:
    """Garante que o dataset bruto existe localmente e está íntegro."""
    settings = settings or load_settings()
    dataset = settings["dataset"]
    path = resolve_path(dataset["raw_file"])

    if not path.exists():
        logger.info("Dataset bruto ausente; baixando de %s", dataset["url"])
        download_raw_dataset(dataset["url"], path)

    df = read_raw_dataset(path, dataset["header_row"])
    verify_dataset(df, dataset["content_sha256"], dataset["expected_shape"])
    logger.info("Dataset íntegro: %s", path)
    return df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    try:
        df = ensure_raw_dataset(settings)
    except DatasetIntegrityError as exc:
        logger.error("Falha na verificação do dataset: %s", exc)
        return 1

    target = settings["dataset"]["target_column"]
    print(f"Dataset: {settings['dataset']['name']} ({settings['dataset']['source']})")
    print(f"Linhas: {len(df):,} | Colunas: {df.shape[1]}")
    print(f"Taxa de inadimplência: {df[target].mean():.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
