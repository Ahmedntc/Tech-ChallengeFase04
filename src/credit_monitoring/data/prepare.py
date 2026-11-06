"""Preparação do dataset: padronização, limpeza, pseudonimização e separação.

Gera duas camadas a partir do dado bruto da UCI:

- **Referência** (``data/reference/``): única base usada no treino do modelo.
- **Pool de Produção** (``data/production/``): nunca visto no treino; é a base
  dos lotes mensais simulados onde o drift vai acontecer.

Uso: ``python -m credit_monitoring.data.prepare``
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import sys
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_monitoring.config import load_settings, resolve_path
from credit_monitoring.contracts.gate import format_summary, validate_batch
from credit_monitoring.data.columns import ID_COLUMN, MONETARY, RAW_COLUMN_MAP, TARGET
from credit_monitoring.data.download import ensure_raw_dataset

logger = logging.getLogger(__name__)

# Códigos não documentados (0) ou "desconhecido" (5, 6) viram "outros" (4).
EDUCATION_REMAP = {0: 4, 5: 4, 6: 4}
# Código não documentado (0) vira "outros" (3).
MARRIAGE_REMAP = {0: 3}


@dataclass
class CleaningReport:
    rows_in: int
    education_remapped: int
    marriage_remapped: int
    duplicates_removed: int
    rows_out: int


def standardize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Renomeia as colunas da UCI para nomes legíveis e padronizados."""
    return raw.rename(columns=RAW_COLUMN_MAP)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Corrige códigos inválidos, remove duplicatas e padroniza tipos."""
    rows_in = len(df)
    education_mask = df["education"].isin(EDUCATION_REMAP.keys())
    marriage_mask = df["marriage"].isin(MARRIAGE_REMAP.keys())
    df = df.assign(
        education=df["education"].replace(EDUCATION_REMAP),
        marriage=df["marriage"].replace(MARRIAGE_REMAP),
    )
    # Registros idênticos (ignorando o ID) não trazem informação nova e, se um cair
    # no treino e o outro no teste, inflam artificialmente as métricas.
    duplicated = df.drop(columns="id").duplicated(keep="first")
    df = df.loc[~duplicated].reset_index(drop=True)
    df[MONETARY] = df[MONETARY].astype("float64")

    report = CleaningReport(
        rows_in=rows_in,
        education_remapped=int(education_mask.sum()),
        marriage_remapped=int(marriage_mask.sum()),
        duplicates_removed=int(duplicated.sum()),
        rows_out=len(df),
    )
    return df, report


def load_pseudonymization_key(settings: dict[str, Any]) -> bytes:
    """Obtém a chave secreta da pseudonimização.

    Ordem: variável de ambiente; senão, arquivo local fora do controle de versão
    (gerado na primeira execução). A chave nunca fica junto dos dados (LGPD, art. 13, §4º).
    """
    cfg = settings["pseudonymization"]
    from_env = os.environ.get(cfg["key_env_var"])
    if from_env:
        return from_env.encode("utf-8")

    key_file = resolve_path(cfg["key_file"])
    if not key_file.exists():
        key_file.parent.mkdir(parents=True, exist_ok=True)
        key_file.write_text(secrets.token_hex(32), encoding="utf-8")
        key_file.chmod(0o600)
        logger.warning("Chave de pseudonimização gerada em %s (não versionar).", key_file)
    return key_file.read_text(encoding="utf-8").strip().encode("utf-8")


def pseudonymize_ids(ids: pd.Series, key: bytes) -> pd.Series:
    """Substitui o identificador por um HMAC-SHA256 truncado (irreversível sem a chave)."""
    return ids.astype(str).map(
        lambda value: hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
    )


def apply_pseudonymization(df: pd.DataFrame, key: bytes) -> pd.DataFrame:
    """Troca a coluna ``id`` original pelo ``client_id`` pseudonimizado."""
    df = df.copy()
    df.insert(0, ID_COLUMN, pseudonymize_ids(df.pop("id"), key))
    return df


def split_reference_production(
    df: pd.DataFrame, production_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separação única e reprodutível, estratificada pelo alvo."""
    reference, production = train_test_split(
        df, test_size=production_fraction, stratify=df[TARGET], random_state=seed
    )
    return reference.reset_index(drop=True), production.reset_index(drop=True)


def prepare(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Executa a preparação completa e grava as camadas de Referência e Produção."""
    settings = settings or load_settings()
    raw = ensure_raw_dataset(settings)
    df, cleaning = clean(standardize_columns(raw))
    df = apply_pseudonymization(df, load_pseudonymization_key(settings))

    # O dataset preparado precisa cumprir o contrato antes de virar base de treino.
    validation = validate_batch(df, "dataset_preparado", labeled=True)
    if not validation.passed:
        raise ValueError("Dataset preparado viola o contrato:\n" + format_summary(validation))

    reference, production = split_reference_production(
        validation.data,
        settings["split"]["production_fraction"],
        settings["project"]["random_seed"],
    )
    paths = settings["paths"]
    outputs = {"reference": reference, "production_pool": production}
    for name, frame in outputs.items():
        path = resolve_path(paths[f"{name}_file"])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        logger.info("Gravado %s: %s (%d linhas)", name, path, len(frame))

    return {
        "cleaning": asdict(cleaning),
        "reference_rows": len(reference),
        "production_pool_rows": len(production),
        "reference_default_rate": float(reference[TARGET].mean()),
        "production_default_rate": float(production[TARGET].mean()),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    summary = prepare()
    cleaning = summary["cleaning"]
    print("Limpeza do dado bruto:")
    print(f"  linhas recebidas ............ {cleaning['rows_in']:>6}")
    print(f"  education remapeada (0/5/6) . {cleaning['education_remapped']:>6}")
    print(f"  marriage remapeada (0) ...... {cleaning['marriage_remapped']:>6}")
    print(f"  duplicatas removidas ........ {cleaning['duplicates_removed']:>6}")
    print(f"  linhas finais ............... {cleaning['rows_out']:>6}")
    print("Separação:")
    print(
        f"  Referência ...... {summary['reference_rows']:>6} linhas "
        f"(inadimplência {summary['reference_default_rate']:.2%})"
    )
    print(
        f"  Pool de Produção  {summary['production_pool_rows']:>6} linhas "
        f"(inadimplência {summary['production_default_rate']:.2%})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
