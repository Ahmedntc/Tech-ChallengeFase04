"""Privacidade: risco de reidentificação e retenção de dados.

**Risco de reidentificação (k-anonimato).** O ``client_id`` é pseudonimizado, mas um
cliente ainda pode ser reconhecido pela combinação de atributos comuns (idade, sexo,
escolaridade...). Um registro é *k-anônimo* quando pelo menos k clientes compartilham
a mesma combinação. Quanto mais registros com k pequeno, maior o risco.

**Retenção.** Cada pasta de dados tem um prazo em dias (``governance.retention`` no
``settings.yaml``). O comando lista os arquivos vencidos e, com ``--apply``, apaga.
Por padrão só lista, para que nada seja apagado sem confirmação.

Uso: ``python -m credit_monitoring.governance.privacy [--apply]``
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from credit_monitoring.config import load_settings, resolve_path
from credit_monitoring.monitoring.logger import configure_logging, log_event

logger = logging.getLogger(__name__)

DAY = 86_400
KEEP = {".gitkeep"}


def k_anonymity(df: pd.DataFrame, quasi_identifiers: list[str], k_min: int) -> dict[str, Any]:
    """Tamanho da classe de cada registro na combinação dos quase-identificadores."""
    sizes = df.groupby(quasi_identifiers, dropna=False)[quasi_identifiers[0]].transform("size")
    return {
        "quasi_identifiers": quasi_identifiers,
        "records": len(df),
        "classes": int(df.groupby(quasi_identifiers, dropna=False).ngroups),
        "min_k": int(sizes.min()),
        "share_below_k": float((sizes < k_min).mean()),
        "share_unique": float((sizes == 1).mean()),
    }


def reidentification_risk(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """k-anonimato para combinações crescentes de quase-identificadores."""
    rows = []
    for extra in [[], *[[col] for col in cfg["extra_identifiers"]]]:
        rows.append(k_anonymity(df, [*cfg["quasi_identifiers"], *extra], cfg["k_min"]))
    return pd.DataFrame(rows)


@dataclass
class ExpiredFile:
    path: Path
    age_days: float
    limit_days: int


def expired_files(settings: dict[str, Any], now: float | None = None) -> list[ExpiredFile]:
    """Arquivos mais antigos que o prazo de retenção da sua pasta."""
    now = now or time.time()
    expired = []
    for folder, days in settings["governance"]["retention"].items():
        root = resolve_path(folder)
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name not in KEEP):
            age = (now - path.stat().st_mtime) / DAY
            if age > days:
                expired.append(ExpiredFile(path, age, days))
    return expired


def apply_retention(
    settings: dict[str, Any], apply: bool = False, now: float | None = None
) -> list[ExpiredFile]:
    """Lista (e, com ``apply=True``, apaga) os arquivos vencidos. Registra no log central."""
    expired = expired_files(settings, now)
    if apply:
        for item in expired:
            item.path.unlink()
    log_event(
        logger,
        logging.INFO,
        "retencao",
        f"Retenção: {len(expired)} arquivo(s) vencido(s)"
        + (" apagado(s)." if apply else " (simulação, nada apagado)."),
        expired=len(expired),
        applied=apply,
        folders=sorted({str(item.path.parent) for item in expired}),
    )
    return expired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aplica o plano de retenção de dados.")
    parser.add_argument("--apply", action="store_true", help="Apaga os arquivos vencidos")
    args = parser.parse_args(argv)
    settings = load_settings()
    configure_logging(settings)

    print("Prazos de retenção:")
    for folder, days in settings["governance"]["retention"].items():
        print(f"  {folder:<32} {days:>5} dias")
    expired = apply_retention(settings, apply=args.apply)
    action = "apagados" if args.apply else "a apagar (use --apply para confirmar)"
    print(f"\nArquivos vencidos {action}: {len(expired)}")
    for item in expired:
        print(f"  {item.path} ({item.age_days:.0f} dias; limite {item.limit_days})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
