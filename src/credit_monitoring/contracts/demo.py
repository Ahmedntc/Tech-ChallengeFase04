"""Demonstração do contrato: um lote válido é aprovado e um lote corrompido é bloqueado.

Os dois lotes chegam como CSV na área de entrada (``data/incoming/``), como viriam
de um sistema de origem, e passam pelo mesmo gate de ingestão.

Uso: ``python -m credit_monitoring.contracts.demo``
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import pandas as pd

from credit_monitoring.config import load_settings, resolve_path
from credit_monitoring.contracts.corrupt import make_corrupted_batch
from credit_monitoring.contracts.gate import format_summary, ingest_batch
from credit_monitoring.data.columns import TARGET

VALID_BATCH = "lote_valido"
CORRUPTED_BATCH = "lote_corrompido"


def build_demo_batches(settings: dict[str, Any]) -> dict[str, str]:
    """Gera os lotes de demonstração a partir do Pool de Produção."""
    seed = settings["project"]["random_seed"]
    size = settings["contract_demo"]["batch_size"]
    pool = pd.read_parquet(resolve_path(settings["paths"]["production_pool_file"]))
    # Lotes de produção chegam sem o alvo: a inadimplência só é conhecida meses depois.
    features = pool.drop(columns=[TARGET])
    sample = features.sample(2 * size, random_state=seed)
    valid, base_for_corruption = sample.iloc[:size], sample.iloc[size:]
    corrupted, manifest = make_corrupted_batch(base_for_corruption, seed)

    incoming = resolve_path(settings["paths"]["incoming_dir"])
    incoming.mkdir(parents=True, exist_ok=True)
    valid.to_csv(incoming / f"{VALID_BATCH}.csv", index=False)
    corrupted.to_csv(incoming / f"{CORRUPTED_BATCH}.csv", index=False)
    return manifest


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    manifest = build_demo_batches(settings)
    incoming = resolve_path(settings["paths"]["incoming_dir"])

    print("Erros injetados no lote corrompido:")
    for rule_id, description in manifest.items():
        print(f"  {rule_id}: {description}")

    results = {}
    for name in (VALID_BATCH, CORRUPTED_BATCH):
        results[name] = ingest_batch(incoming / f"{name}.csv", settings)
        print()
        print(format_summary(results[name]))

    reports_dir = settings["paths"]["validation_reports_dir"]
    print(f"\nRelatórios em {reports_dir}/ (HTML e JSON).")

    missed = set(manifest) - set(results[CORRUPTED_BATCH].violated_rules)
    ok = results[VALID_BATCH].passed and not results[CORRUPTED_BATCH].passed and not missed
    if not ok:
        print(f"Resultado inesperado. Regras não detectadas: {sorted(missed)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
