"""Simulação da passagem do tempo em produção.

Cada mês recebe um lote amostrado do Pool de Produção (nunca visto no treino) e
transformado segundo um cenário econômico declarado em ``config/settings.yaml``.
Os mecanismos são separados de propósito, para que a análise causal (Etapa 4)
consiga atribuir a degradação a cada um:

| Mecanismo | Variáveis afetadas | Rótulo |
|---|---|---|
| Inflação | faturas (``bill_amt_*``) sobem | mantido: risco real igual |
| Pagamentos | ``pay_amt_*`` variam (renda real menor) | mantido |
| Expansão de crédito | limites (``limit_bal``) maiores | mantido |
| Novos perfis | mais clientes jovens no lote | mantido (amostra real) |
| Desemprego | parte da carteira passa a atrasar | sorteado pela taxa da Referência para o atraso |
| Concept drift | nenhuma | clientes em dia e muito endividados passam a inadimplir mais |

Os lotes chegam sem o alvo em ``data/incoming/`` (como viriam de um sistema de
origem). Os rótulos, conhecidos só depois, ficam em ``data/production/labels/``.

Uso: ``python -m credit_monitoring.drift.simulate``
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_monitoring.config import load_settings, resolve_path
from credit_monitoring.contracts.corrupt import make_corrupted_batch
from credit_monitoring.data.columns import BILL_AMT, FEATURES, ID_COLUMN, PAY_AMT, TARGET
from credit_monitoring.monitoring.logger import configure_logging

logger = logging.getLogger(__name__)

PARTNER_SUFFIX = "_parceiro"


@dataclass(frozen=True)
class MonthScenario:
    month: str
    title: str
    narrative: str = ""
    # Aumento nominal das faturas (0,10 = +10%).
    inflation: float = 0.0
    # Variação nominal dos pagamentos (negativo = clientes pagam menos).
    payment_change: float = 0.0
    # Aumento dos limites de crédito concedidos.
    limit_expansion: float = 0.0
    # Participação de clientes jovens no lote (None = proporção natural do Pool).
    young_share: float | None = None
    # Fração da carteira (entre quem está em dia) que passa a atrasar.
    unemployment: float = 0.0
    # Probabilidade extra de inadimplência no segmento em dia e muito endividado.
    concept_drift: float = 0.0
    # Lote extra enviado por um parceiro: "corrompido" (deve ser barrado pelo contrato)
    # ou "vazio" (arquivo de 0 byte, simulando uma transferência interrompida).
    partner_batch: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MonthScenario:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def load_scenarios(settings: dict[str, Any]) -> list[MonthScenario]:
    return [MonthScenario.from_dict(m) for m in settings["simulation"]["months"]]


def sample_batch(
    pool: pd.DataFrame, size: int, young_share: float | None, young_age_max: int, seed: int
) -> pd.DataFrame:
    """Amostra o lote do mês. Com ``young_share``, controla a participação de jovens."""
    if young_share is None:
        return pool.sample(size, random_state=seed).reset_index(drop=True)
    young = pool[pool["age"] <= young_age_max]
    others = pool[pool["age"] > young_age_max]
    n_young = round(size * young_share)
    batch = pd.concat(
        [
            young.sample(n_young, random_state=seed),
            others.sample(size - n_young, random_state=seed),
        ]
    )
    return batch.sample(frac=1, random_state=seed).reset_index(drop=True)


def apply_inflation(df: pd.DataFrame, inflation: float, rng: np.random.Generator) -> pd.DataFrame:
    """Faturas sobem com a inflação; cada cliente sente um pouco diferente (ruído de 3%)."""
    if inflation == 0:
        return df
    df = df.copy()
    factor = (1 + inflation) * np.exp(rng.normal(0, 0.03, size=(len(df), 1)))
    df[BILL_AMT] = np.round(df[BILL_AMT].to_numpy() * factor)
    return df


def apply_payment_change(df: pd.DataFrame, change: float) -> pd.DataFrame:
    if change == 0:
        return df
    df = df.copy()
    df[PAY_AMT] = np.round(df[PAY_AMT] * (1 + change))
    return df


def apply_limit_expansion(df: pd.DataFrame, expansion: float) -> pd.DataFrame:
    """Limites maiores, arredondados para múltiplos de 10.000 como na política original."""
    if expansion == 0:
        return df
    df = df.copy()
    df["limit_bal"] = np.round(df["limit_bal"] * (1 + expansion) / 10_000) * 10_000
    return df


def apply_unemployment(
    df: pd.DataFrame,
    labels: pd.Series,
    rate: float,
    default_rate_by_delay: dict[int, float],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Series, int]:
    """Parte dos clientes em dia passa a atrasar de 1 a 3 meses.

    O atraso é coerente no histórico (atraso de k meses agora implica k−1 no mês anterior
    e assim por diante) e os pagamentos desses meses zeram. O rótulo é sorteado com a
    taxa de inadimplência observada na Referência para o mesmo atraso, preservando a
    relação entre atraso e inadimplência que o modelo aprendeu.
    """
    if rate == 0:
        return df, labels, 0
    df, labels = df.copy(), labels.copy()
    on_time = df.index[df["pay_status_1"] <= 0]
    affected = rng.choice(on_time, size=round(len(df) * rate), replace=False)
    delays = rng.choice([1, 2, 3], size=len(affected), p=[0.4, 0.4, 0.2])

    for month in range(1, int(delays.max(initial=0)) + 1):
        reached = delays >= month
        rows = affected[reached]
        status_col = f"pay_status_{month}"
        df.loc[rows, status_col] = np.maximum(
            df.loc[rows, status_col].to_numpy(), delays[reached] - month + 1
        )
        df.loc[rows, f"pay_amt_{month}"] = 0.0
    default_probability = np.array([default_rate_by_delay[int(d)] for d in delays])
    labels.loc[affected[rng.random(len(affected)) < default_probability]] = 1
    return df, labels, len(affected)


def concept_segment(df: pd.DataFrame, settings: dict[str, Any]) -> pd.Series:
    """Clientes em dia e muito endividados: parecem seguros para o modelo."""
    segment = settings["simulation"]["concept_segment"]
    utilization = df["bill_amt_1"] / df["limit_bal"]
    return (df["pay_status_1"] <= segment["max_pay_status"]) & (
        utilization >= segment["min_utilization"]
    )


def apply_concept_drift(
    df: pd.DataFrame,
    labels: pd.Series,
    probability: float,
    settings: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[pd.Series, int]:
    """Muda a relação X→y: no segmento, adimplentes passam a inadimplir sem mudar as features."""
    if probability == 0:
        return labels, 0
    labels = labels.copy()
    candidates = labels.index[concept_segment(df, settings) & (labels == 0)]
    flipped = candidates[rng.random(len(candidates)) < probability]
    labels.loc[flipped] = 1
    return labels, len(flipped)


def simulate_month(
    scenario: MonthScenario,
    pool: pd.DataFrame,
    default_rate_by_delay: dict[int, float],
    settings: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Gera o lote (features) e os rótulos de um mês."""
    sim = settings["simulation"]
    rng = np.random.default_rng(seed)
    batch = sample_batch(pool, sim["batch_size"], scenario.young_share, sim["young_age_max"], seed)
    labels = batch.pop(TARGET)
    original_default_rate = float(labels.mean())

    batch = apply_inflation(batch, scenario.inflation, rng)
    batch = apply_payment_change(batch, scenario.payment_change)
    batch = apply_limit_expansion(batch, scenario.limit_expansion)
    batch, labels, n_unemployed = apply_unemployment(
        batch, labels, scenario.unemployment, default_rate_by_delay, rng
    )
    labels, n_concept = apply_concept_drift(batch, labels, scenario.concept_drift, settings, rng)

    stats = {
        "rows": len(batch),
        "young_share": float((batch["age"] <= sim["young_age_max"]).mean()),
        "mean_bill_amt_1": float(batch["bill_amt_1"].mean()),
        "mean_limit_bal": float(batch["limit_bal"].mean()),
        "clients_with_new_delay": n_unemployed,
        "labels_changed_by_concept_drift": n_concept,
        "default_rate_before_shocks": original_default_rate,
        "default_rate": float(labels.mean()),
    }
    return batch[[ID_COLUMN, *FEATURES]], labels, stats


def write_partner_batch(kind: str, path: Path, pool: pd.DataFrame, seed: int) -> None:
    """Grava o lote extra do parceiro: com dados inválidos ou como arquivo vazio."""
    if kind == "corrompido":
        partner = pool.drop(columns=[TARGET]).sample(300, random_state=seed)
        corrupted, _ = make_corrupted_batch(partner, seed)
        corrupted.to_csv(path, index=False)
    elif kind == "vazio":
        path.write_bytes(b"")
    else:
        raise ValueError(f"Tipo de lote de parceiro desconhecido: {kind}")


def simulate(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Gera todos os meses do cenário e grava lotes, rótulos e manifesto."""
    settings = settings or load_settings()
    paths = settings["paths"]
    seed = settings["project"]["random_seed"]
    pool = pd.read_parquet(resolve_path(paths["production_pool_file"]))
    reference = pd.read_parquet(resolve_path(paths["reference_file"]))
    default_rate_by_delay = (
        reference.groupby("pay_status_1")[TARGET].mean().reindex([1, 2, 3]).to_dict()
    )

    incoming = resolve_path(paths["incoming_dir"])
    labels_dir = resolve_path(paths["labels_dir"])
    incoming.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"seed": seed, "months": []}
    for i, scenario in enumerate(load_scenarios(settings), start=1):
        month_seed = seed + i
        batch, labels, stats = simulate_month(
            scenario, pool, default_rate_by_delay, settings, month_seed
        )
        batch.to_csv(incoming / f"{scenario.month}.csv", index=False)
        pd.DataFrame({ID_COLUMN: batch[ID_COLUMN], TARGET: labels}).to_parquet(
            labels_dir / f"{scenario.month}.parquet", index=False
        )
        files = [f"{scenario.month}.csv"]
        if scenario.partner_batch:
            name = f"{scenario.month}{PARTNER_SUFFIX}.csv"
            write_partner_batch(scenario.partner_batch, incoming / name, pool, month_seed)
            files.append(name)
        manifest["months"].append({**asdict(scenario), "files": files, "stats": stats})
        logger.info(
            "%s (%s): %d registros, inadimplência %.1f%%",
            scenario.month,
            scenario.title,
            stats["rows"],
            100 * stats["default_rate"],
        )

    manifest_path = resolve_path(paths["simulation_manifest"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    configure_logging()
    manifest = simulate()
    print(f"\n{'mês':<5}{'cenário':<28}{'jovens':>8}{'fatura média':>14}{'inadimplência':>15}")
    for month in manifest["months"]:
        s = month["stats"]
        print(
            f"{month['month']:<5}{month['title']:<28}{s['young_share']:>8.1%}"
            f"{s['mean_bill_amt_1']:>14,.0f}{s['default_rate']:>15.1%}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
