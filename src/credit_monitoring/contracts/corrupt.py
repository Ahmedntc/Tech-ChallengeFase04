"""Gerador de lote corrompido para demonstrar o contrato bloqueando a ingestão.

Cada tipo de erro é injetado em registros diferentes, e o manifesto devolvido
diz exatamente o que foi injetado, para conferir se o contrato pega tudo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Coluna que não existe no contrato: simula um sistema de origem vazando dado pessoal.
LEAKED_COLUMN = "cpf"


def make_corrupted_batch(clean: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict[str, str]]:
    """Injeta violações de todas as regras rígidas (R1 a R7) em um lote válido."""
    if len(clean) < 30:
        raise ValueError("O lote base precisa de pelo menos 30 registros.")
    df = clean.reset_index(drop=True).astype(object)
    rows = iter(np.random.default_rng(seed).permutation(len(df)).tolist())

    def take(n: int) -> list[int]:
        return [next(rows) for _ in range(n)]

    manifest: dict[str, str] = {}

    df[LEAKED_COLUMN] = [f"000.000.{i % 1000:03d}-00" for i in range(len(df))]
    df.loc[take(1), "age"] = "trinta"
    manifest["R1"] = f"coluna extra '{LEAKED_COLUMN}' (dado pessoal vazado) e idade em texto"

    df.loc[take(3), "limit_bal"] = np.nan
    df.loc[take(1), "education"] = np.nan
    manifest["R2"] = "limite de crédito nulo (3) e escolaridade nula (1)"

    df.loc[take(2), "age"] = 15
    df.loc[take(1), "age"] = 130
    manifest["R3"] = "idade 15 (2) e 130 (1)"

    df.loc[take(2), "limit_bal"] = -5000.0
    df.loc[take(1), "limit_bal"] = 0.0
    manifest["R4"] = "limite de crédito negativo (2) e zero (1)"

    df.loc[take(1), "sex"] = 7
    df.loc[take(1), "education"] = 9
    df.loc[take(1), "pay_status_1"] = 12
    manifest["R5"] = "sexo 7, escolaridade 9 e status de pagamento 12"

    df.loc[take(2), "pay_amt_2"] = -100.0
    manifest["R6"] = "pagamento negativo (2)"

    duplicated = df.loc[take(3)]
    df = pd.concat([df, duplicated], ignore_index=True)
    manifest["R7"] = "3 registros reenviados em duplicidade"

    return df, manifest
