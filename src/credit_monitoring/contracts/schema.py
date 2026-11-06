"""Contrato de dados das propostas de crédito (Pandera).

O contrato é a fronteira entre os sistemas de origem e o modelo: qualquer lote
que viole uma regra **rígida** é bloqueado inteiro e enviado para quarentena.
Regras de **alerta** apenas sinalizam situações incomuns, sem bloquear.

Os limites abaixo descrevem o que é *fisicamente plausível* para o negócio, e
não a distribuição observada no treino: um lote com drift (ex.: limites mais
altos por inflação) continua válido. Mudança de distribuição é tratada pelo
monitoramento de drift, não pelo contrato.

Cada checagem leva no nome o ID da regra (ex.: ``R3_idade``), o que permite
associar cada falha reportada pelo Pandera à regra correspondente.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa

from credit_monitoring.data.columns import (
    BILL_AMT,
    FEATURES,
    ID_COLUMN,
    PAY_AMT,
    PAY_STATUS,
    TARGET,
)

AGE_RANGE = (18, 100)
LIMIT_BAL_MAX = 5_000_000
PAY_STATUS_RANGE = (-2, 9)
DOMAINS = {
    "sex": [1, 2],
    "education": [1, 2, 3, 4],
    "marriage": [1, 2, 3],
    TARGET: [0, 1],
}
OVER_LIMIT_FACTOR = 1.5


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    description: str
    blocking: bool


RULES: dict[str, Rule] = {
    rule.rule_id: rule
    for rule in [
        Rule(
            "R1",
            "Schema estrito",
            "Todas as colunas obrigatórias presentes, nenhuma coluna extra "
            "(evita vazamento de dados não previstos, como CPF) e valores numéricos válidos.",
            True,
        ),
        Rule("R2", "Sem valores nulos", "Nenhuma coluna pode ter valores ausentes.", True),
        Rule(
            "R3",
            "Idade válida",
            f"Idade entre {AGE_RANGE[0]} e {AGE_RANGE[1]} anos.",
            True,
        ),
        Rule(
            "R4",
            "Limite de crédito válido",
            f"Limite de crédito maior que zero e até {LIMIT_BAL_MAX:_}.".replace("_", "."),
            True,
        ),
        Rule(
            "R5",
            "Domínios categóricos",
            "Sexo em {1, 2}, escolaridade em {1, 2, 3, 4}, estado civil em {1, 2, 3}, "
            f"status de pagamento entre {PAY_STATUS_RANGE[0]} e {PAY_STATUS_RANGE[1]} "
            "e alvo (quando presente) em {0, 1}.",
            True,
        ),
        Rule(
            "R6",
            "Pagamentos não negativos",
            "Valores pagos (pay_amt_*) maiores ou iguais a zero.",
            True,
        ),
        Rule(
            "R7",
            "Sem duplicatas",
            "client_id único e nenhum registro idêntico a outro (reenvio em duplicidade).",
            True,
        ),
        Rule(
            "W1",
            "Fatura negativa",
            "Fatura com saldo credor (valor negativo): possível, mas incomum.",
            False,
        ),
        Rule(
            "W2",
            "Uso acima do limite",
            f"Fatura mais recente acima de {str(OVER_LIMIT_FACTOR).replace('.', ',')}x "
            "o limite de crédito.",
            False,
        ),
    ]
}


def _value_check(name: str, condition: Callable[[pd.Series], pd.Series]) -> pa.Check:
    """Checagem de valor tolerante a tipo.

    Valores não numéricos são ignorados aqui porque já reprovam na R1 (tipo). Sem isso,
    um único texto inválido numa coluna impediria o Pandera de checar os demais valores
    dela, escondendo outras violações. Nulos também são ignorados (cobertos pela R2).
    """

    def check(series: pd.Series) -> pd.Series:
        values = pd.to_numeric(series, errors="coerce")
        return values.isna() | condition(values)

    return pa.Check(check, name=name)


def _domain(column: str) -> pa.Check:
    allowed = DOMAINS[column]
    return _value_check("R5_dominio", lambda values: values.isin(allowed))


def _int_column(*checks: pa.Check) -> pa.Column:
    return pa.Column(int, checks=list(checks), nullable=False, coerce=True)


def _float_column(*checks: pa.Check) -> pa.Column:
    return pa.Column(float, checks=list(checks), nullable=False, coerce=True)


def _no_duplicate_records(df: pd.DataFrame) -> pd.Series:
    """Marca como inválida toda repetição de um registro (ignorando o client_id).

    Só compara as colunas do contrato e só roda com todas elas presentes: sem uma
    coluna, registros diferentes poderiam parecer iguais (a ausência já reprova na R1).
    """
    if not set(FEATURES) <= set(df.columns):
        return pd.Series(True, index=df.index)
    columns = [col for col in (*FEATURES, TARGET) if col in df.columns]
    return ~df[columns].duplicated(keep="first")


def build_contract(labeled: bool = False) -> pa.DataFrameSchema:
    """Contrato rígido. ``labeled=True`` inclui o alvo (dataset de treino)."""
    columns: dict[str, pa.Column] = {
        ID_COLUMN: pa.Column(str, nullable=False, unique=True, report_duplicates="exclude_first"),
        "limit_bal": _float_column(
            _value_check("R4_limite_credito", lambda v: (v > 0) & (v <= LIMIT_BAL_MAX))
        ),
        "sex": _int_column(_domain("sex")),
        "education": _int_column(_domain("education")),
        "marriage": _int_column(_domain("marriage")),
        "age": _int_column(_value_check("R3_idade", lambda v: v.between(*AGE_RANGE))),
        **{
            col: _int_column(
                _value_check("R5_status_pagamento", lambda v: v.between(*PAY_STATUS_RANGE))
            )
            for col in PAY_STATUS
        },
        **{col: _float_column() for col in BILL_AMT},
        **{
            col: _float_column(_value_check("R6_pagamento_negativo", lambda v: v >= 0))
            for col in PAY_AMT
        },
    }
    if labeled:
        columns[TARGET] = _int_column(_domain(TARGET))

    return pa.DataFrameSchema(
        columns,
        checks=[pa.Check(_no_duplicate_records, name="R7_registro_duplicado")],
        strict=True,
        name="credit_labeled_contract" if labeled else "credit_features_contract",
    )


def _bill_within_limit(df: pd.DataFrame) -> pd.Series:
    return df["bill_amt_1"] <= OVER_LIMIT_FACTOR * df["limit_bal"]


def build_warning_checks() -> pa.DataFrameSchema:
    """Regras de alerta. Só são avaliadas em lotes que já passaram no contrato."""
    return pa.DataFrameSchema(
        {
            col: pa.Column(float, pa.Check(lambda s: s >= 0, name="W1_fatura_negativa"))
            for col in BILL_AMT
        },
        checks=[pa.Check(_bill_within_limit, name="W2_uso_acima_do_limite")],
        name="credit_warning_checks",
    )
