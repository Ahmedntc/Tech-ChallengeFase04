"""Explicabilidade das decisões de crédito (LGPD, art. 20).

O titular pode pedir a revisão de uma decisão automatizada e o controlador deve
informar, quando solicitado, os critérios usados. Para isso, cada score é decomposto
na contribuição de cada variável:

- **XGBoost:** valores SHAP exatos, calculados pela própria biblioteca
  (``pred_contribs``). A soma das contribuições mais o valor de base é igual ao
  log-odds do score, então a explicação reproduz a decisão.
- **Regressão Logística:** coeficiente × valor padronizado (também exato).

Os **motivos da decisão** são as variáveis que mais aumentaram o risco do cliente,
em linguagem simples.

Uso: ``python -m credit_monitoring.governance.explain LOTE.csv CLIENT_ID``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline

from credit_monitoring.config import load_settings, resolve_path
from credit_monitoring.contracts.gate import load_batch
from credit_monitoring.data.columns import ID_COLUMN
from credit_monitoring.models.artifacts import ModelBundle
from credit_monitoring.reporting import num, pct

MONTH_NAMES = {1: "no último mês", 2: "há 2 meses", 3: "há 3 meses"}
EDUCATION = {1: "pós-graduação", 2: "universidade", 3: "ensino médio", 4: "outros"}
MARRIAGE = {1: "casado(a)", 2: "solteiro(a)", 3: "outros"}
SEX = {1: "masculino", 2: "feminino"}


def signed(value: float, digits: int = 3) -> str:
    return ("+" if value >= 0 else "−") + num(abs(value), digits)


def _when(k: int) -> str:
    return MONTH_NAMES.get(k, f"há {k} meses")


def feature_label(feature: str) -> str:
    """Nome da variável em linguagem simples."""
    fixed = {
        "limit_bal": "Limite de crédito",
        "sex": "Sexo",
        "education": "Escolaridade",
        "marriage": "Estado civil",
        "age": "Idade",
    }
    if feature in fixed:
        return fixed[feature]
    kind, k = feature.rsplit("_", 1)
    k = int(k)
    return {
        "pay_status": f"Situação do pagamento {_when(k)}",
        "bill_amt": f"Valor da fatura {_when(k)}",
        "pay_amt": f"Valor pago {_when(k)}",
    }[kind]


def describe_value(feature: str, value: Any) -> str:
    """Valor da variável em linguagem simples."""
    if feature.startswith("pay_status"):
        status = int(value)
        if status >= 1:
            return f"atraso de {status} {'mês' if status == 1 else 'meses'}"
        # -1 é documentado na fonte; -2 e 0 seguem a interpretação usual na literatura.
        return {-2: "sem uso do cartão", -1: "pago em dia", 0: "pagamento parcial (rotativo)"}[
            status
        ]
    if feature.startswith(("bill_amt", "pay_amt")) or feature == "limit_bal":
        return f"NT$ {value:,.0f}".replace(",", ".")
    if feature == "age":
        return f"{int(value)} anos"
    labels = {"education": EDUCATION, "marriage": MARRIAGE, "sex": SEX}
    return labels[feature].get(int(value), str(value))


def contributions(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """Contribuição de cada variável para o log-odds do score (positivo = mais risco).

    Inclui a coluna ``_base``: o valor de partida, igual para todos os clientes.
    """
    X = df[bundle.features]
    estimator = bundle.estimator
    if isinstance(estimator, xgb.XGBClassifier):
        values = estimator.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return pd.DataFrame(values, columns=[*bundle.features, "_base"], index=df.index)
    if isinstance(estimator, Pipeline):
        scaler, model = estimator.named_steps["scaler"], estimator.named_steps["model"]
        values = scaler.transform(X) * model.coef_[0]
        frame = pd.DataFrame(values, columns=bundle.features, index=df.index)
        frame["_base"] = model.intercept_[0]
        return frame
    raise TypeError(f"Modelo sem suporte a explicação: {type(estimator).__name__}")


def global_importance(bundle: ModelBundle, df: pd.DataFrame) -> pd.Series:
    """Importância global: média do valor absoluto das contribuições."""
    contrib = contributions(bundle, df).drop(columns="_base")
    return contrib.abs().mean().sort_values(ascending=False)


def reason_codes(
    bundle: ModelBundle, df: pd.DataFrame, top_n: int = 3
) -> list[list[dict[str, Any]]]:
    """Para cada cliente, as variáveis que mais aumentaram o risco, da maior para a menor."""
    contrib = contributions(bundle, df).drop(columns="_base")
    result = []
    for idx, row in contrib.iterrows():
        top = row[row > 0].sort_values(ascending=False).head(top_n)
        result.append(
            [
                {
                    "feature": feature,
                    "motivo": feature_label(feature),
                    "valor": describe_value(feature, df.loc[idx, feature]),
                    "contribuicao": float(value),
                }
                for feature, value in top.items()
            ]
        )
    return result


def explain_client(bundle: ModelBundle, df: pd.DataFrame, client_id: str) -> dict[str, Any]:
    """Explicação completa de uma decisão, para atender a um pedido de revisão."""
    rows = df[df[ID_COLUMN] == client_id]
    if rows.empty:
        raise KeyError(f"Cliente {client_id} não encontrado no lote.")
    row = rows.iloc[[0]]
    score = float(bundle.predict_proba(row)[0])
    contrib = contributions(bundle, row).iloc[0]
    ordered = contrib.drop("_base").sort_values(key=np.abs, ascending=False)
    return {
        "client_id": client_id,
        "model_version": bundle.version,
        "score": score,
        "threshold": bundle.threshold,
        "decisao": "recusado" if score >= bundle.threshold else "aprovado",
        "motivos": reason_codes(bundle, row)[0],
        "contribuicoes": [
            {
                "variavel": feature_label(feature),
                "valor": describe_value(feature, row.iloc[0][feature]),
                "contribuicao": float(value),
            }
            for feature, value in ordered.items()
        ],
        "base": float(contrib["_base"]),
    }


def format_explanation(explanation: dict[str, Any]) -> str:
    lines = [
        f"Cliente {explanation['client_id']} · modelo {explanation['model_version']}",
        f"Decisão: {explanation['decisao'].upper()} "
        f"(score {pct(explanation['score'])}, limiar {pct(explanation['threshold'])})",
        "",
        "Principais motivos que aumentaram o risco:",
    ]
    for i, reason in enumerate(explanation["motivos"], start=1):
        lines.append(f"  {i}. {reason['motivo']}: {reason['valor']}")
    if not explanation["motivos"]:
        lines.append("  nenhuma variável aumentou o risco deste cliente")
    lines += ["", "Contribuição de cada variável (log-odds; positivo = mais risco):"]
    for item in explanation["contribuicoes"]:
        lines.append(f"  {signed(item['contribuicao'])}  {item['variavel']}: {item['valor']}")
    lines.append(f"  {signed(explanation['base'])}  valor de base do modelo")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explica a decisão de crédito de um cliente.")
    parser.add_argument("batch", type=Path, help="Lote com o cliente (.csv ou .parquet)")
    parser.add_argument("client_id", help="client_id pseudonimizado")
    args = parser.parse_args(argv)

    settings = load_settings()
    bundle = ModelBundle.load(resolve_path(settings["paths"]["model_file"]))
    try:
        explanation = explain_client(bundle, load_batch(args.batch), args.client_id)
    except KeyError as exc:
        print(exc.args[0])
        return 1
    print(format_explanation(explanation))
    return 0


if __name__ == "__main__":
    sys.exit(main())
