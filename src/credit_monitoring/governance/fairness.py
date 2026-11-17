"""Equidade do modelo entre grupos (LGPD, art. 6º, IX: não discriminação).

Para cada atributo (sexo, faixa etária, estado civil, escolaridade), compara os
grupos em:

- **taxa de aprovação** → *impacto desproporcional* (DI): menor taxa / maior taxa.
  A referência usual é a regra dos 4/5 (DI ≥ 0,80).
- **taxa de recusa indevida**: entre os clientes que **pagaram em dia**, quantos o
  modelo recusou. A diferença entre grupos é a de *igualdade de oportunidade*: bons
  pagadores devem ter a mesma chance de aprovação, qualquer que seja o grupo.
- **AUC por grupo:** se o modelo ordena o risco igualmente bem em todos os grupos.

Grupos com menos de ``min_group_size`` clientes aparecem nas tabelas, mas ficam
fora dos indicadores, porque suas taxas são instáveis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from credit_monitoring.data.columns import FEATURES, TARGET
from credit_monitoring.governance.explain import EDUCATION, MARRIAGE, SEX
from credit_monitoring.models.train import split_reference

AGE_BAND = "age_band"
ATTRIBUTE_LABELS = {
    "sex": "Sexo",
    AGE_BAND: "Faixa etária",
    "marriage": "Estado civil",
    "education": "Escolaridade",
}
GROUP_LABELS = {"sex": SEX, "marriage": MARRIAGE, "education": EDUCATION}


def add_age_band(df: pd.DataFrame, cuts: list[int]) -> pd.DataFrame:
    """Faixas etárias: até c1, c1+1 a c2, ..., acima do último corte."""
    edges = [0, *cuts, 200]
    labels = [f"até {cuts[0]}"]
    labels += [f"{lo + 1} a {hi}" for lo, hi in zip(cuts, cuts[1:], strict=False)]
    labels += [f"{cuts[-1] + 1} ou mais"]
    return df.assign(**{AGE_BAND: pd.cut(df["age"], edges, labels=labels).astype(str)})


def group_label(attribute: str, value: Any) -> str:
    mapping = GROUP_LABELS.get(attribute)
    return mapping.get(int(value), str(value)) if mapping else str(value)


def group_metrics(df: pd.DataFrame, attribute: str, threshold: float) -> pd.DataFrame:
    """Métricas por grupo. ``df`` precisa de ``score`` e do alvo."""
    rows = []
    refused = df["score"] >= threshold
    for value, group in df.groupby(attribute, sort=True):
        is_refused = refused.loc[group.index]
        good = group[TARGET] == 0
        rows.append(
            {
                "attribute": attribute,
                "group": group_label(attribute, value),
                "n": len(group),
                "default_rate": float(group[TARGET].mean()),
                "mean_score": float(group["score"].mean()),
                "approval_rate": float(1 - is_refused.mean()),
                "false_refusal_rate": float(is_refused[good].mean()),
                "recall": float(is_refused[~good].mean()) if (~good).any() else np.nan,
                "auc": (
                    float(roc_auc_score(group[TARGET], group["score"]))
                    if group[TARGET].nunique() == 2
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def indicators(groups: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Indicadores de equidade de um atributo, só com os grupos de tamanho confiável."""
    valid = groups[groups["n"] >= cfg["min_group_size"]]
    approval = valid["approval_rate"]
    refusal = valid["false_refusal_rate"]
    result = {
        "attribute": groups["attribute"].iloc[0],
        "groups_used": len(valid),
        "disparate_impact": float(approval.min() / approval.max()),
        "equal_opportunity_gap": float(refusal.max() - refusal.min()),
        "auc_gap": float(valid["auc"].max() - valid["auc"].min()),
        "most_refused_group": valid.loc[refusal.idxmax(), "group"],
    }
    result["passes_disparate_impact"] = result["disparate_impact"] >= cfg["disparate_impact_min"]
    result["passes_equal_opportunity"] = (
        result["equal_opportunity_gap"] <= cfg["equal_opportunity_max_gap"]
    )
    return result


def evaluate(
    df: pd.DataFrame, threshold: float, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tabela por grupo e indicadores para todos os atributos configurados."""
    df = add_age_band(df, cfg["age_bands"])
    tables = [group_metrics(df, attr, threshold) for attr in cfg["attributes"]]
    return (
        pd.concat(tables, ignore_index=True),
        pd.DataFrame([indicators(t, cfg) for t in tables]),
    )


def _fit_and_evaluate(
    settings: dict[str, Any],
    features: list[str],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    refusal_rate: float,
    seed: int,
) -> dict[str, float]:
    """Treina o XGBoost com as features dadas e mede desempenho e equidade no teste.

    O limiar é o quantil do score na validação que produz a taxa de recusa pedida.
    """
    model = XGBClassifier(
        **settings["model"]["xgboost"], eval_metric="logloss", random_state=seed
    ).fit(train[features], train[TARGET])
    threshold = float(np.quantile(model.predict_proba(valid[features])[:, 1], 1 - refusal_rate))
    scored = test.assign(score=model.predict_proba(test[features])[:, 1])
    _, ind = evaluate(scored, threshold, settings["governance"]["fairness"])
    row = {
        "auc": float(roc_auc_score(test[TARGET], scored["score"])),
        "approval_rate": float((scored["score"] < threshold).mean()),
    }
    for item in ind.itertuples():
        row[f"di_{item.attribute}"] = item.disparate_impact
        row[f"eo_{item.attribute}"] = item.equal_opportunity_gap
    return row


def compare_variants(
    settings: dict[str, Any], reference: pd.DataFrame, refusal_rate: float
) -> pd.DataFrame:
    """Mitigação: treina variantes do modelo sem certos atributos e compara a equidade.

    Todas as variantes são comparadas com a **mesma taxa de recusa** (``refusal_rate``, a
    do modelo em produção na validação). Sem isso, um limiar mais permissivo pareceria
    "mais justo" só por recusar menos gente. Cada variante é treinada com várias sementes.
    """
    cfg = settings["governance"]["fairness"]
    train, valid, test = split_reference(reference, settings)
    rows = []
    for name, excluded in cfg["variants"].items():
        features = [f for f in FEATURES if f not in excluded]
        for seed in cfg["seeds"]:
            row = _fit_and_evaluate(settings, features, train, valid, test, refusal_rate, seed)
            rows.append({"variant": name, "excluded": ", ".join(excluded) or "nenhum", **row})
    runs = pd.DataFrame(rows)
    summary = runs.groupby(["variant", "excluded"], sort=False).mean()
    summary["auc_std"] = runs.groupby("variant", sort=False)["auc"].std().to_numpy()
    return summary.reset_index()


# Métricas em que um valor maior é melhor (nas demais, menor é melhor).
HIGHER_IS_BETTER = ("auc", "di_")


def cross_validate_variants(
    settings: dict[str, Any], reference: pd.DataFrame, refusal_rate: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validação cruzada repetida e pareada entre duas variantes do modelo.

    A Referência é dividida em k partições estratificadas, r vezes com embaralhamentos
    diferentes. Em cada rodada, uma partição é o teste e o restante vira treino (80%) e
    validação (20%, usada só para fixar o limiar na mesma taxa de recusa). As duas
    variantes são avaliadas **no mesmo teste**, então a diferença entre elas não sofre
    com a variação de amostra de uma partição para outra.

    Devolve as métricas por rodada e, para cada métrica, a diferença média
    (candidata − base), o intervalo de 95% (t de Student, aproximado porque as rodadas
    compartilham dados de treino) e em quantas rodadas a candidata foi melhor.
    """
    from scipy import stats
    from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

    cfg = settings["governance"]["fairness"]
    cv = cfg["cross_validation"]
    seed = settings["project"]["random_seed"]
    compared = {name: cfg["variants"][name] for name in (cv["baseline"], cv["candidate"])}
    folds = RepeatedStratifiedKFold(
        n_splits=cv["splits"], n_repeats=cv["repeats"], random_state=seed
    )
    rows = []
    for fold, (train_idx, test_idx) in enumerate(folds.split(reference, reference[TARGET])):
        rest, test = reference.iloc[train_idx], reference.iloc[test_idx]
        train, valid = train_test_split(
            rest,
            test_size=settings["split"]["valid_size"],
            stratify=rest[TARGET],
            random_state=seed + fold,
        )
        for name, excluded in compared.items():
            features = [f for f in FEATURES if f not in excluded]
            row = _fit_and_evaluate(settings, features, train, valid, test, refusal_rate, seed)
            rows.append({"variant": name, "fold": fold, **row})
    runs = pd.DataFrame(rows)

    metrics = [c for c in runs.columns if c not in ("variant", "fold")]
    by_variant = runs.pivot(index="fold", columns="variant", values=metrics)
    summary = []
    for metric in metrics:
        base = by_variant[(metric, cv["baseline"])]
        candidate = by_variant[(metric, cv["candidate"])]
        diff = candidate - base
        n = len(diff)
        half_width = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
        better = diff > 0 if metric.startswith(HIGHER_IS_BETTER) else diff < 0
        summary.append(
            {
                "metric": metric,
                "baseline": float(base.mean()),
                "candidate": float(candidate.mean()),
                "difference": float(diff.mean()),
                "ci_low": float(diff.mean() - half_width),
                "ci_high": float(diff.mean() + half_width),
                "folds_better": int(better.sum()),
                "folds": n,
            }
        )
    return runs, pd.DataFrame(summary)


def sampling_noise(
    pool: pd.DataFrame,
    scorer: Any,
    threshold: float,
    size: int,
    cfg: dict[str, Any],
    repetitions: int = 30,
) -> pd.Series:
    """Quanto a diferença de recusa indevida oscila só por amostragem, sem nenhum drift.

    Sorteia lotes do tamanho dado, sem alteração, e devolve a meia-largura do intervalo
    central de 90% da diferença, por atributo. Mudanças menores que isso, entre duas
    janelas desse tamanho, não se distinguem do acaso.
    """
    gaps = []
    for seed in range(repetitions):
        sample = pool.sample(size, random_state=seed)
        _, ind = evaluate(sample.assign(score=scorer(sample)), threshold, cfg)
        gaps.append(ind.set_index("attribute")["equal_opportunity_gap"])
    frame = pd.DataFrame(gaps)
    return (frame.quantile(0.95) - frame.quantile(0.05)) / 2


def production_fairness(
    batches: dict[str, pd.DataFrame], threshold: float, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Indicadores de equidade por janela de produção (lotes já rotulados).

    ``batches`` traz, por janela, as features, o score (``score``) e o alvo de cada cliente.
    """
    rows = []
    for window, df in batches.items():
        _, ind = evaluate(df, threshold, cfg)
        for item in ind.itertuples():
            rows.append(
                {
                    "window": window,
                    "n": len(df),
                    "attribute": item.attribute,
                    "disparate_impact": item.disparate_impact,
                    "equal_opportunity_gap": item.equal_opportunity_gap,
                    "auc_gap": item.auc_gap,
                    "most_refused_group": item.most_refused_group,
                }
            )
    return pd.DataFrame(rows)
