"""Análise causal da degradação do modelo, por intervenção.

O monitoramento (Etapa 2) mostra **correlações**: no mês da crise, várias variáveis
mudaram e o desempenho caiu. Ele não diz qual mudança **causou** a queda. Como a
simulação é controlada, dá para fazer o que em produção não se pode: **intervir**.
Cada mecanismo do cenário (inflação, pagamentos, limites, perfil jovem, desemprego,
concept drift) pode ser ligado ou desligado, mantendo todo o resto igual, e o
efeito é medido diretamente.

Com 6 mecanismos há 64 combinações. O efeito de cada mecanismo é o seu **valor de
Shapley**: a média do efeito de ligá-lo, considerando todas as combinações dos
outros. Assim, as interações (por exemplo, inflação aumentando o número de clientes
muito endividados que o concept drift atinge) são divididas de forma justa, e a
soma dos efeitos é exatamente a variação total. Cada combinação é simulada com
várias sementes, e o resultado é a média, com o erro padrão.

O módulo também testa a **prescrição**: recalibrar ou retreinar o modelo com dados
recentes recupera o desempenho?
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from credit_monitoring.data.columns import TARGET
from credit_monitoring.drift.simulate import MonthScenario, load_scenarios, simulate_month
from credit_monitoring.models.artifacts import ModelBundle
from credit_monitoring.models.metrics import best_f1_threshold, classification_metrics
from credit_monitoring.models.train import split_reference

# Mecanismo → (rótulo, campo do cenário, valor que o desliga).
MECHANISMS: dict[str, tuple[str, str, Any]] = {
    "inflacao": ("Inflação nas faturas", "inflation", 0.0),
    "pagamentos": ("Queda nos pagamentos", "payment_change", 0.0),
    "limites": ("Expansão dos limites", "limit_expansion", 0.0),
    "perfil_jovem": ("Mais clientes jovens", "young_share", None),
    "desemprego": ("Desemprego (novos atrasos)", "unemployment", 0.0),
    "concept_drift": ("Concept drift (risco que o modelo não vê)", "concept_drift", 0.0),
}
DATA_DRIFT = [m for m in MECHANISMS if m != "concept_drift"]
METRICS = {
    "auc": "AUC",
    "calibration_gap": "Inadimplência real − score médio",
    "mean_score": "Score médio",
    "actual_default_rate": "Inadimplência real",
    "approval_rate": "Taxa de aprovação",
}


def active_mechanisms(scenario: MonthScenario) -> list[str]:
    return [
        name
        for name, (_, field, neutral) in MECHANISMS.items()
        if getattr(scenario, field) != neutral
    ]


def with_only(scenario: MonthScenario, keep: frozenset[str]) -> MonthScenario:
    """O cenário com apenas os mecanismos de ``keep`` ligados."""
    changes = {
        field: neutral for name, (_, field, neutral) in MECHANISMS.items() if name not in keep
    }
    return replace(scenario, **changes)


def outcome(y: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, scores)),
        "calibration_gap": float(y.mean() - scores.mean()),
        "mean_score": float(scores.mean()),
        "actual_default_rate": float(y.mean()),
        "approval_rate": float((scores < threshold).mean()),
    }


def coalition_values(
    scenario: MonthScenario,
    bundle: ModelBundle,
    pool: pd.DataFrame,
    default_rate_by_delay: dict[int, float],
    settings: dict[str, Any],
    seeds: list[int],
) -> dict[frozenset[str], pd.DataFrame]:
    """Simula cada combinação de mecanismos com cada semente. Uma linha por semente."""
    players = active_mechanisms(scenario)
    values = {}
    for size in range(len(players) + 1):
        for coalition in itertools.combinations(players, size):
            keep = frozenset(coalition)
            rows = []
            for seed in seeds:
                X, y, _ = simulate_month(
                    with_only(scenario, keep), pool, default_rate_by_delay, settings, seed
                )
                rows.append(outcome(y, bundle.predict_proba(X), bundle.threshold))
            values[keep] = pd.DataFrame(rows)
    return values


def shapley(values: dict[frozenset[str], pd.DataFrame], players: list[str]) -> pd.DataFrame:
    """Valor de Shapley de cada jogador, por semente (linhas) e métrica (colunas)."""
    n = len(players)
    result = {}
    for player in players:
        others = [p for p in players if p != player]
        total = 0
        for size in range(n):
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for coalition in itertools.combinations(others, size):
                without = frozenset(coalition)
                total = total + weight * (values[without | {player}] - values[without])
        result[player] = total
    return pd.concat(result, names=["mechanism", "seed"])


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Média e erro padrão de cada efeito, entre as sementes."""
    grouped = per_seed.groupby(level="mechanism", sort=False)
    mean, sem = grouped.mean(), grouped.sem()
    return mean.join(sem, rsuffix="_se")


def attribute_degradation(
    settings: dict[str, Any],
    bundle: ModelBundle,
    pool: pd.DataFrame,
    default_rate_by_delay: dict[int, float],
) -> dict[str, Any]:
    """Atribui a mudança nas métricas do mês analisado aos mecanismos do cenário."""
    cfg = settings["governance"]["causal"]
    scenario = next(s for s in load_scenarios(settings) if s.month == cfg["month"])
    seeds = list(range(cfg["first_seed"], cfg["first_seed"] + cfg["seeds"]))
    players = active_mechanisms(scenario)
    values = coalition_values(scenario, bundle, pool, default_rate_by_delay, settings, seeds)

    by_mechanism = summarize(shapley(values, players))
    by_mechanism.insert(0, "label", [MECHANISMS[m][0] for m in by_mechanism.index])

    # Os mesmos resultados agrupados em dois blocos: data drift x concept drift.
    groups = {"data_drift": [p for p in players if p in DATA_DRIFT], "concept_drift": []}
    groups["concept_drift"] = [p for p in players if p not in DATA_DRIFT]
    grouped_values = {
        frozenset(name for name in groups if name in chosen): values[
            frozenset(p for name in chosen for p in groups[name])
        ]
        for size in range(3)
        for chosen in itertools.combinations(groups, size)
    }
    by_group = summarize(shapley(grouped_values, list(groups)))

    baseline = values[frozenset()].mean()
    full = values[frozenset(players)].mean()
    return {
        "month": scenario.month,
        "seeds": len(seeds),
        "simulations": len(values) * len(seeds),
        "baseline": baseline.to_dict(),
        "full": full.to_dict(),
        "by_mechanism": by_mechanism,
        "by_group": by_group,
    }


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def retraining_experiment(
    settings: dict[str, Any],
    bundle: ModelBundle,
    reference: pd.DataFrame,
    pool: pd.DataFrame,
    default_rate_by_delay: dict[int, float],
) -> pd.DataFrame:
    """Prescrição: o que acontece se o modelo for recalibrado ou retreinado na crise?

    O Pool de Produção é dividido em duas metades de clientes distintos. Na primeira,
    a crise já aconteceu e os rótulos chegaram (dados para ajustar). Na segunda, a
    crise continua (dados para avaliar). Três respostas são comparadas:

    - **manter** o modelo atual;
    - **recalibrar**: ajustar só a escala do score com os rótulos recentes;
    - **retreinar**: treinar de novo com a Referência + a crise observada, com peso
      maior para os dados recentes.
    """
    cfg = settings["governance"]["retraining"]
    scenario = next(
        s for s in load_scenarios(settings) if s.month == settings["governance"]["causal"]["month"]
    )
    features = bundle.features
    train, _, _ = split_reference(reference, settings)
    rows = []
    for rep in range(cfg["repetitions"]):
        observed_pool, future_pool = train_test_split(
            pool, test_size=0.5, stratify=pool[TARGET], random_state=rep
        )
        X_obs, y_obs, _ = simulate_month(
            scenario,
            observed_pool.reset_index(drop=True),
            default_rate_by_delay,
            settings,
            500 + rep,
        )
        X_new, y_new, _ = simulate_month(
            scenario, future_pool.reset_index(drop=True), default_rate_by_delay, settings, 900 + rep
        )

        current = bundle.predict_proba(X_new)
        rows.append(
            {"response": "manter", **classification_metrics(y_new, current, bundle.threshold)}
        )

        calibrator = LogisticRegression().fit(_logit(bundle.predict_proba(X_obs)), y_obs)
        recalibrated_obs = calibrator.predict_proba(_logit(bundle.predict_proba(X_obs)))[:, 1]
        recalibrated = calibrator.predict_proba(_logit(current))[:, 1]
        threshold = best_f1_threshold(y_obs, recalibrated_obs)
        rows.append(
            {"response": "recalibrar", **classification_metrics(y_new, recalibrated, threshold)}
        )

        weights = np.r_[np.ones(len(train)), np.full(len(X_obs), cfg["recent_weight"])]
        model = XGBClassifier(
            **settings["model"]["xgboost"],
            eval_metric="logloss",
            random_state=settings["project"]["random_seed"],
        ).fit(
            pd.concat([train[features], X_obs[features]]),
            pd.concat([train[TARGET], y_obs]),
            sample_weight=weights,
        )
        threshold = best_f1_threshold(y_obs, model.predict_proba(X_obs[features])[:, 1])
        retrained = model.predict_proba(X_new[features])[:, 1]
        rows.append(
            {"response": "retreinar", **classification_metrics(y_new, retrained, threshold)}
        )

    runs = pd.DataFrame(rows)
    runs["calibration_gap"] = runs["actual_default_rate"] - runs["mean_score"]
    runs["approval_rate"] = 1 - runs["predicted_default_rate"]
    columns = ["auc", "ks", "recall", "precision", "mean_score", "actual_default_rate"]
    columns += ["calibration_gap", "approval_rate"]
    summary = runs.groupby("response", sort=False)[columns].mean()
    summary["auc_se"] = runs.groupby("response", sort=False)["auc"].sem()
    return summary.reset_index()
