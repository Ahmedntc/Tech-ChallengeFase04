import copy
import itertools

import pandas as pd
import pytest

from credit_monitoring.data.columns import TARGET
from credit_monitoring.drift.simulate import load_scenarios
from credit_monitoring.governance.causal import (
    MECHANISMS,
    active_mechanisms,
    attribute_degradation,
    retraining_experiment,
    shapley,
    with_only,
)
from credit_monitoring.models.artifacts import ModelBundle


def test_active_mechanisms_follow_the_scenario(base_settings):
    scenarios = {s.month: s for s in load_scenarios(base_settings)}
    assert active_mechanisms(scenarios["M1"]) == []
    assert active_mechanisms(scenarios["M2"]) == ["inflacao"]
    assert active_mechanisms(scenarios["M6"]) == list(MECHANISMS)


def test_with_only_switches_off_every_other_mechanism(base_settings):
    crisis = load_scenarios(base_settings)[-1]
    only = with_only(crisis, frozenset({"concept_drift"}))
    assert active_mechanisms(only) == ["concept_drift"]
    assert only.concept_drift == crisis.concept_drift
    assert active_mechanisms(with_only(crisis, frozenset())) == []


def test_shapley_is_efficient_symmetric_and_splits_interactions():
    players = ["a", "b", "c"]
    effect = {"a": 1.0, "b": 2.0, "c": 0.0}

    def value(coalition):
        # Efeitos aditivos + interação de 0,6 entre a e b, dividida igualmente entre os dois.
        total = sum(effect[p] for p in coalition) + (0.6 if {"a", "b"} <= coalition else 0)
        return pd.DataFrame({"m": [total, total]})

    values = {
        frozenset(c): value(set(c)) for k in range(4) for c in itertools.combinations(players, k)
    }
    result = shapley(values, players).groupby(level="mechanism").mean()["m"]

    assert result["a"] == pytest.approx(1.3)
    assert result["b"] == pytest.approx(2.3)
    assert result["c"] == pytest.approx(0.0)
    assert result.sum() == pytest.approx(value(set(players))["m"][0] - value(set())["m"][0])


@pytest.fixture
def small_crisis(pipeline_run, base_settings):
    """Modelo e dados da execução reduzida do pipeline, com o cenário de crise (M6) disponível."""
    cfg = copy.deepcopy(pipeline_run[0])
    cfg["simulation"]["months"].append(base_settings["simulation"]["months"][-1])
    bundle = ModelBundle.load(cfg["paths"]["model_file"])
    pool = pd.read_parquet(cfg["paths"]["production_pool_file"])
    reference = pd.read_parquet(cfg["paths"]["reference_file"])
    rates = reference.groupby("pay_status_1")[TARGET].mean().reindex([1, 2, 3]).to_dict()
    return cfg, bundle, pool, reference, rates


def test_attribution_adds_up_and_blames_unemployment_for_the_score_shift(small_crisis):
    cfg, bundle, pool, _, rates = small_crisis
    cfg["governance"]["causal"].update(month="M5", seeds=2)

    result = attribute_degradation(cfg, bundle, pool, rates)

    effects = result["by_mechanism"]
    assert result["simulations"] == 2**5 * 2
    for metric in ("auc", "calibration_gap", "mean_score", "actual_default_rate"):
        total = result["full"][metric] - result["baseline"][metric]
        assert effects[metric].sum() == pytest.approx(total, abs=1e-9)
        assert result["by_group"][metric].sum() == pytest.approx(total, abs=1e-9)
    assert effects["mean_score"].idxmax() == "desemprego"
    assert result["by_group"].loc["concept_drift"].abs().max() == pytest.approx(0.0)


def test_retraining_experiment_compares_three_responses(small_crisis):
    cfg, bundle, pool, reference, rates = small_crisis
    cfg["governance"]["causal"]["month"] = "M6"
    cfg["governance"]["retraining"]["repetitions"] = 1

    summary = retraining_experiment(cfg, bundle, reference, pool, rates).set_index("response")

    assert summary.index.tolist() == ["manter", "recalibrar", "retreinar"]
    # Recalibrar não muda a ordenação (AUC), só corrige a escala do score.
    assert summary.loc["recalibrar", "auc"] == pytest.approx(summary.loc["manter", "auc"])
    assert summary.loc["manter", "calibration_gap"] > 0.05
    assert abs(summary.loc["recalibrar", "calibration_gap"]) < 0.05
    assert summary.loc["retreinar", "auc"] > summary.loc["manter", "auc"]
