import numpy as np
import pandas as pd
import pytest

from credit_monitoring.contracts.gate import validate_batch
from credit_monitoring.data.columns import BILL_AMT, FEATURES, ID_COLUMN, TARGET
from credit_monitoring.drift.simulate import (
    MonthScenario,
    apply_concept_drift,
    apply_inflation,
    apply_limit_expansion,
    apply_unemployment,
    concept_segment,
    load_scenarios,
    sample_batch,
    simulate_month,
)

RATES = {1: 0.34, 2: 0.69, 3: 0.77}


@pytest.fixture
def pool(prepared_df) -> pd.DataFrame:
    return prepared_df.sample(4000, random_state=0).reset_index(drop=True)


@pytest.fixture
def features(pool) -> pd.DataFrame:
    return pool.drop(columns=[TARGET]).head(1000)


def test_sample_batch_controls_the_share_of_young_clients(pool):
    batch = sample_batch(pool, 1000, young_share=0.5, young_age_max=28, seed=1)
    assert len(batch) == 1000
    assert (batch["age"] <= 28).mean() == pytest.approx(0.5)
    assert batch[ID_COLUMN].is_unique


def test_inflation_raises_bills_and_keeps_other_columns(features):
    inflated = apply_inflation(features, 0.20, np.random.default_rng(0))
    ratio = inflated["bill_amt_1"].abs().sum() / features["bill_amt_1"].abs().sum()
    assert ratio == pytest.approx(1.20, abs=0.01)
    untouched = [c for c in features.columns if c not in BILL_AMT]
    assert inflated[untouched].equals(features[untouched])


def test_limit_expansion_keeps_multiples_of_ten_thousand(features):
    expanded = apply_limit_expansion(features, 0.25)
    assert (expanded["limit_bal"] % 10_000 == 0).all()
    assert expanded["limit_bal"].mean() > features["limit_bal"].mean() * 1.2


def test_unemployment_creates_coherent_delays_and_only_adds_defaults(pool):
    df = pool.drop(columns=[TARGET]).head(1000)
    labels = pool[TARGET].head(1000)
    shocked, new_labels, n = apply_unemployment(df, labels, 0.2, RATES, np.random.default_rng(0))

    assert n == 200
    changed = shocked.index[shocked["pay_status_1"] != df["pay_status_1"]]
    assert len(changed) == n
    assert (shocked.loc[changed, "pay_status_1"] >= 1).all()
    assert (shocked.loc[changed, "pay_amt_1"] == 0).all()
    # Atraso de 2+ meses agora implica atraso no mês anterior.
    two_plus = changed[shocked.loc[changed, "pay_status_1"] >= 2]
    assert (shocked.loc[two_plus, "pay_status_2"] >= 1).all()
    assert (new_labels >= labels).all()
    assert new_labels.sum() > labels.sum()


def test_concept_drift_changes_labels_but_never_features(pool, base_settings):
    df = pool.drop(columns=[TARGET]).head(2000)
    labels = pool[TARGET].head(2000)
    new_labels, flipped = apply_concept_drift(
        df, labels, 0.5, base_settings, np.random.default_rng(0)
    )
    segment = concept_segment(df, base_settings)
    changed = new_labels != labels
    assert flipped == changed.sum() > 0
    assert segment[changed].all()
    assert (new_labels[changed] == 1).all()


def test_month_without_shocks_reproduces_the_pool(pool, base_settings):
    scenario = MonthScenario(month="M0", title="controle")
    batch, labels, stats = simulate_month(scenario, pool, RATES, base_settings, seed=3)
    original = pool.set_index(ID_COLUMN).loc[batch[ID_COLUMN]]
    assert np.allclose(batch[FEATURES].to_numpy(), original[FEATURES].to_numpy())
    assert labels.tolist() == original[TARGET].tolist()
    assert stats["default_rate"] == stats["default_rate_before_shocks"]


def test_every_configured_month_respects_the_data_contract(pool, base_settings):
    """Drift muda a distribuição, mas não pode gerar dado inválido."""
    for i, scenario in enumerate(load_scenarios(base_settings), start=1):
        batch, labels, _ = simulate_month(scenario, pool, RATES, base_settings, seed=i)
        result = validate_batch(batch, scenario.month)
        assert result.passed, (scenario.month, result.violated_rules)
        assert set(labels.unique()) <= {0, 1}


def test_crisis_month_raises_the_default_rate(pool, base_settings):
    crisis = load_scenarios(base_settings)[-1]
    _, labels, stats = simulate_month(crisis, pool, RATES, base_settings, seed=7)
    assert stats["default_rate"] > stats["default_rate_before_shocks"] + 0.08
    assert stats["labels_changed_by_concept_drift"] > 0
    assert labels.mean() == pytest.approx(stats["default_rate"])
