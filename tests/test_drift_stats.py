import numpy as np
import pandas as pd
import pytest

from credit_monitoring.drift.stats import drift_severity, psi


@pytest.fixture
def skewed() -> pd.Series:
    """Variável de cauda longa, como faturas."""
    return pd.Series(np.random.default_rng(0).lognormal(10, 1, 5000))


def test_psi_is_near_zero_for_the_same_distribution(skewed):
    other = pd.Series(np.random.default_rng(1).lognormal(10, 1, 5000))
    assert psi(skewed, other) < 0.01


def test_psi_grows_with_the_size_of_the_shift(skewed):
    small = psi(skewed, skewed * 1.1)
    large = psi(skewed, skewed * 2.0)
    assert 0 < small < large
    assert large > 0.25


def test_psi_detects_a_shift_in_discrete_codes():
    reference = pd.Series([0] * 80 + [1] * 15 + [2] * 5)
    current = pd.Series([0] * 50 + [1] * 30 + [2] * 20)
    assert psi(reference, current) > 0.25


def test_psi_counts_values_never_seen_in_the_reference():
    reference = pd.Series([0, 1] * 50)
    current = pd.Series([0, 1] * 40 + [5] * 20)
    assert psi(reference, current) > 0.25


@pytest.mark.parametrize(
    ("psi_value", "ks_pvalue", "expected"),
    [
        (0.30, 0.5, "significativo"),
        (0.15, 0.5, "moderado"),
        (0.05, 0.001, "leve"),
        (0.05, 0.5, "estável"),
    ],
)
def test_severity_combines_magnitude_and_significance(psi_value, ks_pvalue, expected):
    assert drift_severity(psi_value, ks_pvalue, 0.10, 0.25, 0.01) == expected
