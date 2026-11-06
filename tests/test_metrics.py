import numpy as np
import pytest

from credit_monitoring.models.metrics import (
    best_f1_threshold,
    classification_metrics,
    ks_statistic,
)


def test_ks_is_one_for_perfect_separation():
    assert ks_statistic(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_ks_is_low_for_uninformative_scores():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    assert ks_statistic(y, rng.random(5000)) < 0.05


def test_best_f1_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    threshold = best_f1_threshold(y, scores)
    assert 0.3 < threshold <= 0.7


def test_classification_metrics_are_consistent():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.2, 0.6, 0.4, 0.9])
    m = classification_metrics(y, scores, threshold=0.5)

    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == 0.5
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)
    assert m["actual_default_rate"] == 0.5
