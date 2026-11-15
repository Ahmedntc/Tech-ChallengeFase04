import numpy as np
import pytest

from credit_monitoring.data.columns import FEATURES, ID_COLUMN, TARGET
from credit_monitoring.governance.explain import (
    contributions,
    describe_value,
    explain_client,
    feature_label,
    format_explanation,
    global_importance,
    reason_codes,
)
from credit_monitoring.models.train import train


@pytest.fixture(scope="module")
def trained(base_settings, prepared_df, tmp_path_factory):
    """Os dois candidatos treinados numa amostra (XGBoost e Regressão Logística)."""
    from tests.conftest import isolated_settings

    cfg = isolated_settings(base_settings, tmp_path_factory.mktemp("explain"))
    cfg["model"]["xgboost"]["n_estimators"] = 30
    reference = prepared_df.sample(4000, random_state=1).reset_index(drop=True)
    bundle, results = train(cfg, reference=reference, track=False)
    return bundle, results, reference


def _logit(p):
    return np.log(p / (1 - p))


def test_xgboost_contributions_add_up_to_the_score(trained):
    bundle, results, reference = trained
    xgb_result = next(r for r in results if r.name == "xgboost")
    bundle.estimator = xgb_result.estimator
    sample = reference.head(50)

    contrib = contributions(bundle, sample)

    assert list(contrib.columns) == [*bundle.features, "_base"]
    assert np.allclose(contrib.sum(axis=1), _logit(bundle.predict_proba(sample)), atol=1e-4)


def test_logistic_regression_contributions_add_up_to_the_score(trained):
    bundle, results, reference = trained
    bundle.estimator = next(r for r in results if r.name == "logistic_regression").estimator
    sample = reference.head(50)
    contrib = contributions(bundle, sample)
    assert np.allclose(contrib.sum(axis=1), _logit(bundle.predict_proba(sample)), atol=1e-6)


def test_reason_codes_are_the_largest_risk_increasing_contributions(trained):
    bundle, results, reference = trained
    bundle.estimator = next(r for r in results if r.name == "xgboost").estimator
    sample = reference.head(30)
    contrib = contributions(bundle, sample).drop(columns="_base")

    for (_, row), reasons in zip(contrib.iterrows(), reason_codes(bundle, sample), strict=True):
        values = [r["contribuicao"] for r in reasons]
        assert values == sorted(values, reverse=True)
        assert all(v > 0 for v in values)
        expected = row[row > 0].sort_values(ascending=False).head(3).index.tolist()
        assert [r["feature"] for r in reasons] == expected


def test_global_importance_ranks_payment_status_first(trained):
    bundle, results, reference = trained
    bundle.estimator = next(r for r in results if r.name == "xgboost").estimator
    importance = global_importance(bundle, reference.head(500))
    assert importance.index[0] == "pay_status_1"
    assert (importance >= 0).all()


def test_explain_client_reproduces_the_decision(trained):
    bundle, results, reference = trained
    bundle.estimator = next(r for r in results if r.name == "xgboost").estimator
    batch = reference.drop(columns=[TARGET]).head(100)
    scores = bundle.predict_proba(batch)
    client = batch.iloc[int(np.argmax(scores))][ID_COLUMN]

    explanation = explain_client(bundle, batch, client)

    assert explanation["decisao"] == "recusado"
    assert explanation["score"] == pytest.approx(scores.max())
    total = sum(c["contribuicao"] for c in explanation["contribuicoes"]) + explanation["base"]
    assert total == pytest.approx(_logit(scores.max()), abs=1e-4)
    text = format_explanation(explanation)
    assert "RECUSADO" in text and "Principais motivos" in text
    with pytest.raises(KeyError):
        explain_client(bundle, batch, "inexistente")


def test_every_feature_has_a_readable_label():
    labels = {feature_label(f) for f in FEATURES}
    assert len(labels) == len(FEATURES)
    assert feature_label("pay_status_1") == "Situação do pagamento no último mês"
    assert feature_label("bill_amt_3") == "Valor da fatura há 3 meses"


@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        ("pay_status_1", 2, "atraso de 2 meses"),
        ("pay_status_1", 1, "atraso de 1 mês"),
        ("pay_status_2", -1, "pago em dia"),
        ("pay_amt_1", 15000.0, "NT$ 15.000"),
        ("age", 34, "34 anos"),
        ("education", 3, "ensino médio"),
    ],
)
def test_values_are_described_in_plain_language(feature, value, expected):
    assert describe_value(feature, value) == expected
