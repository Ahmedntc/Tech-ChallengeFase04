import numpy as np
import pandas as pd
import pytest

from credit_monitoring.data.columns import TARGET
from credit_monitoring.governance.fairness import (
    add_age_band,
    compare_variants,
    cross_validate_variants,
    group_metrics,
    indicators,
    sampling_noise,
)

CFG = {
    "attributes": ["sex"],
    "age_bands": [28, 45],
    "min_group_size": 3,
    "disparate_impact_min": 0.8,
    "equal_opportunity_max_gap": 0.05,
}


def _frame() -> pd.DataFrame:
    # Grupo 1: 4 bons pagadores, 2 recusados (score alto). Grupo 2: 4 bons, nenhum recusado.
    return pd.DataFrame(
        {
            "sex": [1] * 6 + [2] * 6 + [1],
            TARGET: [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
            "score": [0.9, 0.8, 0.1, 0.1, 0.9, 0.2, 0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1],
        }
    )


def test_age_bands_cover_every_age():
    df = add_age_band(pd.DataFrame({"age": [21, 28, 29, 45, 46, 79]}), [28, 45])
    assert df["age_band"].tolist() == [
        "até 28",
        "até 28",
        "29 a 45",
        "29 a 45",
        "46 ou mais",
        "46 ou mais",
    ]


def test_group_metrics_count_refusals_of_good_payers():
    groups = group_metrics(_frame(), "sex", threshold=0.5).set_index("group")

    male, female = groups.loc["masculino"], groups.loc["feminino"]
    assert (male["n"], female["n"]) == (7, 6)
    assert male["false_refusal_rate"] == pytest.approx(2 / 5)
    assert female["false_refusal_rate"] == 0
    assert male["recall"] == pytest.approx(1 / 2)
    assert female["approval_rate"] == pytest.approx(5 / 6)


def test_indicators_flag_unequal_opportunity_and_name_the_group():
    result = indicators(group_metrics(_frame(), "sex", 0.5), CFG)
    assert result["equal_opportunity_gap"] == pytest.approx(0.4)
    assert not result["passes_equal_opportunity"]
    assert result["most_refused_group"] == "masculino"
    assert result["disparate_impact"] == pytest.approx((4 / 7) / (5 / 6))
    assert not result["passes_disparate_impact"]


def test_small_groups_stay_out_of_the_indicators():
    df = pd.concat([_frame(), pd.DataFrame({"sex": [9], TARGET: [0], "score": [0.99]})])
    result = indicators(group_metrics(df, "sex", 0.5), CFG)
    assert result["groups_used"] == 2


def test_variants_are_compared_at_the_same_approval_rate(settings, prepared_df):
    settings["model"]["xgboost"]["n_estimators"] = 20
    gov = settings["governance"]["fairness"]
    gov["seeds"] = [1]
    gov["variants"] = {"atual": ["sex", "marriage"], "todos": []}
    reference = prepared_df.sample(5000, random_state=0).reset_index(drop=True)

    summary = compare_variants(settings, reference, refusal_rate=0.25)

    assert summary["variant"].tolist() == ["atual", "todos"]
    # O limiar vem da validação; no teste a taxa fica próxima, e igual entre as variantes.
    assert np.allclose(summary["approval_rate"], 0.75, atol=0.08)
    assert abs(summary["approval_rate"].diff().iloc[-1]) < 0.03
    assert {"di_sex", "eo_sex", "eo_age_band", "eo_education"} <= set(summary.columns)


def test_sampling_noise_shrinks_with_larger_samples(prepared_df, base_settings):
    pool = prepared_df.sample(6000, random_state=0)
    cfg = {**base_settings["governance"]["fairness"], "attributes": ["sex"]}

    def scorer(df):
        return df["pay_status_1"].clip(lower=0) / 8

    small = sampling_noise(pool, scorer, 0.1, 300, cfg, repetitions=15)
    large = sampling_noise(pool, scorer, 0.1, 3000, cfg, repetitions=15)
    assert large["sex"] < small["sex"]


def test_cross_validation_compares_both_variants_on_the_same_folds(settings, prepared_df):
    settings["model"]["xgboost"]["n_estimators"] = 20
    gov = settings["governance"]["fairness"]
    gov["variants"] = {"base": ["sex", "marriage"], "nova": ["sex", "marriage", "age"]}
    gov["cross_validation"] = {"splits": 2, "repeats": 2, "baseline": "base", "candidate": "nova"}
    reference = prepared_df.sample(4000, random_state=0).reset_index(drop=True)

    runs, summary = cross_validate_variants(settings, reference, refusal_rate=0.25)

    assert len(runs) == 2 * 2 * 2
    assert runs.groupby("fold")["variant"].apply(set).eq({"base", "nova"}).all()
    auc = summary.set_index("metric").loc["auc"]
    assert auc["folds"] == 4
    assert auc["difference"] == pytest.approx(auc["candidate"] - auc["baseline"])
    assert auc["ci_low"] <= auc["difference"] <= auc["ci_high"]
    assert {"eo_age_band", "eo_education", "di_sex"} <= set(summary["metric"])
