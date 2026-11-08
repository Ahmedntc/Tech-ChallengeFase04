import json

import numpy as np
import pytest

from credit_monitoring.models.artifacts import ModelBundle
from credit_monitoring.models.train import model_features, train


@pytest.fixture
def fast_settings(settings):
    settings["model"]["xgboost"]["n_estimators"] = 30
    return settings


def test_excluded_features_stay_out_of_the_model(settings):
    features = model_features(settings)
    assert not set(settings["model"]["excluded_features"]) & set(features)
    assert "age" in features


def test_training_produces_usable_and_reloadable_bundle(fast_settings, prepared_df):
    reference = prepared_df.sample(4000, random_state=0).reset_index(drop=True)
    bundle, results = train(fast_settings, reference=reference, track=False)

    assert {r.name for r in results} == {"logistic_regression", "xgboost"}
    champion = max(results, key=lambda r: r.valid_metrics["auc"])
    assert bundle.metadata["champion"] == champion.name
    assert champion.test_metrics["auc"] > 0.65

    scores = bundle.predict_proba(reference)
    assert scores.shape == (len(reference),)
    assert ((scores >= 0) & (scores <= 1)).all()

    reloaded = ModelBundle.load(fast_settings["paths"]["model_file"])
    assert np.allclose(reloaded.predict_proba(reference), scores)

    with open(fast_settings["paths"]["model_metadata_file"], encoding="utf-8") as fh:
        metadata = json.load(fh)
    assert metadata["features"] == bundle.features
    assert "mlflow" not in metadata


def test_training_rejects_reference_that_violates_the_contract(fast_settings, prepared_df):
    reference = prepared_df.sample(500, random_state=0).reset_index(drop=True)
    reference.loc[0, "age"] = 12
    with pytest.raises(ValueError, match="viola o contrato"):
        train(fast_settings, reference=reference, track=False)
