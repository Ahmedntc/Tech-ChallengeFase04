"""Treino do modelo baseline de risco de crédito.

1. Lê o dataset de Referência e o valida contra o contrato (com alvo): lixo não entra no treino.
2. Divide em treino, validação e teste, de forma estratificada.
3. Treina dois candidatos: Regressão Logística (baseline interpretável) e XGBoost.
4. Escolhe o campeão pela AUC na validação e o limiar de decisão (máximo F1), também na validação.
5. Reporta as métricas no teste, que não participa de nenhuma escolha.
6. Salva o artefato e os metadados e registra os experimentos no MLflow.

Uso: ``python -m credit_monitoring.models.train [--no-mlflow]``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from credit_monitoring.config import load_settings, resolve_path
from credit_monitoring.contracts.gate import format_summary, validate_batch
from credit_monitoring.data.columns import FEATURES, ID_COLUMN, TARGET
from credit_monitoring.data.download import content_fingerprint
from credit_monitoring.models.artifacts import ModelBundle
from credit_monitoring.models.metrics import best_f1_threshold, classification_metrics

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    name: str
    estimator: Any
    threshold: float
    valid_metrics: dict[str, float]
    test_metrics: dict[str, float]


def model_features(settings: dict[str, Any]) -> list[str]:
    excluded = set(settings["model"]["excluded_features"])
    return [col for col in FEATURES if col not in excluded]


def build_candidates(settings: dict[str, Any]) -> dict[str, Any]:
    seed = settings["project"]["random_seed"]
    lr = settings["model"]["logistic_regression"]
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(C=lr["C"], max_iter=lr["max_iter"], random_state=seed),
                ),
            ]
        ),
        "xgboost": XGBClassifier(
            **settings["model"]["xgboost"], eval_metric="logloss", random_state=seed
        ),
    }


def split_reference(
    reference: pd.DataFrame, settings: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = settings["project"]["random_seed"]
    split = settings["split"]
    train_valid, test = train_test_split(
        reference, test_size=split["test_size"], stratify=reference[TARGET], random_state=seed
    )
    train, valid = train_test_split(
        train_valid,
        test_size=split["valid_size"],
        stratify=train_valid[TARGET],
        random_state=seed,
    )
    return train, valid, test


def fit_candidates(
    candidates: dict[str, Any],
    features: list[str],
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> list[CandidateResult]:
    results = []
    for name, estimator in candidates.items():
        estimator.fit(train[features], train[TARGET])
        valid_score = estimator.predict_proba(valid[features])[:, 1]
        threshold = best_f1_threshold(valid[TARGET], valid_score)
        test_score = estimator.predict_proba(test[features])[:, 1]
        results.append(
            CandidateResult(
                name=name,
                estimator=estimator,
                threshold=threshold,
                valid_metrics=classification_metrics(valid[TARGET], valid_score, threshold),
                test_metrics=classification_metrics(test[TARGET], test_score, threshold),
            )
        )
        logger.info(
            "%s: AUC validação %.4f | AUC teste %.4f",
            name,
            results[-1].valid_metrics["auc"],
            results[-1].test_metrics["auc"],
        )
    return results


def log_to_mlflow(
    settings: dict[str, Any],
    results: list[CandidateResult],
    champion: CandidateResult,
    features: list[str],
    example: pd.DataFrame,
) -> dict[str, Any]:
    """Um run por candidato; o campeão é registrado no Model Registry com o alias 'champion'."""
    from credit_monitoring.monitoring.tracking import mlflow, use_experiment

    use_experiment(settings, "training_experiment")
    registered_name = settings["mlflow"]["registered_model"]
    model_uris = {}
    for result in results:
        with mlflow.start_run(run_name=result.name):
            mlflow.set_tags(
                {
                    "candidate": result.name,
                    "champion": str(result.name == champion.name).lower(),
                    "stage": "baseline",
                }
            )
            mlflow.log_params(
                {
                    **settings["model"][result.name],
                    "model_type": result.name,
                    "n_features": len(features),
                    "excluded_features": ",".join(settings["model"]["excluded_features"]),
                    "threshold": round(result.threshold, 6),
                }
            )
            mlflow.log_metrics(
                {f"valid_{k}": v for k, v in result.valid_metrics.items()}
                | {f"test_{k}": v for k, v in result.test_metrics.items()}
            )
            # Cada modelo no seu formato nativo e seguro (sem pickle): XGBoost em JSON/UBJ,
            # scikit-learn via skops.
            flavor = (
                mlflow.xgboost if isinstance(result.estimator, XGBClassifier) else mlflow.sklearn
            )
            with warnings.catch_warnings():
                # A saída do predict é inteira (classe 0/1); o aviso sobre nulos não se aplica.
                warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")
                info = flavor.log_model(
                    result.estimator,
                    name="model",
                    input_example=example,
                    pip_requirements=str(resolve_path("requirements.txt")),
                )
            model_uris[result.name] = info.model_uri

    version = mlflow.register_model(model_uris[champion.name], registered_name)
    mlflow.MlflowClient().set_registered_model_alias(registered_name, "champion", version.version)
    return {
        "experiment": settings["mlflow"]["training_experiment"],
        "registered_model": registered_name,
        "registered_version": int(version.version),
    }


def train(
    settings: dict[str, Any] | None = None,
    reference: pd.DataFrame | None = None,
    track: bool = True,
) -> tuple[ModelBundle, list[CandidateResult]]:
    settings = settings or load_settings()
    paths = settings["paths"]
    if reference is None:
        reference = pd.read_parquet(resolve_path(paths["reference_file"]))

    validation = validate_batch(reference, "referencia", labeled=True)
    if not validation.passed:
        raise ValueError("Dataset de Referência viola o contrato:\n" + format_summary(validation))
    reference = validation.data

    features = model_features(settings)
    train_df, valid_df, test_df = split_reference(reference, settings)
    results = fit_candidates(build_candidates(settings), features, train_df, valid_df, test_df)
    champion = max(results, key=lambda r: r.valid_metrics["auc"])

    trained_at = datetime.now(UTC)
    metadata: dict[str, Any] = {
        "model_name": settings["mlflow"]["registered_model"],
        "version": trained_at.strftime("v%Y%m%d%H%M%S"),
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "champion": champion.name,
        "selection_criterion": "maior AUC na validação",
        "threshold": champion.threshold,
        "threshold_criterion": "máximo F1 na validação",
        "features": features,
        "excluded_features": settings["model"]["excluded_features"],
        "excluded_features_reason": (
            "Minimização e não discriminação (LGPD, art. 6º, III e IX): mantidos só para "
            "auditoria de vieses."
        ),
        "data": {
            "reference_fingerprint": content_fingerprint(reference.drop(columns=[ID_COLUMN])),
            "n_train": len(train_df),
            "n_valid": len(valid_df),
            "n_test": len(test_df),
            "default_rate_train": float(train_df[TARGET].mean()),
        },
        "candidates": {
            r.name: {"threshold": r.threshold, "valid": r.valid_metrics, "test": r.test_metrics}
            for r in results
        },
        "random_seed": settings["project"]["random_seed"],
        "libraries": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
        },
    }
    if track:
        example = valid_df[features].head(5).astype("float64")
        metadata["mlflow"] = log_to_mlflow(settings, results, champion, features, example)

    bundle = ModelBundle(
        estimator=champion.estimator,
        features=features,
        threshold=champion.threshold,
        version=metadata["version"],
        metadata=metadata,
    )
    bundle.save(resolve_path(paths["model_file"]))
    metadata_path = resolve_path(paths["model_metadata_file"])
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return bundle, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treina o modelo baseline de crédito.")
    parser.add_argument("--no-mlflow", action="store_true", help="Não registra no MLflow")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    bundle, results = train(settings, track=not args.no_mlflow)

    header = f"{'candidato':<22}{'AUC val':>9}{'AUC teste':>11}{'KS':>8}{'Gini':>8}"
    header += f"{'F1':>8}{'Recall':>8}{'Precisão':>10}"
    print("\n" + header)
    for r in results:
        t = r.test_metrics
        print(
            f"{r.name:<22}{r.valid_metrics['auc']:>9.4f}{t['auc']:>11.4f}{t['ks']:>8.4f}"
            f"{t['gini']:>8.4f}{t['f1']:>8.4f}{t['recall']:>8.4f}{t['precision']:>10.4f}"
        )
    print(f"\nCampeão: {bundle.metadata['champion']} | limiar {bundle.threshold:.4f}")
    print(f"Versão: {bundle.version}")
    print(f"Modelo: {settings['paths']['model_file']}")
    print(f"Metadados: {settings['paths']['model_metadata_file']}")
    if "mlflow" in bundle.metadata:
        info = bundle.metadata["mlflow"]
        print(f"MLflow: {info['registered_model']} v{info['registered_version']} (alias champion)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
