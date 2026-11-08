"""Métricas de classificação usadas no treino e, depois, no monitoramento."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """KS de Kolmogorov-Smirnov: maior distância entre as curvas acumuladas de bons e maus.

    Métrica clássica de separação em modelos de crédito (equivale a max(TPR - FPR)).
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Limiar de decisão que maximiza o F1 (escolhido na validação, nunca no teste)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    return float(thresholds[int(np.argmax(f1))])


def classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float
) -> dict[str, float]:
    """Métricas independentes de limiar (AUC, KS, Gini, Brier) e no limiar escolhido."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_score)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, y_score),
        "brier": float(brier_score_loss(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "predicted_default_rate": float(y_pred.mean()),
        "actual_default_rate": float(y_true.mean()),
        "mean_score": float(y_score.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
