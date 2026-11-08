"""Artefato do modelo: estimador + contrato de uso (features, limiar e versão)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    """Tudo o que é preciso para usar o modelo em produção de forma consistente."""

    estimator: Any
    features: list[str]
    threshold: float
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidade de inadimplência, usando só as features do treino, na mesma ordem."""
        return self.estimator.predict_proba(df[self.features])[:, 1]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df) >= self.threshold).astype(int)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path) -> ModelBundle:
        return joblib.load(path)
