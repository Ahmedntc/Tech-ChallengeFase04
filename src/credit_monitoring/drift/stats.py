"""Estatísticas de drift: PSI com faixas por quantis da referência.

O PSI (Population Stability Index) mede *quanto* a distribuição mudou:

    PSI = Σ (atual_i − esperado_i) · ln(atual_i / esperado_i)

As faixas são os decis da referência, como é praxe em risco de crédito. Faixas de
largura igual (padrão do Evidently) concentram quase todos os valores de variáveis
de cauda longa, como faturas, na primeira faixa, e deixam o PSI cego a mudanças
relevantes. Variáveis com poucos valores distintos (códigos e status) usam cada
valor como uma categoria.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-4
MAX_CATEGORIES = 15


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / len(values)


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI entre a distribuição de referência e a atual."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    categories = np.unique(ref)

    if len(categories) <= MAX_CATEGORIES:
        expected = np.array([(ref == value).mean() for value in categories])
        actual = np.array([(cur == value).mean() for value in categories])
        # Valores que nunca apareceram na referência formam uma faixa própria.
        expected = np.append(expected, 0.0)
        actual = np.append(actual, (~np.isin(cur, categories)).mean())
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
        expected = _proportions(ref, edges)
        actual = _proportions(cur, edges)

    expected = np.clip(expected, EPSILON, None)
    actual = np.clip(actual, EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_severity(
    psi_value: float,
    ks_pvalue: float,
    psi_moderate: float,
    psi_significant: float,
    ks_alpha: float,
) -> str:
    """Classifica o drift de uma variável combinando magnitude (PSI) e significância (KS).

    - ``significativo``: PSI acima do limite alto.
    - ``moderado``: PSI acima do limite de atenção.
    - ``leve``: o KS detecta mudança estatisticamente significativa, mas ainda pequena
      em magnitude (detecção precoce).
    - ``estável``: nenhum dos dois sinais.
    """
    if psi_value >= psi_significant:
        return "significativo"
    if psi_value >= psi_moderate:
        return "moderado"
    if ks_pvalue < ks_alpha:
        return "leve"
    return "estável"
