"""Optional lightweight chronological baselines."""
from __future__ import annotations

import numpy as np

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
except ImportError:  # pragma: no cover
    HistGradientBoostingClassifier = None
    LogisticRegression = None


class LogisticBaseline:
    def __init__(self, random_state: int = 42) -> None:
        if LogisticRegression is None:
            raise ImportError("scikit-learn is required for baselines")
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class GradientBoostingBaseline:
    def __init__(self, random_state: int = 42) -> None:
        if HistGradientBoostingClassifier is None:
            raise ImportError("scikit-learn is required for baselines")
        self.model = HistGradientBoostingClassifier(max_iter=150, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingBaseline":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """Use the final timestep as a causal tabular baseline representation."""
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError("sequence input must have shape (samples, sequence, features)")
    return X[:, -1, :]
