"""Gradient-boosting mean-reversion classifier (CPU, sklearn).

Chosen as the default/robust model: trains in seconds on the local Ryzen 7
CPU, handles missing values natively, and is far less prone to overfitting on
a few thousand intraday samples than a deep network -- a reasonable default
before reaching for the optional GPU-trained LSTM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from core.config import MLConfig


class GradientBoostingReversionModel:
    def __init__(self, ml_config: MLConfig) -> None:
        self.config = ml_config
        self.model = HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=1.0,
            random_state=ml_config.random_state,
        )
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingReversionModel":
        if y.nunique() < 2:
            # Degenerate fold (e.g. all-reversion or all-continuation labels): skip
            # training and fall back to a constant-probability predictor downstream.
            self._is_fitted = False
            return self
        self.model.fit(X, y)
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            return pd.Series(0.5, index=X.index)
        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=X.index)
