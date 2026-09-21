"""Feature engineering for the ML mean-reversion/trend-continuation gate.

All features are computed from information available at (or before) bar t,
so they can be used both for offline training and for real-time inference in
the live trader without look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import MLConfig


def build_feature_matrix(signals: pd.DataFrame, symbol_a: str, symbol_b: str, ml_config: MLConfig) -> pd.DataFrame:
    """Build causal features from the strategy's signal frame (must already contain
    ``zscore``, ``spread``, ``hedge_ratio``, ``market_trend`` from ``generate_signals``)."""
    w = ml_config.feature_lookback
    zscore = signals["zscore"]
    spread = signals["spread"]
    hedge_ratio = signals["hedge_ratio"]
    price_a = signals[symbol_a]
    price_b = signals[symbol_b]

    features = pd.DataFrame(index=signals.index)
    features["zscore"] = zscore
    features["zscore_abs"] = zscore.abs()
    features["zscore_roc"] = zscore.diff(w)
    features["zscore_velocity"] = zscore.diff()
    features["spread_vol"] = spread.diff().rolling(w).std()
    features["market_trend"] = signals["market_trend"]
    features["hedge_ratio_change"] = hedge_ratio.pct_change(w)
    features["price_a_momentum"] = price_a.pct_change(w)
    features["price_b_momentum"] = price_b.pct_change(w)
    features["price_a_vol"] = price_a.pct_change().rolling(w).std()
    features["price_b_vol"] = price_b.pct_change().rolling(w).std()
    return features


def build_reversion_labels(zscore: pd.Series, ml_config: MLConfig) -> pd.Series:
    """Binary label: 1 if the z-score excursion at time t meaningfully shrinks by
    time t + horizon (mean reversion), 0 if it persists or grows (trend continuation).
    NaN where the excursion is too small to be a real entry candidate, or where the
    future horizon falls outside the available data (uses ``.shift(-horizon)``, i.e.
    future information -- only valid for constructing training labels, never at
    inference time)."""
    horizon = ml_config.label_horizon
    future_abs_z = zscore.shift(-horizon).abs()
    current_abs_z = zscore.abs()

    label = pd.Series(np.nan, index=zscore.index)
    candidate = current_abs_z >= ml_config.min_abs_zscore_for_label
    reverted = future_abs_z <= (current_abs_z * ml_config.reversion_shrink)

    label[candidate] = reverted[candidate].astype(float)
    return label


def build_training_dataset(
    signals: pd.DataFrame, symbol_a: str, symbol_b: str, ml_config: MLConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned on the same index, with NaNs (missing features or
    unlabeled bars) dropped."""
    X = build_feature_matrix(signals, symbol_a, symbol_b, ml_config)
    y = build_reversion_labels(signals["zscore"], ml_config)

    combined = X.copy()
    combined["__label__"] = y
    combined = combined.dropna()

    return combined.drop(columns="__label__"), combined["__label__"].astype(int)
