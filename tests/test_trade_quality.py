import numpy as np
import pandas as pd
from datetime import timedelta

from core.config import TradeQualityConfig
from core.ml.trade_quality import (
    TRADE_QUALITY_FEATURES,
    build_directional_labels,
    build_entry_features,
    purged_entry_splits,
)


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=400, freq="1h", tz="UTC")
    rng = np.random.default_rng(7)
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, len(index)))
    volume = rng.uniform(80.0, 120.0, len(index))
    return pd.DataFrame({"open": close, "high": close + 0.2, "low": close - 0.2, "close": close, "volume": volume}, index=index)


def test_directional_labels_resolve_long_and_short_with_costs() -> None:
    frame = _frame()
    signals = frame.copy()
    signals["position"] = 0.0
    signals.iloc[300, signals.columns.get_loc("position")] = 1.0
    signals.iloc[350, signals.columns.get_loc("position")] = -1.0
    config = TradeQualityConfig(horizon_bars=8, stop_atr_multiple=1.0, take_profit_atr_multiple=1.0)
    features = build_entry_features(signals, config)
    labels = build_directional_labels(frame, features, config)
    assert set(labels["direction"]) == {1, -1}
    assert {"label", "t1", "realized_r", "mfe", "mae"}.issubset(labels.columns)


def test_features_are_entry_known_and_future_labels_are_not_features() -> None:
    frame = _frame()
    signals = frame.copy()
    signals["position"] = 0.0
    signals.iloc[300, signals.columns.get_loc("position")] = 1.0
    config = TradeQualityConfig(horizon_bars=8)
    features = build_entry_features(signals, config)
    assert tuple(TRADE_QUALITY_FEATURES) == tuple(features.columns)
    assert not any(name in features.columns for name in ("label", "realized_return", "mfe", "mae", "t1"))


def test_purged_entry_splits_are_chronological_and_embargoed() -> None:
    index = pd.date_range("2026-01-01", periods=60, freq="1h", tz="UTC")
    dataset = pd.DataFrame({"t1": index + timedelta(hours=2)}, index=index)
    folds = purged_entry_splits(dataset, n_splits=3, embargo_bars=3)
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert train_idx.max(initial=-1) < test_idx.min()
        assert dataset.index[test_idx].min() > dataset.index[train_idx].max()
