import numpy as np
import pandas as pd

from core.ml.phase3_pipeline import (
    HORIZON_DELTAS,
    TIMEFRAME_DELTAS,
    build_context_features,
    build_labels,
    build_local_features,
    closed_candles,
)


def _ohlcv(periods: int = 180) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    close = pd.Series(100 + np.arange(periods, dtype=float), index=index)
    return pd.DataFrame({"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0}, index=index)


def test_closed_candles_are_available_only_at_close() -> None:
    source = _ohlcv(3)
    closed = closed_candles(source, "1h")
    assert closed.index[0] == source.index[0] + TIMEFRAME_DELTAS["1h"]
    assert closed.index[0] > source.index[0]


def test_local_features_do_not_use_future_rows() -> None:
    source = _ohlcv()
    baseline, _ = build_local_features(closed_candles(source, "1h"), "btc_")
    changed = source.copy()
    changed.iloc[-20:, changed.columns.get_loc("close")] *= 4
    changed.iloc[-20:, changed.columns.get_loc("high")] *= 4
    changed_features, _ = build_local_features(closed_candles(changed, "1h"), "btc_")
    cutoff = -25
    pd.testing.assert_frame_equal(baseline.iloc[:cutoff], changed_features.iloc[:cutoff])


def test_context_features_use_backward_alignment_and_target_is_not_context() -> None:
    target = closed_candles(_ohlcv(), "1h")
    other = closed_candles(_ohlcv(), "1h")
    context, provenance = build_context_features({"BTC/USDT": target, "ETH/USDT": other}, target.index, "BTC/USDT")
    assert all("btc_usdt" not in column for column in context.columns)
    assert "context_ETH_USDT_return_1".lower() in {column.lower() for column in context.columns}
    assert provenance[next(iter(context.columns))]["availability_rule"].startswith("backward")


def test_labels_are_future_only_and_horizon_is_timeframe_aware() -> None:
    target = closed_candles(_ohlcv(20), "4h")
    labels = build_labels(target, "8h", "4h")
    assert labels["future_return"].iloc[-2:].isna().all()
    assert labels.index.equals(target.index)
    assert "future_return" not in build_local_features(target, "btc_")[0].columns
    assert set(("1h", "2h", "4h", "8h", "12h", "24h", "48h", "3d", "7d", "14d", "21d", "28d")) <= set(HORIZON_DELTAS)