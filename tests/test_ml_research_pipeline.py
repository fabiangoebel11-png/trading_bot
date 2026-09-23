from __future__ import annotations

import numpy as np
import pandas as pd

from core.ml.data import normalize_ohlcv
from core.ml.labeling import triple_barrier_labels
from core.ml.dataset import make_sequences
from core.ml.scoring import score_trade_quality, threshold_metrics


def test_normalize_ohlcv_deduplicates_and_rejects_invalid_prices() -> None:
    index = pd.to_datetime(["2024-01-01 00:00Z", "2024-01-01 00:00Z", "2024-01-01 01:00Z"])
    frame = pd.DataFrame(
        {
            "open": [100, 101, 100],
            "high": [101, 102, 99],
            "low": [99, 100, 101],
            "close": [100, 101, 100],
            "volume": [10, 11, -1],
        },
        index=index,
    )
    normalized = normalize_ohlcv(frame)
    assert len(normalized) == 2
    assert normalized.index.tz is not None
    assert normalized.iloc[0]["volume"] == 11
    assert normalized.iloc[1].isna().all()


def test_trade_quality_score_and_threshold_metrics_are_deterministic() -> None:
    probabilities = np.array([[0.05, 0.05, 0.90], [0.90, 0.05, 0.05]])
    quality = score_trade_quality(probabilities, np.array([0.04, 0.03]), np.array([-0.01, -0.01]))
    assert quality["direction"].tolist() == ["LONG", "SHORT"]
    assert (quality["score"] >= 80).all()
    metrics = threshold_metrics(quality, np.array([1, -1]), thresholds=(80.0, 90.0))
    assert metrics["signals_at_80"] == 2
    assert metrics["precision_at_80"] == 1.0


def test_triple_barrier_reports_full_horizon_excursions() -> None:
    index = pd.date_range("2024-01-01", periods=10, freq="1h")
    close = pd.Series(100.0, index=index)
    high = close.copy()
    low = close.copy()
    high.iloc[1] = 105.0
    low.iloc[2] = 97.0
    labels = triple_barrier_labels(
        close,
        pd.Series(2.0, index=index),
        horizon=3,
        atr_multiple=2.0,
        high=high,
        low=low,
        stop_atr_multiple=1.0,
        take_profit_atr_multiple=2.0,
    )
    assert labels.iloc[0]["label"] == 1
    assert labels.iloc[0]["time_to_barrier"] == 1
    assert labels.iloc[0]["mfe"] >= 0.05
    assert labels.iloc[0]["mae"] <= -0.03
    assert labels.iloc[-1]["label"] != labels.iloc[-1]["label"]
    assert labels.iloc[-1]["mfe"] != labels.iloc[-1]["mfe"]


def test_sequence_construction_is_a_zero_copy_view() -> None:
    values = np.arange(100 * 3, dtype=np.float32).reshape(100, 3)
    sequences = make_sequences(values, 10)
    assert sequences.shape == (91, 10, 3)
    assert np.shares_memory(values, sequences)
