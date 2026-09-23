from __future__ import annotations

import numpy as np
import pandas as pd

from core.ml.baselines import flatten_sequences
from core.ml.calibration import calibrate_validation_logits
from core.ml.multitimeframe import align_closed_features
from core.ml.providers import AssetSpec, provider_for
from core.ml.scoring import score_trade_quality, threshold_metrics
from core.ml.splits import chronological_split, walk_forward_date_splits


def test_closed_alignment_never_uses_future_rows() -> None:
    source = pd.DataFrame(
        {"signal": [1.0, 2.0]},
        index=pd.to_datetime(["2024-01-01 00:15", "2024-01-01 00:30"]),
    )
    target = pd.date_range("2024-01-01 00:10", periods=3, freq="10min")
    aligned = align_closed_features(source, target)
    assert np.isnan(aligned["signal"].iloc[0])
    assert aligned["signal"].iloc[1:].tolist() == [1.0, 2.0]


def test_date_splits_and_walk_forward_are_chronological() -> None:
    timestamps = pd.date_range("2024-01-01", periods=24 * 10, freq="h")
    split = chronological_split(
        timestamps, "2024-01-01", "2024-01-04 23:00", "2024-01-05", "2024-01-06 23:00", "2024-01-07", "2024-01-10 23:00"
    )
    assert set(split.train).isdisjoint(split.validation)
    folds = walk_forward_date_splits(
        timestamps,
        [
            {"train_start": "2024-01-01", "train_end": "2024-01-03", "validation_start": "2024-01-04", "validation_end": "2024-01-05"},
            {"train_start": "2024-01-01", "train_end": "2024-01-05", "validation_start": "2024-01-06", "validation_end": "2024-01-07"},
        ],
    )
    assert len(folds) == 2
    assert timestamps[folds[0].validation].min() > timestamps[folds[0].train].max()


def test_temperature_calibration_is_validation_only_and_serializable() -> None:
    calibrator = calibrate_validation_logits(np.array([[4.0, 0.0], [0.0, 4.0]]), np.array([0, 1]))
    assert calibrator is not None
    assert calibrator.predict_proba(np.zeros((2, 2))).shape == (2, 2)
    assert calibrator.to_dict()["method"] == "temperature"


def test_baseline_and_provider_interfaces_are_configurable() -> None:
    sequences = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert flatten_sequences(sequences).shape == (2, 4)
    provider = provider_for("parquet", cache_dir="data")
    assert isinstance(AssetSpec("BTC", "BTC/USDT", "ccxt", "1h", 30), AssetSpec)
    assert provider.__class__.__name__ == "ParquetProvider"


def test_threshold_metrics_include_realized_r_and_win_rate() -> None:
    quality = score_trade_quality(
        np.array([[0.05, 0.05, 0.90], [0.90, 0.05, 0.05]]),
        np.array([0.04, 0.03]),
        np.array([-0.01, -0.01]),
    )
    metrics = threshold_metrics(quality, np.array([1, -1]), realized_r=np.array([2.0, -1.0]))
    assert metrics["signals_at_80"] == 2
    assert metrics["average_r_at_80"] == 0.5
    assert metrics["win_rate_at_80"] == 0.5
