import numpy as np
import pandas as pd
import pytest

from core.ml.phase3_pipeline import fit_train_only_scaler, inspect_ohlcv_quality
from core.ml.dataset import purged_walk_forward_splits


def _frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2024-01-01 00:00Z", "2024-01-01 01:00Z", "2024-01-01 04:00Z", "2024-01-01 02:00Z"])
    close = pd.Series([10.0, 11.0, 13.0, 12.0], index=index)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0}, index=index)


def test_quality_report_detects_order_and_gap() -> None:
    report = inspect_ohlcv_quality(_frame(), "1h")
    assert report["unsorted"] is True
    assert report["gap_count"] == 1
    assert report["duplicate_timestamps"] == 0


def test_train_only_scaler_does_not_fit_on_validation_values() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    features = pd.DataFrame({"x": [0, 1, 2, 100, 100, 100]}, index=index)
    scaler, transformed = fit_train_only_scaler(features, index[:3])
    assert scaler.mean_[0] == 1.0
    assert transformed.loc[index[3], "x"] > 50


def test_purging_and_embargo_remove_boundary_training_rows() -> None:
    t1_pos = np.arange(30) + 3
    folds = purged_walk_forward_splits(30, t1_pos, n_splits=2, embargo_fraction=0.1)
    assert len(folds) == 2
    for fold in folds:
        assert set(fold.train_idx).isdisjoint(fold.test_idx)
        assert all(t1_pos[idx] < fold.test_idx[0] for idx in fold.train_idx)
        assert all(idx < fold.test_idx[0] - 3 for idx in fold.train_idx)


def test_quality_rejects_missing_required_columns() -> None:
    with pytest.raises(ValueError):
        inspect_ohlcv_quality(_frame().drop(columns="close"), "1h")