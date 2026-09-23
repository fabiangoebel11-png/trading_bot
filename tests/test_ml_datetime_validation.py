from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.ml.date_validation import chronological_purged_split, effective_period, horizon_description, validate_symbol_coverage
from core.ml.features import align_macro_to_intraday


def _align(macro_index, target_index, lag=0):
    macro = pd.DataFrame({"value": np.arange(len(macro_index), dtype=float)}, index=macro_index)
    return align_macro_to_intraday(macro, pd.DatetimeIndex(target_index), lag)


def test_macro_alignment_normalizes_naive_and_aware_timezones() -> None:
    target = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    for macro_index in (
        pd.date_range("2024-01-01", periods=2, freq="h"),
        pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
        pd.date_range("2024-01-01", periods=2, freq="h", tz="America/New_York"),
    ):
        result = _align(macro_index, target)
        assert result.index.tz is not None
        assert result.index.equals(target)
        if macro_index.tz is None or str(macro_index.tz) == "UTC":
            assert result["value"].iloc[0] == 0.0
        else:
            assert pd.isna(result["value"].iloc[0])
        assert np.isfinite(result["macro_data_age_hours"].iloc[-1])


def test_macro_reporting_lag_prevents_lookahead() -> None:
    macro = pd.DataFrame({"value": [10.0, 20.0]}, index=pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC"))
    target = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    result = align_macro_to_intraday(macro, target, reporting_lag_days=1)
    assert result["value"].iloc[0] != 10.0
    assert result["value"].iloc[2] == 10.0


def test_effective_period_clips_to_actual_symbol_history() -> None:
    index = pd.date_range("2025-01-01", periods=10, freq="D", tz="UTC")
    period = effective_period(index, "2010-01-01", "2021-12-31")
    assert period.samples == 0
    with pytest.raises(ValueError, match="insufficient training coverage"):
        validate_symbol_coverage(
            {"QQQ": pd.DataFrame(index=index)},
            train_end="2024-12-31", validation_start="2025-01-01", validation_end="2025-12-31",
            test_start="2026-01-01", test_end=None,
        )


def test_horizon_description_uses_target_timeframe() -> None:
    assert horizon_description("5m", 72).startswith("6 hours")
    assert horizon_description("1h", 72).startswith("3 days")


def test_fixed_split_uses_asset_start_and_purges_label_boundaries() -> None:
    index = pd.date_range("2023-09-25", "2026-01-10", freq="5min", tz="UTC")
    split = chronological_purged_split(
        index, np.arange(len(index)), train_end="2024-12-31 23:59:59",
        validation_start="2025-01-01", validation_end="2025-12-31 23:59:59",
        test_start="2026-01-01", purge_hours=6,
        minimum_train_samples=1, minimum_validation_samples=1, minimum_test_samples=1,
    )
    assert index[split.train_idx].min() == index.min()
    assert index[split.validation_idx].min() >= pd.Timestamp("2025-01-01", tz="UTC")
    assert index[split.test_idx].min() >= pd.Timestamp("2026-01-01", tz="UTC")
    assert index[split.train_idx].max() < pd.Timestamp("2025-01-01", tz="UTC") - pd.Timedelta(hours=6)
    assert index[split.validation_idx].max() < pd.Timestamp("2026-01-01", tz="UTC") - pd.Timedelta(hours=6)


def test_fixed_split_rejects_empty_test_window() -> None:
    index = pd.date_range("2023-09-25", "2025-12-31", freq="h", tz="UTC")
    with pytest.raises(ValueError, match="Insufficient test samples"):
        chronological_purged_split(
            index, np.arange(len(index)), train_end="2024-12-31", validation_start="2025-01-01",
            validation_end="2025-12-31", test_start="2026-01-01",
            minimum_train_samples=1, minimum_validation_samples=1, minimum_test_samples=1,
        )
