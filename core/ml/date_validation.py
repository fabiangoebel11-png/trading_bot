"""Per-symbol chronological dataset coverage validation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import pandas as pd


def _utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class EffectivePeriod:
    requested_start: pd.Timestamp
    requested_end: pd.Timestamp
    effective_start: pd.Timestamp | None
    effective_end: pd.Timestamp | None
    samples: int

    @property
    def usable(self) -> bool:
        return self.samples > 0 and self.effective_start is not None and self.effective_end is not None


@dataclass(frozen=True)
class ChronologicalSplit:
    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray
    train_start: pd.Timestamp | None
    train_end: pd.Timestamp | None
    validation_start: pd.Timestamp | None
    validation_end: pd.Timestamp | None
    test_start: pd.Timestamp | None
    test_end: pd.Timestamp | None


def chronological_purged_split(
    timestamps: pd.DatetimeIndex,
    label_end_positions: np.ndarray,
    *,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str | None = None,
    purge_hours: int = 6,
    minimum_train_samples: int = 1,
    minimum_validation_samples: int = 1,
    minimum_test_samples: int = 1,
) -> ChronologicalSplit:
    """Split sequence timestamps and purge samples whose labels cross boundaries."""
    index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).sort_values()
    if len(index) != len(label_end_positions):
        raise ValueError("timestamps and label_end_positions must have equal length")
    train_end_ts = _utc_timestamp(train_end)
    validation_start_ts = _utc_timestamp(validation_start)
    validation_end_ts = _utc_timestamp(validation_end)
    test_start_ts = _utc_timestamp(test_start)
    test_end_ts = _utc_timestamp(test_end) if test_end else index.max()
    label_end = index[np.clip(np.asarray(label_end_positions, dtype=int), 0, len(index) - 1)]
    purge_delta = pd.Timedelta(hours=purge_hours)

    train = (index <= train_end_ts) & (label_end < validation_start_ts) & (index < validation_start_ts - purge_delta)
    validation = (
        (index >= validation_start_ts)
        & (index <= validation_end_ts)
        & (label_end < test_start_ts)
        & (index < test_start_ts - purge_delta)
    )
    test = (index >= test_start_ts) & (index <= test_end_ts)
    counts = {"train": int(train.sum()), "validation": int(validation.sum()), "test": int(test.sum())}
    minimums = {"train": minimum_train_samples, "validation": minimum_validation_samples, "test": minimum_test_samples}
    for name, count in counts.items():
        if count < minimums[name]:
            raise ValueError(f"Insufficient {name} samples after purge: {count}, need {minimums[name]}")

    def bounds(mask: np.ndarray) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        selected = index[mask]
        return (selected.min(), selected.max()) if len(selected) else (None, None)

    train_bounds = bounds(train)
    validation_bounds = bounds(validation)
    test_bounds = bounds(test)
    return ChronologicalSplit(
        np.flatnonzero(train), np.flatnonzero(validation), np.flatnonzero(test),
        *train_bounds, *validation_bounds, *test_bounds,
    )


def effective_period(index: pd.DatetimeIndex, start: str, end: str) -> EffectivePeriod:
    timestamps = pd.DatetimeIndex(pd.to_datetime(index, utc=True)).sort_values()
    requested_start = _utc_timestamp(start)
    requested_end = _utc_timestamp(end)
    mask = (timestamps >= requested_start) & (timestamps <= requested_end)
    selected = timestamps[mask]
    return EffectivePeriod(
        requested_start=requested_start,
        requested_end=requested_end,
        effective_start=selected.min() if len(selected) else None,
        effective_end=selected.max() if len(selected) else None,
        samples=len(selected),
    )


def validate_symbol_coverage(
    datasets: dict[str, pd.DataFrame],
    *,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str | None,
    min_train_samples: int = 1,
) -> list[dict[str, object]]:
    """Print and validate configured periods without silently moving boundaries."""
    reports: list[dict[str, object]] = []
    for symbol, frame in datasets.items():
        available = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
        periods = {
            "train": effective_period(frame.index, str(available.min()) if len(frame.index) else train_end, train_end),
            "validation": effective_period(frame.index, validation_start, validation_end),
            "test": effective_period(frame.index, test_start, test_end or str(pd.to_datetime(frame.index, utc=True).max())),
        }
        report = {
            "symbol": symbol,
            "available_start": available.min() if len(available) else None,
            "available_end": available.max() if len(available) else None,
            "periods": periods,
        }
        reports.append(report)
        print(f"\n{symbol}")
        print(f"available: {report['available_start']} -> {report['available_end']}")
        for name, period in periods.items():
            print(
                f"requested {name}: {period.requested_start.date()} -> {period.requested_end.date()} | "
                f"effective: {period.effective_start} -> {period.effective_end} | samples: {period.samples}"
            )
        if periods["train"].samples < min_train_samples:
            raise ValueError(
                f"{symbol} has insufficient training coverage: "
                f"{periods['train'].samples} samples in configured train period; "
                "refusing to start model fitting."
            )
    return reports


def horizon_description(timeframe: str, horizon_bars: int) -> str:
    delta = pd.Timedelta(timeframe) * horizon_bars
    total_minutes = int(delta.total_seconds() // 60)
    if total_minutes % (24 * 60) == 0:
        return f"{total_minutes // (24 * 60)} days ({horizon_bars} x {timeframe})"
    if total_minutes % 60 == 0:
        return f"{total_minutes // 60} hours ({horizon_bars} x {timeframe})"
    return f"{total_minutes} minutes ({horizon_bars} x {timeframe})"
