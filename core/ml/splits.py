"""Leakage-aware chronological fixed and walk-forward splits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DateSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def chronological_split(
    timestamps: pd.DatetimeIndex,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    test_start: str,
    test_end: str,
) -> DateSplit:
    """Return positional masks for explicitly configured, non-overlapping dates."""
    timestamps = pd.DatetimeIndex(timestamps)
    ranges = {
        "train": (pd.Timestamp(train_start), pd.Timestamp(train_end)),
        "validation": (pd.Timestamp(validation_start), pd.Timestamp(validation_end)),
        "test": (pd.Timestamp(test_start), pd.Timestamp(test_end)),
    }
    masks: dict[str, np.ndarray] = {}
    for name, (start, end) in ranges.items():
        if end < start:
            raise ValueError(f"{name} end precedes start: {start} > {end}")
        masks[name] = np.flatnonzero((timestamps >= start) & (timestamps <= end))
    if np.intersect1d(masks["train"], masks["validation"]).size:
        raise ValueError("train and validation ranges overlap")
    if np.intersect1d(masks["validation"], masks["test"]).size:
        raise ValueError("validation and test ranges overlap")
    if np.intersect1d(masks["train"], masks["test"]).size:
        raise ValueError("train and test ranges overlap")
    return DateSplit(masks["train"], masks["validation"], masks["test"])


def walk_forward_date_splits(
    timestamps: pd.DatetimeIndex,
    windows: list[dict[str, str]],
) -> list[DateSplit]:
    """Build expanding/rolling chronological folds from configured date windows."""
    result: list[DateSplit] = []
    timestamps = pd.DatetimeIndex(timestamps)
    for window in windows:
        train_start = pd.Timestamp(window["train_start"])
        train_end = pd.Timestamp(window["train_end"])
        validation_start = pd.Timestamp(window["validation_start"])
        validation_end = pd.Timestamp(window["validation_end"])
        if train_end >= validation_start:
            raise ValueError("walk-forward train_end must precede validation_start")
        train = np.flatnonzero((timestamps >= train_start) & (timestamps <= train_end))
        validation = np.flatnonzero((timestamps >= validation_start) & (timestamps <= validation_end))
        result.append(DateSplit(train=train, validation=validation, test=np.empty(0, dtype=int)))
    return result
