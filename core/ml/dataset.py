"""Sequence construction and purged/embargoed walk-forward cross-validation.

Two independent overfitting guards live here, both needed because the triple-
barrier labels have *overlapping* information windows (label at bar t can
depend on data up to t + horizon):

- **Purging**: drop any training row whose label window ``[t, t1]`` overlaps
  the test fold's bar range. Without this, a naive time-split leaks the test
  fold's near-future price action into training rows just before the split.
- **Embargo**: additionally drop a small buffer of training rows immediately
  before the test fold starts, since features built from rolling windows are
  autocorrelated across nearby bars even once labels no longer overlap.

Folds are walk-forward (train = everything before the fold, purged/embargoed),
not shuffled k-fold -- consistent with ``core/validation.py``'s existing
walk-forward harness, and required for time series where shuffling would let
the model see the future during training.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Turn a (n, features) matrix into (n - seq_len + 1, seq_len, features)
    sliding windows. Window ``i`` covers rows ``[i, i + seq_len - 1]`` and is
    labeled with the target at its last row -- causal, no future rows used."""
    n = X.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1]), dtype=X.dtype)
    return np.stack([X[i : i + seq_len] for i in range(n - seq_len + 1)])


@dataclass
class Fold:
    train_idx: np.ndarray  # positional indices into the sequence array
    test_idx: np.ndarray


def purged_walk_forward_splits(n_sequences: int, t1_pos: np.ndarray, n_splits: int, embargo_fraction: float) -> list[Fold]:
    """``t1_pos`` is the positional index (into the same 0..n_sequences-1 space)
    at which each sequence's label outcome becomes known. Produces
    ``n_splits`` expanding-window folds: fold ``i``'s test block is roughly
    ``1/(n_splits+1)`` of the data, and its train block is everything before
    it, purged of any row whose ``t1_pos`` reaches into the test block, plus
    an embargo buffer right before the test block starts.
    """
    if n_sequences < (n_splits + 1) * 10:
        raise ValueError(f"Only {n_sequences} sequences available, too few for {n_splits} purged folds.")

    fold_size = n_sequences // (n_splits + 1)
    embargo = max(1, int(n_sequences * embargo_fraction))

    folds = []
    for i in range(1, n_splits + 1):
        test_start = i * fold_size
        test_end = n_sequences if i == n_splits else (i + 1) * fold_size
        test_idx = np.arange(test_start, test_end)

        candidate_train = np.arange(0, test_start)
        # Purge: drop training rows whose label window reaches into the test block.
        purge_mask = t1_pos[candidate_train] < test_start
        # Embargo: additionally drop a buffer immediately before the test block.
        embargo_mask = candidate_train < (test_start - embargo)
        train_idx = candidate_train[purge_mask & embargo_mask]

        folds.append(Fold(train_idx=train_idx, test_idx=test_idx))
    return folds


def build_row_level_dataset(
    features: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Align features (causal, no NaNs at inference time once warmed up) with
    triple-barrier labels, drop unusable rows (feature warm-up / label tail
    with no complete future window), and resolve each label's ``t1`` timestamp
    to a *positional* index within the cleaned, row-aligned frame -- this is
    the ``t1_pos`` purging needs. ``method="bfill"`` is used deliberately: if a
    touch timestamp itself got dropped (e.g. right at the tail), rounding up
    to the next surviving row only ever purges *more* training data, never
    less, which is the safe direction for a leakage guard.
    """
    combined = features.join(labels, how="inner").dropna()
    X = combined.drop(columns=["label", "t1"])
    y = combined["label"].astype(int)
    t1_pos = combined.index.get_indexer(combined["t1"], method="bfill")
    t1_pos = np.where(t1_pos < 0, len(combined) - 1, t1_pos)
    return X, y, t1_pos.astype(np.int64)


def time_decay_sample_weights(timestamps: pd.DatetimeIndex, half_life_days: float | None) -> np.ndarray:
    """Exponential recency weighting for the training loss: ``weight = 0.5 **
    (age_days / half_life_days)``, where ``age_days`` is measured from the
    *most recent* timestamp in ``timestamps`` (i.e. relative to whatever slice
    of history is being fit, not wall-clock "today"). Standard non-stationary-
    market practice (recency weighting is common in quant/HFT covariance and
    signal estimation) -- lets a widened training window span multiple market
    regimes (e.g. the 2020/2021 bull) without that older, structurally
    different microstructure dominating the gradient signal over the current
    regime.

    Deliberately a function of *time only* -- never of the realized label,
    return, or direction -- so a bull-market bar and a bear-market bar of the
    same age get exactly the same weight. That symmetry is intentional: this
    is recency weighting, not "weight profitable regimes higher" (which would
    be look-ahead/profit-fitting, not a real training technique).

    Returns an all-ones array (uniform weighting, i.e. a no-op) if
    ``half_life_days`` is ``None``/``0``.
    """
    if not half_life_days:
        return np.ones(len(timestamps), dtype=np.float64)
    reference = timestamps.max()
    age_days = (reference - timestamps).total_seconds() / 86400.0
    return np.exp(-np.log(2.0) * age_days / half_life_days)
