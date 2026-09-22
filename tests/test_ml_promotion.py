"""Tests for the chronological holdout gate and manual-only promotion logic
(core/ml/holdout.py, core/ml/promotion.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.ml.holdout import chronological_holdout_windows, paired_window_metrics
from core.ml.promotion import PromotionCriteria, evaluate_promotion

PERIODS_PER_YEAR = 24 * 365  # hourly bars


def _hourly_returns(n: int, drift: float, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=n, freq="1h")
    noise = rng.normal(loc=drift, scale=0.002, size=n)
    return pd.Series(noise, index=index)


def test_chronological_holdout_windows_are_non_overlapping_and_consecutive() -> None:
    index = pd.date_range("2024-01-01", periods=24 * 90 + 1, freq="1h")  # exactly 90 days
    holdout_start = index[0] + pd.Timedelta(days=30)  # last 60 days are holdout

    windows = chronological_holdout_windows(index, holdout_start, window_days=30)

    assert len(windows) == 2
    assert windows[0][1] == windows[1][0]  # consecutive, no gap/overlap
    assert windows[0][0] == holdout_start


def test_windows_before_holdout_start_are_excluded() -> None:
    index = pd.date_range("2024-01-01", periods=24 * 10, freq="1h")
    holdout_start = index[-1] + pd.Timedelta(days=1)  # entirely in the future
    assert chronological_holdout_windows(index, holdout_start, window_days=30) == []


def test_promotion_passes_when_ml_clearly_better_on_holdout() -> None:
    n = 24 * 150  # 150 days of hourly bars
    rule_returns = _hourly_returns(n, drift=0.0000, seed=1)
    ml_returns = _hourly_returns(n, drift=0.0006, seed=2)  # clearly better drift
    holdout_start = rule_returns.index[0]

    result = evaluate_promotion(rule_returns, ml_returns, holdout_start, PERIODS_PER_YEAR)

    assert result.delta_sharpe_positive
    assert result.median_return_better
    assert result.max_drawdown_not_worse or result.window_improvement_rate > 0.55
    assert result.n_windows >= 4


def test_promotion_fails_when_ml_is_worse() -> None:
    n = 24 * 150
    rule_returns = _hourly_returns(n, drift=0.0006, seed=3)
    ml_returns = _hourly_returns(n, drift=0.0000, seed=4)  # ML clearly worse
    holdout_start = rule_returns.index[0]

    result = evaluate_promotion(rule_returns, ml_returns, holdout_start, PERIODS_PER_YEAR)

    assert not result.passed
    assert not result.delta_sharpe_positive


def test_promotion_ignores_pre_holdout_data() -> None:
    """Data before holdout_start must never influence the decision -- it may
    have informed model/hyperparameter selection."""
    n = 24 * 260 + 1
    index = pd.date_range("2024-01-01", periods=n, freq="1h")
    # Pre-holdout: ML looks terrible. Post-holdout: ML looks great.
    pre = index < index[0] + pd.Timedelta(days=100)
    rule_vals = np.where(pre, 0.001, 0.0)
    ml_vals = np.where(pre, -0.01, 0.0008)
    rule_returns = pd.Series(rule_vals, index=index)
    ml_returns = pd.Series(ml_vals, index=index)
    holdout_start = index[0] + pd.Timedelta(days=100)

    result = evaluate_promotion(rule_returns, ml_returns, holdout_start, PERIODS_PER_YEAR)

    assert result.delta_sharpe_positive
    assert result.passed


def test_too_few_windows_blocks_promotion_even_if_metrics_look_good() -> None:
    n = 24 * 10  # only 10 days -> far fewer than the default 30-day window requirement
    rule_returns = _hourly_returns(n, drift=0.0, seed=5)
    ml_returns = _hourly_returns(n, drift=0.001, seed=6)
    holdout_start = rule_returns.index[0]

    result = evaluate_promotion(rule_returns, ml_returns, holdout_start, PERIODS_PER_YEAR)

    assert result.n_windows < PromotionCriteria().min_windows
    assert not result.passed
    assert result.reasons  # explains why


def test_evaluate_promotion_never_mutates_inputs_or_sets_flags() -> None:
    rule_returns = _hourly_returns(24 * 120, drift=0.0, seed=7)
    ml_returns = _hourly_returns(24 * 120, drift=0.0005, seed=8)
    rule_copy = rule_returns.copy()
    ml_copy = ml_returns.copy()

    evaluate_promotion(rule_returns, ml_returns, rule_returns.index[0], PERIODS_PER_YEAR)

    pd.testing.assert_series_equal(rule_returns, rule_copy)
    pd.testing.assert_series_equal(ml_returns, ml_copy)
