"""Chronological holdout gate for ML shadow-mode promotion.

Unlike the training-time purged walk-forward CV (``core/ml/dataset.py``,
which only concerns model *fitting*), this module operates strictly on data
that never informed the model in any way: the model/hyperparameters are
frozen (persisted checkpoint + ``training_cutoff`` in the model meta, see
``core/ml/train.py``), and only bars strictly after that cutoff -- collected
live via ``core/ml/shadow.py: ShadowLogger`` in production, or sliced from
historical data after an explicit ``holdout_start`` for research -- are
allowed to influence the promotion decision. No retuning may happen on this
data; ``core/ml/promotion.py: evaluate_promotion`` is read-only.
"""
from __future__ import annotations

import pandas as pd

from core.metrics import max_drawdown, sharpe_ratio


def chronological_holdout_windows(
    index: pd.DatetimeIndex, holdout_start: pd.Timestamp, window_days: int = 30
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Non-overlapping, consecutive windows fully within
    ``[holdout_start, index[-1]]``. Deliberately non-overlapping (unlike the
    Monte Carlo bootstrap in ``core/monte_carlo.py``): promotion is decided on
    truly independent chronological blocks, not resampled/overlapping ones."""
    holdout_index = index[index >= holdout_start]
    if holdout_index.empty:
        return []
    start = holdout_index[0]
    end = holdout_index[-1]
    step = pd.Timedelta(days=window_days)
    windows = []
    cursor = start
    while cursor + step <= end + pd.Timedelta(seconds=1):
        windows.append((cursor, cursor + step))
        cursor += step
    return windows


def window_metrics(returns: pd.Series, periods_per_year: int) -> dict:
    cumulative = (1 + returns).cumprod()
    return {
        "sharpe": sharpe_ratio(returns, periods_per_year),
        "total_return_pct": float((cumulative.iloc[-1] - 1) * 100) if len(cumulative) else 0.0,
        "max_drawdown_pct": max_drawdown(cumulative) * 100 if len(cumulative) else 0.0,
        "n_periods": int(len(returns)),
    }


def paired_window_metrics(
    rule_returns: pd.Series,
    ml_returns: pd.Series,
    holdout_start: pd.Timestamp,
    periods_per_year: int,
    window_days: int = 30,
    min_periods_per_window: int = 24,
) -> pd.DataFrame:
    """One row per chronological holdout window with rule/ml/delta metrics.
    Windows with fewer than ``min_periods_per_window`` bars (e.g. a partial
    trailing window while shadow logging is still ongoing) are skipped."""
    index = rule_returns.index.union(ml_returns.index)
    windows = chronological_holdout_windows(index, holdout_start, window_days)
    rows = []
    for start, end in windows:
        rule_window = rule_returns[(rule_returns.index >= start) & (rule_returns.index < end)]
        ml_window = ml_returns[(ml_returns.index >= start) & (ml_returns.index < end)]
        if len(rule_window) < min_periods_per_window or len(ml_window) < min_periods_per_window:
            continue
        rule_m = window_metrics(rule_window, periods_per_year)
        ml_m = window_metrics(ml_window, periods_per_year)
        rows.append(
            {
                "window_start": start,
                "window_end": end,
                "rule_sharpe": rule_m["sharpe"],
                "ml_sharpe": ml_m["sharpe"],
                "delta_sharpe": ml_m["sharpe"] - rule_m["sharpe"],
                "rule_return_pct": rule_m["total_return_pct"],
                "ml_return_pct": ml_m["total_return_pct"],
                "delta_return_pct": ml_m["total_return_pct"] - rule_m["total_return_pct"],
                "rule_max_drawdown_pct": rule_m["max_drawdown_pct"],
                "ml_max_drawdown_pct": ml_m["max_drawdown_pct"],
                "delta_max_drawdown_pct": ml_m["max_drawdown_pct"] - rule_m["max_drawdown_pct"],
            }
        )
    return pd.DataFrame(rows)
