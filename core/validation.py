"""Walk-forward validation harness to guard against overfitting.

Rather than fitting the strategy once on the full history and reporting
in-sample results, the timeline is split into rolling train/test folds. Each
fold's out-of-sample segment is backtested independently using only
information available up to the start of the fold, and results are
aggregated to judge robustness across regimes.

Folds are independent by construction, so the hybrid (ML-retraining) variant
can optionally run them in parallel across CPU cores (``WalkForwardConfig.n_jobs``)
to make use of a multi-core Ryzen 7 host during CPU-bound GBM training.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

import pandas as pd

from core.backtester import _periods_per_year, run_backtest, run_backtest_from_signals
from core.config import TradingBotConfig
from core.hybrid_strategy import apply_ml_gate, train_ml_gate
from core.macro import apply_macro_gate, fetch_macro_data
from core.metrics import summarize_performance
from core.strategy import generate_signals

FoldBounds = tuple[int, int, int, int]


@dataclass
class WalkForwardFold:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    metrics: dict


def make_folds(n_rows: int, config: TradingBotConfig) -> list[FoldBounds]:
    wf = config.walk_forward
    folds = []
    train_start = 0
    while True:
        train_end = train_start + wf.train_size
        test_end = train_end + wf.test_size
        if test_end > n_rows:
            break
        folds.append((train_start, train_end, train_end, test_end))
        train_start += wf.step_size
    return folds


def _resolve_macro_df(config: TradingBotConfig, macro_df: pd.DataFrame | None) -> pd.DataFrame | None:
    if not config.macro.enabled:
        return None
    return macro_df if macro_df is not None else fetch_macro_data(config.macro)


def _run_baseline_fold(
    df: pd.DataFrame, config: TradingBotConfig, macro_df: pd.DataFrame | None, bounds: FoldBounds
) -> WalkForwardFold:
    train_start, train_end, test_start, test_end = bounds
    periods_per_year = _periods_per_year(config.data.resample_to or config.data.base_timeframe)

    # Include the training window purely to warm up rolling windows (hedge ratio,
    # z-score, ATR) with no look-ahead beyond the test segment; only the test
    # segment's performance is scored.
    window_df = df.iloc[train_start:test_end].copy()
    backtest = run_backtest(window_df, config, macro_df)
    test_segment = backtest.loc[df.index[test_start] : df.index[test_end - 1]]

    metrics = summarize_performance(
        test_segment["strategy_return"], periods_per_year, test_segment["trade"] * test_segment["strategy_return"]
    )
    return WalkForwardFold(train_start, train_end, test_start, test_end, metrics)


def run_walk_forward(
    df: pd.DataFrame, config: TradingBotConfig, macro_df: pd.DataFrame | None = None
) -> list[WalkForwardFold]:
    """Run the strategy fold-by-fold. The strategy itself is purely rule-based
    (rolling OLS + rolling z-score), so no explicit parameter re-fitting step is
    required beyond feeding each fold's data through the same rolling windows;
    what changes per fold is which out-of-sample segment is scored."""
    folds_idx = make_folds(len(df), config)
    if len(folds_idx) < config.walk_forward.min_folds:
        raise ValueError(
            f"Only {len(folds_idx)} walk-forward folds available, "
            f"need >= {config.walk_forward.min_folds}. Provide more history or shrink fold sizes."
        )

    macro_df = _resolve_macro_df(config, macro_df)
    return [_run_baseline_fold(df, config, macro_df, bounds) for bounds in folds_idx]


def summarize_folds(folds: list[WalkForwardFold]) -> pd.DataFrame:
    rows = []
    for i, fold in enumerate(folds):
        row = {"fold": i, **fold.metrics}
        rows.append(row)
    return pd.DataFrame(rows)


def _run_hybrid_fold(
    df: pd.DataFrame, config: TradingBotConfig, macro_df: pd.DataFrame | None, bounds: FoldBounds
) -> WalkForwardFold:
    """Train the ML gate on the fold's train segment only (purged, see
    ``train_ml_gate``) and score strictly on the held-out test segment. Defined at
    module level (rather than nested) so it can be pickled for ``ProcessPoolExecutor``."""
    train_start, train_end, test_start, test_end = bounds
    periods_per_year = _periods_per_year(config.data.resample_to or config.data.base_timeframe)

    window_df = df.iloc[train_start:test_end].copy()
    signals = generate_signals(window_df, config.data.symbol_a, config.data.symbol_b, config.strategy)

    test_start_ts = df.index[test_start]
    model = train_ml_gate(signals, test_start_ts, config)
    gated_signals = apply_ml_gate(signals, model, config)
    gated_signals = apply_macro_gate(gated_signals, config, macro_df)

    backtest = run_backtest_from_signals(gated_signals, config)
    test_segment = backtest.loc[df.index[test_start] : df.index[test_end - 1]]

    metrics = summarize_performance(
        test_segment["strategy_return"], periods_per_year, test_segment["trade"] * test_segment["strategy_return"]
    )
    metrics["mean_ml_proba"] = float(test_segment["ml_proba"].mean())
    return WalkForwardFold(train_start, train_end, test_start, test_end, metrics)


def run_walk_forward_hybrid(
    df: pd.DataFrame, config: TradingBotConfig, macro_df: pd.DataFrame | None = None
) -> list[WalkForwardFold]:
    """Walk-forward validation for the ML-gated hybrid strategy: the ML model is
    re-trained from scratch on each fold's train segment only (strictly before the
    test segment's start timestamp, with a purge/embargo for the label horizon) and
    evaluated purely out-of-sample. Set ``config.walk_forward.n_jobs`` to -1 to
    parallelize independent folds across all Ryzen 7 CPU cores."""
    folds_idx = make_folds(len(df), config)
    if len(folds_idx) < config.walk_forward.min_folds:
        raise ValueError(
            f"Only {len(folds_idx)} walk-forward folds available, "
            f"need >= {config.walk_forward.min_folds}. Provide more history or shrink fold sizes."
        )

    macro_df = _resolve_macro_df(config, macro_df)
    n_jobs = config.walk_forward.n_jobs

    if n_jobs == 1:
        return [_run_hybrid_fold(df, config, macro_df, bounds) for bounds in folds_idx]

    max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    worker = partial(_run_hybrid_fold, df, config, macro_df)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, folds_idx))

