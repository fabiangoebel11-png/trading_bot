"""Walk-forward validation harness to guard against overfitting.

Rather than fitting the strategy once on the full history and reporting
in-sample results, the timeline is split into rolling train/test folds. Each
fold's out-of-sample segment is backtested independently using only
information available up to the start of the fold, and results are
aggregated to judge robustness across regimes.

Folds are independent by construction, so they can optionally run in parallel
across CPU cores (``WalkForwardConfig.n_jobs``) on a multi-core Ryzen 7 host.

Note: the ML-gated hybrid walk-forward (mean-reversion probability model) was
retired together with the pairs-trading strategy it gated; see git history /
``core/hybrid_strategy.py`` if that approach needs to be revived for a future
mean-reversion component.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial

import pandas as pd

from core.backtester import _periods_per_year, run_backtest
from core.config import TradingBotConfig
from core.macro import fetch_macro_data
from core.funding import load_portfolio_funding
from core.metrics import summarize_performance

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


def _resolve_funding_df(
    config: TradingBotConfig, funding_df: dict[str, pd.Series] | None
) -> dict[str, pd.Series] | None:
    if not config.funding.enabled:
        return None
    return funding_df if funding_df is not None else load_portfolio_funding(config.data, config.funding)


def _run_baseline_fold(
    multi_ohlc: dict[str, pd.DataFrame],
    config: TradingBotConfig,
    macro_df: pd.DataFrame | None,
    funding_df: dict[str, pd.Series] | None,
    bounds: FoldBounds,
) -> WalkForwardFold:
    train_start, train_end, test_start, test_end = bounds
    periods_per_year = _periods_per_year(config.data.resample_to or config.data.base_timeframe)
    index = next(iter(multi_ohlc.values())).index

    # Include the training window purely to warm up rolling windows (EMA, Donchian,
    # ATR) with no look-ahead beyond the test segment; only the test segment's
    # performance is scored.
    window = {symbol: df.iloc[train_start:test_end].copy() for symbol, df in multi_ohlc.items()}
    backtest = run_backtest(window, config, macro_df, funding_df)
    test_segment = backtest.loc[index[test_start] : index[test_end - 1]]

    metrics = summarize_performance(
        test_segment["strategy_return"], periods_per_year, test_segment["trade"] * test_segment["strategy_return"]
    )
    return WalkForwardFold(train_start, train_end, test_start, test_end, metrics)


def run_walk_forward(
    multi_ohlc: dict[str, pd.DataFrame],
    config: TradingBotConfig,
    macro_df: pd.DataFrame | None = None,
    funding_df: dict[str, pd.Series] | None = None,
) -> list[WalkForwardFold]:
    """Run the trend-following strategy fold-by-fold. The strategy is purely
    rule-based (EMA regime filter + Donchian breakout), so no explicit parameter
    re-fitting is required beyond feeding each fold's data through the same
    rolling windows; what changes per fold is which out-of-sample segment is
    scored. Set ``config.walk_forward.n_jobs`` to -1 to parallelize independent
    folds across all Ryzen 7 CPU cores."""
    index = next(iter(multi_ohlc.values())).index
    folds_idx = make_folds(len(index), config)
    if len(folds_idx) < config.walk_forward.min_folds:
        raise ValueError(
            f"Only {len(folds_idx)} walk-forward folds available, "
            f"need >= {config.walk_forward.min_folds}. Provide more history or shrink fold sizes."
        )

    macro_df = _resolve_macro_df(config, macro_df)
    funding_df = _resolve_funding_df(config, funding_df)
    n_jobs = config.walk_forward.n_jobs

    if n_jobs == 1:
        return [_run_baseline_fold(multi_ohlc, config, macro_df, funding_df, bounds) for bounds in folds_idx]

    max_workers = os.cpu_count() if n_jobs == -1 else n_jobs
    worker = partial(_run_baseline_fold, multi_ohlc, config, macro_df, funding_df)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(worker, folds_idx))


def summarize_folds(folds: list[WalkForwardFold]) -> pd.DataFrame:
    rows = []
    for i, fold in enumerate(folds):
        row = {"fold": i, **fold.metrics}
        rows.append(row)
    return pd.DataFrame(rows)


