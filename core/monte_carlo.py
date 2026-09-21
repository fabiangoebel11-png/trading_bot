"""Monte Carlo random-start stress test.

Rather than literally re-running the full walk-forward pipeline N times (the
hedge ratio / z-score / ML gate are causal and computed once on the full
history regardless of which sub-window we later score), this module computes
the full-history backtest exactly once and then resamples ``n_runs`` random
contiguous windows from the resulting daily/intraday return series. For each
window it computes the standard performance metrics (Sharpe, Sortino, total
return, max drawdown, profit factor). The resulting distribution answers the
practical question "does this strategy have an edge regardless of which
period you happened to start trading it in?" -- which is what a random-start
Monte Carlo test is meant to probe.

Caveat (documented, not hidden): because windows are drawn from a single
historical path and heavily overlap for large ``n_runs``, this is an
*overlapping block bootstrap*, not an independent Monte Carlo simulation --
neighboring windows share most of their bars and are therefore highly
correlated. Treat the resulting p-value/confidence numbers as directional,
not as textbook-rigorous independent-sample statistics.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.backtester import _periods_per_year, run_backtest
from core.config import TradingBotConfig

try:
    import torch

    from core.ml.device import get_device
except ImportError:  # pragma: no cover - exercised only without the "ml" extra
    torch = None
    get_device = None


@dataclass
class MonteCarloSummary:
    runs: pd.DataFrame
    mean_sharpe: float
    median_sharpe: float
    std_sharpe: float
    pct_profitable: float
    mean_total_return_pct: float
    t_stat_sharpe: float
    p_value_sharpe: float
    initial_capital_usdt: float
    mean_final_capital_usdt: float
    median_final_capital_usdt: float
    worst_case_max_drawdown_usdt: float
    mean_max_drawdown_usdt: float


def _z_two_sided_p_value(z: float) -> float:
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def _gather_windows_numpy(returns: np.ndarray, starts: np.ndarray, window_size: int) -> np.ndarray:
    idx = starts[:, None] + np.arange(window_size)[None, :]
    return returns[idx]


def _batched_metrics_numpy(windowed: np.ndarray, periods_per_year: int, initial_capital: float) -> dict[str, np.ndarray]:
    cumulative = np.cumprod(1 + windowed, axis=1)
    total_return = cumulative[:, -1] - 1.0
    mean = windowed.mean(axis=1)
    std = windowed.std(axis=1, ddof=1)
    sharpe = np.divide(mean, std, out=np.zeros_like(mean), where=std > 0) * math.sqrt(periods_per_year)

    downside = np.minimum(windowed, 0.0)
    downside_std = downside.std(axis=1, ddof=1)
    sortino = np.divide(mean, downside_std, out=np.zeros_like(mean), where=downside_std > 0) * math.sqrt(
        periods_per_year
    )

    running_max = np.maximum.accumulate(cumulative, axis=1)
    drawdown = cumulative / running_max - 1.0
    max_dd = drawdown.min(axis=1)

    # --- Money management: same window expressed as a currency equity curve ---
    capital_curve = initial_capital * cumulative
    capital_running_max = initial_capital * running_max
    max_dd_usdt = (capital_curve - capital_running_max).min(axis=1)
    final_capital_usdt = capital_curve[:, -1]

    gains = np.clip(windowed, 0, None).sum(axis=1)
    losses = -np.clip(windowed, None, 0).sum(axis=1)
    profit_factor = np.divide(gains, losses, out=np.full_like(gains, np.inf), where=losses > 0)

    return {
        "total_return_pct": total_return * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100,
        "profit_factor": profit_factor,
        "mean_return": mean,
        "final_capital_usdt": final_capital_usdt,
        "max_drawdown_usdt": max_dd_usdt,
    }


def _batched_metrics_gpu(windowed_np: np.ndarray, periods_per_year: int, initial_capital: float) -> dict[str, np.ndarray]:
    device = get_device()
    windowed = torch.tensor(windowed_np, device=device, dtype=torch.float64)

    cumulative = torch.cumprod(1 + windowed, dim=1)
    total_return = cumulative[:, -1] - 1.0
    mean = windowed.mean(dim=1)
    std = windowed.std(dim=1, unbiased=True)
    sharpe = torch.where(std > 0, mean / std, torch.zeros_like(mean)) * math.sqrt(periods_per_year)

    downside = torch.clamp(windowed, max=0.0)
    downside_std = downside.std(dim=1, unbiased=True)
    sortino = torch.where(downside_std > 0, mean / downside_std, torch.zeros_like(mean)) * math.sqrt(
        periods_per_year
    )

    running_max = torch.cummax(cumulative, dim=1).values
    drawdown = cumulative / running_max - 1.0
    max_dd = drawdown.min(dim=1).values

    # --- Money management: same window expressed as a currency equity curve ---
    capital_curve = initial_capital * cumulative
    capital_running_max = initial_capital * running_max
    max_dd_usdt = (capital_curve - capital_running_max).min(dim=1).values
    final_capital_usdt = capital_curve[:, -1]

    gains = torch.clamp(windowed, min=0).sum(dim=1)
    losses = -torch.clamp(windowed, max=0).sum(dim=1)
    profit_factor = torch.where(losses > 0, gains / losses, torch.full_like(gains, float("inf")))

    return {
        "total_return_pct": (total_return * 100).cpu().numpy(),
        "sharpe": sharpe.cpu().numpy(),
        "sortino": sortino.cpu().numpy(),
        "max_drawdown_pct": (max_dd * 100).cpu().numpy(),
        "profit_factor": profit_factor.cpu().numpy(),
        "mean_return": mean.cpu().numpy(),
        "final_capital_usdt": final_capital_usdt.cpu().numpy(),
        "max_drawdown_usdt": max_dd_usdt.cpu().numpy(),
    }


def run_monte_carlo_stress_test(
    multi_ohlc: dict[str, pd.DataFrame],
    config: TradingBotConfig,
    macro_df: pd.DataFrame | None = None,
    funding_df: dict[str, pd.Series] | None = None,
    precomputed_backtest: pd.DataFrame | None = None,
) -> MonteCarloSummary:
    """Run the full-history multi-asset backtest once, then batch-resample
    ``n_runs`` random windows of ``window_days`` from the resulting portfolio
    return series (on the RTX 3070 via PyTorch if available and
    ``config.monte_carlo.use_gpu`` is set, otherwise a fully vectorized NumPy
    fallback -- both O(n_runs) with no Python-level loop over runs)."""
    mc_config = config.monte_carlo
    periods_per_year = _periods_per_year(config.data.resample_to or config.data.base_timeframe)

    backtest = (
        precomputed_backtest
        if precomputed_backtest is not None
        else run_backtest(multi_ohlc, config, macro_df, funding_df)
    )
    returns = backtest["strategy_return"].to_numpy(dtype=np.float64)

    window_size = int(mc_config.window_days * periods_per_year / 365)
    if window_size >= len(returns):
        raise ValueError(
            f"window_days={mc_config.window_days} ({window_size} bars) exceeds the available "
            f"history ({len(returns)} bars); shrink window_days or provide more data."
        )

    rng = np.random.default_rng(mc_config.random_seed)
    starts = rng.integers(0, len(returns) - window_size, size=mc_config.n_runs)

    windowed = _gather_windows_numpy(returns, starts, window_size)
    initial_capital = config.capital.initial_capital_usdt

    use_gpu = mc_config.use_gpu and torch is not None and torch.cuda.is_available()
    metrics = _batched_metrics_gpu(windowed, periods_per_year, initial_capital) if use_gpu else _batched_metrics_numpy(
        windowed, periods_per_year, initial_capital
    )

    runs = pd.DataFrame(metrics)
    runs.insert(0, "start_ts", backtest.index[starts])
    runs.insert(0, "start_idx", starts)

    sharpe = runs["sharpe"].to_numpy()
    mean_sharpe = float(sharpe.mean())
    median_sharpe = float(np.median(sharpe))
    std_sharpe = float(sharpe.std(ddof=1))
    n = len(sharpe)
    t_stat = mean_sharpe / (std_sharpe / math.sqrt(n)) if std_sharpe > 0 else 0.0
    p_value = _z_two_sided_p_value(t_stat)

    return MonteCarloSummary(
        runs=runs,
        mean_sharpe=mean_sharpe,
        median_sharpe=median_sharpe,
        std_sharpe=std_sharpe,
        pct_profitable=float((runs["total_return_pct"] > 0).mean() * 100),
        mean_total_return_pct=float(runs["total_return_pct"].mean()),
        t_stat_sharpe=t_stat,
        p_value_sharpe=p_value,
        initial_capital_usdt=initial_capital,
        mean_final_capital_usdt=float(runs["final_capital_usdt"].mean()),
        median_final_capital_usdt=float(runs["final_capital_usdt"].median()),
        worst_case_max_drawdown_usdt=float(runs["max_drawdown_usdt"].min()),
        mean_max_drawdown_usdt=float(runs["max_drawdown_usdt"].mean()),
    )


def plot_monte_carlo_distribution(summary: MonteCarloSummary, config: TradingBotConfig, filename: str = "monte_carlo.png") -> str:
    """Save histograms of the Sharpe/return/profit-factor distribution to disk
    (headless-VPS friendly: no interactive display, just a saved PNG).

    Writes to a temporary file first and then atomically renames it onto the
    final path. This works around a transient Windows issue where writing
    directly to the target filename can fail with ``OSError: [Errno 22]
    Invalid argument`` if another process (antivirus scan, file indexer, an
    open image viewer) is momentarily holding a handle on a same-named file
    from a previous run.
    """
    import time
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(config.monte_carlo.output_dir, exist_ok=True)
    path = os.path.join(config.monte_carlo.output_dir, filename)
    tmp_path = os.path.join(config.monte_carlo.output_dir, f".{filename}.tmp")
    file_format = os.path.splitext(filename)[1].lstrip(".") or "png"

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].hist(summary.runs["sharpe"], bins=40, color="steelblue")
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_title("Sharpe-Verteilung (OOS-Fenster)")

    axes[1].hist(summary.runs["total_return_pct"], bins=40, color="seagreen")
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Gesamtrendite % pro Fenster")

    finite_pf = summary.runs["profit_factor"].replace([np.inf, -np.inf], np.nan).dropna()
    axes[2].hist(finite_pf, bins=40, color="darkorange")
    axes[2].axvline(1, color="red", linestyle="--")
    axes[2].set_title("Profit-Faktor pro Fenster")

    axes[3].hist(summary.runs["max_drawdown_usdt"], bins=40, color="firebrick")
    axes[3].axvline(0, color="black", linestyle="--")
    axes[3].set_title(f"Max Drawdown in USDT (Startkapital {summary.initial_capital_usdt:,.0f})")

    fig.tight_layout()

    last_error: OSError | None = None
    for attempt in range(5):
        try:
            fig.savefig(tmp_path, dpi=120, format=file_format)
            os.replace(tmp_path, path)
            last_error = None
            break
        except OSError as exc:
            last_error = exc
            time.sleep(0.3 * (attempt + 1))
    plt.close(fig)

    if last_error is not None:
        raise last_error
    return path
