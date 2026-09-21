"""Statistical-arbitrage pairs strategy: rolling OLS hedge ratio + z-score signal.

The strategy is purely rule-based (no look-ahead): every value at index ``i``
only uses information available up to and including candle ``i``. The
backtester is responsible for applying the additional latency shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.config import StrategyConfig


def rolling_hedge_ratio(price_a: pd.Series, price_b: pd.Series, window: int) -> pd.Series:
    """Rolling OLS hedge ratio beta_t = argmin || price_a - beta * price_b ||, fit on the
    trailing ``window`` candles (no intercept, matching the classic spread definition).

    Vectorized via rolling sums (beta = sum(x*y) / sum(x*x)) instead of a per-bar Python
    loop: turns an O(n * window) computation into O(n), which matters once walk-forward
    validation re-fits this on every fold across a multi-year, sub-hourly history."""
    # .shift(1): beta at bar t only uses the trailing window strictly *before* t
    # (rows t-window .. t-1), matching the original per-bar loop and avoiding any
    # same-bar look-ahead where the hedge ratio would "see" the price it is priced against.
    xy_sum = (price_a * price_b).rolling(window=window).sum().shift(1)
    xx_sum = (price_b * price_b).rolling(window=window).sum().shift(1)
    hedge_ratio = xy_sum / xx_sum.replace(0, np.nan)
    return hedge_ratio


def rolling_hedge_ratio_sm(price_a: pd.Series, price_b: pd.Series, window: int) -> pd.Series:
    """Reference implementation using statsmodels OLS (slower, used for validation/tests)."""
    hedge_ratios = pd.Series(index=price_a.index, dtype=float)
    for i in range(window, len(price_a)):
        y = price_a.iloc[i - window : i]
        x = price_b.iloc[i - window : i]
        model = sm.OLS(y, x).fit()
        hedge_ratios.iloc[i] = model.params.iloc[0]
    return hedge_ratios


def compute_spread_and_zscore(
    price_a: pd.Series, price_b: pd.Series, hedge_ratio: pd.Series, z_window: int
) -> pd.DataFrame:
    spread = price_a - hedge_ratio * price_b
    rolling_mean = spread.rolling(window=z_window).mean()
    rolling_std = spread.rolling(window=z_window).std()
    zscore = (spread - rolling_mean) / rolling_std
    return pd.DataFrame({"spread": spread, "zscore": zscore})


def compute_trend_filter(price_a: pd.Series, window: int) -> pd.Series:
    """Relative distance of price_a from its own SMA; large values indicate a strong
    directional trend during which mean-reversion trades are disabled."""
    sma = price_a.rolling(window=window).mean()
    return (price_a - sma).abs() / sma


def generate_signals(
    df: pd.DataFrame,
    symbol_a: str,
    symbol_b: str,
    config: StrategyConfig,
) -> pd.DataFrame:
    """Compute hedge ratio, z-score, trend filter and the resulting position series.

    Returns a copy of ``df`` enriched with: hedge_ratio, spread, zscore,
    market_trend, position (in {-1, 0, 1}, no look-ahead).
    """
    out = df.copy()
    price_a = out[symbol_a]
    price_b = out[symbol_b]

    out["hedge_ratio"] = rolling_hedge_ratio(price_a, price_b, config.hedge_ratio_window)
    spread_z = compute_spread_and_zscore(price_a, price_b, out["hedge_ratio"], config.zscore_window)
    out["spread"] = spread_z["spread"]
    out["zscore"] = spread_z["zscore"]
    out["market_trend"] = compute_trend_filter(price_a, config.trend_filter_window)

    out = out.dropna()

    positions = []
    current_pos = 0
    for z, trending in zip(out["zscore"], out["market_trend"] > config.trend_threshold):
        if abs(z) > config.stop_loss_z:
            current_pos = 0
        elif current_pos != 0:
            if (current_pos == 1 and z >= -config.exit_z) or (
                current_pos == -1 and z <= config.exit_z
            ):
                current_pos = 0
        elif not trending:
            if z < -config.entry_z:
                current_pos = 1
            elif z > config.entry_z:
                current_pos = -1
        positions.append(current_pos)

    out["position"] = positions
    return out
