"""Dynamic risk management: ATR-based stop-loss distance and volatility-targeted leverage.

These are independent of the entry/exit signal in ``strategy.py`` and are applied
by the backtester / live trader as an additional safety layer on top of the
z-score stop-loss already encoded in the strategy signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import RiskConfig


def spread_atr(spread: pd.Series, window: int) -> pd.Series:
    """Average True Range analogue for a synthetic spread series (no OHLC available),
    approximated as the rolling mean absolute change."""
    diff = spread.diff().abs()
    return diff.rolling(window=window).mean()


def dynamic_stop_distance(spread: pd.Series, config: RiskConfig) -> pd.Series:
    """Stop-loss distance in spread units, scaled by recent volatility (ATR)."""
    atr = spread_atr(spread, config.atr_window)
    return atr * config.atr_stop_multiplier


def dynamic_leverage(
    spread_returns: pd.Series, config: RiskConfig, periods_per_year: int
) -> pd.Series:
    """Inverse-volatility leverage sizing: scale leverage so the realized annualized
    volatility of the position matches ``vol_target_annualized``, clamped to
    [min_leverage, max_leverage]."""
    realized_vol = spread_returns.rolling(window=config.atr_window).std() * np.sqrt(periods_per_year)
    realized_vol = realized_vol.replace(0, np.nan)

    raw_leverage = config.vol_target_annualized / realized_vol
    leverage = raw_leverage.clip(lower=config.min_leverage, upper=config.max_leverage)
    return leverage.fillna(config.base_leverage)


def apply_dynamic_stop(
    positions: pd.Series, spread: pd.Series, entry_spread: pd.Series, stop_distance: pd.Series
) -> pd.Series:
    """Force-flatten a position once the adverse excursion of the spread exceeds the
    dynamic ATR-based stop distance. ``entry_spread`` is the spread value recorded at
    the bar the current position was opened (forward-filled)."""
    adverse_move = (spread - entry_spread) * np.sign(positions.replace(0, np.nan))
    stopped_out = (adverse_move < -stop_distance) & (positions != 0)
    adjusted = positions.copy()
    adjusted[stopped_out] = 0
    return adjusted
