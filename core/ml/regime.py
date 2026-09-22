"""Causal BTC-price regime classification shared by research and live ML gating."""
from __future__ import annotations

import pandas as pd

from core.config import TrendMLConfig


def classify_regimes(close: pd.Series, config: TrendMLConfig) -> pd.Series:
    """Return ``crash``, ``strong_trend`` or ``sideways`` for every bar.

    Every value uses only the current and prior BTC closes. Crash takes
    precedence over trend because it is the stricter risk regime.
    """
    ema = close.ewm(span=config.regime_ema_span, adjust=False).mean()
    trend_strength = (close / ema - 1.0).abs()
    drawdown = close / close.rolling(config.regime_drawdown_lookback_hours, min_periods=1).max() - 1.0
    regimes = pd.Series("sideways", index=close.index, dtype="object")
    regimes.loc[trend_strength >= config.regime_trend_strength] = "strong_trend"
    regimes.loc[drawdown <= config.regime_crash_drawdown] = "crash"
    return regimes


def ml_regime_active(close: pd.Series, config: TrendMLConfig) -> pd.Series:
    """Whether ML confirmation is allowed to modify the raw rule signal."""
    if not config.regime_gate_enabled:
        return pd.Series(True, index=close.index)
    return classify_regimes(close, config).isin(["crash", "strong_trend"])