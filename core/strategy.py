"""Trend-following / breakout strategy for crypto perpetual futures.

Why this replaces the mean-reversion pairs approach: a Monte Carlo random-start
stress test (``research/run_monte_carlo_stress_test.py``, 1000 windows) falsified
the prior 10m BTC/ETH stat-arb pairs strategy -- 0% profitable windows, mean
return ~ -70%, Sharpe consistently between -4 and -10. Crypto majors spend a
large share of time in strong, persistent directional trends; a mean-reversion
book systematically fights those moves (fights the trend, pays fees + funding
while waiting for a reversion that often doesn't arrive before the stop is hit).

This module implements a classic trend-following / breakout system instead:
a Donchian-channel breakout for entries/exits, confirmed by an EMA regime
filter, plus a minimum-volatility gate to avoid overtrading dead/choppy
markets. This is the standard building block of managed-futures/CTA systems,
which have a long, well-documented track record specifically in trending,
fat-tailed markets such as commodities and crypto.

Every signal is strictly causal: all rolling/EMA statistics at bar t use only
data up to and including t, and the Donchian breakout bands are additionally
shifted by one bar so a breakout is measured against the *prior* channel, not
one that already includes the breakout bar itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import TradingBotConfig, TrendConfig


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Average True Range: rolling mean of the true range (max of high-low,
    |high-prev_close|, |low-prev_close|)."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(window=window).mean()


def compute_donchian_channels(
    high: pd.Series, low: pd.Series, entry_window: int, exit_window: int
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Return (upper_entry, lower_entry, upper_exit, lower_exit). All bands use
    ``.shift(1)`` so bar t's bands are computed strictly from bars before t --
    a breakout at t is measured against the channel that existed *before* t."""
    upper_entry = high.rolling(window=entry_window).max().shift(1)
    lower_entry = low.rolling(window=entry_window).min().shift(1)
    upper_exit = high.rolling(window=exit_window).max().shift(1)
    lower_exit = low.rolling(window=exit_window).min().shift(1)
    return upper_entry, lower_entry, upper_exit, lower_exit


def generate_trend_signals(ohlc: pd.DataFrame, config: TrendConfig) -> pd.DataFrame:
    """``ohlc`` must have columns: open, high, low, close (single symbol).

    Entry: breakout above the trailing Donchian high while the EMA regime filter
    confirms an uptrend (long), or breakout below the trailing Donchian low while
    the regime filter confirms a downtrend (short, if ``allow_short``).
    Exit: price closes back through the tighter exit channel, or the EMA regime
    flips against the open position.
    A minimum-ATR% filter blocks new entries during dead/low-volatility chop,
    where breakouts are mostly noise and just generate fee-eating whipsaws.
    """
    close = ohlc["close"]
    high = ohlc["high"]
    low = ohlc["low"]

    out = ohlc.copy()
    out["ema_fast"] = close.ewm(span=config.fast_ma_window, adjust=False).mean()
    out["ema_slow"] = close.ewm(span=config.slow_ma_window, adjust=False).mean()
    out["trend_bias"] = np.sign(out["ema_fast"] - out["ema_slow"])
    out["atr"] = compute_atr(high, low, close, config.atr_window)

    upper_entry, lower_entry, upper_exit, lower_exit = compute_donchian_channels(
        high, low, config.donchian_entry_window, config.donchian_exit_window
    )
    out["donchian_upper_entry"] = upper_entry
    out["donchian_lower_entry"] = lower_entry
    out["donchian_upper_exit"] = upper_exit
    out["donchian_lower_exit"] = lower_exit

    out = out.dropna(subset=["ema_slow", "donchian_upper_entry", "atr"])

    positions = np.zeros(len(out), dtype=float)
    current_pos = 0.0
    closes = out["close"].to_numpy()
    trends = out["trend_bias"].to_numpy()
    atrs = out["atr"].to_numpy()
    up_entry = out["donchian_upper_entry"].to_numpy()
    low_entry = out["donchian_lower_entry"].to_numpy()
    up_exit = out["donchian_upper_exit"].to_numpy()
    low_exit = out["donchian_lower_exit"].to_numpy()

    for i in range(len(out)):
        c = closes[i]
        trend = trends[i]
        vol_ok = (atrs[i] / c) >= config.min_atr_pct if c else False

        if current_pos > 0 and (c < low_exit[i] or trend < 0):
            current_pos = 0.0
        elif current_pos < 0 and (c > up_exit[i] or trend > 0):
            current_pos = 0.0

        if current_pos == 0.0 and vol_ok:
            if c > up_entry[i] and trend > 0:
                current_pos = 1.0
            elif config.allow_short and c < low_entry[i] and trend < 0:
                current_pos = -1.0

        positions[i] = current_pos

    out["position"] = positions
    return out


def generate_portfolio_signals(
    multi_ohlc: dict[str, pd.DataFrame], 
    config: TrendConfig, 
    full_config: TradingBotConfig | None = None
) -> dict[str, pd.DataFrame]:
    """Generiert Trend-Signale und wendet optional ML-Konfidenzskalierung an."""
    signals_dict = {
        symbol: generate_trend_signals(ohlc, config)
        for symbol, ohlc in multi_ohlc.items()
    }

    # ML-Konfidenz-Skalierung, falls aktiviert.
    #
    # Uses ONLY the stitched, purged-walk-forward out-of-sample confidence
    # (``core.ml.inference.load_oos_confidence``) -- never the full-history
    # final model. This function is exclusively the BACKTESTING path
    # (``core/backtester.py`` is its only caller; live trading goes through
    # ``execution/live_trader.py``'s own ``_apply_ml_confirmation``, which
    # correctly uses the final model for genuine prospective inference).
    # Using the final model here would score e.g. a 2020 bar with a model
    # that has already seen 2024-2026 data during training -- an in-sample
    # leak that silently inflates every backtested/Monte-Carlo Sharpe number
    # (found in a 2026-09-22 audit; see ``core/ml/train.py:
    # train_symbol_model`` for where the OOS series is produced/saved).
    if full_config is not None and getattr(full_config, "ml", None) and getattr(full_config.ml, "enabled", False):
        try:
            from core.ml.inference import apply_regime_gated_ml_confirmation, load_oos_confidence

            for symbol, sig_df in signals_dict.items():
                oos_confidence = load_oos_confidence(symbol, full_config.ml)
                if oos_confidence is None:
                    print(
                        f"  [ML] {symbol}: keine Out-of-Sample-Konfidenz gefunden "
                        "(noch nicht/veraltet trainiert) -- nutze rohes Signal."
                    )
                    continue
                conf = oos_confidence.reindex(sig_df.index)
                signals_dict[symbol] = apply_regime_gated_ml_confirmation(sig_df, conf, full_config.ml)
                print(f"  [ML] {symbol}: Out-of-Sample-Konfidenz (purged walk-forward, leak-frei) angewendet.")
        except Exception as e:
            print(f"  [ML-Warnung] Konfidenz-Skalierung übersprungen: {e}")

    return signals_dict