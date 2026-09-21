"""Macro regime filter: liquid, near-24h-tradable index futures (S&P 500 E-mini,
VIX) as a trend/regime confirmation and systemic risk-off trade-blocker for the
crypto trend-following book.

Two independent signals are derived from daily macro closes:
- ``risk_off``: equity drawdown beyond a threshold OR a VIX volatility spike ->
  flattens crypto positions outright (systemic-shock protection).
- ``trend_bias``: EMA-based macro trend direction (+1 up / -1 down) -> crypto
  positions that oppose the prevailing macro trend are *scaled down* (not
  fully blocked, since crypto can and does decouple from equities), while
  positions aligned with the macro trend are left untouched.

Causality: macro data is daily-close based (real feeds report with a delay),
so both signals are shifted forward by ``reporting_lag_days`` before being
forward-filled onto the intraday crypto index -- an intraday bar can therefore
only ever see a macro signal that was already known in the past.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import MacroConfig, TradingBotConfig


def _cache_path(config: MacroConfig) -> Path:
    symbols = list(config.symbols) + ([config.world_symbol] if config.world_symbol else [])
    tag = "_".join(s.replace("^", "").replace("=", "") for s in symbols)
    return Path(config.cache_dir) / f"macro_{tag}_{config.lookback_days}d.csv"


def _download_close(ticker: str, lookback_days: int) -> pd.Series:
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.Timedelta(days=lookback_days)
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)["Close"]
    except Exception as exc:  # noqa: BLE001 - network layer, fall back below
        print(f"  - Fehler beim Laden von {ticker}: {exc}")
        return pd.Series(dtype=float)
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return data.dropna()


def fetch_macro_data(config: MacroConfig, use_cache: bool = True) -> pd.DataFrame:
    """Fetch daily close prices for the configured macro symbols (equity index
    future, VIX, ...) plus the broad world-equity proxy. Falls back to
    ``fallback_symbol`` (e.g. the ^GSPC cash index) if the primary futures
    ticker returns insufficient data."""
    path = _cache_path(config)
    if use_cache and path.exists():
        print(f"⚡ Lade Makro-Cache: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    print(f"🌐 Lade Makro-Daten {config.symbols} ({config.lookback_days} Tage)...")
    equity = _download_close(config.symbols[0], config.lookback_days)
    if len(equity) < config.trend_slow_window:
        print(f"  - Zu wenig Daten für {config.symbols[0]}, Fallback auf {config.fallback_symbol}...")
        equity = _download_close(config.fallback_symbol, config.lookback_days)

    vix_ticker = next((s for s in config.symbols if "VIX" in s.upper()), "^VIX")
    vix = _download_close(vix_ticker, config.lookback_days)

    world = pd.Series(dtype=float)
    if config.world_symbol:
        world = _download_close(config.world_symbol, config.lookback_days)

    data = pd.DataFrame({"equity": equity, "vix": vix, "world": world}).ffill().dropna(how="all")

    if use_cache and not data.empty:
        os.makedirs(config.cache_dir, exist_ok=True)
        data.to_csv(path)
    return data


def compute_risk_off_mask(macro_df: pd.DataFrame, config: MacroConfig) -> pd.Series:
    """Daily risk-off flag: equity drawdown beyond threshold OR VIX z-score spike,
    OR the broad world-equity proxy suffering the same kind of drawdown (a global
    risk-off shock that doesn't happen to show up in the US E-mini alone)."""
    equity = macro_df["equity"]
    rolling_high = equity.rolling(config.drawdown_window, min_periods=1).max()
    drawdown = equity / rolling_high - 1.0
    equity_risk_off = drawdown <= config.drawdown_threshold

    world_risk_off = pd.Series(False, index=macro_df.index)
    if "world" in macro_df.columns and macro_df["world"].notna().any():
        world = macro_df["world"]
        world_high = world.rolling(config.drawdown_window, min_periods=1).max()
        world_drawdown = world / world_high - 1.0
        world_risk_off = world_drawdown <= config.drawdown_threshold

    vix_risk_off = pd.Series(False, index=macro_df.index)
    if "vix" in macro_df.columns and macro_df["vix"].notna().any():
        vix = macro_df["vix"]
        vix_mean = vix.rolling(config.vix_zscore_window).mean()
        vix_std = vix.rolling(config.vix_zscore_window).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        vix_risk_off = vix_z >= config.vix_zscore_threshold

    risk_off = (equity_risk_off | vix_risk_off | world_risk_off).fillna(False)
    # Shift so a given day's flag only becomes visible from the next day onward.
    return risk_off.shift(config.reporting_lag_days).fillna(False)


def compute_trend_bias(macro_df: pd.DataFrame, config: MacroConfig) -> pd.Series:
    """Daily EMA-crossover trend direction, softly confirmed by the broad world-
    equity proxy: +1 uptrend, -1 downtrend, 0 while undefined (warm-up) *or* when
    the US E-mini and the world proxy disagree on direction. Averaging two
    independent indices this way is a deliberately soft filter -- it only takes a
    firm stance when both agree, instead of being sensitive to the noise of a
    single index."""
    equity = macro_df["equity"]
    ema_fast = equity.ewm(span=config.trend_fast_window, adjust=False).mean()
    ema_slow = equity.ewm(span=config.trend_slow_window, adjust=False).mean()
    equity_bias = np.sign(ema_fast - ema_slow).fillna(0.0)

    if "world" in macro_df.columns and macro_df["world"].notna().any():
        world = macro_df["world"]
        world_fast = world.ewm(span=config.trend_fast_window, adjust=False).mean()
        world_slow = world.ewm(span=config.trend_slow_window, adjust=False).mean()
        world_bias = np.sign(world_fast - world_slow).fillna(0.0)
        bias = np.sign(equity_bias + world_bias)
    else:
        bias = equity_bias

    return bias.shift(config.reporting_lag_days).fillna(0.0)


def _align_daily_to_index(daily: pd.Series, target_index: pd.DatetimeIndex, fill_value) -> pd.Series:
    """Forward-fill a daily macro signal onto the intraday crypto index. Uses only
    past daily values for every intraday timestamp (``ffill`` never looks forward)."""
    daily = daily.sort_index()
    union_index = daily.index.union(target_index)
    aligned = daily.reindex(union_index).ffill().reindex(target_index)
    return aligned.fillna(fill_value)


def apply_macro_gate(
    signals: pd.DataFrame, config: TradingBotConfig, macro_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Flatten positions during systemic risk-off regimes, and scale down (not
    fully block) positions that oppose the macro trend. No-op if
    ``config.macro.enabled`` is False, so the crypto-only pipeline is unaffected
    unless the user explicitly opts in."""
    if not config.macro.enabled:
        return signals

    if macro_df is None:
        macro_df = fetch_macro_data(config.macro)
    if macro_df.empty:
        return signals

    risk_off_daily = compute_risk_off_mask(macro_df, config.macro)
    trend_bias_daily = compute_trend_bias(macro_df, config.macro)

    risk_off = _align_daily_to_index(risk_off_daily, signals.index, False).astype(bool)
    trend_bias = _align_daily_to_index(trend_bias_daily, signals.index, 0.0)

    out = signals.copy()
    out["macro_risk_off"] = risk_off
    out["macro_trend_bias"] = trend_bias

    position = out["position"].astype(float)
    opposes_macro = (np.sign(position) != 0) & (np.sign(position) != np.sign(trend_bias)) & (trend_bias != 0)
    position = np.where(opposes_macro, position * config.macro.counter_trend_scale, position)
    position = np.where(risk_off, 0.0, position)

    out["position"] = position
    return out

