"""Multi-asset macro regime filter (optional "risk-off" trade-blocking layer).

Uses external markets (equity indices, VIX) that are independent of the crypto
pair's own price action as a systemic-risk gauge: when equities are in a sharp
drawdown or volatility spikes, new stat-arb entries are blocked and open
positions are flattened, regardless of what the pair's own z-score says.

Causality: macro data is daily-close based (real feeds report with a delay),
so the risk-off flag is (a) shifted forward by ``reporting_lag_days`` before
(b) being forward-filled onto the intraday crypto index -- an intraday bar can
therefore only ever see a macro flag that was already known in the past.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from core.config import MacroConfig, TradingBotConfig


def _cache_path(config: MacroConfig) -> Path:
    tag = "_".join(s.replace("^", "") for s in config.symbols)
    return Path(config.cache_dir) / f"macro_{tag}_{config.lookback_days}d.csv"


def fetch_macro_data(config: MacroConfig, use_cache: bool = True) -> pd.DataFrame:
    """Fetch daily close prices for the configured macro symbols (equity index, VIX, ...)."""
    path = _cache_path(config)
    if use_cache and path.exists():
        print(f"⚡ Lade Makro-Cache: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.Timedelta(days=config.lookback_days)
    print(f"🌐 Lade Makro-Daten {config.symbols} ({config.lookback_days} Tage)...")
    raw = yf.download(config.symbols, start=start, end=end, progress=False, auto_adjust=True)["Close"]

    if isinstance(raw, pd.Series):
        raw = raw.to_frame(config.symbols[0])
    data = raw.ffill().dropna(how="all")

    if use_cache:
        os.makedirs(config.cache_dir, exist_ok=True)
        data.to_csv(path)
    return data


def compute_risk_off_mask(macro_df: pd.DataFrame, config: MacroConfig) -> pd.Series:
    """Daily risk-off flag: equity drawdown beyond threshold OR VIX z-score spike."""
    equity_col = config.symbols[0]
    equity = macro_df[equity_col]
    rolling_high = equity.rolling(config.drawdown_window, min_periods=1).max()
    drawdown = equity / rolling_high - 1.0
    equity_risk_off = drawdown <= config.drawdown_threshold

    vix_risk_off = pd.Series(False, index=macro_df.index)
    vix_col = next((s for s in config.symbols if "VIX" in s.upper()), None)
    if vix_col and vix_col in macro_df.columns:
        vix = macro_df[vix_col]
        vix_mean = vix.rolling(config.vix_zscore_window).mean()
        vix_std = vix.rolling(config.vix_zscore_window).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        vix_risk_off = vix_z >= config.vix_zscore_threshold

    risk_off = (equity_risk_off | vix_risk_off).fillna(False)
    # Shift so a given day's flag only becomes visible from the next day onward.
    return risk_off.shift(config.reporting_lag_days).fillna(False)


def align_risk_off_to_index(risk_off_daily: pd.Series, target_index: pd.DatetimeIndex) -> pd.Series:
    """Forward-fill the daily risk-off flag onto the intraday crypto index. Uses only
    past daily flags for every intraday timestamp (``ffill`` never looks forward)."""
    daily = risk_off_daily.sort_index()
    union_index = daily.index.union(target_index)
    aligned = daily.reindex(union_index).ffill().reindex(target_index)
    return aligned.fillna(False).astype(bool)


def apply_macro_gate(
    signals: pd.DataFrame, config: TradingBotConfig, macro_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Zero out the position wherever the macro risk-off flag is active. No-op if
    ``config.macro.enabled`` is False (default), so the crypto-only pipeline is
    unaffected unless the user explicitly opts in."""
    if not config.macro.enabled:
        return signals

    if macro_df is None:
        macro_df = fetch_macro_data(config.macro)

    risk_off_daily = compute_risk_off_mask(macro_df, config.macro)
    risk_off_aligned = align_risk_off_to_index(risk_off_daily, signals.index)

    out = signals.copy()
    out["risk_off"] = risk_off_aligned
    out["position"] = np.where(out["risk_off"], 0, out["position"])
    return out
