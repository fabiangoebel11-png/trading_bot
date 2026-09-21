"""Event-driven-style vectorized backtester with realistic execution assumptions:

- Latency buffer: signal computed on candle close is only executable on the
  *next* candle (``latency_candles``), simulating worst-case order-to-fill delay.
- Maker-first execution: trades are assumed filled as resting post-only limit
  orders (``maker_fee``) with a configurable fill probability; the unfilled
  fraction is charged the taker fee plus slippage (simulating a fallback
  market order once the price runs away from the resting limit).
- Dynamic ATR-based stop-loss and volatility-targeted leverage from ``core.risk``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import TradingBotConfig
from core.macro import apply_macro_gate
from core.risk import apply_dynamic_stop, dynamic_leverage, dynamic_stop_distance
from core.strategy import generate_signals


def _periods_per_year(resample_to: str) -> int:
    minutes = pd.Timedelta(resample_to).total_seconds() / 60
    return int((365 * 24 * 60) / minutes)


def prepare_signals(df: pd.DataFrame, config: TradingBotConfig, macro_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compute the raw rule-based stat-arb signal (hedge ratio, z-score, position),
    then apply the optional macro risk-off gate. Kept separate from execution/cost
    simulation so ML-gated positions (see ``core.hybrid_strategy``) can be plugged
    in before ``run_backtest_from_signals``."""
    signals = generate_signals(df, config.data.symbol_a, config.data.symbol_b, config.strategy)
    return apply_macro_gate(signals, config, macro_df)


def run_backtest_from_signals(signals: pd.DataFrame, config: TradingBotConfig) -> pd.DataFrame:
    """Apply the risk/execution/cost model to an already-computed ``position`` column.
    ``signals`` must contain: symbol_a, symbol_b, hedge_ratio, spread, position."""
    symbol_a, symbol_b = config.data.symbol_a, config.data.symbol_b
    periods_per_year = _periods_per_year(config.data.resample_to or config.data.base_timeframe)

    price_a = signals[symbol_a]
    price_b = signals[symbol_b]
    hedge_ratio = signals["hedge_ratio"]
    spread = signals["spread"]

    # --- Dynamic risk layer on top of the raw z-score signal ---
    entry_spread = spread.where(signals["position"].diff().fillna(signals["position"]) != 0)
    entry_spread = entry_spread.ffill()
    stop_distance = dynamic_stop_distance(spread, config.risk)
    position = apply_dynamic_stop(signals["position"], spread, entry_spread, stop_distance)

    # --- Worst-case latency: execute N candles after the signal was generated ---
    execution_position = position.shift(config.execution.latency_candles).fillna(0)

    # --- Volatility-targeted dynamic leverage (>= configured minimum) ---
    spread_returns = spread.diff() / price_a.replace(0, np.nan)
    leverage = dynamic_leverage(spread_returns, config.risk, periods_per_year)

    # --- PnL of the market-neutral spread position ---
    d_price_a = price_a.diff()
    d_price_b = price_b.diff()
    capital_base = price_a + hedge_ratio.abs() * price_b

    raw_pnl = execution_position * (d_price_a - hedge_ratio * d_price_b) * leverage
    raw_return = (raw_pnl / capital_base).fillna(0)

    # --- Transaction costs: maker-first fill model with taker fallback ---
    trade_flag = execution_position.diff().abs().fillna(0)
    fill_prob = config.execution.maker_fill_probability
    blended_fee = (
        fill_prob * config.execution.maker_fee
        + (1 - fill_prob)
        * (config.execution.taker_fee + config.execution.slippage_bps_on_taker_fallback / 10_000)
    )
    cost = trade_flag * blended_fee * leverage

    net_return = raw_return - cost

    result = signals.copy()
    result["execution_position"] = execution_position
    result["leverage"] = leverage
    result["trade"] = trade_flag
    result["strategy_return"] = net_return
    result["cumulative_strategy"] = (1 + net_return).cumprod()
    result["cumulative_market"] = (1 + price_a.pct_change().fillna(0)).cumprod()
    return result


def run_backtest(df: pd.DataFrame, config: TradingBotConfig, macro_df: pd.DataFrame | None = None) -> pd.DataFrame:
    signals = prepare_signals(df, config, macro_df)
    return run_backtest_from_signals(signals, config)
