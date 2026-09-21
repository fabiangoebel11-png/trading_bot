"""Multi-asset trend-following backtester, trading a single **shared wallet**
across up to 3 coins (not 3 independent sub-accounts) with realistic execution
assumptions:

- Latency buffer: signal computed on candle close is only executable on the
  *next* candle (``latency_candles``), simulating worst-case order-to-fill delay.
- Maker-first execution: trades are assumed filled as resting post-only limit
  orders (``maker_fee``) with a configurable fill probability; the unfilled
  fraction is charged the taker fee plus slippage (simulating a fallback
  market order once the price runs away from the resting limit).
- ATR-trailing stop-loss with a post-exit cooldown.
- Capital allocation & shared-wallet risk (Konzept 1 & 2): each symbol's
  *candidate* leverage is first computed independently (vol-targeted, capped so
  a *standalone* stop-out couldn't exceed the wallet's whole risk budget), then
  every bar's active (non-flat) symbols are combined via inverse-leverage
  weights renormalized to sum to 1 *across only the currently active symbols*
  -- so 1 active signal deploys the full wallet, 2 or 3 simultaneous signals
  split it proportionally to how much risk-capital each currently needs. A
  second, defensive scaling pass (``core.risk.shared_wallet_risk_scale``)
  guarantees the combined implied loss, if every currently open position's
  stop is hit on the same bar, never exceeds
  ``config.capital.max_portfolio_risk_pct`` of the shared wallet.
- A fixed starting capital (``config.capital.initial_capital_usdt``) is applied
  on top of the pct-return curve purely for reporting/stress-testing purposes
  (equity/drawdown in USDT), so money-management limits stay meaningful even
  though sizing itself is computed in relative (leverage) terms.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import TradingBotConfig
from core.funding import apply_funding_costs
from core.macro import apply_macro_gate
from core.risk import (
    absolute_position_leverage_cap,
    apply_dynamic_stop,
    apply_trade_cooldown,
    cold_start_leverage_scale,
    dynamic_leverage,
    dynamic_stop_distance,
    risk_based_leverage_cap,
    shared_wallet_risk_scale,
)
from core.strategy import generate_portfolio_signals


def _periods_per_year(resample_to: str) -> int:
    minutes = pd.Timedelta(resample_to).total_seconds() / 60
    return int((365 * 24 * 60) / minutes)


def prepare_signals(
    multi_ohlc: dict[str, pd.DataFrame], config: TradingBotConfig, macro_df: pd.DataFrame | None = None
) -> dict[str, pd.DataFrame]:
    """Compute the trend/breakout signal for every symbol, then apply the macro
    regime gate (risk-off flatten + counter-trend scaling) independently to each."""
    
    # HIER ist der entscheidende Parameter 'full_config=config' jetzt endlich drin:
    signals = generate_portfolio_signals(multi_ohlc, config.trend, full_config=config)
    
    return {symbol: apply_macro_gate(sig, config, macro_df) for symbol, sig in signals.items()}


def _symbol_candidate_frame(
    sig: pd.DataFrame, config: TradingBotConfig, periods_per_year: int
) -> pd.DataFrame:
    """Phase 1, per symbol: stop/cooldown-adjusted execution position, plus the
    leverage this symbol *could* run if it were the wallet's only open position
    (vol-targeted, independently capped by the whole wallet's risk budget) --
    not yet scaled for however many other symbols happen to be active on the
    same bar. That shared-wallet interaction is resolved across symbols in
    ``run_backtest_from_signals``."""
    close = sig["close"]
    stop_distance = dynamic_stop_distance(close, config.risk)
    position = apply_dynamic_stop(sig["position"], close, stop_distance)
    position = apply_trade_cooldown(position, config.capital.cooldown_candles)

    # --- Worst-case latency: execute N candles after the signal was generated ---
    execution_position = position.shift(config.execution.latency_candles).fillna(0)

    price_returns = close.pct_change()
    vol_leverage = dynamic_leverage(price_returns, config.risk, periods_per_year)
    risk_cap = risk_based_leverage_cap(
        stop_distance, close, config.capital.max_portfolio_risk_pct, config.risk.max_leverage
    )
    leverage_candidate = pd.concat([vol_leverage, risk_cap], axis=1).min(axis=1).clip(
        lower=config.risk.min_leverage
    )
    stop_distance_pct = (stop_distance / close).replace([0, np.inf, -np.inf], np.nan)

    out = pd.DataFrame(index=sig.index)
    out["close"] = close
    out["price_return"] = price_returns
    out["execution_position"] = execution_position
    out["leverage_candidate"] = leverage_candidate
    out["stop_distance_pct"] = stop_distance_pct.fillna(0.0)
    return out


def _finalize_symbol_backtest(
    sig: pd.DataFrame,
    candidate: pd.DataFrame,
    leverage: pd.Series,
    config: TradingBotConfig,
    funding_rate: pd.Series | None,
) -> pd.DataFrame:
    """Phase 2, per symbol: apply the final (shared-wallet-scaled) leverage to
    compute PnL, costs, funding and equity/drawdown in USDT."""
    execution_position = candidate["execution_position"]
    price_returns = candidate["price_return"]

    raw_return = (execution_position * price_returns * leverage).fillna(0)

    # --- Transaction costs: maker-first fill model with taker fallback ---
    trade_flag = execution_position.diff().abs().fillna(0)
    fill_prob = config.execution.maker_fill_probability
    blended_fee = (
        fill_prob * config.execution.maker_fee
        + (1 - fill_prob)
        * (config.execution.taker_fee + config.execution.slippage_bps_on_taker_fallback / 10_000)
    )
    cost = trade_flag * blended_fee * leverage

    result = sig.copy()
    result["execution_position"] = execution_position
    result["leverage"] = leverage
    result["trade"] = trade_flag
    result["strategy_return"] = raw_return - cost
    result["cumulative_market"] = (1 + candidate["close"].pct_change().fillna(0)).cumprod()
    result = apply_funding_costs(result, funding_rate, config)

    # --- Simulated forced liquidation floor ---
    # A single-bar move bigger than 1/leverage (a flash crash/exchange outage
    # candle, e.g. >25% against a 4x position) would otherwise push
    # strategy_return below -100%, and cumprod() would carry that negative
    # equity forward, letting the curve "recover" from a state real margin
    # trading never allows (the exchange auto-liquidates at ~0 equity, it
    # doesn't let you go negative and mean-revert back). Floor each bar's loss
    # at -99% so a catastrophic bar wipes the position out without producing
    # a sign-flipped, mathematically nonsensical equity curve afterwards.
    result["strategy_return"] = result["strategy_return"].clip(lower=-0.99)
    result["cumulative_strategy"] = (1 + result["strategy_return"]).cumprod()

    capital = config.capital.initial_capital_usdt * result["cumulative_strategy"]
    result["capital_usdt"] = capital
    result["drawdown_usdt"] = capital - capital.cummax()
    return result


def _build_portfolio(
    signals: dict[str, pd.DataFrame],
    candidates: dict[str, pd.DataFrame],
    final_leverage_matrix: pd.DataFrame,
    weights_matrix: pd.DataFrame,
    implied_portfolio_risk: pd.Series,
    config: TradingBotConfig,
    funding_df: dict[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Shared finalization step: apply a given (already fully scaled/capped)
    per-symbol leverage matrix and assemble the portfolio-level frame. Split
    out so the cold-start/absolute-notional-cap second pass below can reuse it
    without duplicating the PnL/weighting logic."""
    per_symbol = {
        symbol: _finalize_symbol_backtest(
            sig, candidates[symbol], final_leverage_matrix[symbol], config, funding_df.get(symbol)
        )
        for symbol, sig in signals.items()
    }

    returns_matrix = pd.concat({symbol: r["strategy_return"] for symbol, r in per_symbol.items()}, axis=1)
    trades_matrix = pd.concat({symbol: r["trade"] for symbol, r in per_symbol.items()}, axis=1)

    portfolio = pd.DataFrame(index=returns_matrix.index)
    portfolio["strategy_return"] = (returns_matrix * weights_matrix).sum(axis=1)
    portfolio["cumulative_strategy"] = (1 + portfolio["strategy_return"]).cumprod()
    portfolio["trade"] = trades_matrix.sum(axis=1)
    portfolio["implied_stop_out_risk_pct"] = implied_portfolio_risk

    capital = config.capital.initial_capital_usdt * portfolio["cumulative_strategy"]
    portfolio["capital_usdt"] = capital
    portfolio["drawdown_usdt"] = capital - capital.cummax()

    for symbol, r in per_symbol.items():
        portfolio[f"{symbol}_position"] = r["execution_position"]
        portfolio[f"{symbol}_return"] = r["strategy_return"]
        portfolio[f"{symbol}_weight"] = weights_matrix[symbol]  # Zum Debuggen der dyn. Gewichtung

    return portfolio, per_symbol


def run_backtest_from_signals(
    signals: dict[str, pd.DataFrame],
    config: TradingBotConfig,
    funding_df: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Backtest every symbol's already-computed ``position`` column, sharing one
    wallet across all of them: capital allocation and the combined stop-out
    risk cap are resolved jointly (Konzept 1 & 2) before per-symbol PnL/costs
    are finalized.

    Two additional safety nets (Session 2026-09-22 pre-live review) are
    layered on top of the already wallet-budgeted leverage, both requiring
    this bar's realized ``capital_usdt`` trajectory as an input and therefore
    applied in a second pass to stay causal (see ``core.risk`` docstrings for
    why using *this* bar's own capital would be circular):
      - Cold-start / equity-cushion protection: half leverage until the
        account has ever closed +10% above the initial capital, then
        permanently normal (ratchet, never re-triggers on a later drawdown).
      - Absolute per-symbol notional ceiling: caps USDT exposure regardless of
        how large the wallet grows.
    """
    periods_per_year = _periods_per_year(config.data.resample_to or config.data.base_timeframe)
    funding_df = funding_df or {}

    candidates = {
        symbol: _symbol_candidate_frame(sig, config, periods_per_year) for symbol, sig in signals.items()
    }

    execution_matrix = pd.concat({s: c["execution_position"] for s, c in candidates.items()}, axis=1)
    leverage_candidate_matrix = pd.concat({s: c["leverage_candidate"] for s, c in candidates.items()}, axis=1)
    stop_pct_matrix = pd.concat({s: c["stop_distance_pct"] for s, c in candidates.items()}, axis=1)

    # --- Konzept 1: capital allocation across simultaneously active coins ---
    # Inverse-leverage weights (calmer/lower-risk symbols get more weight),
    # renormalized to sum to 1 across only the symbols with a currently open
    # (non-flat) position -- so 1 active signal deploys the whole shared
    # wallet, 2-3 simultaneous signals split it proportionally, instead of
    # permanently diluting capital across symbols that aren't even trading.
    active_mask = execution_matrix.ne(0)
    active_leverage = leverage_candidate_matrix.where(active_mask, 0.0)
    weight_sum = active_leverage.sum(axis=1)
    weights_matrix = active_leverage.div(weight_sum, axis=0).fillna(0.0)

    # --- Konzept 2: shared-wallet stop-out risk cap ---
    # Defensive second layer on top of each symbol's already wallet-budgeted
    # standalone risk cap; see core.risk.shared_wallet_risk_scale docstring.
    portfolio_scale = shared_wallet_risk_scale(
        weights_matrix, leverage_candidate_matrix, stop_pct_matrix, config.capital.max_portfolio_risk_pct
    )
    final_leverage_matrix = leverage_candidate_matrix.mul(portfolio_scale, axis=0).clip(
        lower=config.risk.min_leverage
    )
    implied_portfolio_risk = (weights_matrix * leverage_candidate_matrix * stop_pct_matrix).sum(axis=1) * portfolio_scale

    # --- Pass 1: reference capital trajectory (pre cold-start/abs-cap) ---
    reference_portfolio, _ = _build_portfolio(
        signals, candidates, final_leverage_matrix, weights_matrix, implied_portfolio_risk, config, funding_df
    )

    # --- Pass 2: cold-start scale + absolute notional cap, both driven by the
    # previous bar's reference capital (causal, not circular) ---
    cold_start_scale = cold_start_leverage_scale(
        reference_portfolio["capital_usdt"],
        config.capital.initial_capital_usdt,
        config.capital.cold_start_buffer_pct,
        config.capital.cold_start_risk_scale,
    )
    prior_capital = reference_portfolio["capital_usdt"].shift(1).fillna(config.capital.initial_capital_usdt)
    abs_cap_matrix = pd.concat(
        {
            symbol: absolute_position_leverage_cap(
                weights_matrix[symbol], prior_capital, config.capital.max_absolute_position_size_usdt
            )
            for symbol in weights_matrix.columns
        },
        axis=1,
    )
    capped_leverage_matrix = final_leverage_matrix.mul(cold_start_scale, axis=0).clip(upper=abs_cap_matrix).clip(
        lower=config.risk.min_leverage
    )
    implied_portfolio_risk_capped = (
        weights_matrix * capped_leverage_matrix * stop_pct_matrix
    ).sum(axis=1)

    portfolio, per_symbol = _build_portfolio(
        signals, candidates, capped_leverage_matrix, weights_matrix, implied_portfolio_risk_capped, config, funding_df
    )
    # Reporting/monitoring: how much the cold-start guard is currently
    # throttling leverage (1.0 = normal, < 1.0 = still in the cold-start
    # phase) -- lets the live dashboard/backtest report show exactly when the
    # equity cushion was built and the guard permanently lifted.
    portfolio["cold_start_scale"] = cold_start_scale
    portfolio.attrs["per_symbol"] = per_symbol
    return portfolio



def run_backtest(
    multi_ohlc: dict[str, pd.DataFrame],
    config: TradingBotConfig,
    macro_df: pd.DataFrame | None = None,
    funding_df: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    signals = prepare_signals(multi_ohlc, config, macro_df)
    return run_backtest_from_signals(signals, config, funding_df)

