"""Dynamic risk management: ATR-based stop-loss distance and volatility-targeted leverage.

Generic over any traded price-like series (originally written for a mean-reversion
spread, now reused directly on each symbol's close price for the trend-following
book). Independent of the entry/exit signal in ``strategy.py``; applied by the
backtester/live trader as an additional risk layer on top of the strategy's own
Donchian-channel exit logic.
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


def apply_dynamic_stop(positions: pd.Series, spread: pd.Series, stop_distance: pd.Series) -> pd.Series:
    """ATR-trailing stop: once a position opens, its stop level ratchets in the
    position's favor every bar (only ever moves up for longs / down for shorts,
    tracking ``price -/+ stop_distance``) and force-flattens the position the bar
    the price crosses it. Unlike a fixed stop anchored to the entry price, this
    locks in open profit as a trend extends instead of only protecting against
    the initial entry level."""
    pos = positions.to_numpy(dtype=float).copy()
    price = spread.to_numpy(dtype=float)
    dist = stop_distance.to_numpy(dtype=float)

    stop_level = np.nan
    prev_pos = 0.0
    for i in range(len(pos)):
        p = pos[i]
        if p == 0.0:
            stop_level = np.nan
        elif p != prev_pos:
            # position just opened (or flipped direction) this bar: seed the stop
            stop_level = price[i] - dist[i] if p > 0 else price[i] + dist[i]
        else:
            candidate = price[i] - dist[i] if p > 0 else price[i] + dist[i]
            stop_level = candidate if np.isnan(stop_level) else (
                max(stop_level, candidate) if p > 0 else min(stop_level, candidate)
            )
            if (p > 0 and price[i] < stop_level) or (p < 0 and price[i] > stop_level):
                pos[i] = 0.0
                stop_level = np.nan
        prev_pos = pos[i]

    return pd.Series(pos, index=positions.index)


def apply_trade_cooldown(position: pd.Series, cooldown_candles: int) -> pd.Series:
    """Force flat for ``cooldown_candles`` bars after any exit (stop, signal-exit
    or trend-flip), before a new entry in the same symbol is allowed. Reduces
    fee/slippage bleed from being repeatedly whipsawed back in during choppy,
    range-bound conditions right after getting stopped out."""
    if cooldown_candles <= 0:
        return position

    pos = position.to_numpy(dtype=float).copy()
    cooldown_left = 0
    prev = 0.0
    for i in range(len(pos)):
        if cooldown_left > 0:
            pos[i] = 0.0
            cooldown_left -= 1
        if prev != 0.0 and pos[i] == 0.0 and cooldown_left == 0:
            cooldown_left = cooldown_candles
        prev = pos[i]
    return pd.Series(pos, index=position.index)


def risk_based_leverage_cap(
    stop_distance: pd.Series, close: pd.Series, max_risk_pct: float, max_leverage: float
) -> pd.Series:
    """Maximum leverage a symbol could run *if it were the only open position in
    the shared wallet*, such that if its ATR-trailing stop is hit the loss stays
    within ``max_risk_pct`` of equity: leverage * stop_distance_pct <=
    max_risk_pct  =>  leverage <= max_risk_pct / stop_distance_pct. Callers pass
    the wallet-wide budget (``CapitalConfig.max_portfolio_risk_pct``) here; when
    multiple positions are open simultaneously, ``shared_wallet_risk_scale``
    below scales every symbol's leverage down further so the *combined*
    stop-out loss (not just each symbol's standalone worst case) stays within
    that same budget.

    ``stop_distance`` is 0/undefined exactly when ATR is 0 or ``close`` is 0 --
    e.g. a stale/frozen order book, a data gap, or a flash-crash tick -- i.e.
    precisely when we have the *least* reliable read on risk. Previously this
    defaulted to ``max_leverage`` (fail-open into the riskiest possible sizing);
    it must instead fail *closed* to 0 here so the caller's
    ``min(vol_leverage, risk_cap).clip(lower=min_leverage)`` falls back to the
    conservative floor leverage instead of silently disabling this risk layer.
    """
    stop_distance_pct = (stop_distance / close).replace([0, np.inf, -np.inf], np.nan)
    cap = max_risk_pct / stop_distance_pct
    return cap.clip(upper=max_leverage).fillna(0.0)


def shared_wallet_risk_scale(
    weights: pd.DataFrame,
    leverage_candidates: pd.DataFrame,
    stop_distance_pct: pd.DataFrame,
    max_portfolio_risk_pct: float,
) -> pd.Series:
    """Per-bar safety valve enforcing a single, shared-wallet stop-out budget
    across every symbol that is simultaneously open (Konzept 2: 1% of a 500 EUR
    wallet = 5 EUR max combined loss if BTC/ETH/SOL all get stopped out on the
    same bar).

    ``weights`` (inverse-vol/leverage weighted, renormalized to sum to 1 across
    only the currently *active* symbols -- see ``core.backtester``) already
    keeps the *expected* combined stop-out loss at ``max_portfolio_risk_pct`` by
    construction, since each symbol's standalone risk cap
    (``risk_based_leverage_cap``) is itself bounded by that same budget and the
    weights sum to <= 1. This function is the defensive second layer: it
    recomputes the actual implied combined loss per bar
    (``sum(weight_i * leverage_i * stop_distance_pct_i)``) and returns a scale
    factor in (0, 1] that shrinks every symbol's leverage further if that
    combined loss would otherwise exceed the budget (covers edge cases such as
    weights not summing to exactly 1, or stale/partial data for one symbol).
    """
    implied_portfolio_risk = (weights * leverage_candidates * stop_distance_pct).sum(axis=1)
    scale = max_portfolio_risk_pct / implied_portfolio_risk.replace(0, np.nan)
    return scale.clip(upper=1.0).fillna(1.0)


def cold_start_leverage_scale(
    capital_usdt: pd.Series,
    initial_capital_usdt: float,
    buffer_pct: float,
    reduced_scale: float,
) -> pd.Series:
    """Backtest (vectorized) version of the cold-start / equity-cushion guard:
    a loss right at the very start of trading a fixed real-money account is
    far more damaging than the same percentage loss after a profit cushion has
    been built (sequence-of-returns risk) -- there is no earlier equity high
    to fall back on. Scales leverage down to ``reduced_scale`` (e.g. 0.5x)
    for every bar until the account has *ever* closed at least ``buffer_pct``
    above ``initial_capital_usdt``.

    Deliberately a one-way ratchet (``cummax``): once that cushion has been
    reached even once, the scale-down is permanently lifted and never
    reapplied, even if capital later falls back toward or below the initial
    amount -- this only protects the initial base, it must not change normal
    drawdown behavior afterwards (that is handled entirely by the existing
    ATR-trailing stop / risk-cap layers).

    Uses the *previous* bar's capital (``shift(1)``) so a bar's leverage
    decision can never see that same bar's own realized outcome.
    """
    threshold = initial_capital_usdt * (1.0 + buffer_pct)
    prior_capital = capital_usdt.shift(1).fillna(initial_capital_usdt)
    buffer_ever_reached = (prior_capital >= threshold).cummax()
    return pd.Series(np.where(buffer_ever_reached, 1.0, reduced_scale), index=capital_usdt.index)


def cold_start_scale_live(
    current_equity: float,
    buffer_already_reached: bool,
    initial_capital_usdt: float,
    buffer_pct: float,
    reduced_scale: float,
) -> tuple[float, bool]:
    """Stateful live-trading counterpart to ``cold_start_leverage_scale``: the
    caller (``LiveTrader``) persists ``buffer_already_reached`` across polls
    and passes it back in on every call. Once ``True`` it must never be reset
    to ``False`` -- the ratchet semantics are identical to the backtest
    version, just expressed as an explicit carried-forward flag instead of a
    vectorized ``cummax`` over a full history."""
    reached = buffer_already_reached or (current_equity >= initial_capital_usdt * (1.0 + buffer_pct))
    scale = 1.0 if reached else reduced_scale
    return scale, reached


def absolute_position_leverage_cap(
    weight: pd.Series,
    capital_usdt_prior: pd.Series,
    max_absolute_position_size_usdt: float | None,
) -> pd.Series:
    """Backtest (vectorized) leverage ceiling enforcing a hard USDT notional
    cap per symbol (Konzept 2): a symbol's implied notional exposure is
    ``weight * capital_usdt * leverage``; solving for the leverage that keeps
    this at or below ``max_absolute_position_size_usdt`` gives
    ``leverage <= cap / (weight * capital_usdt)``. Independent of how large
    the shared wallet grows via profits/deposits -- pct-of-equity sizing alone
    has no such ceiling.

    Uses the *previous* bar's realized capital (this bar's capital depends on
    this bar's leverage, so using it here would be circular); the one-bar lag
    is immaterial since the cap is a slow-moving structural ceiling, not a
    bar-by-bar risk control. ``None`` disables the cap (no additional
    restriction beyond the existing risk-based/vol-target leverage caps).
    """
    if max_absolute_position_size_usdt is None:
        return pd.Series(np.inf, index=weight.index)
    notional_base = (weight * capital_usdt_prior).replace(0, np.nan)
    cap = max_absolute_position_size_usdt / notional_base
    return cap.fillna(np.inf)


def cap_absolute_notional(
    notional_usdt: float, max_absolute_position_size_usdt: float | None
) -> float:
    """Live-trading scalar counterpart: hard-clip a single order's notional
    (USDT) to the same ``max_absolute_position_size_usdt`` ceiling used in the
    backtester, regardless of current account equity. ``None`` disables the
    cap."""
    if max_absolute_position_size_usdt is None:
        return notional_usdt
    return min(notional_usdt, max_absolute_position_size_usdt)
