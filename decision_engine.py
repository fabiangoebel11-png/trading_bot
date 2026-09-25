"""Deterministic, order-free trading decision support.

This module consumes stored market/signal snapshots and manual position input.
It never fetches data, trains models, writes to a broker, or places orders.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from typing import Any, Iterable


class StrategyFamily(str, Enum):
    TREND_BREAKOUT = "trend_breakout"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    REGIME_ADAPTIVE = "regime_adaptive"


@dataclass(frozen=True)
class StrategyCard:
    strategy_id: str
    version: str
    family: StrategyFamily
    asset_class: str
    timeframes: tuple[str, ...]
    entry: str
    stop: str
    trailing: str
    take_profit: str
    exit: str
    typical_hold: str
    risk_range: str
    model_role: str
    avoid_when: str


STRATEGY_CARDS: tuple[StrategyCard, ...] = (
    StrategyCard(
        "trend_breakout", "1.0.0", StrategyFamily.TREND_BREAKOUT, "crypto/equity", ("1h", "4h"),
        "Price breaks the prior range while trend direction agrees.",
        "ATR floor or measured adverse excursion, whichever is wider.",
        "Ratchet by the initial stop distance after entry; never loosen.",
        "Model MFE or a minimum 0.5R objective.",
        "Stop, target, opposite regime, or breakout failure.", "hours to several days", "0.25%-2.0%", "Scores an existing rule setup; never creates an order.",
        "Low score, stale data, insufficient warmup, or range-bound regime.",
    ),
    StrategyCard(
        "momentum", "1.0.0", StrategyFamily.MOMENTUM, "crypto/equity", ("15m", "1h"),
        "Directional momentum agrees with the active regime.",
        "ATR floor or measured adverse excursion, whichever is wider.",
        "Ratchet after a favorable move; no stop widening.",
        "Model MFE or a minimum 0.5R objective.",
        "Momentum reversal, stop, target, or stale data.", "minutes to hours", "0.25%-1.0%", "Optional stored model score only.",
        "Opposing regime, weak momentum, or excessive volatility.",
    ),
    StrategyCard(
        "mean_reversion", "1.0.0", StrategyFamily.MEAN_REVERSION, "crypto/equity", ("1h", "4h"),
        "Price is extended from a reference range and regime is non-trending.",
        "Outside the invalidation band; never inferred from a tight stop alone.",
        "Only after reversion confirms; otherwise no trailing assumption.",
        "Reference mean or conservative measured target.",
        "Regime becomes directional, stop, or reference mean is reached.", "hours to several days", "0.25%-1.0%", "Not promoted by Phase 2; informational profile only.",
        "Strong trend, breakout, stale data, or missing reference band.",
    ),
    StrategyCard(
        "regime_adaptive", "1.0.0", StrategyFamily.REGIME_ADAPTIVE, "crypto/equity", ("1h", "4h", "1d"),
        "Selects a family only after the current regime is classified.",
        "Inherited from the selected family and risk budget.",
        "Inherited from the selected family; no hidden override.",
        "Inherited from the selected family.",
        "Regime transition, inherited stop/target, or stale data.", "hours to two weeks", "0.25%-1.0%", "Model output is context, not an execution command.",
        "Unknown regime or insufficient data.",
    ),
)


@dataclass(frozen=True)
class MarketSnapshot:
    asset: str
    asset_class: str
    price: float
    atr: float = 0.0
    atr_pct: float = 0.0
    trend: str = "UNKNOWN"
    volatility: str = "UNKNOWN"
    data_status: str = "UNKNOWN"
    timeframe: str = "1h"
    timestamp: str = ""


@dataclass(frozen=True)
class SignalSnapshot:
    asset: str
    direction: str
    score: float
    expected_return: float = 0.0
    expected_mfe: float = 0.0
    expected_mae: float = 0.0
    expected_duration_bars: float = 0.0
    model_type: str = "unknown"
    timestamp: str = ""


@dataclass(frozen=True)
class LeverageAdvice:
    recommended: float
    safe_min: float
    safe_max: float
    aggressive: float
    max_allowed: float
    not_recommended_above: float
    reason: str


@dataclass(frozen=True)
class PositionMath:
    notional: float
    margin: float
    quantity: float
    risk_amount: float
    risk_pct: float
    profit_at_tp: float
    loss_at_stop: float
    liquidation_estimate: float | None
    stop_to_liquidation_pct: float | None
    current_to_liquidation_pct: float | None
    risk_reward: float
    fee_estimate: float
    funding_estimate: float


@dataclass(frozen=True)
class TradePlan:
    asset: str
    direction: str
    strategy_id: str
    strategy_version: str
    timeframe: str
    entry_low: float
    entry_high: float
    entry_reference: float
    initial_stop: float
    tp1: float
    tp2: float
    trailing_distance: float
    expected_hold: str
    signal_quality: float
    risk_amount: float
    notional: float
    margin: float
    leverage: LeverageAdvice
    invalidation: str
    management: tuple[str, ...]
    rationale: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class PositionAnalysis:
    position_id: str
    asset: str
    direction: str
    state: str
    recommendation: str
    reason: str
    current_pnl: float
    current_pnl_pct: float
    risk_to_stop: float
    stop_distance_pct: float
    take_profit_distance_pct: float
    risk_reward: float
    signal_quality: float
    next_actions: tuple[str, ...]


@dataclass(frozen=True)
class ScenarioResult:
    price: float
    change_pct: float
    pnl: float
    pnl_pct: float
    pnl_pct_on_margin: float
    distance_to_stop_pct: float
    state: str
    stop: float
    trailing_stop: float
    recommendation: str
    rationale: str


def _finite(*values: float) -> bool:
    return all(isfinite(float(value)) for value in values)


def _direction(direction: str) -> int:
    if direction == "LONG":
        return 1
    if direction == "SHORT":
        return -1
    raise ValueError("direction must be LONG or SHORT")


def choose_strategy(market: MarketSnapshot, signal: SignalSnapshot) -> StrategyCard | None:
    """Choose a family descriptively; this is not a model or order gate."""
    if market.data_status not in {"LIVE", "FRESH", "AVAILABLE"} or signal.direction not in {"LONG", "SHORT"}:
        return None
    if signal.score < 50.0:
        return None
    if market.trend in {"UP", "DOWN", "BULL", "BEAR"}:
        return next(card for card in STRATEGY_CARDS if card.family is StrategyFamily.TREND_BREAKOUT)
    if market.volatility in {"HIGH", "ELEVATED"}:
        return next(card for card in STRATEGY_CARDS if card.family is StrategyFamily.MOMENTUM)
    return next(card for card in STRATEGY_CARDS if card.family is StrategyFamily.REGIME_ADAPTIVE)


def signal_quality(market: MarketSnapshot, signal: SignalSnapshot, risk_reward: float) -> dict[str, float]:
    """Transparent 0-100 score from stored signal and market facts."""
    trend = 18.0 if signal.direction in {"LONG", "SHORT"} and market.trend in {"UP", "DOWN", "BULL", "BEAR"} else 0.0
    momentum = min(max(signal.score, 0.0), 100.0) * 0.20
    breakout = 16.0 if market.trend in {"UP", "DOWN", "BULL", "BEAR"} else 0.0
    volatility = 8.0 if market.volatility not in {"UNKNOWN", "EXTREME"} else 0.0
    model = min(max(signal.score, 0.0), 100.0) * 0.25
    rr = min(max(risk_reward / 2.0, 0.0), 1.0) * 20.0
    penalty = 8.0 if market.data_status not in {"LIVE", "FRESH", "AVAILABLE"} else 0.0
    components = {"trend": trend, "momentum": momentum, "breakout": breakout, "volatility": volatility, "ml": model, "risk_reward": rr, "penalty": -penalty}
    components["total"] = float(max(0.0, min(100.0, sum(components.values()))))
    return components


def leverage_advice(*, account: float, risk_amount: float, stop_distance_pct: float, requested_notional: float, max_allowed: float = 10.0, margin_cap_pct: float = 0.35) -> LeverageAdvice:
    if not _finite(account, risk_amount, stop_distance_pct, requested_notional, max_allowed) or account <= 0 or stop_distance_pct <= 0:
        raise ValueError("account, risk, stop distance, and notional must be finite and positive")
    risk_notional = risk_amount / stop_distance_pct
    max_margin = account * margin_cap_pct
    required = requested_notional / max_margin if max_margin else max_allowed
    recommended = min(max(1.0, required), max_allowed)
    safe_max = min(max_allowed, max(1.0, risk_notional / max_margin))
    safe_min = 1.0 if requested_notional <= max_margin else min(recommended, max_allowed)
    aggressive = min(max_allowed, max(recommended, safe_max) * 1.25)
    return LeverageAdvice(
        recommended=round(recommended, 4), safe_min=round(safe_min, 4), safe_max=round(safe_max, 4),
        aggressive=round(aggressive, 4), max_allowed=round(max_allowed, 4),
        not_recommended_above=round(safe_max, 4),
        reason="Leverage is derived from requested notional and margin cap; stop risk constrains exposure, not the signal.",
    )


def calculate_position(*, account: float, entry: float, current: float, stop: float, take_profit: float, leverage: float, direction: str, fee_rate: float = 0.0005, funding_rate: float = 0.0) -> PositionMath:
    sign = _direction(direction)
    if not _finite(account, entry, current, stop, take_profit, leverage, fee_rate, funding_rate) or min(account, entry, leverage) <= 0:
        raise ValueError("position inputs must be finite and positive")
    stop_distance = abs(entry - stop) / entry
    notional = account * 0.35 * leverage
    quantity = notional / entry
    risk = notional * stop_distance
    profit = sign * (take_profit - entry) * quantity
    loss = sign * (stop - entry) * quantity
    liquidation = entry * (1.0 - sign / leverage)
    stop_to_liq = abs(stop - liquidation) / entry
    current_to_liq = abs(current - liquidation) / current if current else None
    return PositionMath(
        notional=notional, margin=notional / leverage, quantity=quantity, risk_amount=risk,
        risk_pct=risk / account, profit_at_tp=profit, loss_at_stop=loss,
        liquidation_estimate=liquidation, stop_to_liquidation_pct=stop_to_liq,
        current_to_liquidation_pct=current_to_liq, risk_reward=abs(profit / loss) if loss else 0.0,
        fee_estimate=notional * fee_rate * 2.0, funding_estimate=notional * funding_rate,
    )


def build_trade_plan(market: MarketSnapshot, signal: SignalSnapshot, *, account: float, risk_pct: float = 0.005, max_leverage: float = 10.0) -> TradePlan | None:
    card = choose_strategy(market, signal)
    if card is None or market.price <= 0 or market.atr <= 0:
        return None
    direction = _direction(signal.direction)
    stop_distance = max(market.atr * 1.5, abs(signal.expected_mae) * market.price, market.price * 0.001)
    entry = market.price
    stop = entry - direction * stop_distance
    tp1 = entry + direction * max(abs(signal.expected_mfe) * market.price * 0.5, stop_distance)
    tp2 = entry + direction * max(abs(signal.expected_mfe) * market.price, stop_distance * 2.0)
    rr = abs((tp2 - entry) / (entry - stop)) if entry != stop else 0.0
    quality = signal_quality(market, signal, rr)
    risk_amount = account * risk_pct
    notional = risk_amount / max(stop_distance / entry, 1e-9)
    advice = leverage_advice(account=account, risk_amount=risk_amount, stop_distance_pct=stop_distance / entry, requested_notional=notional, max_allowed=max_leverage)
    return TradePlan(
        asset=market.asset, direction=signal.direction, strategy_id=card.strategy_id, strategy_version=card.version,
        timeframe=market.timeframe, entry_low=min(entry, entry + direction * market.atr * 0.25),
        entry_high=max(entry, entry + direction * market.atr * 0.25), entry_reference=entry,
        initial_stop=stop, tp1=tp1, tp2=tp2, trailing_distance=stop_distance,
        expected_hold=f"{max(signal.expected_duration_bars, 0.0):.1f} bars", signal_quality=quality["total"],
        risk_amount=risk_amount, notional=notional, margin=notional / max(advice.recommended, 1.0), leverage=advice,
        invalidation=f"Close beyond {stop:.6f} or data becomes stale.",
        management=("+1 ATR: protect the position; +1.5 ATR: ratchet trailing stop; +2 ATR: evaluate TP1/TP2.", "Adverse move to the initial stop: exit according to the strategy; never widen the stop."),
        rationale=tuple(key for key, value in (("Trend alignment", quality["trend"]), ("Momentum", quality["momentum"]), ("Breakout", quality["breakout"]), ("Volatility", quality["volatility"])) if value > 0),
        status="ACTIONABLE_SETUP" if quality["total"] >= 60 else "WEAK_SETUP",
    )


def _state(price: float, entry: float, stop: float, tp1: float, tp2: float, direction: int) -> tuple[str, str]:
    if direction == 1:
        if price <= stop:
            return "INVALIDATED", "EXIT: initial stop/invalidation reached"
        if price >= tp2:
            return "TAKE PROFIT", "TAKE PROFIT / trail remainder"
        if price >= tp1:
            return "HOLD + TRAILING", "MOVE STOP / trail the position"
        if price > entry:
            return "HOLD", "HOLD while setup remains valid"
        return "ADVERSE_MOVE", "HOLD only while setup remains valid"
    if price >= stop:
        return "INVALIDATED", "EXIT: initial stop/invalidation reached"
    if price <= tp2:
            return "TAKE PROFIT", "TAKE PROFIT / trail remainder"
    if price <= tp1:
            return "HOLD + TRAILING", "MOVE STOP / trail the position"
    if price < entry:
            return "HOLD", "HOLD while setup remains valid"
    return "ADVERSE_MOVE", "HOLD only while setup remains valid"


def analyze_position(*, position_id: str, asset: str, direction: str, entry: float, current: float, quantity: float, margin: float, stop: float, take_profit: float, signal_quality_value: float = 0.0) -> PositionAnalysis:
    sign = _direction(direction)
    if not _finite(entry, current, quantity, margin, stop, take_profit) or min(entry, quantity, margin) <= 0:
        raise ValueError("position values must be finite and positive")
    pnl = sign * (current - entry) * quantity
    stop_risk = abs(entry - stop) * quantity
    tp_distance = abs(take_profit - current) / current
    stop_distance = abs(stop - current) / current
    rr = abs((take_profit - entry) / (entry - stop)) if entry != stop else 0.0
    state, recommendation = _state(current, entry, stop, take_profit, take_profit, sign)
    reason = "Trend/position state is derived from entry, stop, target, and current price; no order is sent."
    return PositionAnalysis(position_id, asset, direction, state, recommendation, reason, pnl, pnl / margin, stop_risk, stop_distance, tp_distance, rr, signal_quality_value, (f"Below {stop:.6f}: setup invalidated.", f"At {take_profit:.6f}: target zone."))


def scenario_analysis(*, entry: float, current: float, quantity: float, margin: float, stop: float, take_profit: float, direction: str, changes: Iterable[float] = (-0.03, -0.02, -0.01, -0.005, 0.005, 0.01, 0.02, 0.03, 0.05)) -> list[ScenarioResult]:
    sign = _direction(direction)
    if not _finite(entry, current, quantity, margin, stop, take_profit) or min(entry, current, quantity, margin) <= 0:
        raise ValueError("scenario inputs must be finite and positive")
    stop_distance = abs(entry - stop)
    results: list[ScenarioResult] = []
    for change in changes:
        price = current * (1.0 + float(change))
        trailing = price - sign * stop_distance
        state, recommendation = _state(price, entry, stop, take_profit, take_profit, sign)
        pnl = sign * (price - entry) * quantity
        results.append(ScenarioResult(price, float(change), pnl, pnl / entry * 100.0, pnl / margin, abs(price - stop) / price * 100.0, state, stop, trailing, recommendation, f"Deterministic {state} transition at scenario price {price:.6f}."))
    return results


def as_dict(value: Any) -> dict[str, Any]:
    return asdict(value)