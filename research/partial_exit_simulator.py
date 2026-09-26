"""Causal, stateful partial-exit simulation for research only.

The simulator deliberately has no database or exchange dependencies. It uses
the same next-open entry convention as phase 2 and exposes a policy interface
so exit tactics can be compared without changing the production broker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


TP_METHODS = {
    "fixed_pct",
    "atr_multiple",
    "atr_dynamic",
    "mfe_quantile",
    "vol_normalized",
}
STOP_MODES = {
    "unchanged",
    "break_even",
    "break_even_cost_buffer",
    "break_even_then_atr_trailing",
    "atr_trailing",
    "chandelier",
    "donchian",
    "ema",
    "trailing_after_tp2",
    "regime_dynamic",
}


@dataclass(frozen=True)
class TakeProfitSpec:
    method: str = "atr_multiple"
    values: tuple[float, ...] = (1.0, 2.0)
    fee_slippage_buffer: float = 0.0

    def __post_init__(self) -> None:
        if self.method not in TP_METHODS:
            raise ValueError(f"Unsupported TP method: {self.method}")
        if any(value <= 0 or not np.isfinite(value) for value in self.values):
            raise ValueError("TP values must be positive and finite")


@dataclass(frozen=True)
class PartialExitPolicy:
    policy_id: str
    allocations: tuple[float, ...]
    take_profit: TakeProfitSpec | None = None
    stop_mode: str = "unchanged"
    final_exit: str = "signal"
    chandelier_multiple: float = 3.0
    trailing_multiple: float = 2.0
    donchian_window: int = 20
    ema_window: int = 20
    activate_trailing_after_stage: int = 0

    def __post_init__(self) -> None:
        if not self.allocations or any(value <= 0 for value in self.allocations):
            raise ValueError("allocations must contain positive fractions")
        if not np.isclose(sum(self.allocations), 1.0):
            raise ValueError("allocations must sum to 1.0")
        if self.stop_mode not in STOP_MODES:
            raise ValueError(f"Unsupported stop mode: {self.stop_mode}")
        if self.final_exit not in {"signal", "atr_trailing", "chandelier", "donchian", "ema"}:
            raise ValueError(f"Unsupported final exit: {self.final_exit}")
        if len(self.allocations) > 1 and self.take_profit is None:
            raise ValueError("multi-stage policies require a take-profit specification")


@dataclass
class TradeManagementState:
    side: int
    entry_index: int
    entry_price: float
    entry_atr: float
    original_fraction: float = 1.0
    remaining_fraction: float = 1.0
    next_stage: int = 0
    stop_price: float = np.nan
    highest_price: float = -np.inf
    lowest_price: float = np.inf
    realized_gross: float = 0.0
    realized_cost: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    mae: float = 0.0
    mfe: float = 0.0


def _side_price(side: int, price: float, entry_price: float) -> float:
    return side * (price / entry_price - 1.0)


def _execution_price(price: float, side: int, cost_rate: float) -> float:
    return price * (1.0 - side * cost_rate)


def _target_prices(
    state: TradeManagementState,
    spec: TakeProfitSpec,
    features: pd.DataFrame,
    index: int,
    historical_mfe: float | None,
) -> list[float]:
    atr = float(features["atr"].iloc[index])
    if not np.isfinite(atr) or atr <= 0:
        return []
    if spec.method == "fixed_pct":
        distances = list(spec.values)
    elif spec.method == "atr_multiple":
        distances = [value * atr / state.entry_price for value in spec.values]
    elif spec.method == "atr_dynamic":
        vol = float(features["realized_vol"].iloc[index]) if "realized_vol" in features else 0.0
        scale = max(vol, 1e-6)
        distances = [value * atr / state.entry_price * scale / 0.35 for value in spec.values]
    elif spec.method == "mfe_quantile":
        if historical_mfe is None or not np.isfinite(historical_mfe) or historical_mfe <= 0:
            return []
        distances = [historical_mfe * value for value in spec.values]
    else:
        vol = float(features["realized_vol"].iloc[index]) if "realized_vol" in features else 0.0
        distances = [value * max(vol, 1e-6) for value in spec.values]
    return [state.entry_price * (1.0 + state.side * distance) for distance in distances]


def _stop_for_mode(
    state: TradeManagementState,
    policy: PartialExitPolicy,
    features: pd.DataFrame,
    index: int,
    cost_rate: float,
) -> float:
    if policy.stop_mode == "unchanged":
        return state.stop_price
    if policy.stop_mode in {"break_even", "break_even_cost_buffer"}:
        buffer = cost_rate if policy.stop_mode == "break_even_cost_buffer" else 0.0
        return state.entry_price * (1.0 + state.side * buffer)
    atr = float(features["atr"].iloc[index])
    close = float(features["close"].iloc[index])
    if not np.isfinite(atr) or not np.isfinite(close):
        return state.stop_price
    if policy.stop_mode == "break_even_then_atr_trailing" and state.next_stage < policy.activate_trailing_after_stage:
        return state.entry_price * (1.0 + state.side * cost_rate)
    if policy.stop_mode in {"atr_trailing", "trailing_after_tp2", "regime_dynamic", "break_even_then_atr_trailing"}:
        multiple = policy.trailing_multiple
        if policy.stop_mode == "regime_dynamic" and "high_volatility" in features:
            multiple *= 1.25 if bool(features["high_volatility"].iloc[index]) else 0.85
        candidate = close - state.side * multiple * atr
    elif policy.stop_mode == "chandelier":
        candidate = (state.highest_price if state.side > 0 else state.lowest_price) - state.side * policy.chandelier_multiple * atr
    elif policy.stop_mode == "donchian":
        window = features.iloc[max(0, index - policy.donchian_window):index]
        if window.empty:
            return state.stop_price
        candidate = float(window["low"].min() if state.side > 0 else window["high"].max())
    else:
        ema = float(features["ema_fast"].iloc[index]) if "ema_fast" in features else close
        candidate = ema
    if not np.isfinite(candidate):
        return state.stop_price
    if policy.stop_mode == "break_even_then_atr_trailing":
        break_even = state.entry_price * (1.0 + state.side * cost_rate)
        candidate = max(candidate, break_even) if state.side > 0 else min(candidate, break_even)
    if state.side > 0:
        return max(state.stop_price, candidate)
    return min(state.stop_price, candidate)


def _final_exit_hit(state: TradeManagementState, policy: PartialExitPolicy, features: pd.DataFrame, index: int) -> bool:
    if policy.final_exit == "signal":
        return False
    if policy.final_exit == "atr_trailing":
        return state.side > 0 and features["close"].iloc[index] < state.stop_price or state.side < 0 and features["close"].iloc[index] > state.stop_price
    window = features.iloc[max(0, index - policy.donchian_window):index]
    close = float(features["close"].iloc[index])
    if policy.final_exit == "donchian" and not window.empty:
        level = float(window["low"].min() if state.side > 0 else window["high"].max())
        return close < level if state.side > 0 else close > level
    if policy.final_exit == "ema" and "ema_fast" in features:
        ema = float(features["ema_fast"].iloc[index])
        return close < ema if state.side > 0 else close > ema
    if policy.final_exit == "chandelier":
        atr = float(features["atr"].iloc[index])
        level = (state.highest_price if state.side > 0 else state.lowest_price) - state.side * policy.chandelier_multiple * atr
        return close < level if state.side > 0 else close > level
    return False


class PartialExitEngine:
    """Execute one policy causally over a feature frame."""

    def __init__(self, fee_rate: float, slippage_rate: float) -> None:
        self.cost_rate = float(fee_rate + slippage_rate)
        self.fee_rate = float(fee_rate)
        self.slippage_rate = float(slippage_rate)

    def execute(
        self,
        features: pd.DataFrame,
        policy: PartialExitPolicy,
        stop_multiple: float,
        historical_mfe: float | None = None,
        execution_delay_bars: int = 0,
        adverse_fill_rate: float = 0.0,
        random_fill_deviation_rate: float = 0.0,
        random_seed: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        required = {"open", "high", "low", "close", "atr", "signal_position"}
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"missing feature columns: {sorted(missing)}")
        index = features.index
        bar_returns = np.zeros(len(features), dtype=float)
        gross_returns = np.zeros(len(features), dtype=float)
        cost_returns = np.zeros(len(features), dtype=float)
        exposure = np.zeros(len(features), dtype=float)
        trades: list[dict[str, Any]] = []
        state: TradeManagementState | None = None
        previous_close = np.nan
        blocked = False
        if execution_delay_bars < 0:
            raise ValueError("execution_delay_bars must be non-negative")
        if adverse_fill_rate < 0 or random_fill_deviation_rate < 0:
            raise ValueError("fill deviation rates must be non-negative")
        fill_rng = np.random.default_rng(random_seed)

        def _stressed_fill_price(price: float, side: int, is_entry: bool = False) -> float:
            direction = 1.0 if is_entry else -1.0
            executed = price * (1.0 + direction * side * self.slippage_rate)
            if adverse_fill_rate:
                executed *= 1.0 + direction * side * adverse_fill_rate
            if random_fill_deviation_rate:
                deviation = float(fill_rng.normal(0.0, random_fill_deviation_rate))
                executed *= 1.0 + direction * side * deviation
            return executed

        def close_fraction(i: int, price: float, fraction: float, reason: str) -> tuple[float, float, float]:
            nonlocal state
            if state is None or fraction <= 0:
                return 0.0, 0.0, 0.0
            fraction = min(fraction, state.remaining_fraction)
            executed_price = _stressed_fill_price(price, state.side)
            gross = fraction * _side_price(state.side, executed_price, state.entry_price)
            cost = fraction * self.fee_rate
            state.realized_gross += gross
            state.realized_cost += cost
            state.remaining_fraction -= fraction
            state.events.append(
                {
                    "stage": state.next_stage + 1,
                    "fraction": fraction,
                    "price": executed_price,
                    "reason": reason,
                    "timestamp": index[i],
                }
            )
            state.next_stage += 1
            if state.remaining_fraction <= 1e-12:
                trades.append(
                    {
                        "entry_timestamp": index[state.entry_index],
                        "exit_timestamp": index[i],
                        "side": state.side,
                        "entry_price": state.entry_price,
                        "exit_price": executed_price,
                        "gross_return": state.realized_gross,
                        "cost_return": state.realized_cost,
                        "net_return": state.realized_gross - state.realized_cost,
                        "mae": state.mae,
                        "mfe": state.mfe,
                        "duration_bars": i - state.entry_index + 1,
                        "exit_reason": reason,
                        "partial_events": list(state.events),
                    }
                )
                state = None
            return executed_price, gross, cost

        for i in range(1, len(features)):
            row = features.iloc[i]
            if not np.isfinite(row["open"]) or not np.isfinite(row["close"]):
                previous_close = row["close"] if np.isfinite(row["close"]) else previous_close
                continue
            signal_index = i - 1 - execution_delay_bars
            desired = float(features["signal_position"].iloc[signal_index]) if signal_index >= 0 else 0.0
            if desired == 0.0:
                blocked = False
            current_gross = 0.0
            current_cost = 0.0
            opened_this_bar = False
            if state is not None:
                side = state.side
                signal_exit = desired != side
                stop_hit = (side > 0 and row["low"] <= state.stop_price) or (side < 0 and row["high"] >= state.stop_price)
                targets = _target_prices(state, policy.take_profit, features, i, historical_mfe) if policy.take_profit is not None else []
                target_hit = (
                    state.next_stage < len(policy.allocations) - 1
                    and bool(targets)
                    and (row["high"] >= targets[state.next_stage] if side > 0 else row["low"] <= targets[state.next_stage])
                )
                if not signal_exit or stop_hit or target_hit:
                    state.highest_price = max(state.highest_price, float(row["high"]))
                    state.lowest_price = min(state.lowest_price, float(row["low"]))
                    state.mae = min(state.mae, _side_price(side, state.highest_price if side < 0 else state.lowest_price, state.entry_price))
                    state.mfe = max(state.mfe, _side_price(side, state.lowest_price if side < 0 else state.highest_price, state.entry_price))
                if stop_hit:
                    before = state.remaining_fraction
                    stop_price = state.stop_price
                    fill_price, _, close_cost = close_fraction(i, stop_price, before, "stop")
                    current_gross = before * side * (fill_price / previous_close - 1.0) if np.isfinite(previous_close) else 0.0
                    current_cost = close_cost
                    blocked = True
                elif target_hit:
                    target = targets[state.next_stage]
                    fraction = policy.allocations[state.next_stage]
                    fill_price, _, close_cost = close_fraction(i, target, fraction, f"tp{state.next_stage + 1}")
                    if np.isfinite(previous_close):
                        current_gross += fraction * side * (fill_price / previous_close - 1.0)
                    current_cost += close_cost
                    if state is not None and policy.stop_mode != "unchanged" and state.next_stage >= policy.activate_trailing_after_stage:
                        state.stop_price = _stop_for_mode(state, policy, features, i, self.cost_rate)
                if state is not None and desired != side:
                    before = state.remaining_fraction
                    fill_price_value, _, close_cost = close_fraction(i, float(row["open"]), before, "signal_exit")
                    if np.isfinite(previous_close):
                        current_gross += before * side * (fill_price_value / previous_close - 1.0)
                    current_cost += close_cost
                elif state is not None and _final_exit_hit(state, policy, features, i):
                    before = state.remaining_fraction
                    fill_price_value, _, close_cost = close_fraction(i, float(row["open"]), before, "final_exit")
                    if np.isfinite(previous_close):
                        current_gross += before * side * (fill_price_value / previous_close - 1.0)
                    current_cost += close_cost
                elif state is not None and policy.stop_mode != "unchanged" and state.next_stage > 0:
                    state.stop_price = _stop_for_mode(state, policy, features, i, self.cost_rate)
            if state is None and desired != 0.0 and not blocked:
                entry_price = _stressed_fill_price(float(row["open"]), int(desired), is_entry=True)
                entry_atr = float(features["atr"].iloc[i - 1])
                if np.isfinite(entry_atr) and entry_atr > 0:
                    state = TradeManagementState(
                        side=int(desired),
                        entry_index=i,
                        entry_price=entry_price,
                        entry_atr=entry_atr,
                        realized_cost=self.fee_rate,
                        stop_price=entry_price - desired * stop_multiple * entry_atr,
                    )
                    state.highest_price = entry_price
                    state.lowest_price = entry_price
                    current_cost += self.fee_rate
                    opened_this_bar = True
            if state is not None:
                exposure[i] = state.side * state.remaining_fraction
                if opened_this_bar:
                    current_gross += state.remaining_fraction * state.side * (float(row["close"]) / state.entry_price - 1.0)
                elif np.isfinite(previous_close):
                    current_gross += state.remaining_fraction * state.side * (float(row["close"]) / previous_close - 1.0)
            gross_returns[i] = current_gross
            cost_returns[i] = -current_cost
            bar_returns[i] = current_gross - current_cost
            previous_close = float(row["close"])
        if state is not None and len(features) > 1:
            close_fraction(len(features) - 1, float(features["close"].iloc[-1]), state.remaining_fraction, "end_of_data")
        result = pd.DataFrame(
            {
                "gross_return": gross_returns,
                "strategy_return_1x": bar_returns,
                "cost_return": cost_returns,
                "turnover": np.abs(np.diff(np.r_[0.0, exposure])),
                "position": exposure,
                "atr_pct": features["atr"] / features["close"],
                "realized_vol": features["realized_vol"] if "realized_vol" in features else 0.0,
            },
            index=index,
        )
        return result, pd.DataFrame(trades)


def default_partial_exit_policies() -> list[PartialExitPolicy]:
    """Return the requested interpretable policy families, not tuned winners."""
    atr = TakeProfitSpec("atr_multiple", (1.0, 2.0))
    return [
        PartialExitPolicy("baseline_full_exit", (1.0,), None),
        PartialExitPolicy("runner_25_25_50", (0.25, 0.25, 0.50), atr),
        PartialExitPolicy("runner_33_33_34", (0.33, 0.33, 0.34), atr),
        PartialExitPolicy("runner_50_50", (0.50, 0.50), TakeProfitSpec("atr_multiple", (1.5,))),
        PartialExitPolicy("runner_20_30_50", (0.20, 0.30, 0.50), TakeProfitSpec("atr_multiple", (1.0, 2.0))),
        PartialExitPolicy("partial_breakeven", (0.25, 0.25, 0.50), atr, "break_even"),
        PartialExitPolicy("partial_atr_trailing", (0.25, 0.25, 0.50), atr, "atr_trailing", "atr_trailing"),
        PartialExitPolicy("partial_chandelier", (0.25, 0.25, 0.50), atr, "chandelier", "chandelier"),
        PartialExitPolicy("partial_donchian_runner", (0.25, 0.25, 0.50), atr, "donchian", "donchian"),
        PartialExitPolicy("partial_ema_runner", (0.25, 0.25, 0.50), atr, "ema", "ema"),
    ]