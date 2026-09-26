from __future__ import annotations

import pytest
import pandas as pd

from research.partial_exit_simulator import (
    PartialExitEngine,
    PartialExitPolicy,
    TakeProfitSpec,
)


def _features(signals, highs, lows, closes=None, atr=None):
    periods = len(signals)
    index = pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0] * periods,
            "high": highs,
            "low": lows,
            "close": closes or [100.0] * periods,
            "atr": atr or [1.0] * periods,
            "signal_position": signals,
            "realized_vol": [0.1] * periods,
        },
        index=index,
    )


def test_long_partial_stages_and_residual_stop_are_accounted_once():
    features = _features(
        [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
        [100.0, 100.0, 102.0, 103.0, 104.0, 104.0],
        [100.0, 99.0, 99.0, 99.0, 99.0, 90.0],
        [100.0, 100.0, 101.0, 102.0, 103.0, 90.0],
    )
    policy = PartialExitPolicy("test", (0.25, 0.25, 0.50), TakeProfitSpec("atr_multiple", (1.0, 2.0)))

    _, trades = PartialExitEngine(0.00035, 0.0002).execute(features, policy, 2.0)

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert [event["reason"] for event in trade["partial_events"]] == ["tp1", "tp2", "stop"]
    assert [event["fraction"] for event in trade["partial_events"]] == [0.25, 0.25, 0.5]
    assert sum(event["fraction"] for event in trade["partial_events"]) == 1.0
    assert trade["mae"] <= 0.0
    assert trade["mfe"] >= 0.0
    assert trade["cost_return"] > 0.0


def test_short_partial_exit_uses_short_direction_for_targets_and_excursions():
    features = _features(
        [0.0, -1.0, -1.0, -1.0, 0.0],
        [100.0, 101.0, 99.0, 98.0, 98.0],
        [100.0, 99.0, 98.0, 97.0, 97.0],
        [100.0, 100.0, 98.5, 97.5, 97.0],
    )
    policy = PartialExitPolicy("short", (0.5, 0.5), TakeProfitSpec("atr_multiple", (1.0,)))

    _, trades = PartialExitEngine(0.00035, 0.0002).execute(features, policy, 2.0)

    assert len(trades) == 1
    trade = trades.iloc[0]
    assert trade["side"] == -1
    assert trade["partial_events"][0]["reason"] == "tp1"
    assert trade["mae"] <= 0.0
    assert trade["mfe"] >= 0.0


def test_same_bar_stop_has_priority_over_take_profit():
    features = _features(
        [1.0, 1.0, 1.0, 0.0],
        [100.0, 100.0, 103.0, 100.0],
        [100.0, 100.0, 97.0, 100.0],
        [100.0, 100.0, 100.0, 100.0],
    )
    policy = PartialExitPolicy("same_bar", (0.5, 0.5), TakeProfitSpec("atr_multiple", (1.0,)))

    _, trades = PartialExitEngine(0.00035, 0.0002).execute(features, policy, 2.0)

    assert len(trades) == 1
    assert [event["reason"] for event in trades.iloc[0]["partial_events"]] == ["stop"]


def test_signal_exit_does_not_use_future_exit_bar_extremes():
    base = _features(
        [1.0, 1.0, 0.0, 0.0, 0.0],
        [100.0, 101.0, 101.0, 100.0, 100.0],
        [100.0, 99.5, 99.0, 100.0, 100.0],
    )
    altered = base.copy()
    altered.iloc[3, altered.columns.get_loc("high")] = 10_000.0
    altered.iloc[3, altered.columns.get_loc("low")] = 0.01
    policy = PartialExitPolicy("baseline", (1.0,))

    _, base_trades = PartialExitEngine(0.00035, 0.0002).execute(base, policy, 1000.0)
    _, altered_trades = PartialExitEngine(0.00035, 0.0002).execute(altered, policy, 1000.0)

    assert base_trades.iloc[0]["exit_reason"] == "signal_exit"
    assert altered_trades.iloc[0]["exit_reason"] == "signal_exit"
    assert altered_trades.iloc[0]["mae"] == base_trades.iloc[0]["mae"]
    assert altered_trades.iloc[0]["mfe"] == base_trades.iloc[0]["mfe"]


def test_realized_target_is_in_bar_returns_and_trade_costs_include_entry_fee():
    features = _features(
        [0.0, 1.0, 1.0, 1.0],
        [100.0, 100.0, 102.0, 102.0],
        [100.0, 99.0, 99.0, 99.0],
        [100.0, 100.0, 100.0, 101.0],
    )
    policy = PartialExitPolicy("accounting", (0.25, 0.75), TakeProfitSpec("atr_multiple", (1.0,)))

    result, trades = PartialExitEngine(0.00035, 0.0002).execute(features, policy, 2.0)

    entry_price = 100.0 * (1.0 + 0.0002)
    target_fill = (entry_price + 1.0) * (1.0 - 0.0002)
    expected_gross = 0.25 * (target_fill / 100.0 - 1.0) + 0.75 * (101.0 / 100.0 - 1.0)
    assert result.iloc[3]["strategy_return_1x"] == pytest.approx(expected_gross - 0.25 * 0.00035)
    trade = trades.iloc[0]
    assert trade["net_return"] == pytest.approx(trade["gross_return"] - trade["cost_return"])


def test_entry_bar_return_starts_at_fill_price_not_previous_close():
    features = _features(
        [1.0, 1.0, 1.0, 0.0],
        [100.0, 112.0, 113.0, 113.0],
        [100.0, 108.0, 109.0, 113.0],
        [100.0, 111.0, 112.0, 113.0],
    )
    features["open"] = [100.0, 110.0, 111.0, 113.0]
    policy = PartialExitPolicy("entry_fill", (1.0,))

    result, _ = PartialExitEngine(0.00035, 0.0002).execute(features, policy, 2.0)

    entry_fill = 110.0 * (1.0 + 0.0002)
    assert result.iloc[1]["gross_return"] == pytest.approx(111.0 / entry_fill - 1.0)