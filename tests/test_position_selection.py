"""Tests for the position-selection/ranking and discrete-exchange-leverage
helpers (core/risk.py) backing the testnet-near execution architecture."""
from __future__ import annotations

import pandas as pd
import pytest

from core.risk import (
    cap_total_notional,
    compute_position_score,
    resolve_shared_wallet_weights,
    score_proportional_weights,
    select_discrete_exchange_leverage,
    select_top_active_symbols,
    total_notional_scale,
)


def test_compute_position_score_matches_formula() -> None:
    trend_strength = pd.Series([0.02, -0.10, 0.0])
    atr_pct = pd.Series([0.01, 0.02, 0.005])
    score = compute_position_score(trend_strength, atr_pct)
    assert score.iloc[0] == pytest.approx(2.0)
    assert score.iloc[1] == pytest.approx(5.0)
    assert score.iloc[2] == pytest.approx(0.0)


def test_compute_position_score_avoids_div_by_zero() -> None:
    score = compute_position_score(pd.Series([0.05]), pd.Series([0.0]), epsilon=1e-6)
    assert score.iloc[0] == pytest.approx(0.05 / 1e-6)


def test_select_top_active_symbols_no_cap_is_noop() -> None:
    active = {"BTC/USDT", "ETH/USDT", "SOL/USDT"}
    assert select_top_active_symbols({"BTC/USDT": 1, "ETH/USDT": 2, "SOL/USDT": 3}, active, None) == active


def test_select_top_active_symbols_picks_highest_scores() -> None:
    scores = {"BTC/USDT": 5.0, "ETH/USDT": 1.0, "SOL/USDT": 3.0}
    selected = select_top_active_symbols(scores, set(scores), max_active_positions=2)
    assert selected == {"BTC/USDT", "SOL/USDT"}


def test_select_top_active_symbols_deterministic_tiebreak() -> None:
    scores = {"BTC/USDT": 1.0, "ETH/USDT": 1.0}
    selected = select_top_active_symbols(scores, set(scores), max_active_positions=1)
    assert selected == {"BTC/USDT"}  # alphabetically first wins the tie


def test_score_proportional_weights_sum_to_one_and_favor_strongest() -> None:
    scores = {"BTC/USDT": 3.0, "ETH/USDT": 1.0}
    weights = score_proportional_weights(scores, {"BTC/USDT", "ETH/USDT"})
    assert weights["BTC/USDT"] == pytest.approx(0.75)
    assert weights["ETH/USDT"] == pytest.approx(0.25)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_score_proportional_weights_equal_split_when_scores_zero() -> None:
    weights = score_proportional_weights({"BTC/USDT": 0.0, "ETH/USDT": 0.0}, {"BTC/USDT", "ETH/USDT"})
    assert weights == {"BTC/USDT": 0.5, "ETH/USDT": 0.5}


def test_resolve_shared_wallet_weights_default_is_inverse_leverage() -> None:
    candidates = {
        "BTC/USDT": {"target_position": 1, "leverage_candidate": 2.0, "stop_pct": 0.02, "close": 100.0},
        "ETH/USDT": {"target_position": 1, "leverage_candidate": 4.0, "stop_pct": 0.03, "close": 50.0},
        "SOL/USDT": {"target_position": 0, "leverage_candidate": 3.0, "stop_pct": 0.01, "close": 20.0},
    }
    weights = resolve_shared_wallet_weights(candidates, selection_enabled=False)
    assert "SOL/USDT" not in weights
    assert weights["BTC/USDT"] == pytest.approx(2.0 / 6.0)
    assert weights["ETH/USDT"] == pytest.approx(4.0 / 6.0)


def test_resolve_shared_wallet_weights_selection_picks_top_and_uses_score(monkeypatch) -> None:
    candidates = {
        "BTC/USDT": {
            "target_position": 1, "leverage_candidate": 2.0, "stop_pct": 0.02, "close": 100.0, "score": 10.0
        },
        "ETH/USDT": {
            "target_position": 1, "leverage_candidate": 4.0, "stop_pct": 0.03, "close": 50.0, "score": 1.0
        },
    }
    weights = resolve_shared_wallet_weights(candidates, selection_enabled=True, max_active_positions=1)
    assert weights == {"BTC/USDT": 1.0}  # only the higher-score symbol survives top-1


def test_select_discrete_exchange_leverage_rounds_up_to_smallest_covering_step() -> None:
    assert select_discrete_exchange_leverage(2.3, [1, 2, 3, 5, 10], risk_leverage_cap=10) == 3


def test_select_discrete_exchange_leverage_never_exceeds_risk_cap() -> None:
    # Required leverage would need a step of 10, but the risk cap only allows up to 5.
    assert select_discrete_exchange_leverage(8.0, [1, 2, 3, 5, 10], risk_leverage_cap=5) == 5


def test_select_discrete_exchange_leverage_falls_back_without_steps() -> None:
    assert select_discrete_exchange_leverage(3.5, [], risk_leverage_cap=4.0) == 3.5


def test_cap_total_notional_scales_down_proportionally() -> None:
    signed = {"BTC/USDT": 300.0, "ETH/USDT": -200.0}
    capped = cap_total_notional(signed, max_total_notional_usdt=250.0)
    assert sum(abs(v) for v in capped.values()) == pytest.approx(250.0)
    assert capped["BTC/USDT"] / capped["ETH/USDT"] == pytest.approx(300.0 / -200.0)


def test_cap_total_notional_noop_when_within_budget() -> None:
    signed = {"BTC/USDT": 100.0, "ETH/USDT": -50.0}
    assert cap_total_notional(signed, max_total_notional_usdt=1000.0) == signed


def test_cap_total_notional_disabled_is_noop() -> None:
    signed = {"BTC/USDT": 100.0}
    assert cap_total_notional(signed, max_total_notional_usdt=None) == signed


def test_total_notional_scale_vectorized_matches_scalar_cap() -> None:
    weights = pd.DataFrame({"BTC/USDT": [0.6], "ETH/USDT": [0.4]})
    leverage = pd.DataFrame({"BTC/USDT": [4.0], "ETH/USDT": [4.0]})
    capital_prior = pd.Series([1000.0])
    # implied notional = (0.6*4 + 0.4*4) * 1000 = 4000; cap at 2000 -> scale 0.5
    scale = total_notional_scale(weights, leverage, capital_prior, max_total_notional_usdt=2000.0)
    assert scale.iloc[0] == pytest.approx(0.5)
