"""Tests for the portfolio-level (shared-wallet) shadow simulation that backs
a genuine ML promotion decision (core/ml/shadow.py: PortfolioShadowSimulator,
PortfolioShadowLogger, is_funding_event_hour)."""
from __future__ import annotations

import pandas as pd
import pytest

from core.config import CapitalConfig, ExecutionConfig
from core.ml.shadow import (
    PORTFOLIO_SHADOW_LOG_COLUMNS,
    PortfolioShadowLogger,
    PortfolioShadowSimulator,
    is_funding_event_hour,
)


def _candidate(target_position: int, leverage: float, stop_pct: float, close: float) -> dict:
    return {
        "target_position": target_position,
        "leverage_candidate": leverage,
        "stop_pct": stop_pct,
        "close": close,
    }


def test_is_funding_event_hour_matches_binance_default_schedule() -> None:
    assert is_funding_event_hour(pd.Timestamp("2024-01-01T00:00"))
    assert is_funding_event_hour(pd.Timestamp("2024-01-01T08:00"))
    assert is_funding_event_hour(pd.Timestamp("2024-01-01T16:00"))
    assert not is_funding_event_hour(pd.Timestamp("2024-01-01T01:00"))
    assert not is_funding_event_hour(pd.Timestamp("2024-01-01T23:00"))


def test_simulator_first_bar_has_no_pnl_but_opens_notional() -> None:
    sim = PortfolioShadowSimulator(
        initial_capital_usdt=1000.0,
        cold_start_buffer_pct=0.10,
        cold_start_risk_scale=0.5,
        max_portfolio_risk_pct=0.01,
        max_absolute_position_size_usdt=None,
        fee=0.0004,
    )
    result = sim.step({"BTC/USDT": _candidate(1, 2.0, 0.02, 100.0)})
    # No prior bar to compare against -> no price-driven pnl, but a fee is
    # charged for opening the new notional.
    assert result["return"] < 0.0
    assert result["fee_cost_usdt"] > 0.0
    assert result["funding_cost_usdt"] == 0.0
    assert sim.capital < 1000.0


def test_simulator_realizes_pnl_from_prior_bar_position_only(monkeypatch) -> None:
    """Causal: this bar's price move only affects PnL through the position
    that was ALREADY held entering this bar (previous poll's notional), never
    a same-bar look-ahead."""
    sim = PortfolioShadowSimulator(1000.0, 0.10, 0.5, 0.01, None, fee=0.0)
    sim.step({"BTC/USDT": _candidate(1, 2.0, 0.02, 100.0)})
    capital_after_open = sim.capital

    # Price rises 10%; the position opened on bar 1 should now show a gain.
    result = sim.step({"BTC/USDT": _candidate(1, 2.0, 0.02, 110.0)})
    assert result["return"] > 0.0
    assert sim.capital > capital_after_open


def test_simulator_applies_funding_cost_only_on_event_bars() -> None:
    sim = PortfolioShadowSimulator(1000.0, 0.10, 0.5, 0.01, None, fee=0.0)
    sim.step({"BTC/USDT": _candidate(1, 2.0, 0.02, 100.0)})

    no_event = sim.step(
        {"BTC/USDT": _candidate(1, 2.0, 0.02, 100.0)}, funding_rates={"BTC/USDT": 0.001}, funding_event=False
    )
    assert no_event["funding_cost_usdt"] == 0.0

    event = sim.step(
        {"BTC/USDT": _candidate(1, 2.0, 0.02, 100.0)}, funding_rates={"BTC/USDT": 0.001}, funding_event=True
    )
    assert event["funding_cost_usdt"] > 0.0  # long pays positive funding


def test_simulator_never_reuses_real_equity_two_independent_trajectories_diverge() -> None:
    """Two simulators seeded identically but fed different candidate streams
    (as rule vs. ML would diverge) must produce independent capital paths --
    proving neither reuses a shared/real equity number."""
    rule_sim = PortfolioShadowSimulator(1000.0, 0.10, 0.5, 0.01, None, fee=0.0002)
    ml_sim = PortfolioShadowSimulator(1000.0, 0.10, 0.5, 0.01, None, fee=0.0002)

    rule_sim.step({"BTC/USDT": _candidate(1, 2.0, 0.02, 100.0)})
    ml_sim.step({"BTC/USDT": _candidate(1, 2.0, 0.02, 100.0)})

    rule_sim.step({"BTC/USDT": _candidate(1, 2.0, 0.02, 105.0)})
    ml_sim.step({"BTC/USDT": _candidate(0, 2.0, 0.02, 105.0)})  # ML flat this bar -- diverges from rule

    assert rule_sim.capital != ml_sim.capital


def test_portfolio_shadow_logger_persists_both_trajectories(tmp_path) -> None:
    path = tmp_path / "portfolio_shadow.csv"
    logger = PortfolioShadowLogger(str(path), CapitalConfig(), ExecutionConfig())

    logger.step_and_log(
        pd.Timestamp("2024-01-01T00:00"),
        {"BTC/USDT": _candidate(1, 2.0, 0.02, 100.0)},
        {"BTC/USDT": _candidate(0, 2.0, 0.02, 100.0)},
    )
    logger.step_and_log(
        pd.Timestamp("2024-01-01T01:00"),
        {"BTC/USDT": _candidate(1, 2.0, 0.02, 102.0)},
        {"BTC/USDT": _candidate(1, 2.0, 0.02, 102.0)},
    )

    logged = logger.read_all()
    assert list(logged.columns) == PORTFOLIO_SHADOW_LOG_COLUMNS
    assert len(logged) == 2
    # rule and ml capital diverged because the candidates differed on bar 1.
    assert logged.iloc[-1]["rule_capital_usdt"] != logged.iloc[-1]["ml_capital_usdt"]


def test_portfolio_shadow_logger_ml_mirrors_rule_when_ml_candidates_missing(tmp_path) -> None:
    """If the caller passes ``ml_candidates=None`` (e.g. ml disabled this
    poll), the ML trajectory must track the rule trajectory instead of
    silently freezing or crashing."""
    path = tmp_path / "portfolio_shadow.csv"
    logger = PortfolioShadowLogger(str(path), CapitalConfig(), ExecutionConfig())

    logger.step_and_log(
        pd.Timestamp("2024-01-01T00:00"), {"BTC/USDT": _candidate(1, 2.0, 0.02, 100.0)}, None
    )
    logged = logger.read_all()
    assert logged.iloc[0]["rule_capital_usdt"] == pytest.approx(logged.iloc[0]["ml_capital_usdt"])
