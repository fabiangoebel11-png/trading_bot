import pytest
from datetime import datetime, timedelta, timezone

from paper_broker import PaperBroker
from risk_engine import TradeSetup, compute_risk_parameters, compute_strategy_profiles


def test_conservative_and_aggressive_profiles_diverge_when_setup_justifies_it() -> None:
    setup = TradeSetup(
        asset="BTC/USDT",
        asset_class="crypto",
        direction="LONG",
        entry_price=84_000.0,
        atr=110.0,
        expected_mae=-0.02,
        expected_mfe=0.04,
        score=85.0,
        capital_eur=500.0,
    )
    profiles = compute_strategy_profiles(setup)

    assert profiles["aggressive"].position_size_eur > profiles["conservative"].position_size_eur
    assert profiles["aggressive"].risk_amount_eur != profiles["conservative"].risk_amount_eur


def test_recommended_profile_and_dynamic_exits_are_exposed() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=84_000.0,
            atr=110.0,
            expected_mae=-0.02,
            expected_mfe=0.04,
            score=90.0,
            capital_eur=500.0,
            risk_per_trade_pct=0.005,
            max_position_fraction=1.0,
        )
    )

    assert not result.rejected
    assert result.take_profit_1_price > result.entry_price
    assert result.take_profit_2_price > result.take_profit_1_price
    assert result.trailing_stop_trigger_pct > 0.0
    assert result.trailing_stop_price > result.entry_price


def test_profile_leverage_adapts_to_stop_distance_and_asset_limits() -> None:
    crypto = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=84_000.0,
            atr=110.0,
            expected_mae=-0.005,
            expected_mfe=0.026,
            score=100.0,
            capital_eur=500.0,
        )
    )
    equity = compute_risk_parameters(
        TradeSetup(
            asset="QQQ",
            asset_class="equity",
            direction="LONG",
            entry_price=500.0,
            atr=5.0,
            expected_mae=-0.02,
            expected_mfe=0.04,
            score=90.0,
            capital_eur=500.0,
        )
    )

    assert 1.0 < crypto.leverage <= 10.0
    assert equity.leverage == 1.0
    assert crypto.leverage <= crypto.max_allowed_leverage
    assert equity.leverage <= equity.max_allowed_leverage
    assert crypto.leverage != 1.0


def test_risk_sizing_uses_leverage_instead_of_forcing_one_x() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=84_000.0,
            atr=110.0,
            expected_mae=-0.035,
            expected_mfe=0.026,
            score=56.8,
            capital_eur=500.0,
        )
    )

    assert not result.rejected
    assert result.instrument_type == "crypto_perpetual"
    assert 1.0 < result.leverage <= result.max_allowed_leverage
    assert result.position_size_eur == result.margin_eur * result.leverage
    assert result.margin_eur <= 500.0 * 0.35
    assert result.risk_amount_eur == result.position_size_eur * result.stop_distance_pct


def test_eur_capital_uses_usd_fx_for_usd_quoted_quantity() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="QQQ",
            asset_class="equity",
            direction="LONG",
            entry_price=100.0,
            atr=1.0,
            expected_mae=-0.01,
            expected_mfe=0.02,
            score=100.0,
            capital_eur=1_000.0,
            risk_per_trade_pct=0.01,
            max_position_fraction=1.0,
            max_portfolio_risk_pct=1.0,
            usd_per_eur=1.25,
        )
    )

    assert not result.rejected
    assert result.risk_amount_eur == pytest.approx(10.0)
    assert result.position_size_eur == pytest.approx(1_000.0 / 1.5)
    assert result.quantity == pytest.approx(result.position_size_eur * 1.25 / 100.0)


def test_risk_engine_rejects_invalid_usd_per_eur_rate() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="QQQ",
            asset_class="equity",
            direction="LONG",
            entry_price=100.0,
            atr=1.0,
            expected_mae=-0.01,
            expected_mfe=0.02,
            score=100.0,
            capital_eur=1_000.0,
            usd_per_eur=0.0,
        )
    )

    assert result.rejected
    assert "exchange rate" in result.rejection_reason


def test_tighter_stop_can_require_more_leverage_but_never_exceeds_ceiling() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=84_000.0,
            atr=110.0,
            expected_mae=-0.005,
            expected_mfe=0.026,
            score=100.0,
            capital_eur=500.0,
        )
    )

    assert not result.rejected
    assert 1.0 < result.leverage <= 10.0
    assert result.position_size_eur <= result.risk_based_notional_eur


def test_selected_leverage_override_is_applied_without_duplication() -> None:
    recommended = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=84_000.0,
            atr=110.0,
            expected_mae=-0.02,
            expected_mfe=0.04,
            score=90.0,
            capital_eur=500.0,
        )
    )
    selected = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=84_000.0,
            atr=110.0,
            expected_mae=-0.02,
            expected_mfe=0.04,
            score=90.0,
            capital_eur=500.0,
            risk_per_trade_pct=0.005,
            max_position_fraction=1.0,
            selected_leverage=5.0,
        )
    )

    assert not recommended.rejected
    assert not selected.rejected
    assert recommended.leverage != selected.leverage
    assert selected.leverage == 5.0
    assert selected.margin_eur == selected.position_size_eur / selected.leverage
    assert selected.leverage <= selected.max_allowed_leverage


def test_selected_leverage_rejects_invalid_or_capped_values() -> None:
    rejected_cap = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=84_000.0,
            atr=110.0,
            expected_mae=-0.02,
            expected_mfe=0.04,
            score=90.0,
            capital_eur=500.0,
            selected_leverage=15.0,
        )
    )
    rejected_zero = compute_risk_parameters(
        TradeSetup(
            asset="QQQ",
            asset_class="equity",
            direction="LONG",
            entry_price=500.0,
            atr=5.0,
            expected_mae=-0.02,
            expected_mfe=0.04,
            score=90.0,
            capital_eur=500.0,
            selected_leverage=0.0,
        )
    )

    assert rejected_cap.rejected
    assert "exceeds" in rejected_cap.rejection_reason.lower()
    assert rejected_zero.rejected
    assert "selected leverage" in rejected_zero.rejection_reason.lower()


def test_plain_equity_underlying_has_no_synthetic_liquidation() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="QQQ",
            asset_class="equity",
            direction="LONG",
            entry_price=500.0,
            atr=5.0,
            expected_mae=-0.02,
            expected_mfe=0.04,
            score=90.0,
            capital_eur=500.0,
        )
    )

    assert not result.rejected
    assert result.instrument_type == "equity_underlying"
    assert result.max_allowed_leverage == 1.0
    assert result.leverage == 1.0
    assert result.liquidation_price is None
    assert result.safety_barrier_price is None
    assert result.knockout_barrier_price == 0.0


def test_leverage_changes_margin_but_not_risk_or_notional() -> None:
    common = {
        "asset": "BTC/USDT",
        "asset_class": "crypto",
        "direction": "LONG",
        "entry_price": 100.0,
        "atr": 2.0,
        "expected_mae": -0.50,
        "expected_mfe": 4.0,
        "score": 100.0,
        "capital_eur": 500.0,
    }
    low = compute_risk_parameters(TradeSetup(**common, selected_leverage=2.0))
    high = compute_risk_parameters(TradeSetup(**common, selected_leverage=4.0))

    assert not low.rejected and not high.rejected
    assert low.stop_distance_pct == high.stop_distance_pct == pytest.approx(0.03)
    assert low.take_profit_1_price == pytest.approx(104.5)
    assert low.take_profit_2_price == pytest.approx(107.5)
    assert low.position_size_eur == high.position_size_eur
    assert low.risk_amount_eur == high.risk_amount_eur
    assert low.margin_eur == pytest.approx(high.margin_eur * 2.0)


def test_drawdown_reduces_risk_to_zero_at_account_limit() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=84_000.0,
            atr=110.0,
            expected_mae=-0.02,
            expected_mfe=0.04,
            score=90.0,
            capital_eur=500.0,
            drawdown_pct=0.10,
        )
    )

    assert result.rejected
    assert "risk budget" in result.rejection_reason


def test_knockout_barrier_is_beyond_stop_and_directional() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=100.0,
            atr=2.0,
            expected_mae=-0.03,
            expected_mfe=0.05,
            score=80.0,
            capital_eur=500.0,
        )
    )

    assert result.knockout_barrier_price < result.stop_loss_price < result.entry_price


@pytest.mark.parametrize("atr", [0.0, -1.0])
def test_non_positive_atr_fails_closed(atr: float) -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=100.0,
            atr=atr,
            expected_mae=0.0,
            expected_mfe=0.0,
            score=100.0,
            capital_eur=500.0,
        )
    )

    assert result.rejected
    assert "ATR" in result.rejection_reason
    assert result.position_size_eur == 0.0


@pytest.mark.parametrize("direction", ["LONG", "SHORT"])
def test_extreme_atr_rejects_non_positive_exit_prices(direction: str) -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction=direction,
            entry_price=100.0,
            atr=80.0 if direction == "LONG" else 30.0,
            expected_mae=0.0,
            expected_mfe=0.0,
            score=100.0,
            capital_eur=500.0,
        )
    )

    assert result.rejected
    assert "non-positive" in result.rejection_reason


def test_leverage_is_capped_by_stop_and_liquidation_safety() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=100.0,
            atr=10.0,
            expected_mae=0.0,
            expected_mfe=0.0,
            score=100.0,
            capital_eur=500.0,
            selected_leverage=10.0,
        )
    )

    assert result.rejected
    assert "stop-safe max" in result.rejection_reason


def test_atr_below_price_resolution_rejects_collapsed_exit_levels() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=1e20,
            atr=1e-20,
            expected_mae=0.0,
            expected_mfe=0.0,
            score=100.0,
            capital_eur=500.0,
        )
    )

    assert result.rejected
    assert "price precision" in result.rejection_reason


@pytest.mark.parametrize("available_margin", [float("nan"), -1.0])
def test_invalid_available_margin_fails_closed(available_margin: float) -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=100.0,
            atr=2.0,
            expected_mae=0.0,
            expected_mfe=0.0,
            score=100.0,
            capital_eur=500.0,
            available_margin_eur=available_margin,
        )
    )

    assert result.rejected
    assert "available margin" in result.rejection_reason


def test_equity_paper_broker_rejects_opposing_model_directions() -> None:
    broker = PaperBroker.__new__(PaperBroker)
    now = datetime.now(timezone.utc).isoformat()
    signals = {
        "swing": {"swing_opportunity_score": 90.0, "score": 90.0, "direction": "LONG", "updated_at": now},
        "intraday": {"score": 90.0, "direction": "SHORT", "updated_at": now},
    }
    broker._signal = lambda _conn, _asset, model_type: signals.get(model_type)
    broker._finalize_setup = lambda *_args: "allowed"

    result = broker._build_setup(None, "QQQ", "equity", {}, {}, 0.0)

    assert result is None
    signals["intraday"]["direction"] = "LONG"
    assert broker._build_setup(None, "QQQ", "equity", {}, {}, 0.0) == "allowed"
    signals["intraday"]["updated_at"] = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    assert broker._build_setup(None, "QQQ", "equity", {}, {}, 0.0) is None


def test_paper_broker_market_and_model_state_have_freshness_limits() -> None:
    broker = PaperBroker.__new__(PaperBroker)
    now = datetime.now(timezone.utc)
    market = {
        "market_open": 1, "feed_healthy": 1, "freshness_ok": 1, "warmup_ready": 1,
        "data_age_seconds": 0.0, "updated_at": now.isoformat(),
    }
    assert broker._market_is_fresh(market, now)
    assert not broker._market_is_fresh({**market, "market_open": 0}, now)
    assert not broker._market_is_fresh({**market, "data_age_seconds": 180.0}, now)

    stale_signal = {"updated_at": (now - timedelta(minutes=30)).isoformat()}
    fresh_signal = {"updated_at": now.isoformat()}
    assert not broker._is_fresh_record(stale_signal, now, max_age_seconds=20 * 60)
    assert broker._is_fresh_record(fresh_signal, now, max_age_seconds=20 * 60)


def test_selected_leverage_cannot_exceed_available_margin() -> None:
    result = compute_risk_parameters(
        TradeSetup(
            asset="BTC/USDT",
            asset_class="crypto",
            direction="LONG",
            entry_price=100.0,
            atr=1.0,
            expected_mae=0.0,
            expected_mfe=0.0,
            score=100.0,
            capital_eur=1_000.0,
            available_margin_eur=100.0,
            max_position_fraction=1.0,
            selected_leverage=1.0,
        )
    )

    assert result.rejected
    assert "available margin budget" in result.rejection_reason
