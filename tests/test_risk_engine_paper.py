from risk_engine import TradeSetup, compute_risk_parameters


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
    assert result.leverage == 1.0
    assert result.leverage < result.max_allowed_leverage
    assert result.position_size_eur == result.margin_eur * result.leverage
    assert result.margin_eur <= 500.0 * 0.35
    assert result.risk_amount_eur == result.position_size_eur * result.stop_distance_pct


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
