from core.architecture import horizon_to_bars, proxy_target_status, trading_day_horizon_to_bars
from core.foundation_models import BaseForecastModel, ForecastConsensus, ForecastResult


def test_crypto_horizon_bars_match_requested_multi_horizon_contract() -> None:
    assert horizon_to_bars("1d", "4h") == 6
    assert horizon_to_bars("3d", "4h") == 18
    assert horizon_to_bars("5d", "4h") == 30
    assert horizon_to_bars("7d", "4h") == 42
    assert horizon_to_bars("14d", "4h") == 84


def test_equity_daily_horizons_use_trading_days_not_calendar_days() -> None:
    assert trading_day_horizon_to_bars("1d") == 1
    assert trading_day_horizon_to_bars("3d") == 3
    assert trading_day_horizon_to_bars("5d") == 5
    assert trading_day_horizon_to_bars("10d") == 10


def test_proxy_aliases_are_not_treated_as_real_equity_targets() -> None:
    assert proxy_target_status("SPY").is_real_target is True
    assert proxy_target_status("QQQ").is_real_target is True
    assert proxy_target_status("SP500_PROXY").is_real_target is False
    assert proxy_target_status("NASDAQ100_PROXY").is_real_target is False


def test_foundation_model_contract_and_consensus_gate() -> None:
    model = BaseForecastModel(
        model_id="chronos_2",
        asset="BTC/USDT",
        mode="INTRADAY",
        native_timeframe="1h",
        forecast_horizons=(1, 4, 8, 12, 24),
        model_version="v1",
    )
    assert model.supports_horizon(4)
    result = ForecastResult(
        model_id="chronos_2",
        asset="BTC/USDT",
        mode="INTRADAY",
        native_timeframe="1h",
        forecast_horizons=(1, 4, 8),
        expected_returns={1: 0.01, 4: 0.02},
        direction_probabilities={1: {"LONG": 0.6, "SHORT": 0.4}, 4: {"LONG": 0.55, "SHORT": 0.45}},
        score=0.8,
        data_timestamp="2026-09-24T00:00:00Z",
    )
    assert result.expected_return_for(4) == 0.02
    consensus = ForecastConsensus(
        model_available=True,
        data_valid=True,
        context_valid=True,
        forecast_valid=True,
        model_version_registered=True,
        validation_status="ACCEPTABLE",
    )
    assert consensus.is_valid()
