import pytest
import numpy as np
import pandas as pd

import model_integration
from model_integration import ModelForecast, _forecast_from_quantiles, _model_agreement, infer_chronos2_forecast


def test_chronos_quantiles_become_explicit_model_output_and_score() -> None:
    forecast = _forecast_from_quantiles(
        100.0, 102.0, 95.0, 110.0,
        asset="BTC/USDT", category="SWING", timeframe="4h", horizon=42, model_name="test-chronos-2",
    )
    assert forecast.status == "AVAILABLE"
    assert forecast.model_id == "chronos2"
    assert forecast.model_version == "test-chronos-2"
    assert forecast.direction == "LONG"
    assert forecast.expected_return == pytest.approx(0.02)
    assert forecast.expected_adverse_move == pytest.approx(-0.05)
    assert forecast.expected_favorable_move == pytest.approx(0.10)
    assert 0.0 <= forecast.model_score <= 100.0
    assert forecast.category == "SWING"
    assert forecast.forecast_horizon == "next 42 4h bars"


def test_chronos_quantiles_reject_nonfinite_output() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        _forecast_from_quantiles(
            100.0, float("nan"), 95.0, 110.0,
            asset="BTC/USDT", category="INTRADAY", timeframe="1h", horizon=4, model_name="test",
        )


def test_explicit_chronos_route_does_not_fallback_to_tcn(monkeypatch) -> None:
    monkeypatch.setattr(model_integration, "_chronos2_quantile_forecast", lambda _frame, _horizon: (102.0, 95.0, 110.0, "test-chronos-2"))
    frame = pd.concat([_frame()] * 8, ignore_index=True)
    frame.index = pd.date_range("2026-01-01", periods=len(frame), freq="h", tz="UTC")
    forecast = infer_chronos2_forecast("BTC/USDT", "1h", frame, horizon="4h")

    assert forecast.status == "AVAILABLE"
    assert forecast.model_id == "chronos2"
    assert forecast.horizon == "4h"


def test_model_agreement_never_averages_conflicting_directions() -> None:
    tcn = ModelForecast("AVAILABLE", "tcn", "v", "BTC/USDT", "INTRADAY", "1h", "next 4 1h bars", "LONG", 80.0, 0.01, -0.01, 0.02, "LOW", "2026-01-01T00:00:00", "test")
    chronos = ModelForecast("AVAILABLE", "chronos2", "v", "BTC/USDT", "INTRADAY", "1h", "next 4 1h bars", "SHORT", 70.0, -0.01, -0.02, 0.01, "LOW", "2026-01-01T00:00:00", "test")
    agreement, score = _model_agreement(tcn, chronos)
    assert agreement == "CONFLICT"
    assert score is None


class _FakeHorizonModel:
    forecast_horizons = (1, 4, 8, 12, 24)

    def predict_trade_outputs_by_horizon(self, _sequences):
        result = {}
        for horizon in self.forecast_horizons:
            result[horizon] = {
                "probabilities": np.array([[0.1, 0.2, 0.7]], dtype=np.float32),
                "opportunity": np.array([0.5], dtype=np.float32),
                "expected_return": np.array([float(horizon)], dtype=np.float32),
                "expected_mfe": np.array([float(horizon) + 1.0], dtype=np.float32),
                "expected_mae": np.array([-float(horizon) - 1.0], dtype=np.float32),
                "expected_duration": np.array([float(horizon)], dtype=np.float32),
            }
        return result


def _patch_fake_tcn(monkeypatch, internal_symbol: str) -> None:
    monkeypatch.setattr(model_integration, "load_symbol_model", lambda _symbol, _config: (_FakeHorizonModel(), {"feature_columns": ["feature"], "sequence_length": 4, "model_version": "test"}))
    monkeypatch.setattr(model_integration, "build_feature_matrix", lambda frame, _macro, _config: pd.DataFrame({"feature": np.ones(len(frame))}, index=frame.index))
    monkeypatch.setattr(model_integration, "make_sequences", lambda values, sequence_length: values[-sequence_length:][None, :, :])


def _frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=index)


@pytest.mark.parametrize("requested,expected", [("1h", 1), ("4h", 4), ("8h", 8), ("12h", 12), ("24h", 24)])
def test_tcn_router_uses_requested_horizon_head(monkeypatch, requested: str, expected: int) -> None:
    _patch_fake_tcn(monkeypatch, "BTC/USDT")

    forecast = model_integration._infer_tcn_forecast(
        "BTC/USDT", "1h", _frame(), horizon=requested, category="INTRADAY", model_root=None
    )

    assert forecast.status == "AVAILABLE"
    assert forecast.asset == "BTC/USDT"
    assert forecast.horizon == requested
    assert forecast.expected_return == pytest.approx(float(expected))
    assert forecast.as_dict()["horizon"] == requested


def test_tcn_router_does_not_fallback_to_primary_head(monkeypatch) -> None:
    _patch_fake_tcn(monkeypatch, "BTC/USDT")

    forecast = model_integration._infer_tcn_forecast(
        "BTC/USDT", "1h", _frame(), horizon="48h", category="INTRADAY", model_root=None
    )

    assert forecast.status == "MODEL_UNAVAILABLE"
    assert forecast.expected_return is None
    assert "unavailable" in forecast.reason


@pytest.mark.parametrize("asset,internal", [("SPY", "SP500_PROXY"), ("QQQ", "NASDAQ100_PROXY")])
def test_equity_intraday_tcn_is_disabled(monkeypatch, asset: str, internal: str) -> None:
    calls = []
    _patch_fake_tcn(monkeypatch, internal)
    monkeypatch.setattr(model_integration, "load_symbol_model", lambda symbol, _config: (calls.append(symbol) or (_FakeHorizonModel(), {"feature_columns": ["feature"], "sequence_length": 4, "model_version": "test"})))

    forecast = model_integration._infer_tcn_forecast(
        asset, "1h", _frame(), horizon="4h", category="INTRADAY", model_root=None
    )

    assert forecast.status == "MODEL_UNAVAILABLE"
    assert "TCN is disabled" in forecast.reason
    assert calls == []
    assert forecast.as_dict()["asset"] == asset


@pytest.mark.parametrize("asset", ["SPY", "QQQ"])
def test_registered_swing_route_uses_public_asset_and_requested_horizon(monkeypatch, asset: str) -> None:
    monkeypatch.setattr(
        model_integration,
        "_infer_swing_forecast",
        lambda requested_asset, timeframe, frame, *, horizon, model_root: ModelForecast(
            "AVAILABLE", f"model2_swing:{requested_asset}", "swing-v1", requested_asset, "SWING", timeframe,
            str(horizon), "LONG", 80.0, 0.02, -0.01, 0.03, "MEDIUM", "2026-01-01T00:00:00Z",
            "registered swing", horizon=str(horizon), asset_class="equity", data_quality="AVAILABLE", confidence="MEDIUM", expected_duration=5.0,
        ),
    )

    forecast = model_integration._infer_registered_forecast(
        asset, "1d", _frame(), horizon="5 trading days", category="SWING", model_root=None, chronos_reason="",
    )

    assert forecast.asset == asset
    assert forecast.horizon == "5 trading days"
    assert forecast.model_id == f"model2_swing:{asset}"
    assert forecast.as_dict()["expected_duration"] == 5.0
