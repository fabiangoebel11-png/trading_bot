from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd

import live_daemon
import state_db
from core.ml.assistant_forecasts import AssistantForecast, PUBLIC_HORIZONS, combine_model_forecasts, direct_swing_forecasts, direct_tcn_forecasts
from core.ml.providers import AssetSpec
from model_integration import ModelForecast


def _tcn_output() -> dict:
    return {
        "probabilities": (0.2, 0.1, 0.7), "expected_return": 0.01, "expected_mfe": 0.02,
        "expected_mae": -0.01, "expected_duration": 4.0, "opportunity_score": 65.0,
        "direction": "LONG",
    }


def _chronos_forecast(score: float) -> AssistantForecast:
    now = datetime.now(timezone.utc).isoformat()
    return AssistantForecast(
        asset="BTC/USDT", asset_class="crypto", timeframe="1h", horizon="4h",
        forecast_source="CHRONOS_2", model="chronos2", model_version="v1", direction="LONG",
        probability_short=None, probability_neutral=None, probability_long=None,
        expected_return=0.01, expected_mfe=None, expected_mae=None, expected_duration=None,
        opportunity_score=score, confidence="MEDIUM", forecast_timestamp=now, target_timestamp=None,
        data_quality="AVAILABLE", forecast_status="PARTIALLY_VALIDATED", quality_status="PARTIALLY_VALIDATED",
        usable_for_decision=False, freshness="FRESH", forecast_age_minutes=0.0,
        p10_price=99.0, p50_price=101.0, p90_price=103.0,
    )


def test_only_strong_two_model_agreement_creates_crypto_paper_signal() -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    tcn = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn_crypto_1h", model_version="v1",
        timestamp=timestamp, outputs={4: _tcn_output()},
    )[1]
    weak = live_daemon._require_chronos_directional_edge(
        combine_model_forecasts([tcn, _chronos_forecast(50.5)])
    )
    strong = live_daemon._require_chronos_directional_edge(
        combine_model_forecasts([tcn, _chronos_forecast(live_daemon.MIN_CHRONOS_SIGNAL_SCORE)])
    )

    assert weak.forecast_status == "DEGRADED"
    assert not weak.usable_for_decision
    assert live_daemon._crypto_signal_from_consensus([weak], datetime.now(timezone.utc))["direction"] == "UNCERTAIN"
    assert strong.forecast_status == "MODEL_AGREEMENT"
    assert strong.usable_for_decision
    signal = live_daemon._crypto_signal_from_consensus([strong], datetime.now(timezone.utc))
    assert signal["direction"] == "LONG"
    assert signal["score"] >= live_daemon.MIN_CHRONOS_SIGNAL_SCORE


def test_equity_intraday_signal_uses_both_consensus_components() -> None:
    now = datetime.now(timezone.utc)
    forecast = SimpleNamespace(
        horizon="4h", forecast_status="MODEL_AGREEMENT", usable_for_decision=True,
        direction="LONG", forecast_timestamp=now.isoformat(), expected_return=0.02,
        expected_duration=None, expected_mfe=None, expected_mae=None, model_version="chronos-v1",
        individual_forecasts=(
            {"model": "RULE_STRATEGY", "direction": "LONG", "score": 88.0},
            {"model": "chronos2", "direction": "LONG", "opportunity_score": 72.0},
        ),
    )

    signal = live_daemon._equity_intraday_signal_from_consensus([forecast], now)

    assert signal["direction"] == "LONG"
    assert signal["score"] == 72.0
    assert signal["rule_score"] == 88.0
    assert signal["expected_duration"] == 4.0


def test_complete_four_asset_matrix_persists_all_public_horizons(tmp_path) -> None:
    database = tmp_path / "state.db"
    state_db.init_db(database)
    crypto = direct_tcn_forecasts(asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1", timestamp="2026-01-01T00:00:00+00:00", outputs={1: _tcn_output(), 4: _tcn_output(), 8: _tcn_output(), 12: _tcn_output(), 24: _tcn_output()})
    equity = direct_swing_forecasts(asset="SPY", model="swing", model_version="v1", timestamp="2026-01-01T00:00:00+00:00", outputs={day: {"expected_return": 0.01, "expected_mfe": 0.02, "expected_mae": -0.01, "expected_duration": float(day), "opportunity_score": 60.0} for day in (1, 3, 5, 10, 20)})
    with state_db.connect(database) as connection:
        for asset, forecasts in (("BTC/USDT", crypto), ("ETH/USDT", [item.__class__(**{**item.as_dict(), "asset": "ETH/USDT"}) for item in crypto]), ("SPY", equity), ("QQQ", [item.__class__(**{**item.as_dict(), "asset": "QQQ"}) for item in equity])):
            state_db.replace_assistant_forecasts(connection, asset, [item.as_dict() for item in forecasts])
        rows = connection.execute("SELECT asset, horizon FROM assistant_forecasts").fetchall()

    assert len(rows) == 36
    for asset in ("BTC/USDT", "ETH/USDT", "SPY", "QQQ"):
        assert {row["horizon"] for row in rows if row["asset"] == asset} == set(PUBLIC_HORIZONS)


def test_swing_forecast_is_returned_below_alert_threshold(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "spy_swing.pt"
    checkpoint.touch()
    timestamps = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    dataset = SimpleNamespace(
        timestamps=timestamps,
        X_daily=np.zeros((2, 3), dtype=np.float32), X_entry=np.zeros((2, 3), dtype=np.float32),
        branch_mask=np.zeros(2, dtype=np.float32), metadata={"daily_data_end": timestamps[-1].isoformat()},
    )

    class FakeModel:
        def predict(self, *_args):
            return {
                "returns": np.full((1, 5), 0.001), "mfe": np.array([0.002]), "mae": np.array([-0.002]),
                "entry": np.array([-1.0]), "duration": np.full((1, 5), 1.0),
            }

    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    daemon.swing_cfg = SimpleNamespace(swing=SimpleNamespace(model_dir=str(tmp_path), market_cache_dir="unused", daily_lookback=90, target_horizons=[1, 3, 5, 10, 20], macro_symbols=[], swing_score_weight=0.7, entry_score_weight=0.3))
    daemon._swing_models = {}
    daemon._notified_swing = set()
    daemon._pending_notifications = []
    daemon._write_snapshot = lambda *_args: None
    daemon._persist_strategy_profiles = lambda *_args: None
    monkeypatch.setattr(live_daemon, "build_swing_dataset", lambda *_args: dataset)
    monkeypatch.setattr(live_daemon, "load_swing_model", lambda *_args: FakeModel())

    result = daemon._refresh_swing(SimpleNamespace(), "SPY", pd.Timestamp("2026-01-02", tz="UTC").to_pydatetime())

    assert result is not None
    _, forecasts = result
    assert set(forecasts) == {1, 3, 5, 10, 20}


def test_live_asset_specs_are_bounded_by_required_warmup() -> None:
    crypto_config = SimpleNamespace(ml=SimpleNamespace(
        assets=["BTC/USDT"], base_timeframe="1h", higher_timeframes=["4h", "1d"],
        sequence_length=128, market_assets={},
    ))
    crypto_specs = [
        AssetSpec("BTC/USDT", "BTC/USDT", "ccxt", "1h", 3650),
        AssetSpec("BTC/USDT", "BTC/USDT", "ccxt", "4h", 5475),
        AssetSpec("BTC/USDT", "BTC/USDT", "ccxt", "1d", 14600),
    ]

    bounded_crypto = live_daemon._bounded_live_specs(crypto_config, crypto_specs)

    assert {spec.timeframe: spec.history_days for spec in bounded_crypto} == {
        "1h": 62, "4h": 84, "1d": 500,
    }
    assert live_daemon._base_candle_limit(crypto_config.ml, "BTC/USDT") == 1472

    equity_config = SimpleNamespace(ml=SimpleNamespace(
        assets=["NASDAQ100_PROXY"], base_timeframe="1h", higher_timeframes=["4h", "1d"],
        sequence_length=128, market_assets={"NASDAQ100_PROXY": {"derive_from": "5m"}},
    ))
    equity_specs = [
        AssetSpec("NASDAQ100_PROXY", "QQQ", "twelve_data", "5m", 365, session_timezone="America/New_York"),
        AssetSpec("NASDAQ100_PROXY", "QQQ", "local_derived", "1h", 365, session_timezone="America/New_York"),
        AssetSpec("NASDAQ100_PROXY", "QQQ", "yfinance", "1d", 14600, session_timezone="America/New_York"),
    ]

    bounded_equity = live_daemon._bounded_live_specs(equity_config, equity_specs)

    assert {spec.timeframe: spec.history_days for spec in bounded_equity} == {
        "5m": 108, "1h": 108, "1d": 700,
    }
    assert all(spec.history_start is None for spec in bounded_equity)


def test_base_ohlc_is_closed_utc_and_capped(monkeypatch, tmp_path) -> None:
    now = pd.Timestamp.now(tz="UTC").floor("h")
    index = pd.date_range(end=now - pd.Timedelta(hours=1), periods=2000, freq="h")
    frame = pd.DataFrame({column: np.ones(len(index)) for column in ("open", "high", "low", "close", "volume")}, index=index)
    path = tmp_path / "base.parquet"
    path.touch()
    spec = SimpleNamespace(name="BTC/USDT", timeframe="1h")
    config = SimpleNamespace(
        base_timeframe="1h", higher_timeframes=["1d"], sequence_length=128,
        market_cache_dir="unused",
    )
    monkeypatch.setattr(live_daemon, "asset_specs_from_config", lambda _config: [spec])
    monkeypatch.setattr(live_daemon, "canonical_cache_path", lambda *_args: path)
    monkeypatch.setattr(live_daemon, "load_ohlcv_parquet", lambda _path: frame)

    loaded = live_daemon._load_base_ohlc(config, "BTC/USDT")

    assert loaded is not None
    assert len(loaded) == 1472
    assert str(loaded.index.tz) == "UTC"
    assert loaded.index[-1] == index[-1]
    assert live_daemon._closed_candle_timestamp(
        pd.DataFrame(
            {"close": [100.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2026-09-26T08:00:00")]),
        ),
        "1h",
    ) == "2026-09-26T09:00:00+00:00"


def test_chronos_uses_the_same_utc_close_timestamp_as_tcn(monkeypatch) -> None:
    old_naive_timestamp = "2026-09-26T08:00:00"
    output = ModelForecast(
        status="AVAILABLE", model_id="chronos2", model_version="v1", asset="BTC/USDT",
        category="INTRADAY", input_timeframe="1h", forecast_horizon="1h", direction="LONG",
        model_score=65.0, expected_return=0.01, expected_adverse_move=-0.01,
        expected_favorable_move=0.02, uncertainty="MEDIUM", data_timestamp=old_naive_timestamp,
        reason="", forecast_low=99.0, forecast_median=101.0, forecast_high=103.0,
    )
    source_timestamp = "2026-09-26T09:00:00+00:00"
    received_timestamps = []
    monkeypatch.setattr(live_daemon, "infer_chronos2_forecast", lambda *_args, **_kwargs: output)
    monkeypatch.setattr(
        live_daemon,
        "chronos2_forecast",
        lambda **kwargs: received_timestamps.append(kwargs["output"].data_timestamp) or kwargs["horizon"],
    )

    forecasts = live_daemon.LiveDaemon._chronos_forecasts(
        "BTC/USDT", "crypto", pd.DataFrame({"close": [100.0]}), source_timestamp,
    )

    assert set(forecasts) == set(PUBLIC_HORIZONS)
    assert received_timestamps == [source_timestamp] * len(PUBLIC_HORIZONS)


def test_model1_runner_moves_loaded_tcn_to_cuda_when_available(monkeypatch) -> None:
    device = SimpleNamespace(type="cuda")

    class FakeModule:
        def __init__(self):
            self.moved_to = None

        def to(self, target):
            self.moved_to = target
            return self

    class FakeModel:
        def __init__(self):
            self.model = FakeModule()
            self.config = SimpleNamespace(use_amp=True)
            self.device = SimpleNamespace(type="cpu")
            self.use_amp = False

        def to_cpu(self):
            self.device = SimpleNamespace(type="cpu")
            self.use_amp = False

    model = FakeModel()
    monkeypatch.setattr(live_daemon, "load_symbol_model", lambda *_args: (model, {}))
    monkeypatch.setattr(live_daemon, "get_device", lambda: device)
    runner = live_daemon.Model1Runner(SimpleNamespace(context_required=False))

    loaded = runner._model("BTC/USDT")

    assert loaded is not None
    assert model.model.moved_to is device
    assert model.device is device
    assert model.use_amp