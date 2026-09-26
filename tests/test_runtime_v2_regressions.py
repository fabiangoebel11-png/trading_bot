from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import live_daemon
import state_db
from core.config import DataConfig
from core.ml.assistant_forecasts import PUBLIC_HORIZONS, unavailable_forecast
from core.ml.features import build_feature_matrix, fetch_breadth_basket
from core.ml.train import load_prepared_macro_matrix
from core.ml.providers import asset_specs_from_config, canonical_cache_path
from core.ml.data import load_ohlcv_parquet
from core.config import TradingBotConfig
from train import load_config
from dataclasses import replace


def _crypto_runtime_features():
    cfg = load_config("configs/training_tcn_crypto_1h.yaml")
    macro = load_prepared_macro_matrix(cfg.ml)
    breadth = fetch_breadth_basket(cfg.data, cfg.ml, use_cache=True)
    asset = cfg.ml.assets[0]
    specs = [
        s for s in asset_specs_from_config(TradingBotConfig(ml=cfg.ml))
        if s.name == asset and s.timeframe == cfg.ml.base_timeframe
    ]
    ohlc = load_ohlcv_parquet(canonical_cache_path(cfg.ml.market_cache_dir, specs[0]))
    return cfg, macro, breadth, ohlc


def test_crypto_runtime_breadth_corr_is_finite() -> None:
    cfg, macro, breadth, ohlc = _crypto_runtime_features()
    features = build_feature_matrix(ohlc, macro, cfg.ml, breadth)
    assert np.isfinite(features["breadth_corr"].to_numpy()).all()

    window = features.iloc[-128:]
    assert np.isfinite(window.to_numpy()).all(), "final 128-bar TCN window must be finite or be rejected before inference"


def test_stale_breadth_cache_falls_back_to_neutral_values(monkeypatch, tmp_path) -> None:
    cfg = load_config("configs/training_tcn_crypto_1h.yaml")
    cfg.data.cache_dir = str(tmp_path)
    stale_index = pd.date_range("2026-09-01T00:00:00Z", periods=10, freq="h")
    stale_df = pd.DataFrame({"breadth_return": np.linspace(0.0, 0.1, len(stale_index))}, index=stale_index)
    cache_path = tmp_path / "ml_breadth_BNB-USDT_XRP-USDT_ADA-USDT_DOGE-USDT_LINK-USDT_1h_2500d.csv"
    stale_df.to_csv(cache_path)

    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: stale_df)
    result = fetch_breadth_basket(cfg.data, cfg.ml, use_cache=True)

    assert isinstance(result, pd.Series)
    assert len(result) > 0
    assert np.isfinite(result.to_numpy()).all()
    assert result.iloc[-1] == result.iloc[-1]


def test_model1runner_accepts_legacy_1h_equity_metadata(monkeypatch) -> None:
    cfg = load_config("configs/training_tcn_equity_1h.yaml")
    runner = live_daemon.Model1Runner(cfg.ml)

    def fake_load_symbol_model(symbol: str, config):
        return object(), {
            "feature_columns": ["feature"],
            "sequence_length": 1,
            "timeframes": ["1h", "4h"],
            "model_version": "legacy-equity",
        }

    monkeypatch.setattr(live_daemon, "load_symbol_model", fake_load_symbol_model)

    loaded = runner._model("SP500_PROXY")
    assert loaded is not None


def test_model1runner_rejects_true_equity_timeframe_mismatch(monkeypatch) -> None:
    cfg = load_config("configs/training_tcn_equity_1h.yaml")
    runner = live_daemon.Model1Runner(cfg.ml)

    def fake_load_symbol_model(symbol: str, config):
        return object(), {
            "feature_columns": ["feature"],
            "sequence_length": 1,
            "timeframes": ["5m", "15m"],
            "model_version": "bad-equity",
        }

    monkeypatch.setattr(live_daemon, "load_symbol_model", fake_load_symbol_model)

    with pytest.raises(ValueError, match="timeframe"):
        runner._model("SP500_PROXY")


def test_partial_stale_forecasts_keep_runtime_partial(monkeypatch, tmp_path) -> None:
    database = tmp_path / "partial_runtime.db"
    monkeypatch.setattr(state_db, "DB_PATH", database)
    state_db.init_db(database)
    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    daemon.market_states = live_daemon.MarketStateService()
    daemon._startup_freshness = lambda forecast_rows, now: (["QQQ 1h: source session 2024-08-31 != 2026-09-24"], set())

    with state_db.connect(database) as connection:
        for asset, asset_class in (("BTC/USDT", "crypto"), ("ETH/USDT", "crypto"), ("SPY", "equity"), ("QQQ", "equity")):
            forecasts = [
                replace(unavailable_forecast(asset, asset_class, horizon, reason="test source unavailable"), forecast_timestamp=pd.Timestamp.now(tz="UTC").isoformat())
                for horizon in PUBLIC_HORIZONS
            ]
            state_db.replace_assistant_forecasts(connection, asset, [item.as_dict() for item in forecasts])

    assert daemon._refresh_runtime_status() == "PARTIAL_DEGRADATION"
    with state_db.connect(database) as connection:
        row = connection.execute("SELECT status, detail FROM runtime_status WHERE id = 1").fetchone()
    assert row["status"] == "PARTIAL_DEGRADATION"
    assert "QQQ" in row["detail"]


def test_daily_equity_forecasts_do_not_mark_same_session_as_stale(monkeypatch) -> None:
    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    daemon.market_states = type("MarketStates", (), {"state_at": lambda self, asset, now: type("Market", (), {"session_type": "REGULAR"})()})()
    monkeypatch.setattr(daemon, "_latest_equity_session_date", lambda now: pd.Timestamp("2026-09-24").date())

    rows = [{
        "asset": "QQQ",
        "horizon": "5d",
        "timeframe": "1d",
        "forecast_timestamp": pd.Timestamp("2026-09-24 16:00:00Z").isoformat(),
    }]

    stale, market_closed = daemon._startup_freshness(rows, pd.Timestamp("2026-09-25 00:00:00Z"))

    assert stale == []
    assert market_closed == set()
