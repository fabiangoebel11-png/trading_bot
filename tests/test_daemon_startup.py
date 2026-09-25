from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time
from zoneinfo import ZoneInfo

import live_daemon
import pandas as pd
import state_db
from core.ml.assistant_forecasts import PUBLIC_HORIZONS, unavailable_forecast
from core.market_state import MarketStateService


def _latest_equity_session_anchor(now_utc: pd.Timestamp) -> pd.Timestamp:
    service = MarketStateService()
    ny = ZoneInfo("America/New_York")
    local = now_utc.tz_convert(ny)
    market = service.state_at("SPY", now_utc.to_pydatetime())
    if market.session_type in {"REGULAR", "AFTER_HOURS"}:
        return pd.Timestamp(datetime.combine(local.date(), time(12, 0)).replace(tzinfo=ny))
    candidate = local.date()
    if market.is_trading_day and local.time() >= time(16, 0):
        return pd.Timestamp(datetime.combine(candidate, time(12, 0)).replace(tzinfo=ny))
    candidate -= pd.Timedelta(days=1)
    while not service.state_at("SPY", datetime.combine(candidate, time(12, 0), tzinfo=ny)).is_trading_day:
        candidate -= pd.Timedelta(days=1)
    return pd.Timestamp(datetime.combine(candidate, time(12, 0)).replace(tzinfo=ny))


def _write_complete_matrices(database) -> None:
    now = pd.Timestamp.now(tz="UTC")
    with state_db.connect(database) as connection:
        for asset, asset_class in (("BTC/USDT", "crypto"), ("ETH/USDT", "crypto"), ("SPY", "equity"), ("QQQ", "equity")):
            timestamp = now if asset_class == "crypto" else _latest_equity_session_anchor(now).tz_convert("UTC")
            forecasts = [replace(unavailable_forecast(asset, asset_class, horizon, reason="test startup source unavailable"), forecast_timestamp=timestamp.isoformat()) for horizon in PUBLIC_HORIZONS]
            state_db.replace_assistant_forecasts(connection, asset, [forecast.as_dict() for forecast in forecasts])


def test_startup_marks_ready_only_after_all_forecast_matrices_exist(monkeypatch, tmp_path) -> None:
    database = tmp_path / "startup.db"
    monkeypatch.setattr(state_db, "DB_PATH", database)
    state_db.init_db(database)
    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    daemon.market_states = MarketStateService()
    daemon._last_signal_refresh = 0.0
    daemon.price_tick = lambda: None
    daemon.signal_refresh = lambda *, force_refresh: _write_complete_matrices(database)

    assert daemon.initialize() == "READY"
    with state_db.connect(database) as connection:
        assert connection.execute("SELECT status FROM runtime_status WHERE id = 1").fetchone()["status"] == "READY"
        assert connection.execute("SELECT COUNT(*) FROM assistant_forecasts").fetchone()[0] == 36


def test_startup_marks_degraded_when_a_matrix_is_missing(monkeypatch, tmp_path) -> None:
    database = tmp_path / "startup.db"
    monkeypatch.setattr(state_db, "DB_PATH", database)
    state_db.init_db(database)
    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    daemon.market_states = MarketStateService()
    daemon.price_tick = lambda: None
    daemon.signal_refresh = lambda *, force_refresh: None

    assert daemon.initialize() == "DEGRADED"
    with state_db.connect(database) as connection:
        row = connection.execute("SELECT status, detail FROM runtime_status WHERE id = 1").fetchone()
    assert row["status"] == "DEGRADED"
    assert "BTC/USDT" in row["detail"]


def test_after_close_equity_session_is_valid_but_stale_crypto_is_not() -> None:
    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    daemon.market_states = MarketStateService()
    now = pd.Timestamp("2026-09-24T20:21:00+00:00")  # 16:21 New York, after regular close.
    rows = [
        {"asset": "BTC/USDT", "horizon": "1h", "timeframe": "1h", "forecast_timestamp": "2026-09-22T18:00:00+00:00"},
        {"asset": "ETH/USDT", "horizon": "1h", "timeframe": "1h", "forecast_timestamp": "2026-09-22T18:00:00+00:00"},
        {"asset": "SPY", "horizon": "1h", "timeframe": "1h", "forecast_timestamp": "2026-09-24T20:00:00+00:00"},
        {"asset": "QQQ", "horizon": "1d", "timeframe": "1d", "forecast_timestamp": "2026-09-24T20:00:00+00:00"},
    ]

    stale, market_closed = daemon._startup_freshness(rows, now)

    assert {"SPY", "QQQ"} == market_closed
    assert len(stale) == 2
    assert all("USDT" in item for item in stale)