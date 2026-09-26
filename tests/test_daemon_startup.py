from __future__ import annotations

from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import live_daemon
import pandas as pd
import pytest
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


def test_closed_equity_price_tick_preserves_mark_but_marks_it_stale(tmp_path) -> None:
    database = tmp_path / "closed_market.db"
    state_db.init_db(database)
    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    daemon.market_states = MarketStateService()
    now = datetime(2026, 9, 26, 16, 0, tzinfo=timezone.utc)
    prior_quote = datetime(2026, 9, 25, 19, 55, tzinfo=timezone.utc)
    market = daemon.market_states.state_at("SPY", now)

    with state_db.connect(database) as connection:
        connection.execute(
            "INSERT INTO market_state (asset, asset_class, timestamp, market_open, session_type, is_trading_day, "
            "is_holiday, session_progress, last_price, atr, atr_pct, funding_rate, data_age_seconds, feed_healthy, "
            "freshness_ok, warmup_ready, reason, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            ("SPY", "equity", prior_quote.isoformat(), 1, "REGULAR", 1, 0, 1.0, 500.0, 2.0, 0.004, None, 0.0, 1, 1, 1, "", prior_quote.isoformat()),
        )
        daemon._upsert_market_state(connection, "SPY", "equity", 501.0, now, market_snapshot=market)
        row = connection.execute(
            "SELECT timestamp, market_open, last_price, feed_healthy, freshness_ok, data_age_seconds, reason "
            "FROM market_state WHERE asset='SPY'"
        ).fetchone()

    assert row["market_open"] == 0
    assert row["last_price"] == 500.0
    assert row["timestamp"] == prior_quote.isoformat()
    assert row["feed_healthy"] == 0 and row["freshness_ok"] == 0
    assert row["data_age_seconds"] >= 20 * 60 * 60
    assert "MARKET_CLOSED" in row["reason"]


def test_eur_usd_refresh_is_hourly_and_retains_last_success_on_failure(monkeypatch, tmp_path) -> None:
    database = tmp_path / "fx_state.db"
    monkeypatch.setattr(state_db, "DB_PATH", database)
    state_db.init_db(database)
    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    attempts = iter([1.08, None])
    calls = []

    def fetch_rate():
        calls.append(True)
        return next(attempts)

    daemon._fetch_eur_usd_rate = fetch_rate
    first_attempt = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
    with state_db.connect(database) as connection:
        daemon._refresh_eur_usd_rate(connection, first_attempt)
        daemon._refresh_eur_usd_rate(connection, first_attempt + timedelta(minutes=30))

    assert len(calls) == 1
    with state_db.connect(database) as connection:
        daemon._refresh_eur_usd_rate(connection, first_attempt + timedelta(hours=1))
        daemon._refresh_eur_usd_rate(connection, first_attempt + timedelta(hours=1, minutes=30))
        row = connection.execute(
            "SELECT timestamp, last_price, data_age_seconds, freshness_ok, warmup_ready, reason, updated_at "
            "FROM market_state WHERE asset = ?",
            (live_daemon.FX_ASSET,),
        ).fetchone()

    assert len(calls) == 2
    assert row["last_price"] == 1.08
    assert row["timestamp"] == first_attempt.isoformat()
    assert row["data_age_seconds"] == pytest.approx(3600.0)
    assert row["freshness_ok"] == 0
    assert row["warmup_ready"] == 1
    assert "retaining the last successful quote" in row["reason"]
    assert row["updated_at"] == (first_attempt + timedelta(hours=1)).isoformat()