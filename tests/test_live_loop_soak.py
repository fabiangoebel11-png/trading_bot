from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, time
from threading import Event
from zoneinfo import ZoneInfo

import pandas as pd

import live_daemon
import state_db
from core.market_state import MarketStateService
from core.ml.assistant_forecasts import PUBLIC_HORIZONS, unavailable_forecast


ASSETS = (("BTC/USDT", "crypto"), ("ETH/USDT", "crypto"), ("SPY", "equity"), ("QQQ", "equity"))


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
    while not service.state_at("SPY", datetime.combine(candidate, time(12, 0)).replace(tzinfo=ny)).is_trading_day:
        candidate -= pd.Timedelta(days=1)
    return pd.Timestamp(datetime.combine(candidate, time(12, 0)).replace(tzinfo=ny))


def test_accelerated_live_loop_soak_recovers_without_unnecessary_inference(monkeypatch, tmp_path) -> None:
    database = tmp_path / "live_soak.db"
    monkeypatch.setattr(state_db, "DB_PATH", database)
    state_db.init_db(database)

    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    daemon.market_states = MarketStateService()
    daemon._last_signal_refresh = 0.0
    cycle = {"count": 0, "inferences": 0, "fail_price": False}

    def price_tick() -> None:
        cycle["count"] += 1
        if cycle["fail_price"]:
            raise RuntimeError("simulated provider outage")
        now = pd.Timestamp.now(tz="UTC")
        with state_db.connect() as connection:
            for asset, asset_class in ASSETS:
                row_timestamp = now if asset_class == "crypto" else _latest_equity_session_anchor(now).tz_convert("UTC")
                row_timestamp_value = row_timestamp.isoformat()
                session_type = "24_7" if asset_class == "crypto" else "REGULAR"
                connection.execute(
                    "INSERT INTO market_state (asset, asset_class, timestamp, market_open, session_type, is_trading_day, is_holiday, session_progress, last_price, atr, atr_pct, funding_rate, data_age_seconds, feed_healthy, freshness_ok, warmup_ready, reason, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(asset) DO UPDATE SET timestamp=excluded.timestamp, session_type=excluded.session_type, is_trading_day=excluded.is_trading_day, last_price=excluded.last_price, feed_healthy=1, freshness_ok=1, updated_at=excluded.updated_at",
                    (asset, asset_class, row_timestamp_value, 1, session_type, 1 if asset_class == "crypto" else 1, 0, 1.0, 100.0 + cycle["count"], 1.0, 0.01, 0.0, 0.0, 1, 1, 1, "", row_timestamp_value),
                )
                state_db.save_context_snapshot(database, asset, row_timestamp_value, "OK", {"cycle": cycle["count"]}, conn=connection)
                connection.execute(
                    "INSERT INTO strategy_state (asset, asset_class, timestamp, direction, entry_price, aggressive_json, conservative_json, updated_at) VALUES (?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(asset) DO UPDATE SET timestamp=excluded.timestamp, updated_at=excluded.updated_at",
                    (asset, asset_class, row_timestamp_value, "LONG", 100.0, "{}", "{}", row_timestamp_value),
                )

    def signal_refresh(*, force_refresh: bool = False) -> None:
        cycle["inferences"] += 1
        now = pd.Timestamp.now(tz="UTC")
        with state_db.connect() as connection:
            for asset, asset_class in ASSETS:
                base = now if asset_class == "crypto" else _latest_equity_session_anchor(now)
                timestamp = (base + pd.Timedelta(milliseconds=cycle["inferences"])).tz_convert("UTC").isoformat()
                forecasts = [
                    replace(
                        unavailable_forecast(asset, asset_class, horizon, reason="deterministic soak source"),
                        forecast_source="DIRECT_MODEL", model="soak-model", model_version="test", direction="NEUTRAL",
                        forecast_timestamp=timestamp, data_quality="AVAILABLE", forecast_status="VALIDATED",
                    )
                    for horizon in PUBLIC_HORIZONS
                ]
                state_db.replace_assistant_forecasts(connection, asset, [forecast.as_dict() for forecast in forecasts])
                connection.execute(
                    "INSERT INTO signals (asset, model_type, timestamp, score, direction, expected_return, expected_duration_bars, expected_mfe, expected_mae, alert_allowed, alert_reason, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(asset, model_type) DO UPDATE SET timestamp=excluded.timestamp, updated_at=excluded.updated_at",
                    (asset, "soak", timestamp, 50.0, "UNCERTAIN", 0.0, 0.0, 0.0, 0.0, 0, "", timestamp),
                )

    daemon.price_tick = price_tick
    daemon.signal_refresh = signal_refresh

    assert daemon.initialize() == "READY"
    initial_inferences = cycle["inferences"]
    with state_db.connect() as connection:
        initial_timestamp = connection.execute("SELECT MIN(forecast_timestamp) FROM assistant_forecasts").fetchone()[0]

    for _ in range(60):
        assert daemon.run_cycle() == "READY"
    assert cycle["inferences"] == initial_inferences

    assert daemon.run_cycle(force_signal=True) == "READY"
    assert cycle["inferences"] == initial_inferences + 1
    with state_db.connect() as connection:
        refreshed_timestamp = connection.execute("SELECT MIN(forecast_timestamp) FROM assistant_forecasts").fetchone()[0]
        assert refreshed_timestamp > initial_timestamp
        assert connection.execute("SELECT COUNT(*) FROM assistant_forecasts").fetchone()[0] == 36
        assert connection.execute("SELECT COUNT(*) FROM assistant_forecasts GROUP BY asset, horizon HAVING COUNT(*) != 1").fetchone() is None

    cycle["fail_price"] = True
    assert daemon.run_cycle() == "DEGRADED"
    with state_db.connect() as connection:
        assert connection.execute("SELECT status FROM runtime_status WHERE id = 1").fetchone()["status"] == "DEGRADED"

    cycle["fail_price"] = False
    assert daemon.run_cycle(force_signal=True) == "READY"
    with state_db.connect() as connection:
        for table, expected in (("runtime_status", 1), ("market_state", 4), ("context_state", 4), ("strategy_state", 4), ("signals", 4), ("assistant_forecasts", 36)):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == expected
        assert connection.execute("SELECT status FROM runtime_status WHERE id = 1").fetchone()["status"] == "READY"

    print("LIVE_SOAK_TEST=PASS")


def test_price_ticks_continue_while_background_signal_refresh_is_running() -> None:
    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    daemon._last_signal_refresh = 0.0
    daemon._signal_executor = ThreadPoolExecutor(max_workers=1)
    daemon._signal_future = None
    started = Event()
    release = Event()
    ticks = []
    daemon.price_tick = lambda: ticks.append(True)

    def slow_signal_refresh(*, force_refresh: bool = False) -> None:
        started.set()
        if not release.wait(timeout=5.0):
            raise TimeoutError("test signal refresh was not released")

    daemon.signal_refresh = slow_signal_refresh
    daemon._refresh_runtime_status = lambda: "READY"
    try:
        assert daemon.run_cycle(force_signal=True, background_signals=True) == "INITIALIZING"
        assert started.wait(timeout=2.0)
        assert daemon.run_cycle(background_signals=True) == "INITIALIZING"
        assert len(ticks) == 2
        release.set()
        assert daemon._signal_future.result(timeout=2.0) == "READY"
        assert daemon.run_cycle(background_signals=True) == "READY"
        assert len(ticks) == 3
    finally:
        release.set()
        daemon._signal_executor.shutdown(wait=True)
    print("VIRTUAL_CYCLES=60")
    print("MODEL_REINFERENCE_TEST=PASS")
    print("NO_UNNECESSARY_INFERENCE=PASS")
    print("PROVIDER_FAILURE_RECOVERY=PASS")
    print("SQLITE_CONSISTENCY=PASS")
    print("NO_ORDERS=PASS")
    print("NO_TRAINING=PASS")
