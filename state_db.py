"""Shared SQLite schema + connection helper for the live/paper-trading runtime.

This is the ONLY point of contact between the four independent runtime
components (``live_daemon.py``, ``risk_engine.py`` is pure/no I/O,
``paper_broker.py``, ``app.py``): every process opens its own short-lived
connection to the same file, so the GUI/broker/daemon can crash and restart
independently without any shared in-memory state. SQLite's file-level locking
(``PRAGMA journal_mode=WAL``) makes concurrent readers (GUI) and a single
writer (daemon/broker) safe without a separate server process.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
DB_PATH = _PROJECT_ROOT / "data" / "trading_state.db"

SCHEMA = """
-- One row per traded asset: the latest known market/session status.
CREATE TABLE IF NOT EXISTS market_state (
    asset TEXT PRIMARY KEY,
    asset_class TEXT NOT NULL,           -- 'crypto' | 'equity'
    timestamp TEXT NOT NULL,             -- ISO8601 UTC, timestamp of last processed candle
    market_open INTEGER NOT NULL,
    session_type TEXT NOT NULL,          -- REGULAR | PRE_MARKET | AFTER_HOURS | CLOSED | 24_7
    is_trading_day INTEGER NOT NULL,
    is_holiday INTEGER NOT NULL,
    session_progress REAL,
    last_price REAL,
    atr REAL,
    atr_pct REAL,
    funding_rate REAL,                   -- crypto only, latest 8h funding rate (fraction)
    data_age_seconds REAL,
    feed_healthy INTEGER NOT NULL,
    freshness_ok INTEGER NOT NULL,
    warmup_ready INTEGER NOT NULL,
    reason TEXT,
    updated_at TEXT NOT NULL
);

-- One row per (asset, model). Overwritten every daemon cycle -- this is a
-- "current state" table, not a time series (see equity_curve for history).
CREATE TABLE IF NOT EXISTS signals (
    asset TEXT NOT NULL,
    model_type TEXT NOT NULL,            -- 'crypto_sniper' | 'intraday' | 'swing'
    timestamp TEXT NOT NULL,
    score REAL NOT NULL,
    direction TEXT NOT NULL,
    expected_return REAL NOT NULL,
    expected_duration_bars REAL NOT NULL,
    expected_duration_days REAL,
    expected_mfe REAL NOT NULL,
    expected_mae REAL NOT NULL,
    swing_score REAL,
    entry_score REAL,
    swing_opportunity_score REAL,
    alert_allowed INTEGER NOT NULL DEFAULT 0,
    alert_reason TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (asset, model_type)
);

-- Latest always-available what-if risk profiles. JSON keeps the risk engine's
-- pure output extensible without coupling the GUI schema to every parameter.
CREATE TABLE IF NOT EXISTS strategy_state (
    asset TEXT PRIMARY KEY,
    asset_class TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL,
    aggressive_json TEXT NOT NULL,
    conservative_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Singleton row (id = 1): the paper wallet's current cash/equity snapshot.
CREATE TABLE IF NOT EXISTS paper_portfolio (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    initial_capital_eur REAL NOT NULL,
    cash_eur REAL NOT NULL,
    equity_eur REAL NOT NULL,
    realized_pnl_eur REAL NOT NULL,
    fees_paid_eur REAL NOT NULL,
    funding_paid_eur REAL NOT NULL,
    open_trade_count INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

-- Currently open simulated positions. Survives daemon/broker restarts --
-- this table (not process memory) is the single source of truth.
CREATE TABLE IF NOT EXISTS open_trades (
    trade_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    direction TEXT NOT NULL,              -- LONG | SHORT
    model_source TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    quantity REAL NOT NULL,
    notional_eur REAL NOT NULL,
    margin_eur REAL NOT NULL,
    leverage REAL NOT NULL,
    stop_loss_price REAL NOT NULL,
    take_profit_price REAL NOT NULL,
    knockout_barrier_price REAL NOT NULL,
    initial_stop_loss_price REAL NOT NULL,
    current_price REAL NOT NULL,
    unrealized_pnl_eur REAL NOT NULL DEFAULT 0.0,
    entry_score REAL,
    swing_score REAL,
    expected_mfe REAL,
    expected_mae REAL,
    expected_duration_bars REAL,
    opened_reason TEXT NOT NULL,
    last_funding_time TEXT,
    entry_fees_eur REAL NOT NULL DEFAULT 0.0
);

-- Closed trades, append-only.
CREATE TABLE IF NOT EXISTS trade_history (
    trade_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    direction TEXT NOT NULL,
    model_source TEXT NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    exit_price REAL NOT NULL,
    exit_time TEXT NOT NULL,
    quantity REAL NOT NULL,
    notional_eur REAL NOT NULL,
    leverage REAL NOT NULL,
    realized_pnl_eur REAL NOT NULL,
    fees_eur REAL NOT NULL,
    funding_eur REAL NOT NULL,
    exit_reason TEXT NOT NULL,
    opened_reason TEXT NOT NULL
);

-- Time series of total paper equity, for the GUI's equity curve.
CREATE TABLE IF NOT EXISTS equity_curve (
    timestamp TEXT PRIMARY KEY,
    equity_eur REAL NOT NULL,
    cash_eur REAL NOT NULL,
    open_positions INTEGER NOT NULL
);
"""


def init_db(db_path: Path | str = DB_PATH) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def connect(db_path: Path | str = DB_PATH):
    """Short-lived WAL connection with commit/rollback semantics.

    A KeyboardInterrupt or any other exception rolls back the current
    transaction before the connection is closed, so interrupted writers never
    leave a partial transaction open.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()
    finally:
        conn.close()


def ensure_portfolio(db_path: Path | str, initial_capital_eur: float) -> None:
    """Create the singleton portfolio row only if it doesn't exist yet --
    never resets an existing wallet on daemon/broker restart."""
    with connect(db_path) as conn:
        row = conn.execute("SELECT id FROM paper_portfolio WHERE id = 1").fetchone()
        if row is None:
            from datetime import datetime, timezone

            conn.execute(
                "INSERT INTO paper_portfolio (id, initial_capital_eur, cash_eur, equity_eur, "
                "realized_pnl_eur, fees_paid_eur, funding_paid_eur, open_trade_count, updated_at) "
                "VALUES (1, ?, ?, ?, 0.0, 0.0, 0.0, 0, ?)",
                (initial_capital_eur, initial_capital_eur, initial_capital_eur, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()


if __name__ == "__main__":
    init_db()
    print(f"Initialized schema at {DB_PATH}")
