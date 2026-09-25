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

-- Required context health is separate from primary market state so a stale
-- macro series cannot be mistaken for a fresh target candle.
CREATE TABLE IF NOT EXISTS context_state (
    asset TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
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

-- One explicit public forecast contract per asset and horizon. The daemon
-- rewrites this latest-state matrix after each model refresh.
CREATE TABLE IF NOT EXISTS assistant_forecasts (
    asset TEXT NOT NULL,
    horizon TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    forecast_source TEXT NOT NULL,
    model TEXT,
    model_version TEXT,
    direction TEXT,
    probability_short REAL,
    probability_neutral REAL,
    probability_long REAL,
    expected_return REAL,
    expected_mfe REAL,
    expected_mae REAL,
    expected_duration REAL,
    opportunity_score REAL,
    confidence TEXT NOT NULL,
    forecast_timestamp TEXT,
    target_timestamp TEXT,
    data_quality TEXT NOT NULL,
    forecast_status TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    quality_status TEXT NOT NULL DEFAULT 'UNVERIFIED',
    usable_for_decision INTEGER NOT NULL DEFAULT 0,
    freshness TEXT NOT NULL DEFAULT 'UNKNOWN',
    forecast_age_minutes REAL,
    derived_from TEXT,
    derivation_method TEXT,
    model_sources_json TEXT NOT NULL DEFAULT '[]',
    individual_forecasts_json TEXT NOT NULL DEFAULT '[]',
    combination_method TEXT NOT NULL DEFAULT 'SINGLE_MODEL',
    combined_confidence TEXT NOT NULL DEFAULT 'UNKNOWN',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (asset, horizon)
);

-- Singleton runtime lifecycle state written by the order-free live daemon.
CREATE TABLE IF NOT EXISTS runtime_status (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
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
    ,instrument_type TEXT NOT NULL DEFAULT 'unknown'
    ,protection_model TEXT NOT NULL DEFAULT 'unspecified'
    ,liquidation_price REAL
    ,safety_barrier_price REAL
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

-- Immutable strategy definitions. A changed configuration creates a new
-- (strategy_id, strategy_version) pair instead of mutating a referenced row.
CREATE TABLE IF NOT EXISTS strategy_registry (
    strategy_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    name TEXT NOT NULL,
    asset TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (strategy_id, strategy_version)
);

-- Frozen research manifests are separate from operational strategy state.
CREATE TABLE IF NOT EXISTS experiment_manifests (
    experiment_id TEXT PRIMARY KEY,
    manifest_json TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Append-only explanation of every evaluated signal or action.
CREATE TABLE IF NOT EXISTS decision_records (
    decision_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    asset TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    decision TEXT NOT NULL,
    record_json TEXT NOT NULL,
    record_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- User-entered positions for analysis only. This is deliberately separate
-- from open_trades, which belongs to the optional paper simulator.
CREATE TABLE IF NOT EXISTS assistant_positions (
    position_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    asset_class TEXT NOT NULL,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    quantity REAL NOT NULL,
    margin REAL NOT NULL,
    stop_price REAL NOT NULL,
    take_profit_price REAL NOT NULL,
    opened_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN',
    notes TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);

-- Human-readable ideas and their immutable calculation payload.
CREATE TABLE IF NOT EXISTS assistant_trade_ideas (
    idea_id TEXT PRIMARY KEY,
    asset TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT 'MODEL_NOT_AVAILABLE',
    model_version TEXT NOT NULL DEFAULT 'unknown',
    status TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    input_snapshot_json TEXT NOT NULL DEFAULT '{}',
    recommendation TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS assistant_position_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    strategy_version TEXT NOT NULL DEFAULT 'unknown'
);

-- Deterministic what-if results are retained for later review.
CREATE TABLE IF NOT EXISTS assistant_scenarios (
    scenario_id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    scenario_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS system_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def init_db(db_path: Path | str | None = None) -> None:
    db_path = Path(db_path or DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(open_trades)").fetchall()}
        migrations = {
            "instrument_type": "ALTER TABLE open_trades ADD COLUMN instrument_type TEXT NOT NULL DEFAULT 'unknown'",
            "protection_model": "ALTER TABLE open_trades ADD COLUMN protection_model TEXT NOT NULL DEFAULT 'unspecified'",
            "liquidation_price": "ALTER TABLE open_trades ADD COLUMN liquidation_price REAL",
            "safety_barrier_price": "ALTER TABLE open_trades ADD COLUMN safety_barrier_price REAL",
        }
        for column, statement in migrations.items():
            if column not in columns:
                conn.execute(statement)
        idea_columns = {row[1] for row in conn.execute("PRAGMA table_info(assistant_trade_ideas)").fetchall()}
        idea_migrations = {
            "model": "ALTER TABLE assistant_trade_ideas ADD COLUMN model TEXT NOT NULL DEFAULT 'MODEL_NOT_AVAILABLE'",
            "model_version": "ALTER TABLE assistant_trade_ideas ADD COLUMN model_version TEXT NOT NULL DEFAULT 'unknown'",
            "input_snapshot_json": "ALTER TABLE assistant_trade_ideas ADD COLUMN input_snapshot_json TEXT NOT NULL DEFAULT '{}'",
            "recommendation": "ALTER TABLE assistant_trade_ideas ADD COLUMN recommendation TEXT NOT NULL DEFAULT ''",
        }
        for column, statement in idea_migrations.items():
            if column not in idea_columns:
                conn.execute(statement)
        signal_columns = {row[1] for row in conn.execute("PRAGMA table_info(signals)").fetchall()}
        signal_migrations = {
            "model_version": "ALTER TABLE signals ADD COLUMN model_version TEXT",
            "data_timestamp": "ALTER TABLE signals ADD COLUMN data_timestamp TEXT",
            "rule_score": "ALTER TABLE signals ADD COLUMN rule_score REAL",
            "forecast_score": "ALTER TABLE signals ADD COLUMN forecast_score REAL",
            "trade_quality_score": "ALTER TABLE signals ADD COLUMN trade_quality_score REAL",
            "context_score": "ALTER TABLE signals ADD COLUMN context_score REAL",
            "risk_score": "ALTER TABLE signals ADD COLUMN risk_score REAL",
            "combined_opportunity_score": "ALTER TABLE signals ADD COLUMN combined_opportunity_score REAL",
            "strategy_state": "ALTER TABLE signals ADD COLUMN strategy_state TEXT",
            "context_status": "ALTER TABLE signals ADD COLUMN context_status TEXT",
        }
        for column, statement in signal_migrations.items():
            if column not in signal_columns:
                conn.execute(statement)
        forecast_columns = {row[1] for row in conn.execute("PRAGMA table_info(assistant_forecasts)").fetchall()}
        forecast_migrations = {
            "model_sources_json": "ALTER TABLE assistant_forecasts ADD COLUMN model_sources_json TEXT NOT NULL DEFAULT '[]'",
            "individual_forecasts_json": "ALTER TABLE assistant_forecasts ADD COLUMN individual_forecasts_json TEXT NOT NULL DEFAULT '[]'",
            "combination_method": "ALTER TABLE assistant_forecasts ADD COLUMN combination_method TEXT NOT NULL DEFAULT 'SINGLE_MODEL'",
            "combined_confidence": "ALTER TABLE assistant_forecasts ADD COLUMN combined_confidence TEXT NOT NULL DEFAULT 'UNKNOWN'",
            "quality_status": "ALTER TABLE assistant_forecasts ADD COLUMN quality_status TEXT NOT NULL DEFAULT 'UNVERIFIED'",
            "usable_for_decision": "ALTER TABLE assistant_forecasts ADD COLUMN usable_for_decision INTEGER NOT NULL DEFAULT 0",
            "freshness": "ALTER TABLE assistant_forecasts ADD COLUMN freshness TEXT NOT NULL DEFAULT 'UNKNOWN'",
            "forecast_age_minutes": "ALTER TABLE assistant_forecasts ADD COLUMN forecast_age_minutes REAL",
        }
        for column, statement in forecast_migrations.items():
            if column not in forecast_columns:
                conn.execute(statement)
        conn.commit()


@contextmanager
def connect(db_path: Path | str | None = None):
    """Short-lived WAL connection with commit/rollback semantics.

    A KeyboardInterrupt or any other exception rolls back the current
    transaction before the connection is closed, so interrupted writers never
    leave a partial transaction open.
    """
    db_path = Path(db_path or DB_PATH)
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


def save_context_snapshot(db_path: Path | str, asset: str, timestamp: str, status: str, snapshot: dict, *, conn: sqlite3.Connection | None = None) -> None:
    """Persist the latest context health without changing target state."""
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    statement = (
        "INSERT INTO context_state (asset, timestamp, status, snapshot_json, updated_at) VALUES (?,?,?,?,?) "
        "ON CONFLICT(asset) DO UPDATE SET timestamp=excluded.timestamp, status=excluded.status, "
        "snapshot_json=excluded.snapshot_json, updated_at=excluded.updated_at"
    )
    values = (asset, timestamp, status, json.dumps(snapshot, sort_keys=True, default=str), now)
    if conn is not None:
        conn.execute(statement, values)
        return
    with connect(db_path) as conn:
        conn.execute(statement, values)


def replace_assistant_forecasts(conn: sqlite3.Connection, asset: str, forecasts: list[dict]) -> None:
    """Atomically replace an asset's public forecast matrix in an existing transaction."""
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    columns = (
        "asset", "horizon", "asset_class", "timeframe", "forecast_source", "model", "model_version",
        "direction", "probability_short", "probability_neutral", "probability_long", "expected_return",
        "expected_mfe", "expected_mae", "expected_duration", "opportunity_score", "confidence",
        "forecast_timestamp", "target_timestamp", "data_quality", "forecast_status", "reason",
        "quality_status", "usable_for_decision", "freshness", "forecast_age_minutes", "derived_from",
        "derivation_method", "model_sources_json", "individual_forecasts_json", "combination_method",
        "combined_confidence", "updated_at",
    )
    placeholders = ", ".join("?" for _ in columns)
    statement = f"INSERT INTO assistant_forecasts ({', '.join(columns)}) VALUES ({placeholders})"
    conn.execute("DELETE FROM assistant_forecasts WHERE asset = ?", (asset,))
    conn.executemany(
        statement,
        [
            (
                item["asset"], item["horizon"], item["asset_class"], item["timeframe"], item["forecast_source"],
                item["model"], item["model_version"], item["direction"], item["probability_short"],
                item["probability_neutral"], item["probability_long"], item["expected_return"], item["expected_mfe"],
                item["expected_mae"], item["expected_duration"], item["opportunity_score"], item["confidence"],
                item["forecast_timestamp"], item["target_timestamp"], item["data_quality"], item["forecast_status"],
                item.get("reason", ""), item.get("quality_status", "UNVERIFIED"), int(bool(item.get("usable_for_decision", False))),
                item.get("freshness", "UNKNOWN"), item.get("forecast_age_minutes"), item["derived_from"], item["derivation_method"],
                json.dumps(item.get("model_sources", ())), json.dumps(item.get("individual_forecasts", ())),
                item.get("combination_method", "SINGLE_MODEL"), item.get("combined_confidence", "UNKNOWN"), now,
            )
            for item in forecasts
        ],
    )


def set_runtime_status(status: str, detail: str = "", *, db_path: Path | str | None = None) -> None:
    """Record the daemon lifecycle without conflating stale DB rows with readiness."""
    from datetime import datetime, timezone

    db_path = db_path or DB_PATH
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO runtime_status (id, status, detail, updated_at) VALUES (1,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET status=excluded.status, detail=excluded.detail, updated_at=excluded.updated_at",
            (status, detail, datetime.now(timezone.utc).isoformat()),
        )


def register_strategy(db_path: Path | str, strategy) -> str:
    """Insert an immutable strategy version and return its content hash."""
    from datetime import datetime, timezone

    from core.architecture import content_hash, canonical_json, to_payload

    payload = to_payload(strategy)
    digest = content_hash(payload)
    now = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT content_hash FROM strategy_registry WHERE strategy_id = ? AND strategy_version = ?",
            (strategy.strategy_id, strategy.version),
        ).fetchone()
        if existing is not None:
            if existing["content_hash"] != digest:
                raise ValueError("strategy version already exists with different content")
            return digest
        conn.execute(
            "INSERT INTO strategy_registry (strategy_id, strategy_version, name, asset, status, payload_json, content_hash, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (strategy.strategy_id, strategy.version, strategy.name, strategy.asset, strategy.status.value, canonical_json(payload), digest, now, now),
        )
    return digest


def save_experiment_manifest(db_path: Path | str, manifest) -> str:
    """Insert a frozen experiment manifest; conflicting IDs are rejected."""
    from datetime import datetime, timezone

    from core.architecture import canonical_json, content_hash, to_payload

    payload = to_payload(manifest)
    digest = content_hash(payload)
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT content_hash FROM experiment_manifests WHERE experiment_id = ?",
            (manifest.experiment_id,),
        ).fetchone()
        if existing is not None:
            if existing["content_hash"] != digest:
                raise ValueError("experiment ID already exists with different content")
            return digest
        conn.execute(
            "INSERT INTO experiment_manifests (experiment_id, manifest_json, content_hash, created_at) VALUES (?, ?, ?, ?)",
            (manifest.experiment_id, canonical_json(payload), digest, datetime.now(timezone.utc).isoformat()),
        )
    return digest


def append_decision_record(db_path: Path | str, record) -> str:
    """Append one immutable decision record and reject ID reuse with new content."""
    from datetime import datetime, timezone

    from core.architecture import canonical_json, content_hash, to_payload

    payload = to_payload(record)
    digest = content_hash(payload)
    with connect(db_path) as conn:
        existing = conn.execute(
            "SELECT record_hash FROM decision_records WHERE decision_id = ?",
            (record.decision_id,),
        ).fetchone()
        if existing is not None:
            if existing["record_hash"] != digest:
                raise ValueError("decision ID already exists with different content")
            return digest
        conn.execute(
            "INSERT INTO decision_records (decision_id, timestamp, asset, strategy_id, strategy_version, decision, record_json, record_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (record.decision_id, record.timestamp, record.asset, record.strategy_id, record.strategy_version, record.decision.value, canonical_json(payload), digest, datetime.now(timezone.utc).isoformat()),
        )
    return digest


def save_assistant_position(db_path: Path | str, position: dict) -> None:
    """Upsert a manually entered position; never invokes execution code."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO assistant_positions (position_id, asset, asset_class, direction, entry_price, quantity, margin, "
            "stop_price, take_profit_price, opened_at, status, notes, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(position_id) DO UPDATE SET asset=excluded.asset, asset_class=excluded.asset_class, direction=excluded.direction, "
            "entry_price=excluded.entry_price, quantity=excluded.quantity, margin=excluded.margin, stop_price=excluded.stop_price, "
            "take_profit_price=excluded.take_profit_price, status=excluded.status, notes=excluded.notes, updated_at=excluded.updated_at",
            (
                position["position_id"], position["asset"], position.get("asset_class", "unknown"), position["direction"],
                position["entry_price"], position["quantity"], position["margin"], position["stop_price"],
                position["take_profit_price"], position.get("opened_at", now), position.get("status", "OPEN"), position.get("notes", ""), now,
            ),
        )


def save_assistant_trade_idea(db_path: Path | str, idea: dict) -> None:
    """Persist one immutable decision-support idea; IDs cannot be rewritten."""
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    with connect(db_path) as conn:
        plan_json = json.dumps(idea["plan"], sort_keys=True)
        input_snapshot_json = json.dumps(idea.get("input_snapshot", {}), sort_keys=True)
        existing = conn.execute("SELECT * FROM assistant_trade_ideas WHERE idea_id = ?", (idea["idea_id"],)).fetchone()
        if existing is not None:
            if existing["plan_json"] != plan_json or existing["input_snapshot_json"] != input_snapshot_json:
                raise ValueError("trade idea ID already exists with different content")
            return
        conn.execute(
            "INSERT INTO assistant_trade_ideas (idea_id, asset, strategy_id, strategy_version, model, model_version, status, plan_json, input_snapshot_json, recommendation, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                idea["idea_id"], idea["asset"], idea["strategy_id"], idea["strategy_version"], idea.get("model", "MODEL_NOT_AVAILABLE"),
                idea.get("model_version", "unknown"), idea["status"], plan_json, input_snapshot_json, idea.get("recommendation", ""), idea.get("created_at", now), now,
            ),
        )


def append_assistant_position_snapshot(db_path: Path | str, snapshot_id: str, position_id: str, analysis: object, strategy_version: str = "unknown") -> None:
    """Append a point-in-time position analysis without mutating history."""
    import json
    from dataclasses import asdict, is_dataclass
    from datetime import datetime, timezone

    payload = asdict(analysis) if is_dataclass(analysis) else analysis
    encoded = json.dumps(payload, sort_keys=True)
    with connect(db_path) as conn:
        existing = conn.execute("SELECT analysis_json FROM assistant_position_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        if existing is not None:
            if existing["analysis_json"] != encoded:
                raise ValueError("position snapshot ID already exists with different content")
            return
        conn.execute(
            "INSERT INTO assistant_position_snapshots (snapshot_id, position_id, timestamp, analysis_json, strategy_version) VALUES (?, ?, ?, ?, ?)",
            (snapshot_id, position_id, datetime.now(timezone.utc).isoformat(), encoded, strategy_version),
        )


def append_assistant_scenario(db_path: Path | str, scenario_id: str, subject_id: str, result: object) -> None:
    """Append a scenario result; an ID cannot be silently overwritten."""
    import json
    from dataclasses import asdict, is_dataclass
    from datetime import datetime, timezone

    payload = asdict(result) if is_dataclass(result) else result
    with connect(db_path) as conn:
        existing = conn.execute("SELECT scenario_json FROM assistant_scenarios WHERE scenario_id = ?", (scenario_id,)).fetchone()
        encoded = json.dumps(payload, sort_keys=True)
        if existing is not None:
            if existing["scenario_json"] != encoded:
                raise ValueError("scenario ID already exists with different content")
            return
        conn.execute(
            "INSERT INTO assistant_scenarios (scenario_id, subject_id, scenario_json, created_at) VALUES (?, ?, ?, ?)",
            (scenario_id, subject_id, encoded, datetime.now(timezone.utc).isoformat()),
        )


def append_system_event(db_path: Path | str, event_id: str, event_type: str, severity: str, message: str, payload: dict | None = None) -> None:
    import json
    from datetime import datetime, timezone

    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO system_events (event_id, event_type, severity, message, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (event_id, event_type, severity, message, json.dumps(payload or {}, sort_keys=True), datetime.now(timezone.utc).isoformat()),
        )


if __name__ == "__main__":
    init_db()
    print(f"Initialized schema at {DB_PATH}")
