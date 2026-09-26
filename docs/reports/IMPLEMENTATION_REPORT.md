# Trading Decision Assistant: Implementation Report

## Existing inventory

Already present:

- `live_daemon.py`: read-only market collection and Model 1A/1B/2 inference.
- `state_db.py`: WAL SQLite persistence for market state, signals, strategy profiles, paper trades, history, registry, manifests, and decision records.
- `risk_engine.py`: deterministic stop, take-profit, risk budget, notional, margin, leverage ceiling, KO/liquidation estimate, and trailing-stop helpers.
- `paper_broker.py`: a separate simulator that manages virtual trades from SQLite. It does not place exchange orders.
- `app.py`: Streamlit is read-only and does not expose broker-order methods.
- Research phases 1-4.5 and existing model infrastructure are preserved. No training is required for the deterministic assistant layer.

## Implemented in this increment

- `decision_engine.py`: order-free Strategy Cards, strategy-family selection, transparent 0-100 setup score, leverage advice, position mathematics, Trade Plan generation, current-position analysis, and deterministic scenario/state-machine analysis.
- Four strategy families: Trend/Breakout, Momentum, Mean Reversion, and Regime-Adaptive. They are informational profiles; no automatic strategy promotion occurs.

## Still to integrate

- Add user-position, trade-idea, scenario, snapshot, and system-event tables/migrations.
- Add persistence APIs for manual positions and append-only analysis snapshots.
- Replace the current three-tab Streamlit surface with Dashboard, Markets, Trade Ideas, My Positions, Trade Assistant, Strategies, Models, Data, and System views.
- Add data-quality reports and human-readable model/version cards.
- Keep Telegram informational only.

## Explicit boundaries

- No Streamlit order controls.
- No broker order methods are called by the assistant.
- No Windows autostart, service, or scheduler is added.
- No model training, feature engineering, sweep, GPU, TCN, or research redesign is performed.
- No live-readiness or profitability claim is made.

## Training handoff

This increment does not start training and does not require a new model artifact. If a later Trade Ideas view should use a refreshed Model 1/2 artifact, the user starts the existing training command manually and then the daemon consumes the resulting versioned artifact. A missing or stale artifact remains visible as `INSUFFICIENT DATA`/`STALE`; it is never silently treated as approval.