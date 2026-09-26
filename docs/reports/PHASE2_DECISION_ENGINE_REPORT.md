# Phase 2 Decision Engine Report

## Scope

The project now exposes a personal, read-only Trading Decision Assistant. The GUI does not place orders, call the paper broker, start the daemon, download market data, or train models.

## Decision flow

1. Load an existing OHLCV cache and persisted daemon snapshots.
2. Calculate causal ATR, EMA, Donchian, RSI, Bollinger, momentum, and volatility values.
3. Classify the market regime: trend up/down, range, high/low volatility, or unclear.
4. Score the configured strategy families: trend breakout, momentum, mean reversion, and regime adaptive.
5. Keep ML metadata separate from rule-based selection. Missing models remain `MODEL_NOT_AVAILABLE`.
6. Build a risk plan through the existing Risk Engine: entry, stop, targets, trailing method, risk budget, notional, margin, leverage band, and invalidation.
7. Persist displayed ideas immutably with a content-derived ID and input/plan snapshot.

## Position assistant

Manual positions are informational only. The state machine reports `HOLD`, `HOLD + TRAILING`, `TAKE PROFIT`, `INVALIDATED`, or `ADVERSE_MOVE`. Scenario analysis includes negative and positive price moves through +5% and reports PnL, PnL percentage, margin return, stop distance, state, and recommendation.

## Current data status

- BTC/USDT: real cached 1h data is available and produces a concrete plan.
- ETH/USDT: the GUI uses the newest matching cache when available.
- QQQ and SPY: no repository OHLCV cache was found; the GUI reports explicit insufficient/stale data and does not fabricate a setup.
- Models: availability is reported per result and never inferred from a missing artifact.

## Persistence and versioning

Trade ideas are immutable by content. Reusing the same content-derived ID is idempotent; attempting to reuse an ID with changed plan or input data raises an error. Manual position analyses can be appended as point-in-time snapshots.

## Validation

- Focused assistant tests: `8 passed`.
- Full regression suite: `190 passed, 198 warnings`.
- Python compilation: `app.py`, `decision_engine.py`, `decision_pipeline.py`, and `state_db.py` passed.
- Real-data smoke: 79,618 BTC/USDT 1h rows produced an `IntegratedTradePlan`; duplicate persistence resulted in one database row.

## Explicit non-goals

Telegram remains informational infrastructure only. No GUI action sends an order or triggers training. Automatic data acquisition for missing equity caches is intentionally not added; a configured ingestion process must populate those caches before QQQ/SPY can be analyzed.
