# Phase 3 Trading Decision Assistant Report

## Result

The project now provides a practical, read-only Trading Decision Assistant. The GUI does not place orders, call the paper broker, start the daemon, train models, or download data. The daemon remains an independently started process that updates market and signal state; Streamlit reads that state and existing provider caches.

## Markets and timeframes

- BTC/USDT and ETH/USDT: local OHLCV CSV caches are available for 15m, 1h, 4h, and 1d. The assistant calculates regimes, all strategy candidates, selected strategy, direction, plan, risk, leverage, and scenarios.
- QQQ and SPY: existing provider parquet caches are available under `data/market/NASDAQ100_PROXY` and `data/market/SP500_PROXY` for 15m, 1h, 4h, and 1d. The GUI reads these caches without inventing data. The configured provider path remains the daemon-owned updater.
- US equity session state comes from `MarketStateService`; outside the regular session the state is shown as `PRE_MARKET`, `AFTER_HOURS`, or `CLOSED`. Stale or insufficient input does not become an active setup.

## Score model

Every plan separates:

- `RULE_SCORE`: score from causal indicators and strategy-candidate rules.
- `MODEL_SCORE`: optional persisted model score, shown as `NOT_AVAILABLE` when the model result cannot be validated.
- `FINAL_SCORE`: rule score alone when no validated model score exists; otherwise the documented blend `0.65 * RULE_SCORE + 0.35 * MODEL_SCORE`, with a direction-conflict penalty.

The score is strategy evaluation, not a direct BUY/SELL model. All candidates are calculated and sorted; the first valid candidate is selected only after scoring all families.

## Models

Existing artifacts were checked before accepting a model score:

- Model 1A crypto: `data/models/model1a_crypto/*_tcn.pt` plus metadata for BTC/ETH.
- Model 1B equity intraday: `data/models/model1b_equity/*_tcn.pt` plus metadata for NASDAQ100/SP500 proxies.
- Model 2 equity swing: `models/checkpoints/qqq_swing.*` and `spy_swing.*`.

The metadata contains model version, feature columns, and supported timeframes. Equity 1h uses the intraday snapshot; equity 4h/1d uses the swing snapshot. A missing checkpoint, unreadable manifest, or missing persisted model snapshot produces `MODEL_NOT_AVAILABLE`; no fallback or fabricated ML score is used. No training was performed in this integration phase.

## Trade plan

A valid plan exposes asset, category, timeframe, strategy/version, direction, entry zone, initial stop, TP1, TP2, trailing method, expected holding time, Rule/Model/Final scores, risk budget, maximum loss, position size, notional, margin, recommended leverage, leverage range, risk/reward, regime, invalidation, and management rationale.

Leverage is derived after account capital, risk budget, stop distance, position size, notional, and margin. The product/exchange ceiling is separate from the recommendation. Normal equity-underlying data remains capped at 1x unless an explicitly configured leveraged instrument is supplied.

## Position analysis and scenarios

Manual positions accept asset, direction, entry, current price, quantity, margin, leverage, stop, take profit, timeframe, and optional strategy. The assistant reports PnL, margin return, risk to stop, liquidation estimate, risk/reward, state, and recommendation without sending an order.

The state machine is deterministic and shared by scenario calculations. It reports states such as `HOLD`, `HOLD + TRAILING`, `TAKE PROFIT`, `INVALIDATED`, and `ADVERSE_MOVE`. Scenario rows cover +0.5%, +1%, +2%, +3%, +5% and the corresponding negative moves, including price, PnL, PnL percentage, margin return, stop distance, trailing stop, state, and recommendation.

## Persistence and decision records

Displayed valid ideas are persisted idempotently using a content-derived immutable ID. Plan content and input snapshot content cannot be rewritten under an existing ID. Manual position analyses can be appended as point-in-time snapshots. Existing daemon state remains the source for current market/model timestamps.

## Telegram

Telegram is informational only and remains daemon-owned. High-quality swing notifications include the asset, direction, scores, entry, stop, target, protection, and leverage. No Telegram path places an order.

## Validation

- Full regression: `192 passed, 0 failed, 198 warnings`.
- Focused assistant suite: `10 passed`.
- Python compilation: `app.py`, `decision_pipeline.py`, and `decision_engine.py` passed.
- Streamlit headless smoke: server started successfully.
- BTC/ETH cached plan smoke: 1h and 4h paths completed without error.
- QQQ/SPY provider-cache smoke: 15m, 1h, 4h, and 1d caches loaded without network access.
- Immutable persistence smoke: duplicate content produced one row; changed content is rejected.

## Remaining operational limits

The daemon must be started manually to refresh live prices, provider caches, and signal snapshots. Streamlit does not become a data-ingestion process. A model score is unavailable until the daemon has produced a current compatible snapshot. Warnings in the regression suite are existing warnings; no test failed.
