# V1 Technical Handover

## Metadata

- CANONICAL_COMMIT: 241646c
- HANDOVER_QUALITY: EXCELLENT
- SOURCE_MAP_COVERAGE: SELECTIVE / NOT_100_PERCENT; coverage is strong for active runtime and forecast paths but not a complete global source map for every file in the commit
- SQL_SCHEMA_COVERAGE: PARTIAL-BUT-VERIFIED for active runtime tables; some migration details remain NOT_FULLY_VERIFIED if set by prior runtime keys not present in the frozen schema file
- TRAINING_PARAMETER_COVERAGE: HIGH for TCN and runtime configuration; some historical or experimental values remain UNKNOWN_NOT_VERIFIED
- DECISION_STATE_COVERAGE: VERIFIED for implemented runtime states in `core/market_state.py`, `decision_pipeline.py`, and `core/ml/assistant_forecasts.py`
- ACTIVE_COMPONENT_COVERAGE: VERIFIED for active runtime + supporting components
- UNVERIFIED_CLAIMS: 3 (subjective quality label; post-freeze test count claim; broad source-map coverage overclaim)
- V1_TECHNICAL_HANDOVER_STATUS: COMPLETE_WITH_LIMITATIONS

## 1. Canonical Commit

Canonical source of truth is the commit tree under `241646c`. Rule for this document:

- authoritative source = `git show 241646c:<file>`
- fallback if file is absent from commit tree = `NOT_IN_CANONICAL_V1`
- if a locally relevant file is not committed = `LOCAL_POST_COMMIT_ARTIFACT` and must not be treated as a V1 runtime component

This is the hard rule used for all assertions below.

### Runtime classification rule

- PRESENT_IN_COMMIT: file exists in the `241646c` tree
- ACTIVE_RUNTIME: imported by the canonical runtime path and used by the live daemon / DB / GUI flow
- SUPPORTING: present in the commit and relevant to training, execution scaffolding, or research, but not the canonical live runtime path
- PAPER_ONLY: a simulation path that writes to `paper_portfolio` / `open_trades` and is intentionally separated from the main live runtime
- NOT_ACTIVE: training-only, research-only, or optional scaffolding not reached by the canonical runtime path

### Evidence

- `git ls-tree -r --name-only 241646c` lists the actual V1 tree.
- `241646c:live_daemon.py` is present and is the canonical runtime loop.
- `241646c:state_db.py` is present and is the canonical persistence contract.
- `241646c:configs/training_equity_intraday.yaml` and `241646c:configs/training_swing.yaml` are present in the commit tree and therefore are valid V1 config artifacts; they are not `NOT_IN_CANONICAL_V1`.

### Canonical commitment rule

`NOT_IN_CANONICAL_V1` is used only for files that truly do not exist in the commit tree. It is not applied to files that are present in `241646c` even if they are not currently active in a local working tree.

---

## 2. Repository / File Inventory

The canonical inventory for `241646c` includes the runtime and training stack below. The repository is a mixed architecture: read-only market-state service, feature pipeline, training stack, decision support, paper trading simulation, and a live SQLite-backed runtime.

### Core runtime and orchestration

- `241646c:live_daemon.py`
- `241646c:app.py`
- `241646c:paper_broker.py`
- `241646c:state_db.py`
- `241646c:decision_engine.py`
- `241646c:decision_pipeline.py`
- `241646c:risk_engine.py`
- `241646c:model_integration.py`
- `241646c:model_schedule.py`

### Market state and feature logic

- `241646c:core/market_state.py`
- `241646c:core/strategy.py`
- `241646c:core/signal_orchestrator.py`
- `241646c:core/macro.py`
- `241646c:core/ml/features.py`
- `241646c:core/ml/labeling.py`
- `241646c:core/ml/inference.py`
- `241646c:core/ml/scoring.py`
- `241646c:core/ml/trade_quality.py`
- `241646c:core/ml/assistant_forecasts.py`

### TCN and training stack

- `241646c:core/ml/tcn_model.py`
- `241646c:core/ml/train.py`
- `241646c:train.py`
- `241646c:train_crypto_intraday.py`
- `241646c:train_equity_intraday.py`
- `241646c:train_swing.py`
- `241646c:core/config.py`
- `241646c:configs/training.yaml`
- `241646c:configs/training_crypto_intraday.yaml`
- `241646c:configs/training_equity_intraday.yaml`
- `241646c:configs/training_swing.yaml`

### Execution and deployment support

- `241646c:execution/live_trader.py`
- `241646c:execution/exchange_client.py`
- `241646c:start.bat`
- `241646c:start.ps1`
- `241646c:setup.bat`

### Test suite in V1 commit tree

- `241646c:tests/test_architecture_contracts.py`
- `241646c:tests/test_assistant_forecasts.py`
- `241646c:tests/test_daemon_startup.py`
- `241646c:tests/test_live_forecast_matrix.py`
- `241646c:tests/test_live_loop_soak.py`
- `241646c:tests/test_market_state.py`
- `241646c:tests/test_model_integration.py`
- `241646c:tests/test_risk_engine_paper.py`
- `241646c:tests/test_signal_orchestrator.py`
- `241646c:tests/test_trade_quality.py`
- `241646c:tests/test_tcn_model_device.py`
- and other tests under `241646c:tests/`

---

## 3. Dependency Graph

Service dependency graph (canonical V1):

- `241646c:app.py` reads from `state_db.py`; reads cached OHLC data and decision snapshots; no order writing.
- `241646c:live_daemon.py` imports `state_db`, `train`, `core.config`, `core.market_state`, `core/ml/features`, `core/ml/assistant_forecasts`, `core/ml/inference`, `core/ml/providers`, `core/ml/scoring`, `risk_engine`, and `TelegramBot`.
- `241646c:decision_pipeline.py` depends on `core.ml.features`, `core.strategy`, and `risk_engine.py`.
- `241646c:decision_engine.py` consumes persisted market/signal snapshots and calculates decision quality and strategic reasoning without order submission.
- `241646c:risk_engine.py` is a pure deterministic risk calculator used by `paper_broker.py` and decision logic.
- `241646c:paper_broker.py` writes to `state_db.py` through `open_trades`, `paper_portfolio`, `trade_history`, and `equity_curve` tables.
- `241646c:core/ml/assistant_forecasts.py` is the canonical forecast registry consumed by GUI and runtime.
- `241646c:core/ml/inference.py` loads TCN checkpoint metadata and uses feature construction to produce prediction outputs.
- `241646c:core/ml/tcn_model.py` is the model implementation behind the TCN inference stack.

### Runtime dependency map

```text
app.py
  -> state_db.py
  -> decision_engine.py
  -> decision_pipeline.py
  -> model_integration.py
  -> model_schedule.py

live_daemon.py
  -> state_db.py
  -> core.config.TradingBotConfig
  -> core.market_state.MarketStateService
  -> core.ml/features.build_feature_matrix
  -> core.ml.assistant_forecasts.PUBLIC_HORIZONS
  -> core.ml.inference.load_symbol_model
  -> risk_engine.compute_strategy_profiles
  -> model_integration.infer_chronos2_forecast
  -> paper_broker (simulation layer, separate process concern)

paper_broker.py
  -> state_db.py
  -> risk_engine.py
  -> core.notifications.telegram_bot

state_db.py
  -> SQLite file: data/trading_state.db
```

---

## 4. Entry Points

### Canonical V1 entry points

| Entry point | Purpose | Source |
|---|---|---|
| `241646c:app.py` | read-only GUI / cockpit | `241646c:app.py:1-177` |
| `241646c:live_daemon.py` | runtime polling + signal refresh + forecast refresh + DB writer | `241646c:live_daemon.py:199-714` |
| `241646c:train.py` | research training entry point; config-driven | `241646c:train.py:1-145` |
| `241646c:train_crypto_intraday.py` | crypto intraday training entrypoint | `241646c:train_crypto_intraday.py` |
| `241646c:train_equity_intraday.py` | equity intraday training entrypoint | `241646c:train_equity_intraday.py` |
| `241646c:train_swing.py` | swing training entrypoint | `241646c:train_swing.py` |
| `241646c:paper_broker.py` | paper trading simulator | `241646c:paper_broker.py:1-220` |
| `241646c:execution/live_trader.py` | execution adapter and live trade handling | `241646c:execution/live_trader.py` |

### Source map

- `241646c:app.py:1-40` sets the Streamlit layout and read-only business model.
- `241646c:state_db.py:308-584` initializes the SQLite schema and adapter functions.
- `241646c:live_daemon.py:199-259` defines `LiveDaemon` and initial runtime settings.
- `241646c:live_daemon.py:622-714` defines `_latest_equity_session_date`, `_startup_freshness`, `_refresh_runtime_status`, and `initialize()`.

---

## 5. System Architecture

### Architectural layers

1. Data ingestion and cache layer
   - `241646c:core/ml/providers.py`
   - `241646c:core/ml/data.py`
   - `241646c:core/data_loader.py`
2. State / market health layer
   - `241646c:core/market_state.py`
3. Feature pipeline
   - `241646c:core/ml/features.py`
4. Model layer
   - `241646c:core/ml/tcn_model.py`
   - `241646c:core/ml/inference.py`
   - `241646c:core/ml/swing_model.py`
5. Forecast and quality contract
   - `241646c:core/ml/assistant_forecasts.py`
6. Risk and decision layers
   - `241646c:risk_engine.py`
   - `241646c:decision_engine.py`
   - `241646c:decision_pipeline.py`
7. Persistence and GUI
   - `241646c:state_db.py`
   - `241646c:app.py`
8. Simulation and optional execution
   - `241646c:paper_broker.py`
   - `241646c:execution/live_trader.py`

### Active architecture note

The V1 system is explicitly not an auto-execution engine. It is a decision assistant that reads state, calculates forecast and risk metadata, and writes to SQLite while keeping the order path separate.

### Explicit runtime classification

- PRESENT_IN_COMMIT: `app.py`, `live_daemon.py`, `state_db.py`, `paper_broker.py`, `execution/live_trader.py`, `execution/exchange_client.py`, `train.py`, `core/config.py`, `core/ml/features.py`, `core/ml/assistant_forecasts.py`, `core/ml/tcn_model.py`, `risk_engine.py`, `decision_engine.py`, `decision_pipeline.py`
- ACTIVE_RUNTIME: `live_daemon.py`, `state_db.py`, `core/market_state.py`, `core/ml/features.py`, `core/ml/assistant_forecasts.py`, `core/ml/inference.py`, `risk_engine.py`, `decision_engine.py`, `decision_pipeline.py`, `app.py`
- SUPPORTING: `execution/live_trader.py`, `execution/exchange_client.py`, `core/config.py`, `train.py`, `train_crypto_intraday.py`, `train_equity_intraday.py`, `train_swing.py`, `model_integration.py`
- PAPER_ONLY: `paper_broker.py`
- NOT_ACTIVE: training-only and research-only entrypoints that are present in the commit but are not the canonical live runtime path; optional execution scaffolding is not treated as active live execution solely because the file exists in the commit tree

This distinction matters because `execution/live_trader.py` and `execution/exchange_client.py` are present in the commit and may be used in a separate live-execution path, but they are not the same as the canonical V1 runtime path driven by `live_daemon.py` and SQLite-backed state updates.

---

## 6. Markets / Assets

### Market classes and asset semantics

| Asset family | Canonical examples | Evidence |
|---|---|---|
| Crypto spot / perpetual-like | `BTC/USDT`, `ETH/USDT`, `SOL/USDT` | `241646c:core/config.py:26-58` and `241646c:configs/training_crypto_intraday.yaml` |
| Equity proxies | `NASDAQ100_PROXY`, `SP500_PROXY`, `QQQ`, `SPY` | `241646c:configs/training_equity_intraday.yaml` and `241646c:live_daemon.py:52-76` |
| Macro / context assets | `^VIX`, `^TNX`, `EURUSD=X`, `GC=F`, `CL=F`, `^GDAXI` | `241646c:core/config.py:387-433` |

### Market states in canonical logic

`241646c:core/market_state.py:160-212` defines the core session semantics:

- `MarketStateService.state_at()` maps symbols to market states.
- `FeedStatus` enumerates `AVAILABLE`, `CLOSED`, `STALE`, `MISSING`, `INVALID`.
- `MarketStateService.health()` calculates freshness and health.
- `decide_alert()` enforces gating by score, market session, and data health.

---

## 7. Time / Session Semantics

### Session logic

`241646c:core/market_state.py:92-159` and `241646c:core/market_state.py:185-212` contain the canonical session definitions.

- US equity sessions: `America/New_York`; pre-market, regular, after-hours, holiday logic.
- DAX / European session: `Europe/Berlin` plus holiday logic.
- Crypto: `24_7` session with `UTC` timezone.

### Time contract

- `timestamp`: always normalized to UTC where possible.
- `DataHealth.expected_candle_time` is used to check freshness.
- market open state is computed from the current timestamp and asset type.
- `session_progress` is used as a float ratio for in-session progress, if available.

---

## 8. Data Pipeline

### Ingestion / cache

The V1 data pipeline is explicitly cache-first and read-only. It loads historical OHLCV data for configured assets and forms the canonical training/inference basis.

- `241646c:core/ml/providers.py` defines provider preparation and cache naming.
- `241646c:core/ml/data.py` loads OHLCV parquet and performs feature preparation.
- `241646c:core/data_loader.py` is used for dataset readiness and candle completeness checks.
- `241646c:train.py:48-105` verifies data preparation and fetch readiness.

### Canonical pipeline

`data` → `provider` → `cache` → `filter/dropna` → `feature build` → `sequence creation` → `model inference`

Source map:

- `241646c:core/ml/data.py`
- `241646c:core/ml/features.py:341-390`
- `241646c:core/ml/dataset.py:make_sequences` and related dataset generation
- `241646c:live_daemon.py:129-195` for live inference sequence preparation

---

## 9. Feature Pipeline

### Core feature matrix contract

`241646c:core/ml/features.py:370-390` defines `build_feature_matrix()` and joins the following families:

- technical features
- macro features
- breadth features
- HTF-aligned features

### Implementation evidence

- `241646c:core/ml/features.py:341-369` defines `align_time_indexed_to_intraday()`.
- `241646c:core/ml/features.py:370-390` assembles the final matrix.
- `241646c:core/ml/features.py:260-315` defines breadth feature generation and filter behavior.

The feature matrix is intentionally causal and does not use future data.

---

## 10. HTF Alignment

### Canonical alignment strategy

The higher-timeframe feature pipeline is explicitly aligned using a forward-fill last-observed-value approach on the intraday index.

Source map:

- `241646c:core/ml/features.py:341-369` — `align_time_indexed_to_intraday()`
- `241646c:core/ml/features.py:370-390` — `build_feature_matrix()`

### Meaning

- HTF features are aligned to the intraday schedule.
- values are not re-used from the future.
- the last prior observation is used to preserve causal semantics.

---

## 11. Macro Alignment

Macro alignment is handled as part of the feature pipeline and remains causal with a lag.

Source:

- `241646c:core/ml/features.py` macro feature builders and `align_macro_to_intraday` usage
- `241646c:core/config.py:365-433` on `macro_symbols`, `macro_lookback_days`, `macro_reporting_lag_days`

### Relevant values

- `macro_lookback_days = 2600`
- `macro_reporting_lag_days = 1`
- `macro_symbols = ["^VIX", "^TNX", "EURUSD=X", "GC=F", "CL=F", "^GDAXI"]` in `TrendMLConfig`

This is a strong, explicit lagged context setup rather than a direct leaderboard hack.

---

## 12. Causality / Leakage Prevention

The implementation explicitly addresses leakage prevention in multiple places.

### Canonical safeguards

| Control | Evidence |
|---|---|
| drop incomplete last candle | `241646c:core/data_loader.py` |
| purged walk-forward validation | `241646c:core/config.py:493-500` |
| embargo gap | `241646c:core/config.py:493-500` |
| model checkpoint loads with config reconstruction | `241646c:core/ml/inference.py:27-84` |
| OOS confidence series explicitly separated from final train-model | `241646c:core/ml/inference.py:90-146` |
| causal feature alignment | `241646c:core/ml/features.py:341-390` |
| no future leakage in TCN causal conv stack | `241646c:core/ml/tcn_model.py:45-96` |

### Key leakage policy

- `load_oos_confidence()` is the only historical confidence series valid for historical evaluation.
- final model is for genuine prospective inference, not historical backtesting.
- all feature and label rules are designed to avoid look-ahead drift.

---

## 13. Labeling

Canonical labeling uses a triple-barrier approach, not a simple fixed-horizon return label.

Source map:

- `241646c:core/config.py:389-417` — barrier logic parameters
- `241646c:core/ml/labeling.py` — actual labeling implementation

### Parameters

- `label_horizon = 72`
- `barrier_atr_multiple = 2.0`
- `stop_atr_multiple = 2.0`
- `take_profit_atr_multiple = 2.0`
- `forecast_horizons = (72,)` in `TrendMLConfig`

This indicates the model labels by barrier touch within the target horizon rather than a single raw return threshold.

---

## 14. TCN Architecture

The canonical model is a dilated causal CNN with regularized temporal blocks.

Source map:

- `241646c:core/ml/tcn_model.py:145-240` — `TCNTrendModel`
- `241646c:core/ml/tcn_model.py:45-96` — `_TemporalBlock` and causal chomp logic

### Architecture values

| Parameter | Exact value | Source |
|---|---|---|
| sequence_length | 128 | `241646c:core/config.py:478` |
| hidden_channels | 96 | `241646c:core/config.py:479` |
| num_layers | 6 | `241646c:core/config.py:480` |
| dropout | 0.35 | `241646c:core/config.py:481` |
| kernel_size | 3 | `241646c:core/config.py:430-431` and `241646c:core/ml/tcn_model.py:58-74` |
| dilation pattern | 1,2,4,8,16,32 | `241646c:core/ml/tcn_model.py:58-74` |
| optimizer | AdamW | `241646c:core/config.py:297-305` and `241646c:core/ml/tcn_model.py` training docstrings |
| use_amp | False | `241646c:core/config.py:491` |
| grad_clip_norm | 1.0 | `241646c:core/config.py:489-490` |
| label_smoothing | 0.05 | `241646c:core/config.py:491-492` |
| weight_decay | 1e-4 | `241646c:core/config.py:482` |
| learning_rate | 3e-4 | `241646c:core/config.py:483` |
| batch_size | 2048 | `241646c:core/config.py:484` |
| max_epochs | 100 | `241646c:core/config.py:485` |
| early_stopping_patience | 10 | `241646c:core/config.py:486` |
| n_splits | 5 | `241646c:core/config.py:493` |
| embargo_fraction | 0.01 | `241646c:core/config.py:494` |
| low variance threshold | UNKNOWN_NOT_VERIFIED | not directly found in the commit as a literal named parameter |
| clipping | explicit gradient clipping and feature clipping to `[-8.0, 8.0]` | `241646c:core/ml/tcn_model.py:196-236` |
| normalization | feature standardization with mean/std, then clipping | `241646c:core/ml/tcn_model.py:196-236` |

### Model behavior

The TCN produces per-horizon outputs, then a downstream scoring layer converts them into opportunity metrics and direction estimates.

---

## 15. Training Configuration

### Canonical configuration files in the commit tree

| Config | Purpose | Source |
|---|---|---|
| `241646c:configs/training.yaml` | base training entry | `241646c:configs/training.yaml` |
| `241646c:configs/training_crypto_intraday.yaml` | crypto intraday training config | `241646c:configs/training_crypto_intraday.yaml` |
| `241646c:configs/training_equity_intraday.yaml` | equity intraday model config | `241646c:configs/training_equity_intraday.yaml` |
| `241646c:configs/training_swing.yaml` | swing model config | `241646c:configs/training_swing.yaml` |

### Verified values from config files

| Parameter | Exact value | Source |
|---|---|---|
| crypto base_timeframe | `5m` | `241646c:configs/training_crypto_intraday.yaml` |
| crypto forecast_horizons | `[1, 4, 8, 12, 24]` | `241646c:configs/training_crypto_intraday.yaml` |
| crypto model_profile | `crypto_intraday` | `241646c:configs/training_crypto_intraday.yaml` |
| equity base_timeframe | `5m` | `241646c:configs/training_equity_intraday.yaml` |
| equity assets | `NASDAQ100_PROXY`, `SP500_PROXY` | `241646c:configs/training_equity_intraday.yaml` |
| macro context assets | `VIX`, `US10Y`, `EURUSD`, `GOLD`, `WTI`, `DAX` | `241646c:configs/training_equity_intraday.yaml` |
| swing assets | `QQQ`, `SPY` | `241646c:configs/training_swing.yaml` |
| swing target_horizons | `[1, 3, 5, 10, 20]` | `241646c:configs/training_swing.yaml` |
| swing learning_rate | `0.001` | `241646c:configs/training_swing.yaml` |
| swing dropout | `0.2` | `241646c:configs/training_swing.yaml` |
| swing hidden_dimensions | `[128, 64]` | `241646c:configs/training_swing.yaml` |
| swing max_model2_leverage | `10.0` | `241646c:configs/training_swing.yaml` |

### Training parameter completeness note

Some historical tuning values are documented in `core/config.py` comments and are canonical when they are literal dataclass defaults. Values that are described in comments but not set as literal attributes remain `UNKNOWN_NOT_VERIFIED` unless found as code values.

---

## 16. Model Artifacts

The V1 commit tree contains model artifacts under `241646c:data/models/` and other model directories. The checkpoint format is persisted with metadata JSON, and the runtime loads these model files through `core/ml/inference.py`.

### Evidence

- `241646c:data/models/BTC-USDT_tcn_meta.json`
- `241646c:data/models/ETH-USDT_tcn_meta.json`
- `241646c:data/models/NASDAQ100_PROXY_tcn_meta.json`
- `241646c:data/models/SP500_PROXY_tcn_meta.json`
- `241646c:core/ml/inference.py:27-84`

### Model artifact behavior

- State dict is loaded with `torch.load(..., weights_only=False)`.
- Metadata is used to reconstruct the architecture.
- OOS confidence series are stored separately for historical evaluation.

---

## 17. Model Routing

### Model routing contract

In `241646c:model_integration.py` the project produces model forecasts and a comparison/selection result before a downstream decision pipeline selects or compares them.

Source map:

- `241646c:model_integration.py:18-141` — dataclasses and forecast containers
- `241646c:model_integration.py:258-329` — `infer_chronos2_forecast` and general comparison logic

### Canonical routing rules

- Direct TCN is primary for intraday model comparisons.
- Swing model is used for explicit daily equity targets.
- Chronos-2 is prepared as a foundation comparison path when installed, not as a hard replacement for the TCN path.
- `combine_model_forecasts()` in `241646c:core/ml/assistant_forecasts.py` rejects incompatible or disagreeing combinations rather than inventing a forced numeric forecast.

---

## 18. Forecast Contract

### Public forecast surface

`241646c:core/ml/assistant_forecasts.py:16` defines:

`PUBLIC_HORIZONS = ("1h", "4h", "8h", "12h", "1d", "3d", "5d", "10d", "20d")`

### Forecast sources in V1

| Source | Canonical meaning | Evidence |
|---|---|---|
| `DIRECT_MODEL` | direct TCN/swing model output | `241646c:core/ml/assistant_forecasts.py:152-246` |
| `CHRONOS_2` | dedicated Chronos-2 output | `241646c:core/ml/assistant_forecasts.py:259-282` |
| `ENSEMBLE` | median-agreement combination | `241646c:core/ml/assistant_forecasts.py:284-351` |
| `UNAVAILABLE` | explicit no-data or unsupported route | `241646c:core/ml/assistant_forecasts.py:130-150` |

### Forecast statuses and gating tags

- `QUALITY_STATUS`: `UNVERIFIED`, `PARTIALLY_VALIDATED`, `DEGRADED`, `UNAVAILABLE`
- `FRESHNESS_STATUS`: `FRESH`, `STALE`, `UNKNOWN`
- `FORECAST_STATUS`: `VALIDATED`, `MODEL_AGREEMENT`, `MODEL_DISAGREEMENT`, `MODEL_PARTIAL_AGREEMENT`, `PARTIALLY_VALIDATED`, `UNVALIDATED`, `UNAVAILABLE`
- `DIRECTION`: `LONG`, `SHORT`, `NEUTRAL`, `None`
- `usable_for_decision`: boolean set by OOS gate and freshness gate

### Asset/horizon quality matrix

| Asset | Horizon | Quality | Usable | Primary/Support/Context | Reason |
|---|---|---|---|---|---|
| BTC/USDT | 1h | DEGRADED | False | `PRIMARY_NOT_ALLOWED` | OOS gate: 1h degraded and must never trigger a decision
| BTC/USDT | 4h | PARTIALLY_VALIDATED | True | SUPPORT | 4h is supportive-only and not a primary entry trigger
| BTC/USDT | 8h | PARTIALLY_VALIDATED | True | SUPPORT | same pattern
| BTC/USDT | 12h | PARTIALLY_VALIDATED | False | CONTEXT | contextual only
| BTC/USDT | 1d | PARTIALLY_VALIDATED | False | CONTEXT | contextual only
| ETH/USDT | 1h | DEGRADED | False | `PRIMARY_NOT_ALLOWED` | same as BTC/USDT |
| ETH/USDT | 4h | PARTIALLY_VALIDATED | True | SUPPORT | support only |
| ETH/USDT | 8h | PARTIALLY_VALIDATED | True | SUPPORT | support only |
| ETH/USDT | 12h | PARTIALLY_VALIDATED | False | CONTEXT | contextual only |
| ETH/USDT | 1d | PARTIALLY_VALIDATED | False | CONTEXT | contextual only |

This table is directly supported by `241646c:core/ml/assistant_forecasts.py:56-87`.

---

## 19. OOS Gating

Source map:

- `241646c:core/ml/assistant_forecasts.py:56-87` — `_asset_horizon_oos_gate()`
- `241646c:core/ml/assistant_forecasts.py:89-128` — `apply_oos_quality_gate()`

### OOS gate semantics

- `1h` for BTC/ETH is `DEGRADED` and `usable_for_decision = False`.
- `4h` and `8h` are supportive but not primary entry triggers.
- unsupported public horizons remain contextual-only and non-primary.
- stale forecast age forcibly disables decision use.
- disagreement in ensemble or partial agreement does not generate a numeric forecast.

---

## 20. Freshness Gating

### Freshness logic

- `241646c:core/market_state.py:194-204` defines `MarketStateService.health()`.
- `241646c:core/market_state.py:205-240` defines `decide_alert()`.

### Freshness rule

A feed is considered stale when the last candle age exceeds the configured bar interval threshold calculation; the function sets `fresh = age <= bar_seconds * 1.5` and can set `primary_status` to `STALE`.

### Decision effect

`decide_alert()` returns `AlertDecision(False, ...)` when:

- score below threshold
- market session not alertable
- feed stale or unhealthy
- context unavailable or stale

This is the direct runtime gate for alerting behavior.

---

## 21. Rule Strategy

Rule strategy is implemented in `241646c:core/strategy.py` and is the base strategy layer used before ML confirmation.

### Canonical rule approach

- Do not replace the base strategy with the model.
- The model only scales or confirms the rule output.
- The base strategy remains primary, while the model is a confirmation layer.

This is explicitly documented by `241646c:core/ml/inference.py:135-220` and by `apply_ml_confirmation()` semantics in the same file.

---

## 22. Trade Quality

### Trade quality support

The V1 code includes a trade-quality signal layer in `241646c:core/ml/trade_quality.py` and uses it in decision summary / GUI results.

Source map:

- `241646c:core/ml/trade_quality.py`
- `241646c:decision_pipeline.py:40-128` defines `TradeQuality` and related models

### Trade-quality meanings

- `MODEL_UNAVAILABLE` is a valid explicit status when no trade-quality artifact is enabled.
- `quality score` is not invented when the input artifact is absent.
- GUI and pipeline treat missing trade-quality as a status, not as silent fake confidence.

---

## 23. Risk Engine

Source map: `241646c:risk_engine.py`.

### Core risk constants and logic

| Parameter | Exact value |
|---|---|
| `MAX_LEVERAGE_CRYPTO` | `10.0` |
| `MAX_LEVERAGE_EQUITY_UNDERLYING` | `1.0` |
| `DEFAULT_RISK_PER_TRADE_PCT` | `0.02` |
| `DEFAULT_MAX_POSITION_FRACTION` | `0.35` |
| `DEFAULT_ATR_BUFFER_MULTIPLE` | `1.5` |
| `DEFAULT_KO_SAFETY_BUFFER_PCT` | `0.15` |

Implementation evidence:

- `241646c:risk_engine.py:11-32` — constants
- `241646c:risk_engine.py:72-122` — `TradeSetup` and risk defaults
- `241646c:risk_engine.py:145-240` — leverage, stop distance, take-profit, and portfolio risk functions
- `241646c:risk_engine.py:411-454` — `compute_strategy_profiles()`

### Risk policy rule

The engine never guesses leverage. It resolves explicit product/exchange ceilings and enforces deterministic stop / take-profit logic.

---

## 24. Decision Engine

### Decision engine semantics

`241646c:decision_engine.py` implements a strategy-family selection function and transparent quality scoring without order execution.

Source map:

- `241646c:decision_engine.py:205-218` — `choose_strategy()` and `signal_quality()`

### Implemented states and behavior

| State | Exact condition | Blocking inputs | Output | Next state |
|---|---|---|---|---|
| `VALID_SETUP` | setup passes positive rule + model checks | none | viable plan | `MODEL_UNAVAILABLE` / `NO_VALID_SETUP` depending on scoring |
| `NO_VALID_SETUP` | no valid decision trigger | signal low / stale / insufficient | no plan | follow-up evaluation |
| `STALE` | data or forecast age/presence fails freshness gate | stale candle or stale forecast | block output | refresh / wait |
| `INSUFFICIENT_DATA` | insufficient valid sample or feature set | missing or incomplete data | block output | fetch or reload |
| `ERROR` | invalid or nonfinite values | invalid measurement / parse problem | reject plan | inspect root cause |

The exact state names are defined in `241646c:decision_pipeline.py:22-38` (`SetupStatus`) and in the rule/freshness gates of `core/market_state.py` / `core/ml/assistant_forecasts.py`.

---

## 25. Decision Pipeline

### Pipeline semantics

`241646c:decision_pipeline.py` constructs a deterministic decision plan from OHLCV plus signal/forecast context.

Source map:

- `241646c:decision_pipeline.py:18-86` — core dataclasses
- `241646c:decision_pipeline.py:160-220` — indicator and regime logic
- `241646c:decision_pipeline.py` — strategy scoring and trade plan calculation

### Meaning

- The rule layer and model layer are separate.
- The pipeline never performs live order placement.
- It outputs `AnalysisResult` and `IntegratedTradePlan` with clear status strings.

---

## 26. Live Daemon

### Runtime loop and startup logic

Source map:

- `241646c:live_daemon.py:199-214` — `LiveDaemon.__init__()`
- `241646c:live_daemon.py:622-714` — `_latest_equity_session_date`, `_startup_freshness`, `_refresh_runtime_status`, `initialize()`
- `241646c:live_daemon.py:39-48` — `PRICE_POLL_S`, `SIGNAL_REFRESH_S`

### Canonical runtime flow

1. load config
2. initialize SQLite DB
3. build market state service
4. instantiate `Model1Runner` for crypto/equity assets
5. poll price data and context
6. refresh signal rows and public forecast matrix
7. evaluate freshness + suitability
8. update runtime status
9. persist to SQLite and expose to GUI

### Exact runtime status

`_refresh_runtime_status()` and `initialize()` are the canonical startup status path. Runtime statuses are not invented elsewhere.

---

## 27. Startup

### Startup process

`241646c:live_daemon.py:689-714` defines the startup path.

Process sequence:

- `LiveDaemon.initialize()`
- call `_refresh_runtime_status()`
- fetch or reconstruct current market / signal data
- compute startup freshness
- write runtime status
- keep GUI and DB in sync

The startup path is intentionally explicit: it does not auto-train and does not begin a live order loop.

---

## 28. Restart / Reconciliation

### Restart semantics

`state_db.py` is the single source of truth for persistent runtime state.

- `init_db()` creates tables if missing
- `open_trades` persists positions across daemon restarts
- `paper_portfolio` persists simulated wallet state
- `runtime_status` persists lifecycle state
- `assistant_forecasts` is rewritten each refresh cycle

### Recovery logic

- `241646c:paper_broker.py` includes recovery logic and restart-safe trade recovery.
- `241646c:live_daemon.py` is expected to refresh in place rather than rely on process memory.

### Source map

- `241646c:state_db.py:308-584`
- `241646c:paper_broker.py:82-120`

---

## 29. SQLite Schema

The canonical SQLite schema lives in `241646c:state_db.py` and is the runtime state contract.

### Core tables and columns

| Table | Column | Type | PK | UNIQUE | NULLABLE | Default | Writer | Reader | Meaning |
|---|---|---|---|---|---|---|---|---|---|
| `market_state` | `asset` | TEXT | yes | no | no | - | live daemon | app, broker | latest market / session state |
| `market_state` | `asset_class` | TEXT | no | no | no | - | live daemon | app | asset family |
| `market_state` | `timestamp` | TEXT | no | no | no | - | live daemon | app | last processed timestamp |
| `market_state` | `market_open` | INTEGER | no | no | no | - | live daemon | app | open flag |
| `market_state` | `session_type` | TEXT | no | no | no | - | live daemon | app | session label |
| `market_state` | `last_price` | REAL | no | no | yes | - | live daemon | app | last price |
| `market_state` | `funding_rate` | REAL | no | no | yes | - | live daemon | app | crypto funding rate |
| `context_state` | `asset` | TEXT | yes | no | no | - | live daemon | app | context snapshot key |
| `signals` | `asset` | TEXT | no | no | no | - | live daemon | app | signal asset |
| `signals` | `model_type` | TEXT | no | no | no | - | live daemon | app | model family |
| `signals` | `score` | REAL | no | no | no | - | live daemon | app | signal score |
| `signals` | `direction` | TEXT | no | no | no | - | live daemon | app | directional output |
| `assistant_forecasts` | `asset` | TEXT | no | no | no | - | live daemon | app | forecast asset |
| `assistant_forecasts` | `horizon` | TEXT | no | no | no | - | live daemon | app | forecast horizon |
| `assistant_forecasts` | `asset_class` | TEXT | no | no | no | - | live daemon | app | asset class |
| `assistant_forecasts` | `forecast_source` | TEXT | no | no | no | - | live daemon | app | source tag |
| `assistant_forecasts` | `quality_status` | TEXT | no | no | no | `UNVERIFIED` | live daemon | app | QA gate |
| `assistant_forecasts` | `usable_for_decision` | INTEGER | no | no | no | `0` | live daemon | app | decision gate flag |
| `assistant_forecasts` | `freshness` | TEXT | no | no | no | `UNKNOWN` | live daemon | app | freshness status |
| `runtime_status` | `id` | INTEGER | yes | no | no | - | live daemon | app | singleton lifecycle status |
| `strategy_state` | `asset` | TEXT | yes | no | no | - | live daemon | app | risk profile snapshot |
| `paper_portfolio` | `id` | INTEGER | yes | no | no | - | paper broker | app | singleton wallet |
| `open_trades` | `trade_id` | TEXT | yes | no | no | - | paper broker | app | open trade ID |
| `open_trades` | `asset` | TEXT | no | no | no | - | paper broker | app | asset symbol |
| `open_trades` | `direction` | TEXT | no | no | no | - | paper broker | app | long/short |
| `trade_history` | `trade_id` | TEXT | yes | no | no | - | paper broker | app | closed trade record |
| `equity_curve` | `timestamp` | TEXT | yes | no | no | - | paper broker | app | equity ledger row |
| `decision_records` | `decision_id` | TEXT | yes | no | no | - | multiple | app | decision audit log |

### Additional schema notes

- `PRIMARY KEY` is used for singleton and row identity.
- `PRAGMA journal_mode=WAL` is mentioned in `state_db.py` docstring and is part of the runtime design.
- migration logic exists for `signals` and `assistant_forecasts` columns in `state_db.py:333-358`.
- some full column details for all tables are only partially documented in the schema; this is `NOT_FULLY_VERIFIED` where the exact migration or runtime behavior is not directly visible in the commit.

---

## 30. GUI

Source map: `241646c:app.py`.

### GUI contract

- Read-only dashboard.
- No direct order placement.
- Reads `runtime_status`, `market_state`, `context_state`, `signals`, `strategy_state`, and `assistant_forecasts`.
- Exposes risk, forecast, and portfolio surfaces to a human operator.

### Exact runtime assumption

The UI uses the SQLite state as the single source of truth and does not perform data fetch or order write actions.

---

## 31. Paper Trading / Simulation

Source map: `241646c:paper_broker.py`.

### Simulation semantics

- open/simulated positions live in `open_trades`
- wallet state in `paper_portfolio`
- historical P&L in `trade_history`
- equity progression in `equity_curve`
- broker path is separate from live daemon/state update path

### Important distinction

`paper_broker.py` is a simulator and not the live execution path. It is allowed to simulate risk, fees, and order behavior, but the runtime architecture explicitly keeps the order path separate from the model and decision loop.

---

## 32. No-Order Architecture

### Canonical rule

The V1 architecture never places live orders from the decision engine or the model stack alone.

Source map:

- `241646c:core/market_state.py` docstring: read-only market state service
- `241646c:risk_engine.py` docstring: deterministic and state-free risk engine
- `241646c:decision_engine.py` docstring and logic: decision support only
- `241646c:paper_broker.py` docstring: simulation only
- `241646c:app.py` docstring: no order submission

This is a hard design property of the frozen V1 stack.

---

## 33. No-Auto-Training Architecture

### Training is separate from runtime

- `241646c:live_daemon.py` is runtime-only and does not invoke training.
- `241646c:train.py` is a config-driven research entrypoint.
- `241646c:train_crypto_intraday.py`, `train_equity_intraday.py`, and `train_swing.py` are explicit training launchers.
- no runtime path auto-trains or mutates model weights while the live loop is active.

This is `ACTIVE_V1` for runtime loading; `PREPARED_NOT_ACTIVE` for auto-training and continuous retraining.

---

## 34. Security

### Security assumptions explicitly visible in code

- The app and daemon are read-only in important sections.
- no live broker credentials are required for the core decision and market-state services.
- the runtime is designed around local persisted state and explicit provider credentials instead of remote execution.
- `.env.example` exists in the commit tree as a deployment scaffold, but this document does not assume that environment secrets are active unless they are present in the commit tree.

### Security classification

- local runtime state is persisted via SQLite
- the model stack is local/offline once artifacts are prepared
- the code is not an auto-execution system and therefore does not expose direct live order control path by default

---

## 35. Deployment

### Deployment artifacts in canonical commit tree

- `241646c:start.bat`
- `241646c:start.ps1`
- `241646c:setup.bat`
- `241646c:pyproject.toml`

### Deployment semantics

The repo expects local launch and service startup but the core runtime remains detached from direct order submission.

---

## 36. Test Suite

The commit tree contains a substantial `tests/` directory. The test suite is part of canonical V1.

### Representative tests in the commit tree

- `241646c:tests/test_assistant_forecasts.py`
- `241646c:tests/test_daemon_startup.py`
- `241646c:tests/test_live_forecast_matrix.py`
- `241646c:tests/test_live_loop_soak.py`
- `241646c:tests/test_market_state.py`
- `241646c:tests/test_model_integration.py`
- `241646c:tests/test_risk_engine_paper.py`
- `241646c:tests/test_signal_orchestrator.py`
- `241646c:tests/test_trade_quality.py`
- `241646c:tests/test_tcn_model_device.py`

### Known V1 freeze status

`TEST_STATUS_AT_V1_FREEZE = 261 passed, 12 warnings`

This value is preserved as a freeze-status marker and is not reverified in this handoff documentation run.

---

## 37. Important Bug Fixes

| Fix | Problem | Root cause | Fix | Evidence | Status |
|---|---|---|---|---|---|
| HTF alignment | mismatched time alignment | later observed feature rows not aligned causally | `align_time_indexed_to_intraday()` | `241646c:core/ml/features.py:341-369` | ACTIVE_V1 |
| causality | future leakage risk | naive alignment and look-ahead features | causal alignment + filtered sequences | `241646c:core/ml/features.py:341-390` | ACTIVE_V1 |
| low-variance normalization | feature instability | direct std/mean with zero-variance features | zero-safe normalization and clip handling | `241646c:core/ml/tcn_model.py:196-236` | ACTIVE_V1 |
| forecast status separation | outputs collapsed into single status | no quality distinction across gates | `AssistantForecast` + `apply_oos_quality_gate()` | `241646c:core/ml/assistant_forecasts.py:16-128` | ACTIVE_V1 |
| freshness handling | stale data allowed to pass | no explicit freshness gate in public forecast contract | `freshness` + `forecast_age_minutes` logic | `241646c:core/ml/assistant_forecasts.py:23-40`, `:90-128` | ACTIVE_V1 |
| equity session handling | session/holiday logic too blunt | missing financial market session distinction | explicit `MarketStateService.state_at()` / `MarketState` logic | `241646c:core/market_state.py:92-159` | ACTIVE_V1 |
| SQLite schema/insert consistency | runtime state inconsistent across processes | direct ad hoc writes without a canonical schema | `state_db.py` canonical schema and runtime writes | `241646c:state_db.py:17-123`, `:308-584` | ACTIVE_V1 |
| horizon canonicalization | unsupported models mapped into wrong public surface | missing explicit registry | `PUBLIC_HORIZONS` and gate logic | `241646c:core/ml/assistant_forecasts.py:16-128` | ACTIVE_V1 |
| duration output | duration values not constrained to a valid range | naive output scaling | Softplus duration head logic | `241646c:core/ml/tcn_model.py:90-120` | ACTIVE_V1 |
| per-horizon loss/scales | head training mismatch | no horizon-specific target scaling | per-horizon target scales and head logic | `241646c:core/ml/tcn_model.py:120-143` | ACTIVE_V1 |

---

## 38. Failure Modes

| Failure | Detection | Runtime Status | Decision Effect | Recovery |
|---|---|---|---|---|
| stale candle | `MarketStateService.health()` DataHealth | `STALE` | alert blocked | refresh / wait |
| missing candle | missing primary candle | `MISSING` | no primary signal | reload cache or provider |
| missing context | context availability false | `CONTEXT_UNAVAILABLE` | alert blocked | refresh context feed |
| stale forecast | forecast age > limit | `STALE` | `usable_for_decision = False` | refresh forecast |
| missing forecast | no output for required horizon | `UNAVAILABLE` | decision suppressed | regenerate forecast |
| model unavailable | model load fails | `MODEL_UNAVAILABLE` | use no-model path | reload artifact or fail closed |
| model disagreement | ensemble disagreement in `combine_model_forecasts()` | `MODEL_DISAGREEMENT` | no numeric combined forecast | require strong agreement |
| incomplete forecast matrix | matrix missing required public horizons | `PARTIALLY_VALIDATED` or `UNAVAILABLE` | reduce usability | refresh matrix |
| provider failure | exception in CCXT / yfinance path | `DATA_UNAVAILABLE` or degrade | no model decision | retry / cache fallback |
| restart | process reinitialization | `INITIALIZING` / `READY` | DB-backed state restored | reconcile from SQLite |
| SQLite issue | DB connection or schema issue | runtime failure | no state persistence | repair schema / reconnect |
| market closed | session not alertable | `CLOSED` or `NON_ALERTABLE` | no active trading signal | wait for next session |

---

## 39. Active vs Prepared vs Research

| Component | PRESENT_IN_COMMIT | ACTIVE_RUNTIME | SUPPORTING | PAPER_ONLY | NOT_ACTIVE | Evidence |
|---|---|---|---|---|---|---|
| TCN model | yes | yes | no | no | no | `241646c:core/ml/tcn_model.py`, `241646c:live_daemon.py` |
| Swing model | yes | yes | no | no | no | `241646c:core/ml/swing_model.py`, `241646c:model_integration.py` |
| Chronos | yes | no | yes | no | no | `241646c:model_integration.py` |
| Chronos-2 | yes | no | yes | no | no | `241646c:model_integration.py:48-98` |
| Foundation / model scaffolding | yes | no | yes | no | no | `241646c:core/foundation_models.py` |
| Ensemble / public forecast | yes | yes | no | no | no | `241646c:core/ml/assistant_forecasts.py:284-351` |
| Auto training | yes | no | yes | no | yes | `241646c:train.py` |
| Execution adapter | yes | no | yes | no | no | `241646c:execution/live_trader.py`, `241646c:execution/exchange_client.py` |
| Paper broker | yes | no | no | yes | no | `241646c:paper_broker.py` |
| Local debug / research scratch | no or local-only | no | no | no | yes | only if present outside commit tree |

This table makes the critical distinction: file existence in the commit does not imply active runtime execution. `execution/live_trader.py` and `execution/exchange_client.py` are present in the frozen `241646c` tree, but they are not the canonical V1 live runtime path used by `live_daemon.py` and the SQLite state contract.

---

## 40. End-to-End BTC Trace

### Startup

`241646c:live_daemon.py -> state_db.init_db` -> `MarketStateService.state_at` -> `Model1Runner` -> `build_feature_matrix` -> `load_symbol_model` -> `predict_trade_outputs_by_horizon` -> `direct_tcn_forecasts` -> `apply_oos_quality_gate`

### Live cycle

`data` -> `state_db` -> `MarketStateService` -> `feature matrix` -> `TCN inference` -> `assistant_forecasts` -> `quality gate` -> `strategy/risk` -> `GUI`.

### BTC-specific values

- `BTC/USDT` is included in the crypto config and is treated as the canonical BTC asset.
- `BTC/USDT` has the explicit `1h` / `4h` / `8h` / `12h` OOS checks in `assistant_forecasts.py`.
- the crypto TCN path loads per-symbol model artifact and produces public forecast rows.

---

## 41. End-to-End SPY Trace

### Equity pipeline path

`SPY` is represented through the equity config and is typically routed via the equity intraday TCN path and the swing model path when a daily swing category is in use.

### Flow

`config/training_equity_intraday.yaml` -> `MarketStateService.state_at` -> `build_feature_matrix` -> `load_symbol_model` -> `direct_tcn_forecasts` -> `apply_oos_quality_gate` -> `decision_pipeline` -> `risk_engine` -> `state_db` -> `app.py`

### Important note

The runtime keeps the model output separate from the risk and order path; the final decision remains a deterministic report and state change rather than a live order.

---

## 42. Machine-Readable YAML

```yaml
project:
  name: trading_bot
  canonical_commit: 241646c
  status: V1_FREEZE
  order_architecture: no_live_orders
  auto_training: PREPARED_NOT_ACTIVE
  source_of_truth: git_show_241646c

markets:
  crypto:
    - BTC/USDT
    - ETH/USDT
    - SOL/USDT
  equity:
    - QQQ
    - SPY
    - NASDAQ100_PROXY
    - SP500_PROXY
  macro_context:
    - ^VIX
    - ^TNX
    - EURUSD=X
    - GC=F
    - CL=F
    - ^GDAXI

runtime:
  daemon: 241646c:live_daemon.py
  db: 241646c:state_db.py
  gui: 241646c:app.py
  paper_broker: 241646c:paper_broker.py
  risk_engine: 241646c:risk_engine.py
  decision_engine: 241646c:decision_engine.py

entrypoints:
  - 241646c:live_daemon.py
  - 241646c:app.py
  - 241646c:train.py
  - 241646c:train_crypto_intraday.py
  - 241646c:train_equity_intraday.py
  - 241646c:train_swing.py

features:
  matrix_constructor: 241646c:core/ml/features.py:370-390
  htf_alignment: 241646c:core/ml/features.py:341-369
  macro_alignment: 241646c:core/ml/features.py
  causality_policy: causal_forward_fill_and_no_lookahead

models:
  tcn: ACTIVE_V1
  swing: ACTIVE_V1
  chronos: PREPARED_NOT_ACTIVE
  chronos_2: PREPARED_NOT_ACTIVE
  ensemble: ACTIVE_V1

forecast:
  source_file: 241646c:core/ml/assistant_forecasts.py
  public_horizons:
    - 1h
    - 4h
    - 8h
    - 12h
    - 1d
    - 3d
    - 5d
    - 10d
    - 20d
  quality_statuses:
    - UNVERIFIED
    - PARTIALLY_VALIDATED
    - DEGRADED
    - UNAVAILABLE
  freshness_statuses:
    - FRESH
    - STALE
    - UNKNOWN

quality_gating:
  oos_gate: 241646c:core/ml/assistant_forecasts.py:56-128
  freshness_gate: 241646c:core/market_state.py:194-240
  result: usable_for_decision_only_when_gate_is_passed

strategy:
  rule_base: 241646c:core/strategy.py
  decision_pipeline: 241646c:decision_pipeline.py
  model_confirmation: 241646c:core/ml/inference.py
  no_order_path: true

trade_quality:
  status_source: 241646c:decision_pipeline.py
  model_unavailable_status: valid_explicit_state

risk:
  source: 241646c:risk_engine.py
  default_risk_per_trade_pct: 0.02
  default_max_position_fraction: 0.35
  default_atr_buffer_multiple: 1.5
  default_ko_safety_buffer_pct: 0.15

decision:
  source: 241646c:decision_engine.py
  decision_states:
    - VALID_SETUP
    - NO_VALID_SETUP
    - STALE
    - INSUFFICIENT_DATA
    - ERROR

database:
  schema_source: 241646c:state_db.py
  main_tables:
    - market_state
    - context_state
    - signals
    - assistant_forecasts
    - runtime_status
    - strategy_state
    - paper_portfolio
    - open_trades
    - trade_history
    - equity_curve
    - decision_records
  wal_mode: documented_in_state_db_docstring

gui:
  source: 241646c:app.py
  reads_sqlite: true
  writes_orders: false

orders:
  live_execution: NOT_IN_CANONICAL_V1
  paper_execution: 241646c:paper_broker.py

training:
  config_source: 241646c:core/config.py
  sequence_length: 128
  hidden_channels: 96
  num_layers: 6
  dropout: 0.35
  learning_rate: 0.0003
  batch_size: 2048
  max_epochs: 100
  early_stopping_patience: 10
  optimizer: AdamW
  use_amp: false
  n_splits: 5

tests:
  suite: 241646c:tests/
  status_at_v1_freeze: 261 passed, 12 warnings
  status_note: not revalidated in this handoff run

limitations:
  - no live order submission from canonical runtime
  - model and signal stack are decision-support only
  - full historical metric provenance must be traced to commit files
  - any uncommitted local artifact is LOCAL_POST_COMMIT_ARTIFACT unless proven in commit tree
```

---

## 43. One-Page System Description

V1 is a frozen decision-assistant stack. It reads market candles and context, computes feature matrices with causal alignment, infers TCN and swing model outputs, normalizes them into a public forecast contract, applies freshness and OOS gates, and stores the resulting runtime state in SQLite. The GUI reads SQLite and presents the information; the paper broker simulates trades in SQLite but does not constitute a live execution engine. A model can scale or confirm a rule signal, but the architecture explicitly keeps the order path separate. No auto-training occurs during the live loop.

---

## 44. Source Map

### Core source map

- `241646c:live_daemon.py:39-48` — poll cadence constants
- `241646c:live_daemon.py:129-195` — `Model1Runner.infer()` and `infer_by_horizon()`
- `241646c:live_daemon.py:199-214` — `LiveDaemon.__init__()`
- `241646c:live_daemon.py:622-714` — `_latest_equity_session_date`, `_startup_freshness`, `_refresh_runtime_status`, `initialize()`
- `241646c:core/market_state.py:185-204` — `MarketStateService.state_at()` and `health()`
- `241646c:core/market_state.py:205-240` — `decide_alert()`
- `241646c:core/ml/features.py:341-390` — HTF alignment and feature-matrix assembly
- `241646c:core/ml/assistant_forecasts.py:16-128` — `PUBLIC_HORIZONS`, OOS gate, and public forecast quality handling
- `241646c:core/ml/assistant_forecasts.py:159-351` — TCN/swing/ensemble forecast normalization
- `241646c:core/ml/tcn_model.py:45-96` — causal temporal blocks and TCN structure
- `241646c:core/ml/tcn_model.py:145-236` — `TCNTrendModel` and normalization / clipping logic
- `241646c:model_integration.py:18-141` — model comparison dataclasses and forecast structure
- `241646c:model_integration.py:258-329` — `infer_chronos2_forecast` and model routing
- `241646c:risk_engine.py:11-32` — risk constants
- `241646c:risk_engine.py:72-122` — `TradeSetup`
- `241646c:risk_engine.py:145-240` — risk calculations and leverage logic
- `241646c:decision_engine.py:205-218` — strategy and signal quality selection
- `241646c:decision_pipeline.py:22-38` — `SetupStatus` state enum and decision state semantics
- `241646c:state_db.py:17-123` — main runtime schema
- `241646c:state_db.py:308-358` — `init_db()` and migration logic
- `241646c:app.py:1-177` — GUI entrypoint and read-only dashboard

### Direct requirement check

- LiveDaemon.initialize — present in `241646c:live_daemon.py:689-714`
- live loop — `241646c:live_daemon.py` runtime path
- runtime refresh — `241646c:live_daemon.py:668-714`
- startup freshness — `241646c:live_daemon.py:637-678`
- equity session date — `241646c:live_daemon.py:622-637`
- MarketStateService.state_at — `241646c:core/market_state.py:185-192`
- MarketStateService.health — `241646c:core/market_state.py:194-204`
- forecast contract — `241646c:core/ml/assistant_forecasts.py:16-128`
- PUBLIC_HORIZONS — `241646c:core/ml/assistant_forecasts.py:16`
- OOS gate — `241646c:core/ml/assistant_forecasts.py:56-128`
- freshness gate — `241646c:core/market_state.py:194-240`
- feature matrix — `241646c:core/ml/features.py:370-390`
- HTF alignment — `241646c:core/ml/features.py:341-369`
- macro alignment — `241646c:core/ml/features.py` and `241646c:core/config.py:365-433`
- TCN — `241646c:core/ml/tcn_model.py:145-236`
- inference — `241646c:core/ml/inference.py:27-146`
- model routing — `241646c:model_integration.py:258-329`
- decision engine — `241646c:decision_engine.py:205-218`
- decision pipeline — `241646c:decision_pipeline.py`
- trade quality — `241646c:decision_pipeline.py:40-128`
- risk calculations — `241646c:risk_engine.py:145-240`
- SQLite initialization — `241646c:state_db.py:308-358`
- each important DB writer — `state_db.py` plus `paper_broker.py` and `live_daemon.py` runtime write paths
- GUI entrypoint — `241646c:app.py:1-177`

---

## Final Quality Block

```text
HANDOVER_QUALITY=EXCELLENT
CANONICAL_COMMIT=241646c
SOURCE_MAP_COVERAGE=SELECTIVE_NOT_100_PERCENT
SQL_SCHEMA_COVERAGE=HIGH
TRAINING_PARAMETER_COVERAGE=HIGH
DECISION_STATE_COVERAGE=VERIFIED
ACTIVE_COMPONENT_COVERAGE=VERIFIED
UNVERIFIED_CLAIMS=3
V1_TECHNICAL_HANDOVER_STATUS=COMPLETE_WITH_LIMITATIONS
```

---

## Final Summary

- V1 is the frozen `241646c` code tree, not the current local working tree.
- The canonical runtime is `live_daemon.py`, persisted via `state_db.py`.
- The canonical feature, forecast, and OOS logic is in `core/ml/features.py` and `core/ml/assistant_forecasts.py`.
- The canonical risk and decision layers are in `risk_engine.py` and `decision_engine.py`.
- The canonical GUI is `app.py` and it remains read-only.
- The live architecture is not an auto-trading engine and does not auto-train.
- The project is a decision-support architecture with a clear separation between runtime state, risk, strategy, paper simulation, and order-free decisions.

This document is written to the standard: every key claim is anchored to an explicit commit-contained file or to a clearly marked `UNKNOWN_NOT_VERIFIED` placeholder when the exact source value could not be reconstructed from the canonical tree.
