# ML Multi-Asset Dataflow Audit

Audit date: 2026-09-23  
Scope: read-only forensic trace of the current Model 1A/1B/2 training, feature, regime, strategy-selection, and BTC/ETH decision paths.

## Final Classification

**CLEAN_MULTI_ASSET_CONTEXT_ARCHITECTURE_WITH_TARGET_SPECIFIC_MODELS**

No evidence was found that one traded target's raw OHLCV, labels, or model outputs
are pooled into another target's current model. The system is nevertheless not
globally single-asset: Model 1B and Model 2 intentionally use cross-asset macro
context, and the generic ML pipeline can reuse one prepared context matrix across
separate per-symbol models. The current production daemon has a narrower split:
Model 1A (BTC/ETH) is target-specific, while Model 1B/Model 2 (QQQ/SPY) use
Gold, WTI, DAX, VIX, US10Y, and EURUSD context.

This is an architecture/dataflow classification, not a claim that all model
artifacts are statistically strong. Several stored Model 1 artifacts show
directional collapse or weak test metrics, and the current root trainer records a
single `fixed_test` entry despite importing a multi-fold splitter.

## Ten-Point Executive Summary

1. **Current production universe:** Model 1A targets BTC/USDT and ETH/USDT; Model 1B targets QQQ/SPY through logical proxy names; Model 2 targets QQQ/SPY separately.
2. **No pooled target training:** `run_training_pipeline` iterates over target symbols and calls `train_symbol_model` once per symbol. The `assets` array in metadata identifies the configured universe, not a pooled feature table.
3. **Current BTC/ETH features:** stored Model 1A metadata contains target-local technical features plus causal 15m/1h/4h/1d higher-timeframe features. It contains no VIX, Gold, ETF, BTC-peer, ETH-peer, or breadth columns.
4. **Current BTC/ETH live inference:** `live_daemon.py` passes `macro_prices=None`; the dedicated crypto config has empty `context_assets`, empty `macro_symbols`, and `breadth_enabled=false`. The attempted breadth call therefore returns an empty series.
5. **Equity cross-asset context is real:** current Model 1B metadata contains VIX, US10Y, EURUSD, Gold, WTI, DAX, and `macro_data_age_hours` feature families. These are context features for each separate QQQ/SPY model.
6. **Model 2 cross-asset context is real:** its dataset builds daily target-local features plus lagged daily macro features from VIX, US10Y, EURUSD, Gold, WTI, and DAX, with target-local 1h/5m entry branches.
7. **Generic mixed-asset path is not one pooled model:** `configs/training.yaml` configures QQQ, SPY, BTC, and ETH plus macro context, but the trainer still constructs one model per target. Root `data/models/*` artifacts with macro/breadth columns are therefore separate/generic-generation evidence, not proof of current Model 1A live usage.
8. **Regime recognition in the current daemon:** the root daemon does not call the legacy `core/ml/regime.py` gate. Current decisions use Model 1A scores for crypto and Model 1B/Model 2 scores plus matching-asset fusion for equities. The older `execution/live_trader.py`/`core/strategy.py` ML-confirmation path is separate.
9. **Trading-decision coupling:** BTC/ETH can be coupled downstream through the shared paper-wallet risk budget and maximum concurrent trades, but that is portfolio allocation/execution coupling, not cross-asset training leakage or a feature entering the other asset's model.
10. **Integrity status:** feature construction, label separation, macro lagging, and higher-timeframe close labeling are designed causally. However, the root trainer's imported `purged_walk_forward_splits` is not invoked in the inspected current path; current metadata shows `fixed_test`, so multi-fold validation is **PARTIAL_IMPLEMENTATION**, not verified.

## Runtime Dataflow

```text
prepared target OHLCV
    -> target-local technical features
    -> optional macro matrix / optional breadth basket
    -> causal feature matrix
    -> per-target sequence + triple-barrier labels
    -> one checkpoint per target
    -> live daemon inference
    -> SQLite signals
    -> paper broker score/fusion/risk gates
    -> BTC/ETH or QQQ/SPY paper decisions
```

The current root runtime is implemented in [live_daemon.py](../../live_daemon.py),
with feature construction in [core/ml/features.py](../../core/ml/features.py),
per-symbol training in [core/ml/train.py](../../core/ml/train.py), and execution
gates in [paper_broker.py](../../paper_broker.py).

## Asset Usage Matrix

| Path | Target assets | Raw target data pooled? | Cross-asset features | Current decision use | Classification |
|---|---|---:|---|---|---|
| Model 1A crypto | BTC/USDT, ETH/USDT | No; one model per target | None in current feature manifests; target-local HTF only | `crypto_sniper` score, threshold 80 | `CLEAN_TARGET_SPECIFIC` |
| Model 1B equity | QQQ, SPY via proxies | No; one model per proxy | VIX, US10Y, EURUSD, Gold, WTI, DAX | Intraday snapshot; equity entry leg | `CLEAN_MULTI_ASSET_CONTEXT` |
| Model 2 swing | QQQ, SPY | No; one checkpoint per ticker | VIX, US10Y, EURUSD, Gold, WTI, DAX | Swing setup; combined with Model 1B | `CLEAN_MULTI_ASSET_CONTEXT` |
| Generic `training.yaml` | QQQ, SPY, BTC/USDT, ETH/USDT | No; loop remains per symbol | Configured macro context; breadth if enabled | Research/config path, not current root daemon config | `CONFIGURED_NOT_CURRENT_RUNTIME` |
| Legacy/generic `data/models/*` | Mixed symbols, including SOL artifact | Not established as current production path | Macro and/or breadth columns present | Not loaded by current root daemon model directories | `LEGACY_OR_SEPARATE` |
| Shared wallet/paper broker | BTC, ETH, QQQ, SPY as active trades | N/A | No feature input; shared risk state | Open-risk budget, max concurrent trades, sizing | `PORTFOLIO_COUPLING_ONLY` |

## Training and Feature Findings

### Model 1A: BTC/ETH

The dedicated configuration in [configs/training_crypto_intraday.yaml](../../configs/training_crypto_intraday.yaml)
contains only BTC/USDT and ETH/USDT, with no context assets, no macro symbols, and
disabled breadth. Its stored BTC and ETH metadata has the same 51-column family:

- target-local log returns, realized volatility, ATR, EMA/Donchian, RSI, volume, and time encodings;
- causal 15m, 1h, 4h, and 1d higher-timeframe features derived from that target's own OHLCV;
- no `VIX_*`, `TNX_*`, `EURUSDX_*`, `GCF_*`, `CLF_*`, `GDAXI_*`, or `breadth_*` columns.

The metadata `assets` value contains both BTC and ETH because both are configured
for that training run. It does not turn BTC and ETH rows into a pooled sample set.
`train_symbol_model(symbol, ...)` receives one symbol's OHLCV and writes one
symbol checkpoint.

### Model 1B: QQQ/SPY

The dedicated configuration in [configs/training_equity_intraday.yaml](../../configs/training_equity_intraday.yaml)
contains the six context assets. The stored proxy metadata contains the macro
feature families `VIX`, `TNX`, `EURUSDX`, `GCF`, `CLF`, and `GDAXI`, plus the
causal age feature `macro_data_age_hours`. QQQ and SPY still retain separate
target-local technical and higher-timeframe blocks.

### Model 2: swing

The two checkpoints in `models/checkpoints` report 54 features and separate
assets. [core/ml/swing_data.py](../../core/ml/swing_data.py) builds:

- 90-day target-local daily technical history;
- previous-observation-aligned daily macro returns/volatility/availability for VIX, US10Y, EURUSD, Gold, WTI, and DAX;
- target-local 1h and 5m entry branches mapped to New York dates;
- future daily returns/MFE/MAE/duration only as targets, not as feature columns.

The daily macro alignment uses `reindex(...).ffill().shift(1)`, and the intraday
daily lookup uses the prior daily position (`side="left"`). These are intentional
causality controls.

### Generic and legacy artifacts

The older root `data/models` metadata for BTC/ETH contains macro and breadth
columns, while the older SOL artifact has a different macro universe and five
walk-forward entries. Those artifacts demonstrate that the repository has had a
broader cross-asset ML generation. They do not establish current Model 1A usage:
the current daemon loads only `data/models/model1a_crypto` for crypto and
`data/models/model1b_equity` for equity.

## Regime Recognition and Strategy Selection

There are two distinct mechanisms:

1. The legacy TrendML path in [core/ml/inference.py](../../core/ml/inference.py)
   can call `apply_regime_gated_ml_confirmation`, which classifies the passed
   signal close series using [core/ml/regime.py](../../core/ml/regime.py). That
   helper is asset-local at runtime despite an outdated BTC wording in its
   docstring. It scales an existing rule position; it does not create a trade.
2. The current root daemon does not call that helper. It writes Model 1A crypto
   signals directly, writes Model 1B intraday snapshots, and writes Model 2 swing
   snapshots. [core/signal_orchestrator.py](../../core/signal_orchestrator.py)
   requires matching QQQ/SPY assets, same direction, score thresholds, and fresh
   context for equity alert fusion. [paper_broker.py](../../paper_broker.py)
   requires a crypto score threshold or both equity scores before opening a trade.

Therefore, current strategy selection is model-family and asset-specific, with
cross-model agreement for QQQ/SPY. It is not a BTC-derived regime switch for ETH.

## BTC/ETH Decision Trace

For each BTC/ETH refresh, the current root daemon:

1. prepares only the dedicated crypto asset specs;
2. loads that symbol's 5m OHLCV cache;
3. computes ATR and target-local feature columns;
4. passes `None` for macro prices and the empty/disabled breadth result;
5. runs that symbol's own checkpoint on its latest sequence;
6. writes a `crypto_sniper` row for the same asset;
7. lets the paper broker apply score, direction, open-market, health, risk-budget, and concurrency gates.

BTC can affect ETH only after model inference through shared portfolio state such
as open risk and the maximum concurrent trade count. ETH cannot receive BTC's raw
price, label, model score, or feature vector through this root path.

## Temporal and Leakage Controls

Evidence supporting causal intent:

- technical features use current/prior rolling and exponential windows;
- higher-timeframe bars use `label="right", closed="right"` and are aligned after close;
- macro features are shifted by a reporting lag and forward-filled;
- triple-barrier future highs/lows are explicitly labels, with incomplete tail rows removed;
- `build_row_level_dataset` keeps original feature columns and excludes label columns;
- scaler statistics are fitted inside `TCNTrendModel.fit` on the supplied training slice;
- chronological date splitting and purge checks exist in `chronological_purged_split`.

Residual audit findings:

- `core/ml/train.py` imports `purged_walk_forward_splits` but the current root
  `train_symbol_model` path uses one `chronological_purged_split` and appends a
  single `fold: fixed_test` metric. `n_splits=5` is therefore not verified as
  active for current artifacts.
- The final model is fit on train plus validation and is appropriate for
  prospective inference only. Historical scoring must use the OOS confidence
  artifact; this contract exists in `load_oos_confidence`, but current root
  Model 1A/1B daemon inference intentionally uses the final model for live use.
- Exact raw cache coverage was not independently recomputed for every ignored or
  unavailable cache path in this audit. Stored checkpoint metadata is authoritative
  for the feature manifest and recorded cutoff, but not a substitute for a fresh
  data-lineage manifest.

## Artifact Register

| Artifact | Stored evidence | Interpretation |
|---|---|---|
| `data/models/model1a_crypto/BTC-USDT_tcn_meta.json` | symbol BTC/USDT; assets BTC+ETH; no macro/breadth columns; cutoff 2026-09-22 04:20 UTC | current Model 1A BTC checkpoint |
| `data/models/model1a_crypto/ETH-USDT_tcn_meta.json` | symbol ETH/USDT; assets BTC+ETH; no macro/breadth columns; cutoff 2026-09-22 04:25 UTC | current Model 1A ETH checkpoint |
| `data/models/model1b_equity/NASDAQ100_PROXY_tcn_meta.json` | macro columns for six context assets; cutoff 2026-09-21 19:10 UTC | current Model 1B QQQ proxy checkpoint |
| `data/models/model1b_equity/SP500_PROXY_tcn_meta.json` | macro columns for six context assets; cutoff 2026-09-21 19:35 UTC | current Model 1B SPY proxy checkpoint |
| `models/checkpoints/qqq_swing.json` | 54 features; QQQ; six macro symbols; 2026-09-23 training date | current Model 2 QQQ metadata |
| `models/checkpoints/spy_swing.json` | 54 features; SPY; six macro symbols; 2026-09-23 training date | current Model 2 SPY metadata |
| `data/models/BTC-USDT_tcn_meta.json` and peers | mixed macro/breadth manifests and collapsed fixed-test outputs | older/generic generation; not current Model 1A directory |
| `data/models/SOL-USDT_tcn_meta.json` | separate five-fold legacy artifact | legacy/separate evidence, not current root trio |

## Required Follow-Up Checks

This audit makes no code changes. The highest-value verification items are:

1. persist a machine-readable feature/data manifest per training run, including every source asset, cache path, effective range, and feature column;
2. either wire `n_splits`/purged folds into the root trainer or remove the misleading configuration/import and rename metadata fields;
3. record the exact Model 2 split timestamps and purge boundaries in checkpoint metadata;
4. add a runtime assertion that the loaded checkpoint feature manifest matches the active provider/context configuration;
5. maintain separate artifact namespaces and provenance IDs for current production models versus generic/legacy research models.

## Bottom Line

The broader multi-asset design is real and legitimate where it is actually
wired: equity models use macro context, and the generic pipeline supports shared
context across separate target models. The current BTC/ETH production model is
not a pooled cross-asset model and does not currently use Gold, ETFs, or macro
assets as features. BTC/ETH are coupled later by shared-wallet risk and execution
policy only. No `LEAKAGE_FOUND` classification is supported by the inspected
current root path; the main unresolved issue is validation/artifact provenance,
not cross-asset contamination.

## Evidence Register

| ID | Evidence | Finding |
|---|---|---|
| E01 | [configs/training_crypto_intraday.yaml](../../configs/training_crypto_intraday.yaml) | BTC/ETH-only targets; empty macro context; breadth disabled |
| E02 | [configs/training_equity_intraday.yaml](../../configs/training_equity_intraday.yaml) | QQQ/SPY targets plus VIX/US10Y/EURUSD/Gold/WTI/DAX context |
| E03 | [core/ml/train.py](../../core/ml/train.py) | per-symbol training loop; fixed chronological split in current path |
| E04 | [core/ml/features.py](../../core/ml/features.py) | target-local technical/HTF features, optional macro/breadth joins, causal alignment |
| E05 | [live_daemon.py](../../live_daemon.py) | current Model 1A receives `None` macro context; current equity path loads macro matrix |
| E06 | [core/ml/swing_data.py](../../core/ml/swing_data.py) | Model 2 target-local features plus lagged macro context |
| E07 | [core/signal_orchestrator.py](../../core/signal_orchestrator.py) | matching-asset Model 1B/Model 2 fusion for QQQ/SPY |
| E08 | [paper_broker.py](../../paper_broker.py) | score gates and shared-wallet downstream coupling |
| E09 | current Model 1A metadata under `data/models/model1a_crypto` | no macro/breadth feature columns |
| E10 | current Model 1B metadata under `data/models/model1b_equity` | six macro feature families are persisted in the trained manifests |
| E11 | `models/checkpoints/qqq_swing.json`, `spy_swing.json` | 54-feature Model 2 artifacts and six configured macro symbols |
| E12 | [core/ml/inference.py](../../core/ml/inference.py), [core/ml/regime.py](../../core/ml/regime.py) | legacy regime-gated confirmation exists separately from root daemon |
