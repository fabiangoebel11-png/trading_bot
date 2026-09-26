# Frozen Data, Feature, and Score Specification

Status: **FROZEN v1.0**

This document is based on the files currently present in the repository on 2026-09-23. File names are not treated as proof of a source; headers, timestamps, row counts, and intervals were inspected. A session gap in an equity series is not automatically classified as a data error.

## 1. Data inventory

### Target assets

| Asset | Actual source and files | Timeframes observed | First -> last observed | Rows | Gaps / quality | Real or proxy | Usable for |
|---|---|---|---|---:|---|---|---|
| BTC/USDT | CCXT caches: `data/market/BTC_USDT/{5m,15m,1h,4h,1d}/ccxt.parquet`; matching CSV caches under `data/ohlcv_BTC-USDT_*` | 5m, 15m, 1h, 4h, 1d | 2017-08-17 -> 2026-09-22 (1h CCXT: 79,639 rows) | see files | OHLCV NaN rows 0; 1h has 28 interval deviations from the 1h median, 5m 16, 4h 8; requires closed-candle and duplicate checks | Real BTC/USDT market data from CCXT cache | Target for crypto rule analysis and Model 1A inference |
| ETH/USDT | CCXT caches: `data/market/ETH_USDT/{5m,15m,1h,4h,1d}/ccxt.parquet`; matching CSV caches under `data/ohlcv_ETH-USDT_*` | 5m, 15m, 1h, 4h, 1d | 2017-08-17 -> 2026-09-22 (1h CCXT: 79,639 rows) | see files | OHLCV NaN rows 0; 1h has 28 interval deviations, 5m 16, 4h 8; requires closed-candle and duplicate checks | Real ETH/USDT market data from CCXT cache | Target for crypto rule analysis and Model 1A inference |
| QQQ | `data/market/NASDAQ100_PROXY/`: 5m `twelve_data.parquet`, 15m/1h/4h `local_derived.parquet`, 1d `yfinance.parquet`; provider symbol QQQ in `configs/training_equity_intraday.yaml` | 5m, 15m, 1h, 4h, 1d | 2023-09-25 -> 2026-09-23 for intraday/day caches; 1d has 6,928 rows | see files | Intraday session gaps are expected; 5m cache has 9 OHLCV NaN rows and 754 interval deviations; 1h/4h have no NaN rows and 752 session interval deviations; clean before use | Proxy representation of the QQQ ETF, not a crypto/native continuous market | Target for equity rule analysis; Model 1B configured target; Model 2 swing target only through the daemon path |
| SPY | `data/market/SP500_PROXY/`: 5m `twelve_data.parquet`, 15m/1h/4h `local_derived.parquet`, 1d `yfinance.parquet`; provider symbol SPY in `configs/training_equity_intraday.yaml` | 5m, 15m, 1h, 4h, 1d | 2023-09-25 -> 2026-09-23 for intraday/day caches; 1d has 8,470 rows | see files | Intraday session gaps are expected; 5m cache has 11 OHLCV NaN rows and 752 interval deviations; 1h/4h have no NaN rows and 752 session interval deviations; clean before use | Proxy representation of the SPY ETF | Target for equity rule analysis; Model 1B configured target; Model 2 swing target only through the daemon path |

### Context and other observed markets

| Context asset | Actual repository evidence | Timeframe / coverage | Quality and allowed use |
|---|---|---|---|
| ES_F_PROXY | No asset directory or file named `ES_F_PROXY` was found. `ES=F` is a column in `data/ml_macro_ESF_NQF_VIX_URTH_2600d.csv` and `data/ml_macro_ESF_NQF_VIX_URTH_1095d.csv`. | Daily, 2019-08-12 -> 2026-09-22, 1,793 rows in the 2600d cache | Context only if loaded from that combined cache; not a separately verified target feed |
| NQ_F_PROXY | No asset directory or file named `NQ_F_PROXY` was found. `NQ=F` is a column in the same combined ML macro caches. | Daily, 2019-08-12 -> 2026-09-22, 1,793 rows | Context only if loaded from that combined cache; not a separately verified target feed |
| VIX | `data/market/VIX/1d/yfinance.parquet`; also `^VIX` in the combined ML macro cache | Daily, 1990-01-02 -> 2026-09-23, 9,250 rows; combined ML cache 2019-08-12 -> 2026-09-22 | Real yfinance VIX context; equity Model 1B config references `^VIX` |
| URTH | No standalone `data/market/URTH` file was found. `URTH` is a column in `data/ml_macro_ESF_NQF_VIX_URTH_2600d.csv` and `data/ml_macro_ESF_NQF_VIX_URTH_3600d.csv`. | Daily, 2016-11-14 -> 2026-09-22 in the 3600d cache, 2,481 rows | Context only through the combined cache; no standalone provider cache claim |
| US10Y | `data/market/US10Y/1d/yfinance.parquet` (`^TNX`) | Daily, 1968-10-10 -> 2026-09-22, 14,494 rows | Real yfinance rate context; equity config references `^TNX` |
| EURUSD | `data/market/EURUSD/1d/yfinance.parquet` (`EURUSD=X`) | Daily, 2003-12-01 -> 2026-09-23, 5,919 rows | Real yfinance FX context; 128 rows contain OHLCV NaN and require cleaning |
| GOLD | `data/market/GOLD/1d/yfinance.parquet` (`GC=F`) | Daily, 2000-08-30 -> 2026-09-23, 6,541 rows | Real yfinance commodity context; 441 rows contain OHLCV NaN and require cleaning |
| WTI | `data/market/WTI/1d/yfinance.parquet` (`CL=F`) | Daily, 2000-08-23 -> 2026-09-23, 6,549 rows | Real yfinance commodity context; 9 rows contain OHLCV NaN and require cleaning |
| DAX | `data/market/DAX/1d/yfinance.parquet` (`^GDAXI`) | Daily, 1987-12-30 -> 2026-09-23, 9,794 rows | Real yfinance equity-index context; equity config references `^GDAXI` |
| Crypto breadth basket | `data/ml_breadth_BNB-USDT_XRP-USDT_ADA-USDT_DOGE-USDT_LINK-USDT_1h_2500d.csv` | 1h, 2019-11-18 -> 2026-09-22, 59,966 rows | Derived equal-weight breadth return; `breadth_enabled: false` in active Model 1A/1B configs, therefore not an active input |

The `data/market_crypto_2017` directory contains additional BTC/ETH CCXT caches. These are real cached market data, but they are not silently substituted for a configured source; the selected file and timeframe must be recorded with each analysis.

## 2. Frozen target/context separation

Every model request has exactly one target. Context is separate and cannot be used as the target by name substitution.

- Crypto Model 1A: `TARGET = BTC/USDT` or `ETH/USDT`; `CONTEXT = none` in the active YAML (`context_assets: {}`, `macro_symbols: []`, `breadth_enabled: false`). Higher-timeframe features are derived from the target's own OHLCV, not external context assets.
- Equity Model 1B: `TARGET = NASDAQ100_PROXY` (provider symbol QQQ) or `SP500_PROXY` (provider symbol SPY); `CONTEXT = VIX, US10Y, EURUSD, GOLD, WTI, DAX` only when the configured macro cache is available and time-aligned. The active feature metadata confirms macro columns for these context series in the equity artifact.
- Model 2: `TARGET = QQQ` or `SPY`; `CONTEXT = model-specific inputs in `build_swing_dataset` only if present`. Direct assistant activation is **NOT IMPLEMENTED**.

`ES_F_PROXY`, `NQ_F_PROXY`, and `URTH` are not standalone asset IDs in this repository. They may only be mentioned as columns of a verified combined macro cache.

## 3. Model inputs

### Active Model 1A: crypto TCN artifact

- Input asset: BTC/USDT or ETH/USDT.
- Input timeframe: artifact supports `5m`, `15m`, `1h`, `4h`, and `1d`; the current direct assistant smoke uses a closed `1h` frame. The training YAML's base timeframe is `5m`; this configuration/artifact mismatch must be resolved before claiming a single production training timeframe.
- Forecast horizon: persisted metadata has no `label_horizon` value (`null`); the YAML default used by the adapter is 72 bars. The exact horizon is therefore **NOT IMPLEMENTED as persisted artifact metadata**.
- Raw inputs: target OHLCV.
- Derived features from `core/ml/features.py`: multi-lag log returns (1/3/6/12/24/48), realized volatility (12/48/288), ATR percentage, EMA distances and regime (20/100), Donchian distances, RSI(14), volume z-score, cyclical hour/day features, plus causal same-target higher-timeframe features for 15m/1h/4h/1d with data-age columns.
- Optional context: none active. Breadth features are disabled in the active YAML.
- Forbidden inputs: future return, future direction, future MAE/MFE, labels, or any value from the forecast window. The runtime feature builder receives only the closed frame and past-derived rolling/resampled values.

### Active Model 1B: equity TCN artifact

- Input asset: `NASDAQ100_PROXY`/QQQ or `SP500_PROXY`/SPY.
- Input timeframe: artifact supports `5m`, `15m`, `1h`, `4h`, and `1d`; active YAML base is `5m`.
- Forecast horizon: `label_horizon` is not persisted in the inspected metadata. **NOT IMPLEMENTED as a frozen artifact field.**
- Raw inputs: target OHLCV from Twelve Data/yfinance-derived caches.
- Derived features: the target features above, plus causal macro return/volatility/z-score features for VIX, TNX, EURUSD, GOLD, WTI, and DAX, and macro data age; same-target higher-timeframe features.
- Optional context: only the listed macro series when present, aligned causally. Breadth is disabled.
- Direct assistant inference: **NOT IMPLEMENTED**. The active direct bridge currently invokes only the crypto adapter; the equity result may use a persisted daemon snapshot.

### Model 2: equity swing model

- Input assets: QQQ and SPY.
- Configured target horizons: `[1, 3, 5, 10, 20]` in `configs/training_swing.yaml`.
- Raw/derived inputs and exact output contract: implemented in `core/ml/swing_data.py` and `core/ml/swing_model.py`, and used by `live_daemon.py`.
- Direct assistant inference: **NOT IMPLEMENTED**. No direct call from `app.py`/`decision_pipeline.py` is claimed here.

## 4. Model outputs

The actually exercised Model 1 TCN returns exactly: `probabilities`, `opportunity`, `expected_return`, `expected_duration`, `expected_mfe`, and `expected_mae`. The direct adapter exposes expected return, favorable move (MFE), adverse move (MAE), direction, score, uncertainty label, timestamp, and status. It does not expose forecast quantiles or a forecast range. Those are **NOT IMPLEMENTED**.

Model output -> normalization -> model score:

`continuous_opportunity_score()` computes:

- payoff = favorable / max(favorable + abs(adverse), epsilon)
- directional confidence = clip(abs(expected_return) / 0.01, 0, 1)
- `MODEL_SCORE = 100 * (0.55 * opportunity + 0.25 * payoff + 0.20 * directional_confidence)`
- direction = LONG if expected return > 0, SHORT if expected return < 0, otherwise UNCERTAIN

This score is computed from model heads only. It is not copied from the rule score. A missing or invalid model returns `MODEL_UNAVAILABLE` and has no model score.

## 5. Rule score

`select_strategy()` in `decision_pipeline.py` is the sole rule-score source:

- Trend breakout: +35 EMA trend alignment, +40 Donchian breakout agreeing with trend, +25 agreeing momentum.
- Momentum: +40 nonzero momentum, +35 momentum agrees with trend, +25 acceptable volatility regime.
- Mean reversion: +45 RSI/Bollinger extension, +40 range regime, +15 no strong trend.
- Regime adaptive: `0.85 * max(trend_score, momentum_score, mean_score)` when direction is known.

Scores are capped at 100. The selected candidate's score is `RULE_SCORE`. No model output is included in this calculation.

## 6. Risk/setup score

`risk_setup_score()` is separate from prediction quality and is frozen in `decision_pipeline.py`:

- stop quality: `clip(ATR% / stop_distance%, 0, 1)`
- reward quality: `clip((risk_reward - 0.5) / 1.5, 0, 1)`
- adverse-move quality: `clip(1 - abs(MAE) / max(abs(MFE)+abs(MAE), epsilon), 0, 1)`
- exposure quality: `clip(1 - margin / max(500, 1), 0, 1)`
- `RISK_SETUP_SCORE = 100 * (0.30*stop + 0.30*reward + 0.20*adverse + 0.20*exposure)`

The underlying stop, take-profit, portfolio-risk, ATR, volatility, and leverage constraints remain those in `risk_engine.py`. This score is not a forecast probability.

## 7. Final score

Frozen scoring version: **`v1.0`**

`FINAL_SCORE = clip(0.45 * RULE_SCORE + 0.35 * MODEL_SCORE + 0.20 * RISK_SETUP_SCORE, 0, 100)`.

When the model is unavailable, `MODEL_SCORE` is absent in the result and contributes zero; the status remains explicit. A direction disagreement applies the existing deterministic 0.5 preliminary penalty before risk sizing. Weights are constants, not changed during normal operation.

## 8. Registry

| model_id | type | version | asset scope | category | input TF | horizon | features/context | output | training | artifact | validation | active |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `model1a_crypto_tcn` | TCN multitask | `tcn-multitask-v1` | BTC/USDT, ETH/USDT | INTRADAY / multi-TF artifact | 5m configured; 1h direct smoke | 72 bars by adapter; artifact metadata `null` | target OHLCV-derived technical + target HTF; no external context | opportunity, expected return/duration/MFE/MAE, probabilities | TCN checkpoint; exact training method not persisted in inspected metadata | `data/models/model1a_crypto/*_tcn.pt` + `_meta.json` | artifact loads; direct BTC/ETH closed-candle smoke passed; full score calibration **NOT IMPLEMENTED** | ACTIVE for direct crypto path |
| `model1b_equity_tcn` | TCN multitask | `tcn-multitask-v1` | QQQ/SPY proxies | INTRADAY | 5m configured | artifact metadata `null` | target OHLCV + macro contexts + target HTF | same Model 1 heads | exact method metadata **NOT IMPLEMENTED** | `data/models/model1b_equity/*_tcn.pt` + `_meta.json` | artifact metadata present; direct assistant smoke **NOT IMPLEMENTED** | NOT ACTIVE in direct assistant path |
| `model2_equity_swing` | SwingModel | metadata version **NOT IMPLEMENTED** | QQQ, SPY | SWING | daily + intraday inputs per `build_swing_dataset` | configured 1/3/5/10/20 horizons | swing dataset inputs; exact frozen feature list **NOT IMPLEMENTED here** | swing outputs from `swing_model.py` | training metadata **NOT IMPLEMENTED** | `models/checkpoints/{qqq,spy}_swing.pt` | daemon path exists; direct assistant validation **NOT IMPLEMENTED** | NOT ACTIVE in direct assistant path |

## 9. Required separation summary

TARGET and CONTEXT are distinct. Model outputs are distinct from rule outputs. Risk/setup quality is distinct from forecast probability. The final score is versioned and deterministic. Any row marked `NOT IMPLEMENTED` is intentionally not promoted to an active claim until its artifact metadata, input contract, and end-to-end assistant invocation are verified.
