# Phase 3.3 ML Research Design Freeze

Status: FROZEN. This document defines the next research design only. It does not start Phase 4, train a production model, change strategy selection, or modify paper/live execution.

## 1. Actual Local Data

Canonical target inputs are the existing local Parquet caches under `data/market_crypto_2017`.

| Asset | Timeframe | Start UTC | End UTC | Rows | Median interval | Gaps | Duplicate timestamps | Invalid OHLC | Status |
|---|---|---|---|---:|---|---:|---:|---:|---|
| BTC/USDT | 1h | 2017-08-17 04:00 | 2026-09-21 21:00 | 79,618 | 1h | 28 | 0 | 0 | AVAILABLE |
| BTC/USDT | 4h | 2017-08-17 04:00 | 2026-09-22 16:00 | 19,926 | 4h | 8 | 0 | 0 | AVAILABLE |
| ETH/USDT | 1h | 2017-08-17 04:00 | 2026-09-21 21:00 | 79,618 | 1h | 28 | 0 | 0 | AVAILABLE |
| ETH/USDT | 4h | 2017-08-17 04:00 | 2026-09-22 16:00 | 19,926 | 4h | 8 | 0 | 0 | AVAILABLE |

The canonical OHLCV timestamps are timezone-naive in the files and are normalized to UTC by the loader. Crypto data is 24/7. There are no duplicate timestamps or invalid OHLC rows; the reported gaps are retained as gaps and are not forward-filled.

Additional local crypto files exist for SOL, BNB, XRP, ADA, DOGE, and LINK, mostly at 1h/5m. SOL has a shorter real history beginning 2020-08-11. These assets are not added as targets in this freeze. Their presence is recorded, not converted into a research claim.

Daily local market files also exist for DAX, GOLD, EURUSD, NASDAQ-100, SP500, US10Y, VIX, and WTI. They are not part of the default feature set and are not relabeled as crypto prediction targets.

## 2. Target and Context Assets

### Frozen targets

- BTC/USDT
- ETH/USDT

### Frozen context assets

- ES_F_PROXY: ES futures proxy
- NQ_F_PROXY: NQ futures proxy
- VIX: volatility-index proxy
- URTH: world-equity ETF proxy

The names above are intentional. They must not be described as SPY, QQQ, or generic equities. The dataset contract enforces `target_asset != context_asset`, rejects unknown context names, and rejects context assets as targets.

The proxy branch `crypto_context_proxy_v1` remains `OPTIONAL_NOT_PROMOTED` because Phase 3.2 found no robust reproducible incremental value. The default `crypto_core_v1` contains no proxy features.

## 3. Observation Timeframes and Horizons

All requested combinations have more than 19,000 raw labelable rows at the shorter 4h source and more than 79,000 at the 1h source before feature warmup. They are therefore frozen as `READY` by a data sufficiency rule, not by observed performance:

- 1h -> 1h, 2h, 4h, 8h, 12h, 24h, 3d, 7d
- 4h -> 7d, 14d, 21d, 28d

Horizon and observation timeframe remain separate fields. A label uses future observations after the prediction timestamp; the final `horizon` rows cannot receive a complete label and are dropped. Overlapping labels are expected for horizons longer than one bar.

## 4. Label Contract

Primary frozen benchmark label:

`future_return[t,h] = close[t+h] / close[t] - 1`

`direction_raw_v1 = 1 if future_return > 0 else 0`

This is a neutral statistical benchmark, not a trade recommendation. A secondary pre-registered diagnostic is `direction_cost_aware_v1`, using a fixed 20 bps round-trip threshold. The threshold is not tuned from validation or holdout results. Regression labels are future return and future log return. MAE, MFE, stop-hit probability, and holding time are future outcomes only; they cannot become input features.

## 5. Frozen Feature Sets

### `crypto_core_v1` (FROZEN_DEFAULT)

- returns
- momentum
- RSI
- SMA/EMA distances
- volatility
- range
- volume
- breakout structure

All features are computed from closed target candles and causal rolling calculations. No new feature is added because it is theoretically interesting.

### `crypto_context_proxy_v1` (OPTIONAL_NOT_PROMOTED)

The separate proxy branch may add ES_F_PROXY, NQ_F_PROXY, VIX, and URTH through the existing backward as-of adapter. It uses a 2-day conservative availability lag and a 5-day maximum staleness. It remains outside the default research path.

## 6. Leakage Contract

Every feature contract must contain `feature_name`, `source_asset`, `source_timestamp`, `availability_timestamp`, `alignment_timestamp`, `timeframe`, `transformation`, `staleness_limit`, `proxy_or_real`, and `causal_status`.

At prediction timestamp `t`, the required rule is:

`availability_timestamp <= t`

and `t - availability_timestamp <= staleness_limit`.

Alignment is backward as-of. Closed candles are required. Ad-hoc forward fill, backward fill, interpolation, and arbitrary shifts are forbidden unless a feature-specific contract explicitly permits and tests them. Scaling is fit on training rows only.

## 7. Temporal Splits

The final design uses expanding walk-forward validation followed by a frozen final holdout:

| Fold | Train period | Validation period |
|---|---|---|
| wf_2022 | 2017-08-17 through 2021-12-31 | 2022 |
| wf_2023 | 2017-08-17 through 2022-12-31 | 2023 |
| wf_2024 | 2017-08-17 through 2023-12-31 | 2024 |

Final holdout: 2025-01-01 through the frozen data cutoff 2026-09-22. Model, feature, and label selection may use only the walk-forward validation results. The holdout may be reported once after freeze, never used for training or selection, and never used for strategy promotion.

For every fold, training rows are purged when their label end reaches the validation start. One complete target horizon is used as an embargo immediately before every validation or holdout boundary. The same rule applies to 4h, 24h, 7d, and 14d labels and is covered by tests.

## 8. Model Universe

Allowed baseline classes for the next research phase:

- LogisticRegression
- HistGradientBoostingClassifier
- RandomForestClassifier

XGBoost and LightGBM are excluded because they are not installed in the current environment. The repository-owned `TCNTrendModel` is a conditional candidate only: it must first consume the frozen dataset, split, preprocessing, and leakage contracts. No TCN training is started here. No hyperparameter sweep is authorized.

## 9. Evaluation Metrics

Classification: ROC-AUC, PR-AUC, balanced accuracy, log loss, Brier score, and calibration.

Regression: MAE, RMSE, rank correlation, and directional accuracy.

Trading-relevance diagnostics only: mean return conditional on signal, return distribution, signal frequency, turnover proxy, and signal-only Sharpe diagnostic. These are not strategy optimization and do not authorize promotion.

## 10. Research Matrix

The frozen matrix contains 24 `READY` rows: 2 targets x 12 timeframe/horizon combinations. Every row uses `crypto_core_v1` and `direction_raw_v1`. The exact structured matrix is in `data/research/phase3/ml_research_manifest.json`.

No combination is marked READY because of a prior score. A combination is READY because its local canonical source exists, has sufficient labelable history, and passes the recorded basic quality checks.

## 11. Versioned Artifacts

- [ml_research_v1.yaml](../configs/ml_research_v1.yaml)
- [ml_research_manifest.json](../data/research/phase3/ml_research_manifest.json)
- [research_design.py](../core/ml/research_design.py)
- [test_phase3_ml_research_design.py](../tests/test_phase3_ml_research_design.py)

The YAML SHA-256 recorded in the manifest is `73026dd08db19c83059c57c75b65bcebf077280cee941bdcbc1155cb400f6517`.

## 12. Explicit Non-Goals

This freeze starts no Phase 4 training. It performs no TCN mass experiments, hyperparameter sweep, strategy optimization, leverage optimization, strategy promotion, paper-trading change, or live-trading change.
