# Phase 3 Dataset Report

## Reproducible real-data builds

The builder was run against the existing local Parquet cache with no downloads:

| Dataset | Rows | Coverage | Features | Labels | Dropped | Status |
|---|---:|---|---:|---|---:|---|
| BTC/USDT, 1h observation, 4h horizon | 79,559 | 2017-08-17 05:00 UTC to 2026-09-20 22:00 UTC | 31 | `future_return`, `direction` | 59 | AVAILABLE |
| ETH/USDT, 1h observation, 24h horizon | 79,540 | 2017-08-19 11:00 UTC to 2026-09-20 22:00 UTC | 30 | `future_return`, `direction` | 78 | AVAILABLE |

The count difference reflects warm-up, context alignment, and horizon tails. Each dataset has one explicit target and does not add the target as a context feature.

## Time and leakage behavior

- Features are built from current or earlier closed candles only.
- Labels use `shift(-steps)` only in the label builder and are joined after feature construction.
- Future-price mutation tests leave all earlier feature rows unchanged.
- Higher-timeframe availability is represented by close timestamps.
- Context values are backward as-of aligned with a maximum staleness bound.
- The train-only scaler fits only on the supplied training index.

## Splits and validation preparation

The repository already contains purged/embargoed walk-forward logic in `core/ml/dataset.py`. The Phase 3 tests exercise the actual function and verify that training rows whose label windows reach the test block, plus the embargo buffer, are removed. This phase does not run model training or promote any result.

## Tests executed

Focused Phase 3 tests:

- `pytest -q tests/test_phase3_pipeline.py`: 4 passed.
- `pytest -q tests/test_phase3_quality_and_splits.py`: 4 passed.
- Real BTC and ETH build smoke tests: both passed.
- Required real combinations passed: BTC `1h/4h`, BTC `1h/24h`, ETH `1h/7d`, and ETH `4h/14d`; all returned `AVAILABLE`, UTC/increasing indexes, target-specific feature columns, and label exclusion.
- Local OHLCV quality scan: completed for all BTC/ETH 5m, 15m, 1h, 4h, and 1d Parquet files.
- Full repository suite: `pytest -q` -> 139 passed, 0 failed, 0 skipped.

The full repository suite is green after this additive slice.

## Known data issues

- Intraday Crypto files contain real gaps: BTC/ETH 1h each have 28 detected gaps; 4h each have 8; 15m has 33 detected BTC gaps under the interval rule.
- A small number of Crypto candles have zero volume. No gaps are filled and no prices are invented.
- SOL has no verified local Parquet source in the Phase 3 cache.
- Most requested TradFi assets have no standalone local OHLCV source. Existing VIX/equity/world macro CSV proxies require a separate source and availability audit before inclusion.

## Explicitly not done

No mass retraining, hyperparameter search, strategy ranking, leverage optimization, paper-trading change, model promotion, or profitability claim was made. The existing Model 1A/1B and Model 2 paths remain untouched.

## Exact next step

Add a separately audited adapter for the existing macro proxy CSVs, with explicit provider-symbol identity, publication/market-close availability timestamps, session-aware staleness, and target-specific lookahead tests. Only after that should controlled multi-horizon training and model comparison begin.
