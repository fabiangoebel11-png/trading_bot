# Phase 3 Data Inventory

## Scope and method
This inventory uses only files already present under `data/`. No downloader or API call was made. Local OHLCV files were inspected with `core.ml.phase3_pipeline.inspect_ohlcv_quality`; timestamps were normalized to UTC for diagnostics. Missing assets are reported as `INSUFFICIENT_DATA` rather than synthesized.

## Local crypto OHLCV

| Asset | Role | Source/cache | Timeframes | Rows and coverage | Quality |
|---|---|---|---|---|---|
| BTC/USDT | TARGET or CONTEXT | `data/market_crypto_2017/.../ccxt.parquet` | 5m, 15m, 1h, 4h, 1d | 5m 955,386 from 2017-08-17 to 2026-09-22; 15m 318,502; 1h 79,618; 4h 19,926; 1d 3,324 | UTC, sorted, no duplicates, no invalid OHLC; gaps exist on intraday files; a few zero-volume rows |
| ETH/USDT | TARGET or CONTEXT | `data/market_crypto_2017/.../ccxt.parquet` | 5m, 15m, 1h, 4h, 1d | 5m 955,387 from 2017-08-17 to 2026-09-22; 15m 318,498; 1h 79,618; 4h 19,926; 1d 3,324 | UTC, sorted, no duplicates, no invalid OHLC; gaps exist on intraday files; a few zero-volume rows |
| SOL/USDT | Optional TARGET or CONTEXT | No matching local `market_crypto_2017` parquet found | None verified | No local history verified | `INSUFFICIENT_DATA`; not downloaded or backfilled |

Representative verified ranges: BTC and ETH 1h both contain 79,618 rows from 2017-08-17 04:00 UTC through 2026-09-21 21:00 UTC. The 1h quality scan found 28 gaps per asset; 4 zero-volume rows per asset. The 4h scan found 8 gaps per asset; 1 zero-volume row per asset. 15m showed 33 gaps for BTC and the same broad historical coverage; 1d showed no detected gaps under the interval rule.

## Local macro / TradFi data

| Requested asset | Local source found | Usable role in this Phase 3 builder |
|---|---|---|
| Gold | No local OHLCV file found | `INSUFFICIENT_DATA` |
| SPY | No local OHLCV file found | `INSUFFICIENT_DATA` |
| QQQ | No local OHLCV file found | `INSUFFICIENT_DATA` |
| DAX | No local OHLCV file found | `INSUFFICIENT_DATA` |
| VIX | Macro CSVs exist as a historical context proxy, not standalone OHLCV | Available only for a future explicit macro adapter |
| US10Y | No local OHLCV file found | `INSUFFICIENT_DATA` |
| EURUSD | No local OHLCV file found | `INSUFFICIENT_DATA` |
| WTI | No local OHLCV file found | `INSUFFICIENT_DATA` |

Existing macro CSVs include `macro_ESF_VIX_730d.csv`, `macro_ESF_VIX_URTH_2600d.csv`, `macro_ESF_VIX_URTH_3600d.csv`, `macro_ESF_VIX_URTH_730d.csv`, and ML macro caches such as `ml_macro_ESF_NQF_VIX_URTH_1095d.csv`. Their columns are provider/proxy series such as `equity`, `vix`, `world`, `ES=F`, `NQ=F`, `^VIX`, and `URTH`, with a `Date` column. They are not silently reclassified as SPY, QQQ, DAX, Gold, US10Y, EURUSD, or WTI.

## Existing implementation surfaces

- `core/data_loader.py`: CCXT retrieval, caching, resampling, and incomplete-candle handling.
- `core/ml/data.py`: OHLCV normalization, quality reports, and Parquet I/O.
- `core/ml/features.py`: existing causal technical, macro, breadth, and higher-timeframe feature helpers.
- `core/ml/labeling.py`: triple-barrier labels with `t1`, MAE, and MFE.
- `core/ml/dataset.py`: sequence construction plus purged/embargoed splits.
- `core/ml/multitimeframe.py`: backward alignment of closed higher-timeframe features.
- `core/architecture.py`: generic backward `align_asof` contract and versioned feature/label contracts.
- Existing model metadata and OOS files exist under `data/models/`; they are not treated as Phase 3 dataset sources or final validation.

## Inventory conclusion
Real local data is sufficient to build target-specific BTC and ETH datasets across 5m, 15m, 1h, 4h, and 1d observations. It is not sufficient to claim a complete local multi-market dataset for all requested TradFi assets. This phase therefore keeps those assets explicit in the feature-set specification but reports them unavailable until a separately audited local cache or approved download process exists.
