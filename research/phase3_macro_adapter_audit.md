# Phase 3.1 Macro/TradFi Data Adapter Audit

## Scope
This audit inspects local CSV files only. No API request, downloader, training, backtest, model comparison, strategy selection, promotion, or paper-trading change was performed.

## Actual local files

### Daily macro proxy caches

| File | Columns | Rows | Range | Frequency | Meaning |
|---|---|---:|---|---|---|
| `data/macro_ESF_VIX_URTH_3600d.csv` | Date, equity, vix, world | 2,481 | 2016-11-14 to 2026-09-22 | Daily business-session observations | ES-futures-like equity proxy, VIX proxy, URTH world-equity proxy |
| `data/macro_ESF_VIX_URTH_2600d.csv` | Date, equity, vix, world | 1,793 | 2019-08-12 to 2026-09-22 | Daily business-session observations | Same proxy family, shorter history |
| `data/macro_ESF_VIX_URTH_730d.csv` | Date, equity, vix, world | 504 | 2024-09-23 to 2026-09-21 | Daily business-session observations | Same proxy family, recent cache |
| `data/macro_ESF_VIX_730d.csv` | Date, equity, vix | 504 | 2024-09-23 to 2026-09-21 | Daily business-session observations | ES-futures-like and VIX proxies without world proxy |

### ML macro caches

| File | Columns | Rows | Range | Meaning |
|---|---|---:|---|---|
| `data/ml_macro_ESF_NQF_VIX_URTH_2600d.csv` | Date, ES=F, NQ=F, ^VIX, URTH | 1,793 | 2019-08-12 to 2026-09-22 | ES and Nasdaq futures proxies, VIX, world-equity proxy |
| `data/ml_macro_ESF_NQF_VIX_URTH_1095d.csv` | Date, ES=F, NQ=F, ^VIX, URTH | 755 | 2023-09-25 to 2026-09-21 | Same proxy family, shorter history |
| `data/ml_macro_VIX_1095d.csv` | Date, ^VIX | 752 | 2023-09-25 to 2026-09-21 | VIX proxy only |

All inspected files have a UTC-normalized daily `Date`, no duplicate timestamps, sorted timestamps, and no missing values in their stored columns. They contain no weekend rows. Gaps over two days are expected around weekends and exchange holidays; they are not filled by the adapter.

## Availability-time conclusion

The files contain only a date, not an exchange close timestamp, publication timestamp, timezone metadata, or vendor release timestamp. A row dated `D` therefore cannot safely be used at the beginning of `D`. The adapter applies an explicit conservative availability lag, defaulting to two days, then uses the central backward `align_asof` contract. The value is available only when:

`source_date + availability_lag <= alignment_timestamp`

A five-day maximum staleness limit is applied by default. Once exceeded, the aligned value is `NaN`; no unrestricted forward fill occurs. This is conservative but still a proxy policy, not proof of the actual vendor publication time. The features retain `PROXY` identity and are not relabeled as SPY, QQQ, Gold, or any other real asset.

## Session and staleness rules

- Crypto target timeline: continuous 24/7; macro observations remain unchanged when markets are closed, subject to the five-day maximum age.
- US equity/futures/VIX proxies: no weekend rows; Friday/last-session values can be used only after the explicit lag and until staleness expires.
- URTH world proxy: treated identically because the local file lacks its exchange calendar and publication timestamp.
- Holidays and closures: represented as gaps; no calendar-based interpolation or synthetic observation is added.
- A source value with unknown publication time is never visible at its source date.
- Different policies can be passed explicitly to the adapter; there is no hidden global `ffill` rule.

## Proxy status matrix

| Requested context | Status | Actual source / reason |
|---|---|---|
| Gold | INSUFFICIENT_DATA | No verified local Gold file |
| SPY | AVAILABLE_PROXY | ES futures-like `equity` / `ES=F`; not SPY |
| QQQ | AVAILABLE_PROXY | NQ futures `NQ=F`; not QQQ; shorter 2600d cache |
| DAX | INSUFFICIENT_DATA | No verified local DAX series |
| VIX | AVAILABLE_PROXY | `vix` / `^VIX` proxy column |
| US10Y | INSUFFICIENT_DATA | No verified local rate file in these caches |
| EURUSD | INSUFFICIENT_DATA | No verified local FX file |
| WTI | INSUFFICIENT_DATA | No verified local WTI file |
| SOL/USDT | INSUFFICIENT_DATA | No verified local SOL OHLCV file |
| URTH/world equity | AVAILABLE_PROXY | `world` / `URTH`, explicitly treated as URTH proxy |

## Adapter

`core/ml/macro_adapter.py` provides:

- registered local CSV loading;
- UTC timestamp normalization;
- quality diagnostics;
- explicit proxy mapping;
- availability lag;
- central backward/as-of alignment;
- maximum staleness enforcement;
- row-level provenance containing feature, source asset, source file, source timestamp, availability timestamp, alignment timestamp, staleness, and proxy status;
- explicit `INSUFFICIENT_DATA` statuses.

`core/ml/phase3_pipeline.py` accepts the adapter through optional `macro_path`. Without it, the prior crypto-only dataset path is unchanged. Dataset IDs distinguish `no_macro` and `macro_proxy` variants.

## Leakage tests

The focused tests cover:

- future macro value mutation does not alter earlier aligned values;
- later availability time removes a value at the target timestamp;
- stale values become invalid;
- weekend targets do not create a new macro observation;
- proxy identity is preserved;
- target-specific datasets exclude future labels from feature columns.

## Real A/B integration checks

Using `data/macro_ESF_VIX_URTH_3600d.csv` as the long-history proxy cache:

| Dataset | Without macro | With macro | Change |
|---|---:|---:|---:|
| BTC 1h / 4h | 79,559 rows, 31 features | 79,559 rows, 37 features | +6 features, 0 rows |
| BTC 1h / 24h | 79,539 rows, 31 features | 79,539 rows, 37 features | +6 features, 0 rows |
| ETH 1h / 7d | 79,396 rows, 31 features | 79,396 rows, 37 features | +6 features, 0 rows |
| ETH 4h / 14d | 19,788 rows, 31 features | 19,788 rows, 37 features | +6 features, 0 rows |

Macro feature columns are the explicitly named ES futures, VIX, and URTH proxy value/return pairs. Dataset IDs differ between `..._no_macro_1.0.0` and `..._macro_proxy_1.0.0`. No single missing macro market caused uncontrolled row loss.

## Open problems

- Actual publication timestamps and exchange calendars are absent from the CSVs.
- A two-day conservative lag is reproducible but not a verified vendor SLA.
- ES/NQ futures are not interchangeable with SPY/QQQ and must remain proxies.
- The 2600d NQ cache starts in 2019 and should not silently replace the 3600d long-history context.
- No macro model training or performance claim has been made.

## Decision and next phase

The long-history ES/VIX/URTH proxy cache is suitable for controlled research as `AVAILABLE_PROXY` under the explicit conservative lag and staleness policy. It is not equivalent to real SPY, QQQ, Gold, DAX, US10Y, EURUSD, or WTI data. The next phase may compare proxy-enriched and crypto-only datasets under fixed splits, but must first lock the availability policy and provenance into the experiment manifest. No strategy promotion is justified by this adapter audit.
