# Phase 3 Feature Specification

## Feature set
`crypto_multimarket_v1` is defined in `configs/feature_sets/crypto_multimarket_v1.yaml`.

Targets are explicit and separate: `BTC/USDT` or `ETH/USDT`. The other crypto assets are context only when loaded. A target is never inferred from the union of context columns.

## Implemented now

The additive builder in `core/ml/phase3_pipeline.py` implements a deliberately bounded first set:

- Target-local returns and log returns: periods 1, 3, 6, 12, 24.
- Momentum: periods 6, 12, 24 and RSI(14).
- Trend: SMA and EMA distance for 20 and 50 bars.
- Volatility: realized volatility and rolling range for 12, 24, and 48 bars.
- Volume: relative volume over 24 bars.
- Market structure: range position and breakout distance for 20 and 55 bars.
- Context: backward-aligned one-period return and 24-period volatility for each available non-target crypto context asset.
- Labels: future return and direction, kept separate from feature columns.

The current target-local build contains 29 columns; an ETH build with available context contains 30 total features because the available context contributes two columns. Exact counts are emitted in dataset metadata rather than inferred from the YAML.

## Availability and alignment

- Raw candles are normalized to UTC.
- Input candles are shifted to their close timestamp before use.
- Context features use the central backward `align_asof` contract.
- Maximum context staleness is two days for this initial crypto context adapter.
- No new pipeline uses an ad-hoc forward fill, future merge, or synthetic candle.
- An unfinished higher-timeframe candle is not visible before its close.

## Provenance
Every generated feature has a provenance entry containing source asset, source timeframe, calculation, lookback, and availability rule. This is intended to make later model reports traceable to source data.

## Missing data policy
Missing requested assets are recorded as unavailable and do not trigger downloads or artificial values. A target with no local source is `INSUFFICIENT_DATA`. Context assets can be absent; the final dataset still records the requested context list and available columns.

## Not yet implemented

- A standalone audited adapter for the existing macro CSV proxies.
- Local SPY, QQQ, DAX, Gold, US10Y, EURUSD, and WTI OHLCV ingestion.
- All requested 30m/2h/8h/12h derived caches; the builder can validate the contract but does not fabricate missing source files.
- Cross-crypto breadth, ETH/BTC relative-strength, and cross-market divergence features in this first bounded slice.
- Automatic fit/transform persistence for a production training job; the train-only scaler helper is present and tested.
- Model fitting, model ranking, strategy selection, leverage tuning, and promotion.

The feature count is intentionally small enough for controlled research. No broad feature sweep or final strategy claim is made.
