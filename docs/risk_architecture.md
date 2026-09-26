# Adaptive Risk Architecture

## Runtime data flow

```text
Market data
  -> causal features and model outputs
  -> direction, score, confidence, expected MAE/MFE, horizon + explicit unit
  -> instrument and market protection model
  -> allowed risk fraction
  -> stop distance
  -> risk-based notional
  -> margin / portfolio / notional limits
  -> effective leverage
  -> paper order and persisted protection state
```

Leverage is an output of `risk_engine.compute_risk_parameters`, never the input that determines exposure. The risk amount is calculated first from equity, score/confidence, drawdown, volatility and correlation multipliers. The notional is then `risk_amount / stop_distance_pct`, capped before margin and leverage are selected. Effective leverage is the smallest leverage needed to fit that final notional into the available margin allocation and is bounded by the instrument/exchange ceiling.

## Instrument contract

- `crypto_perpetual`: technical ceiling 10x in this pure risk layer; the actual exchange tier still has to be supplied by the execution adapter.
- `equity_underlying`: the current QQQ/SPY data is ordinary ETF/underlying price data, so it is capped at 1x and has no synthetic liquidation or KO price.
- `equity_cfd`, `equity_future`, `equity_margin`, and `crypto_future`: require an explicit broker/exchange maximum. The engine rejects the setup when that condition is missing.

Strategic stop loss, internal safety barrier, exchange liquidation estimate, and emergency flatten are separate concepts. For ordinary ETF underlyings only the strategic stop is applicable.

## Evidence status

The current repository already contains causal trend, ML, purged chronological/OOS, funding, shared-wallet, walk-forward and Monte-Carlo infrastructure. The current Model 2 OOS report provides score-bucket evidence for QQQ; it does not by itself prove a profitable leveraged strategy, a 100x equity product, or a robust multi-asset router.

Promotion remains blocked until a fresh, untouched holdout covers the intended instrument, costs, slippage, funding, session effects and the complete paper-trading path. Missing exchange tiers, empirical correlation estimates, or instrument metadata must be reported as insufficient evidence rather than replaced with a guessed parameter.
