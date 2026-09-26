# Phase 2 Architecture

## 1. Purpose
Phase 2 establishes the versioned contracts required for reproducible research and controlled paper-trading integration. It does not change live daemon, paper broker, risk engine, Telegram, or Streamlit behavior.

## 2. Scope
This phase covers data-universe identity, target and feature roles, observation timeframes, prediction horizons, feature and label versions, strategy identity, experiment manifests, decision records, causal alignment, and backtest request/result contracts.

## 3. Non-goals
This phase does not optimize a strategy, promote a candidate, replace the risk engine, activate partial exits, or claim PAPER-READY status. Existing OOS and research reports remain research evidence only.

## 4. Runtime Boundary
The intended control path remains:

`Strategy -> Signal -> Risk Engine -> Execution/Paper Broker`

The new contracts describe and persist this path; they do not bypass the risk engine.

## 5. Source Of Truth
`core/architecture.py` is the source of truth for the new Python contracts. `configs/data_universe.yaml` is the canonical configured universe for the current repository. SQLite stores immutable references to contracts used by operational or research workflows.

## 6. Data Universe
Each asset declares its asset class, provider, provider symbol, supported timeframes, timezone, trading hours, expected frequency, staleness policy, quality state, and roles. The initial universe includes BTC/USDT, ETH/USDT, SOL/USDT, TradFi instruments, macro proxies, rates, FX, commodities, and volatility indices.

## 7. Target Assets
BTC/USDT, ETH/USDT, and SOL/USDT are explicitly represented as target-capable crypto assets. SPY and QQQ remain target-capable equity assets. A target asset must be selected explicitly by a strategy and cannot be inferred from a feature list.

## 8. Feature Assets
Feature assets include target assets when deliberately used as context, plus cross-asset and macro inputs such as VIX, US10Y, EURUSD, GOLD, WTI, DAX, and BNB/USDT. Their role is explicit and independently auditable.

## 9. Observation Timeframe
A strategy records the bar timeframe from which an observation is formed, for example `1h` or `5m`. This is separate from the prediction horizon. A timeframe is not a claim about how long a position is held.

## 10. Prediction Horizon
`HorizonSpec` validates a concrete duration against H1-H5 classes:

- H1: 15 minutes through 4 hours
- H2: 4 hours through 24 hours
- H3: 24 hours through 7 days
- H4: 7 days through 14 days
- H5: 14 days through 28 days

Boundary ownership must be resolved consistently when a duration lies exactly on a shared boundary.

## 11. Feature Set Contract
A feature set identifies its assets, timeframes, feature groups, lookbacks, alignment rules, normalization policy, label version, and data cutoff. It is immutable once referenced by a manifest or decision record.

## 12. Label Contract
A label specification identifies its version, horizons, label names, and threshold configuration. Labels include future return, direction, probability, thresholded return, MAE, MFE, stop probability, and expected holding time where implemented.

## 13. Strategy Contract
`StrategyConfig` identifies strategy ID, semantic version, asset, direction, asset class, observation timeframe, horizon class, family, model, feature set, filters, stops, exits, risk profile, and lifecycle status. Research status is the default.

## 14. Registry
`strategy_registry` is additive and versioned. A `(strategy_id, strategy_version)` row is immutable: repeating the same content is idempotent, while reusing the pair with different content is rejected. A changed strategy requires a new version.

## 15. Experiment Manifest
An experiment manifest freezes code version, data universe, feature set, labels, strategy version, model configuration, periods, walk-forward settings, OOS and clean-holdout periods, stress assumptions, costs, slippage, funding, and result status. It is persisted by experiment ID with a content hash.

## 16. Decision Record
A decision record captures timestamp, asset, direction, strategy version, horizon, timeframe, regime, model and feature references, expected outcomes, entry and protection values, sizing context, action, and reason codes. `decision_records` is append-only and rejects conflicting reuse of a decision ID.

## 17. Causal Alignment
`align_asof` performs backward-only alignment using `pandas.merge_asof`. Availability lag is applied before joining. A maximum staleness duration can invalidate old context. This contract prevents a feature observed after the prediction timestamp from entering the decision.

## 18. Leakage Checks
`assert_no_future_features` rejects feature rows whose `observed_at` is after the target prediction timestamp. Tests cover timezone-aware timestamps, lagged availability, backward alignment, and stale data behavior. Higher-level validation must still inspect label construction and split boundaries.

## 19. Backtest Request
A backtest request references a complete strategy configuration, dataset identity, execution assumptions, and validation configuration. It must be reproducible from the referenced manifest and must not silently substitute a different feature, label, or strategy version.

## 20. Backtest Result
A result records dataset and strategy references, period boundaries, trade count, return, drawdown, Sharpe-like metrics, cost assumptions, and validation status. Required metric keys and statistical acceptance thresholds remain a follow-up contract before automated promotion.

## 21. Lifecycle And Promotion
The lifecycle is research, validated, paper, shadow, live, disabled, or rejected. Promotion requires clean validation evidence, cost and slippage assumptions, temporal integrity, decision-record coverage, and explicit human review. OOS-consuming robust gates remain research screening and cannot promote a strategy by themselves.

## 22. Delivery Gates
### MUST HAVE

- Stable strategy, feature, label, universe, manifest, decision, and backtest identities.
- Immutable version behavior and deterministic content hashes.
- Explicit target versus feature roles.
- Explicit observation timeframe versus prediction horizon.
- Backward causal alignment with availability lag and staleness handling.
- Future-feature rejection tests.
- Append-only registry, manifest, and decision persistence.
- Full regression suite passing before runtime integration.
- Human-readable audit trail for every promotion decision.

### SHOULD HAVE

- Deserialization support for every persisted contract.
- Dataset checksums and provider/cache provenance.
- Coverage and missingness reports per asset and timeframe.
- Required metric schemas for backtest results.
- Purged and embargoed walk-forward validation as the default research path.
- Automated verification that the clean holdout was not used for candidate selection.
- Read-only UI views for registry, manifests, and decisions.

### OPTIONAL

- A richer data catalog with storage locations and refresh jobs.
- Event-time and availability-time columns for every raw observation.
- Portfolio-level decision records spanning multiple target assets.
- Automated experiment comparison reports.
- A promotion workflow integrated with code review or signed approvals.

## Current Status
The Phase 2 foundation is implemented and tested. The full test suite currently passes with 131 tests. Runtime paper-trading behavior is unchanged. No strategy has been promoted, and the platform is not being declared PAPER-READY.
