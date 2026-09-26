# Asset Isolation Audit

Audit date: 2026-09-23
Scope: forensic, read-only audit of the repository research history for `ETH/USDT breakout_atr_60_1.0`.

## Final Classification

**CLEAN_BUT_MULTI_ASSET_RESEARCH**

This classification is about cross-asset influence on the target ETH candidate. The
target Phase 2 and partial-exit paths evaluate one OHLCV frame at a time and show no
BTC/SOL aggregate feeding ETH parameter values, signals, policy choices, or leverage.
The repository nevertheless contains separate multi-asset portfolio, ML, macro, and
shared-wallet paths. Those paths are not evidence that BTC or SOL selected the target
candidate.

## Executive Summary

| Field | Finding |
|---|---|
| ETH strategy isolated | YES for the Phase 2 and partial-exit target path; PARTIAL for the repository as a whole |
| BTC influenced ETH | NO in the code-traceable target path |
| SOL influenced ETH | NO in the code-traceable target path |
| Direct training leakage | NO for the rule-based target strategy; ML is a separate per-symbol architecture |
| Optimization leakage | NO cross-asset leakage found; the Phase 2 robustness gate did read ETH OOS |
| Selection leakage | YES temporally: Phase 2 `robust` selection includes OOS metrics; NO BTC/SOL selection leakage found |
| OOS contamination | YES |
| Holdout contamination | PARTIAL: the manifest was immutable and no post-manifest tuning occurred, but the same historical OOS bars were read by the prior diagnosis |
| Final category | CLEAN_BUT_MULTI_ASSET_RESEARCH |

## Direct Answer About SOL

SOL appears frequently because it is the default third member of the configured crypto
universe, is included in the Phase 2, partial-exit, diagnosis, and holdout scopes, has
its own cached OHLCV data, and has a separate saved TCN artifact. It is also named in
the ML breadth/universe documentation because the breadth basket explicitly excludes
the traded BTC/ETH/SOL universe.

The evidence does **not** show SOL influencing ETH training, parameter search,
validation, signal features, regime flags, leverage calculations, OOS metrics, or
partial-exit/holdout decisions in the target path. SOL was evaluated in parallel. The
only repository-level coupling found is in separate production-style portfolio and
ML infrastructure: a shared wallet can change portfolio allocation when several
symbols are active, and the ML architecture has shared macro/breadth context. Neither
path selected `breakout_atr_60_1.0`.

The provenance of the human decision to freeze this exact candidate before the
partial-exit study is not recorded as a separate selection log. The code-traceable
fact is that the candidate is passed as a fixed ID, not selected from BTC/SOL results.

## Pipeline and Data-Flow Map

1. Local OHLCV CSVs are loaded per requested symbol by
   [`load_local_ohlcv`](../phase2_strategy_research.py#L77). The Phase 2 loader does
   not download data and does not concatenate symbols.
2. [`build_candidate_grid`](../phase2_strategy_research.py#L102) creates the same
   58 rule candidates for each asset. `breakout_atr_60_1.0` is the candidate with
   lookback 60 and breakout distance 1.0 ATR.
3. [`_features`](../phase2_strategy_research.py#L217) derives ATR, ATR%, returns,
   realized volatility, and EMAs from the current symbol frame only.
4. [`run_phase2`](../phase2_strategy_research.py#L687) loops over `symbol, frame`,
   creates `family_rows` per symbol, computes parameter stability within those rows,
   and then applies the robustness gate.
5. [`_robust_gate`](../phase2_strategy_research.py#L560) uses validation, OOS,
   walk-forward, stress, stability, and drawdown values. This is temporally
   contaminated selection, but not cross-asset pooling.
6. The stored Phase 2 artifact contains six robust rows, all ETH rows. There are no
   robust BTC or SOL rows in the actual artifact.
7. The partial-exit studies pass the fixed candidate ID into a per-symbol execution
   loop. They compare each policy with that symbol's own full-position baseline.
8. The holdout manifest registers BTC, ETH, and SOL, but its comparisons remain
   per-symbol. The historical holdout is explicitly not pristine because the earlier
   diagnosis already read the same OOS bars.

## Exact Trace of `breakout_atr_60_1.0`

| Stage | Evidence | Asset scope | Cross-asset influence |
|---|---|---|---|
| Definition | `build_candidate_grid` creates `breakout_atr_60_1.0` from `(60, 1.0)` | Same grid for BTC, ETH, SOL | None in definition |
| Signal/features | `generate_signal` and `_features` consume one frame | One symbol | None found |
| Phase 2 evaluation | `run_phase2` creates rows inside the symbol loop | Per symbol | None found |
| Robustness | `_robust_gate` checks the row's own validation/OOS/WF/stress | Per row | No BTC/SOL aggregate; ETH OOS is used |
| Stored result | Six robust candidates, all `ETH/USDT` | ETH only in robust output | BTC/SOL did not enter the stored robust set |
| Partial-exit base | `BASE_CANDIDATE_ID` is fixed in the module and looked up directly | BTC, ETH, SOL run separately | No policy selection from other assets |
| Holdout base | Manifest strategy is fixed before policy evaluation | Primary ETH plus secondary BTC/SOL | No cross-asset comparison gate |

## Asset Usage Matrix

| Component | BTC | ETH | SOL | Could change ETH target candidate? | Finding |
|---|---|---|---|---|---|
| Phase 2 raw data | evaluated | evaluated | evaluated | No | Independent frames |
| Phase 2 candidate grid | evaluated | evaluated | evaluated | No | Same predeclared grid |
| Phase 2 parameter stability | own BTC rows | own ETH rows | own SOL rows | No | Peers are same-symbol family rows |
| Phase 2 robustness | no robust row in artifact | six robust rows | no robust row in artifact | No | OOS used, but no pooling |
| Phase 2 leverage study | per-symbol if robust | per-symbol if robust | per-symbol if robust | No | `_leverage_metrics` receives one result |
| Partial-exit diagnosis | parallel | primary | parallel | No | `variant_rows` filtered by symbol |
| Partial-exit holdout | secondary | primary | secondary | No | Separate controls/comparisons |
| Rule-based features | local OHLC | local OHLC | local OHLC | No | No cross-asset columns |
| ML technical features | own model | own model | legacy/separate model artifact | Not in target Phase 2 path | Per-symbol training |
| ML macro/breadth context | shared context | shared context | shared context | Not in target Phase 2 path | Context is not raw BTC/SOL input to ETH |
| Production shared wallet | can be active | can be active | can be active | Yes, for portfolio sizing only | Separate production/backtester path |
| Position selection variant | candidate | candidate | candidate | Yes, if explicitly enabled | Disabled by default and not Phase 2 |

## BTC Trace

- BTC is a default raw-data asset in [`DataConfig`](../../core/config.py#L60) and is
  loaded alongside ETH in the parallel research scopes.
- Phase 2 computes BTC rows separately. The stored Phase 2 artifact has no robust BTC
  row, so BTC did not enter the actual robust candidate set.
- The ETH TCN metadata lists BTC in its asset configuration, but its feature columns
  contain ETH-local technical features, macro context, breadth-derived features, and
  higher-timeframe ETH features. No BTC price or BTC return feature is present.
- The current breadth loader drops configured traded symbols, including BTC/ETH/SOL,
  before making the external breadth basket. This prevents BTC from being reintroduced
  as ETH's own breadth proxy.
- The production backtester does combine active symbols for shared-wallet weights and
  stop-out risk. That can affect ETH portfolio sizing when BTC is active, but this is
  not the Phase 2 candidate search or partial-exit base selection.

## SOL Trace

- SOL is the third default crypto symbol and is explicitly present in the Phase 2,
  partial-exit diagnosis, and holdout defaults.
- SOL gets its own local OHLCV frame, candidate rows, baseline, policy rows, and
  holdout comparisons. These are not concatenated into ETH rows.
- The holdout includes a separately documented historical `SOL/USDT partial_breakeven`
  candidate, classified `PROMISING BUT NOT ROBUST`; it is explicitly not promoted.
- A `SOL-USDT_tcn` model artifact exists, but the current ML config's exact prepared
  asset set is `NASDAQ100_PROXY`, `SP500_PROXY`, `BTC/USDT`, and `ETH/USDT`. The SOL
  artifact is therefore evidence of a separate or legacy model run, not evidence that
  SOL trained the ETH model.
- SOL can participate in shared-wallet portfolio sizing in the production backtester,
  but no such result is used to choose the Phase 2 ETH candidate.

## ML Audit

The target Phase 2 and partial-exit modules do not import or invoke the ML training or
inference path. The rule-based strategy research is therefore not ML-selected.

The separate ML path has these properties:

- [`run_training_pipeline`](../../core/ml/train.py#L340) loads the configured prepared
  asset set, then calls [`train_symbol_model`](../../core/ml/train.py#L165) once per
  symbol. It is not a pooled BTC/ETH/SOL label table.
- The ETH model metadata records assets including BTC/ETH, but its feature columns do
  not contain a BTC or SOL price/return field. Macro features and an external breadth
  summary are shared context.
- [`fetch_breadth_basket`](../../core/ml/features.py#L238) excludes symbols present in
  `data_config.symbols`, and the breadth series is reduced to equal-weight derived
  statistics rather than raw traded-coin features.
- The ML model is disabled by default, and the regime gate and production promotion
  flag are also disabled by default in the current configuration.
- Historical backtesting uses the stitched purged-walk-forward OOS confidence file,
  not the final full-history model, when ML is explicitly enabled. This is a separate
  concern from the rule-based target candidate.

## Regime Audit

- Phase 2 [`_regime_flags`](../phase2_strategy_research.py#L439) uses the current
  symbol's returns, EMAs, rolling volatility, drawdown, crash, and recovery signals.
  The diagnosis stores regime metrics under each symbol key.
- Production macro regimes use equity/VIX/world macro series and are applied to each
  symbol in the production backtester. That global macro path is not used by the
  Phase 2 target candidate.
- The ML regime helper is called with each signal frame's `close` in
  [`apply_regime_gated_ml_confirmation`](../../core/ml/inference.py#L144). The helper's
  documentation mentions BTC, but the traced implementation does not substitute BTC
  close for an ETH close.
- A separate Monte Carlo comparison labels sampled windows by BTC regime. That is an
  analysis stratifier, not an ETH training feature or candidate-selection input.

## Leverage and Portfolio Coupling

- Phase 2 leverage is calculated after a candidate passes its own robustness gate by
  [`_leverage_metrics`](../phase2_strategy_research.py#L578), using one candidate's
  result and one symbol's ATR/volatility series.
- Partial-exit leverage reporting is likewise called from one symbol's policy result.
- The production backtester is different: it constructs per-symbol candidate frames,
  combines active symbols into shared-wallet weights, optionally applies top-N
  selection, and applies a shared stop-out risk scale. See
  [`run_backtest_from_signals`](../../core/backtester.py#L237).
- This portfolio coupling can make BTC/SOL affect ETH allocation in a production
  portfolio backtest, but no evidence connects that path to the selection of
  `breakout_atr_60_1.0` or to the partial-exit holdout decision.

## OOS and Holdout Timeline

| Stage | Time / evidence | Status | Contamination finding |
|---|---|---|---|
| Local data | BTC/ETH 2019-11-17 to 2026-09-21; SOL 2020-08-11 to 2026-09-21 | USED | Asset frames remain separate |
| Phase 2 train/validation/OOS | 50% / 20% / 30% chronological | USED | Per-symbol, but OOS enters robustness gate |
| Candidate robustness/selection | `_robust_gate` requires positive ETH OOS Sharpe/return and OOS drawdown | POTENTIALLY CONTAMINATED | ETH OOS influenced robust classification |
| Partial-exit diagnosis | Reads the historical final 30% and reports ETH/BTC/SOL diagnostics | USED | Descriptive, but not untouched afterward |
| Holdout manifest | 2026-09-23, fixed policies and periods | CLEAN AFTER MANIFEST | `selection_or_tuning_after_manifest` is false |
| Holdout evaluation | ETH/BTC holdout starts 2024-09-02; SOL starts 2024-11-21 | POTENTIALLY CONTAMINATED | Same historical OOS bars were already read |
| Final partial-exit decision | `NO ROBUST PARTIAL EXIT ADVANTAGE FOUND` | USED | No robust policy promoted |

The holdout is not a cross-asset contamination finding. It is a temporal integrity
finding. The stored report itself says `NOT_PRISTINE_PRIOR_DIAGNOSTIC_READ`, with
`holdout_pristine: false`, while also recording that no tuning occurred after the
manifest.

## Partial-Exit Influence Audit

The partial-exit research fixes the entry candidate before evaluating management
policies. In [`run_partial_exit_research`](../partial_exit_research.py#L282), each
symbol builds its own features, baseline, train-derived MFE value, and policy rows.
The diagnosis repeats this at [`run_partial_exit_diagnosis`](../partial_exit_diagnosis.py#L541)
and creates the ETH view by filtering rows whose symbol is ETH. The holdout repeats
the same per-symbol pattern at [`run_holdout`](../partial_exit_holdout.py#L420).

BTC/SOL therefore supplied parallel descriptive evidence, not a pooled objective or
an ETH policy-selection score. The diagnosis explicitly says additional policies were
diagnostic only, and the holdout found no robust policy classification.

## Evidence Register

| ID | Reference | Finding |
|---|---|---|
| E01 | [`phase2_strategy_research.py:77`](../phase2_strategy_research.py#L77) | Local per-symbol OHLCV loader; no network or pooled frame |
| E02 | [`phase2_strategy_research.py:102`](../phase2_strategy_research.py#L102) | Candidate grid is fixed and explicit |
| E03 | [`phase2_strategy_research.py:217`](../phase2_strategy_research.py#L217) | Rule features use one frame |
| E04 | [`phase2_strategy_research.py:560`](../phase2_strategy_research.py#L560) | Robust gate includes ETH OOS metrics |
| E05 | [`phase2_strategy_research.py:687`](../phase2_strategy_research.py#L687) | Candidate rows are built inside the symbol loop |
| E06 | [`partial_exit_research.py:293`](../partial_exit_research.py#L293) | Partial-exit evaluation loops by symbol |
| E07 | [`partial_exit_diagnosis.py:592`](../partial_exit_diagnosis.py#L592) | ETH report is a symbol filter, not a pooled rank |
| E08 | [`partial_exit_holdout.py:393`](../partial_exit_holdout.py#L393) | Historical holdout is explicitly not pristine |
| E09 | [`partial_exit_holdout.py:430`](../partial_exit_holdout.py#L430) | Holdout evaluates each symbol separately |
| E10 | [`core/ml/features.py:256`](../../core/ml/features.py#L256) | Traded symbols are excluded from external breadth |
| E11 | [`core/ml/train.py:358`](../../core/ml/train.py#L358) | ML models are trained per symbol |
| E12 | [`core/strategy.py:145`](../../core/strategy.py#L145) | Historical ML confidence is loaded per symbol |
| E13 | [`core/backtester.py:287`](../../core/backtester.py#L287) | Shared wallet can couple active portfolio symbols |
| E14 | [`core/backtester.py:296`](../../core/backtester.py#L296) | Top-N selection is optional and separate |
| E15 | [ETH model metadata](../../data/models/ETH-USDT_tcn_meta.json#L1) | ETH model artifact records BTC/ETH assets but no BTC/SOL feature column |
| E16 | [Phase 2 artifact](../../data/research/phase2/phase2_strategy_research.json#L1) | Stored robust set contains six ETH candidates only |
| E17 | [Holdout artifact](../partial_exit/partial_exit_holdout.json#L1) | Manifest includes BTC/ETH/SOL; no robust policy result |

## Disposition

No automatic repair, retraining, optimization, parameter change, new rule, or new
holdout definition was performed. Existing results were not overwritten.

Because Phase 2 robustness used historical OOS and the later holdout reused bars that
the diagnosis had already read, the target result should not be described as a clean
pristine prospective selection. The appropriate next research boundary is a genuinely
future ETH-only holdout with the entry candidate and management policies frozen before
the data is read; this is a temporal-integrity recommendation, not an indication that
BTC or SOL leaked into ETH.
