# Phase 4.5: ML vs. Rule-Based Research Decision Gate

## Executive Summary
Decision: **PROCEED TO CONTROLLED PAPER/OOS VALIDATION** (PATH A).
This is a read-only synthesis. It does not train, optimize, promote, or change any trading component.

## Research Timeline
Phase 1 architecture/research audit -> Phase 2 rule-based strategy research -> Phase 2 partial exits and architecture contracts -> Phase 3 data/proxy/label contracts -> Phase 4 ML baseline -> Phase 4.1 analysis -> Phase 4.2 deep diagnostic -> Phase 4.3 generalization diagnostic -> Phase 4.4 label-value diagnostic -> Phase 4.5 decision gate.

## Decision Gate
- **Path:** `PATH A`
- **ML status:** `ML SIGNAL NOT DEMONSTRATED; ML RESEARCH INCONCLUSIVE`
- **Rule-based status:** `RESEARCH-SUFFICIENT FOR CONTROLLED VALIDATION, WITH OOS SELECTION CAVEAT`
- **Live release:** not authorized.
- Qualification: Phase 2 OOS results must not be treated as an untouched promotion holdout; the next validation period must be frozen before observation.

## Evidence Matrix
| Research area | Evidence status | What was shown? | What was not shown? | Open risk | Maturity |
|---|---|---|---|---|---|
| Rule-Based Breakout/ATR | OOS EVIDENCE WITH SELECTION CAVEAT | Phase 2 evaluated 174 rows across BTC/ETH/SOL, with 6 rows marked robust by its diagnostic gate and OOS/stress metrics recorded. | No untouched post-selection promotion result; Phase 2 OOS was part of the research workflow. | A clean, pre-registered OOS/paper period remains necessary. | DIAGNOSTICALLY EVALUATED |
| Partial Exits | NO ROBUST RETURN ADVANTAGE | 19 policies were compared; some variants provide descriptive drawdown/risk differences. | No stable return advantage and no automatic promotion. | Risk benefit must be evaluated only inside the next frozen validation protocol. | DIAGNOSTICALLY EVALUATED |
| Context Proxies | NO_EVIDENCE | Phase 3.2 and Phase 4 paired comparisons measured proxy deltas under causal alignment. | No stable, reproducible incremental proxy advantage. | Proxy staleness and period dependence remain descriptive risks. | DIAGNOSTICALLY EVALUATED |
| ML Direction | SIGNAL_WEAK | Phase 4 baseline metrics were measured on 144 combinations with three walk-forward folds and a separate holdout. | AUC above chance was not converted into a demonstrated tradable net-return signal. | Generalization and execution-value alignment. | DIAGNOSTICALLY EVALUATED |
| ML Generalization | SIGNAL_WEAK | 64/144 showed WF variability, 16/144 weakened within WF, 2/144 had stable WFs followed by holdout decline, and 43/144 had AUC movement without return/coverage movement. Phase 4.2 contains 31 candidate dossiers, 124 candidate-split rows, 112 common Core/Context pairs, 28 Category-A and 3 Category-B candidates. | No stable holdout generalization or model promotion evidence. | Holdout degradation and missing per-prediction telemetry. | DIAGNOSTICALLY EVALUATED |
| Alternative Labels | LABEL_VALUE_UNCLEAR | 96 task/split rows reconstructed future return, log return, direction, cost-aware direction, MAE, and MFE read-only. | No predictive advantage for alternatives; stop-hit probability and holding time remain unavailable. | Raw direction includes sub-cost positive moves. | DIAGNOSTICALLY EVALUATED |
| Data Quality | CONTRACTED, NOT COMPLETE | Frozen universe, causal timestamps, closed candles, no artificial gap filling, train-only scaling, purge and embargo are specified. | A complete production-feed quality/OOS run is not part of these research artifacts. | Gaps, staleness, feed continuity, and funding coverage. | PARTIALLY EXPLORED |
| Risk Architecture | IMPLEMENTATION PRESENT, VALIDATION PENDING | Versioned contracts, strategy registry, manifests, decision records, causal align_asof, and risk/execution boundaries exist in the repository. | This synthesis does not certify exchange connectivity, paper fills, or live readiness. | Paper-mode operational checks and reconciliation. | DIAGNOSTICALLY EVALUATED |

## Rule-Based Research Status
Phase 2 contains 174 candidate rows and reports 58 candidates per asset. BTC, ETH, and SOL were examined; OOS and stress values are descriptive research evidence. The documented OOS selection caveat remains: it is not an untouched post-selection promotion gate.
Partial exits show no robust return advantage. Any risk-management difference is separate and does not promote a policy.

## ML Research Status
Phase 4 baseline: 144 rows; diagnostic classes: `{'MIXED': 112, 'ROBUST CANDIDATE': 31, 'WEAK': 1}`. Phase 4.3 remains `SIGNAL_WEAK`: 64/144 WF-variable, 16/144 weakening within WF, 2/144 stable WF then holdout drop, 43/144 AUC movement without return/coverage movement, despite 128/144 ROC-AUC above 0.5 and 137/144 PR-AUC above the positive-rate baseline. These are not trading proofs.
Phase 4.4 contains 96 rows (24 tasks x four splits). `direction` includes sub-cost positive moves; alternative-label predictive value remains unproven. `stop_hit_probability` and `holding_time` are unavailable.
No additional ML complexity is justified by this evidence. No TCN, Transformer, GPU, new feature, asset, horizon, or sweep is started.

## Readiness
- Research readiness: SUFFICIENT FOR CONTROLLED NEXT VALIDATION, NOT PROMOTION
- Implementation readiness: CONTRACTS AND AUDIT STRUCTURES PRESENT; OPERATIONAL CHECKS REMAIN
- Risk-control readiness: RISK BOUNDARY PRESENT; PAPER EXECUTION VERIFICATION REMAINS
- Trading validation readiness: PROCEED WITH CONTROLLED PAPER/OOS VALIDATION
- Live readiness: NOT DERIVED / NOT AUTHORIZED

## Paper/OOS Checklist
- **Data** [VERIFY NEXT]: feed identity; timestamps; closed candles; missing-data handling; funding availability
- **Signal** [CONTRACTED]: frozen strategy/version; frozen parameters; signal logging; decision record
- **Execution** [VERIFY NEXT]: simulated fees; slippage; latency; order handling; fill assumptions
- **Risk** [IMPLEMENTED/VERIFY NEXT]: stops; notional and wallet risk; drawdown limits; emergency flatten; reconciliation
- **Monitoring** [IMPLEMENTED/VERIFY NEXT]: logs; alerts; health checks; Telegram/Streamlit read-only views

## Research Freeze Proposal
The proposed freeze records the existing target universe, timeframes, horizons, feature/model/strategy/risk/cost assumptions, data universe, and evaluation protocol. It is documentation only and changes no configuration.

## Remaining Critical Gaps
- Freeze the next untouched validation/paper observation period before collecting outcomes.
- Verify feed continuity, closed-candle handling, funding, simulated execution, and reconciliation in paper mode.
- Do not use Phase 2 OOS results as a clean promotion holdout.

## Next Phase
Controlled paper/OOS validation of the frozen rule-based line; ML remains shadow/research-only.

## Explicit Stop Condition
Stop further ML complexity: no TCN, Transformer, GPU, new features, proxies, assets, horizons, or sweeps from this gate.

No profitability guarantee. No live approval. Rule-based and ML results are not directly comparable because their data, horizons, metrics, and evaluation designs differ.
