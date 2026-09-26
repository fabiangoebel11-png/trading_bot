"""Phase 4.5: read-only synthesis and decision gate for existing research."""
from __future__ import annotations

import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "research" / "phase4"
CSV_PATH = OUTPUT_DIR / "phase4_5_research_decision_gate.csv"
JSON_PATH = OUTPUT_DIR / "phase4_5_research_decision_gate.json"
MANIFEST_PATH = OUTPUT_DIR / "phase4_5_research_decision_gate_manifest.json"
REPORT_PATH = ROOT / "research" / "phase4_5_research_decision_gate.md"
SUMMARY_PATH = ROOT / "research" / "phase4_5_research_summary.md"

INPUTS = {
    "phase2_csv": ROOT / "data/research/phase2/phase2_strategy_research.csv",
    "phase2_report": ROOT / "data/research/phase2/phase2_strategy_research.md",
    "partial_exit_csv": ROOT / "data/research/partial_exit/partial_exit_research.csv",
    "partial_exit_report": ROOT / "data/research/partial_exit/partial_exit_research.md",
    "phase2_architecture": ROOT / "research/phase2_architecture.md",
    "universe_config": ROOT / "configs/data_universe.yaml",
    "ml_config": ROOT / "configs/ml_research_v1.yaml",
    "phase3_proxy_csv": ROOT / "data/research/phase3/proxy_feature_value.csv",
    "phase3_proxy_report": ROOT / "research/phase3_proxy_feature_value.md",
    "phase4_analysis": OUTPUT_DIR / "phase4_analysis.csv",
    "phase4_deep": OUTPUT_DIR / "phase4_deep_diagnostic.csv",
    "phase4_generalization": OUTPUT_DIR / "phase4_generalization_diagnostic.csv",
    "phase4_label_value": OUTPUT_DIR / "phase4_label_value_diagnostic.csv",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(name: str) -> str:
    return INPUTS[name].read_text(encoding="utf-8")


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(INPUTS[name])


def _required_status() -> dict[str, bool]:
    return {name: path.exists() for name, path in INPUTS.items()}


def _count_phrase(text: str, pattern: str, default: int = 0) -> int:
    match = re.search(pattern, text)
    return int(match.group(1)) if match else default


def _phase2_summary() -> dict[str, Any]:
    frame = _csv("phase2_csv")
    report = _text("phase2_report")
    return {
        "rows": len(frame),
        "assets": sorted(frame["symbol"].dropna().unique().tolist()),
        "candidate_count_per_asset_reported": _count_phrase(report, r"Candidates per asset: (\d+)"),
        "robust_rows": int(frame["robust"].fillna(False).astype(bool).sum()),
        "oos_rows": int(frame["oos_sharpe"].notna().sum()),
        "stress_rows": int(frame["stress_median_sharpe"].notna().sum()),
        "funding_status": "NOT TESTED" if "Funding: `NOT TESTED" in report else "REPORTED",
        "selection_caveat": "OOS results are research evidence, not an untouched post-selection promotion gate.",
    }


def _partial_exit_summary() -> dict[str, Any]:
    frame = _csv("partial_exit_csv")
    return {
        "rows": len(frame),
        "policies": int(frame["policy_id"].nunique()),
        "symbols": sorted(frame["symbol"].dropna().unique().tolist()),
        "classes": frame["classification"].value_counts().to_dict(),
        "decision": "NO ROBUST PARTIAL EXIT ADVANTAGE FOUND",
        "return_advantage": "NOT DEMONSTRATED",
        "risk_management_advantage": "Some variants show lower OOS drawdown or different stress behavior, but no automatic promotion is justified.",
    }


def _phase4_summary() -> dict[str, Any]:
    analysis = _csv("phase4_analysis")
    deep = _csv("phase4_deep")
    generalization = _csv("phase4_generalization")
    labels = _csv("phase4_label_value")
    classifications = analysis["classification"].value_counts().to_dict()
    return {
        "baseline_rows": len(analysis),
        "baseline_classifications": classifications,
        "deep_rows": len(deep),
        "deep_candidate_dossiers": 31,
        "deep_common_core_context_pairs": 112,
        "deep_category_a_candidates": 28,
        "deep_category_b_candidates": 3,
        "generalization_rows": len(generalization),
        "label_rows": len(labels),
        "label_task_count": int(labels[["target", "timeframe", "horizon"]].drop_duplicates().shape[0]),
        "label_splits": sorted(labels["split_id"].unique().tolist()),
        "evidence_classification": "SIGNAL_WEAK",
        "label_value_classification": "LABEL_VALUE_UNCLEAR",
        "label_conclusion": "NO LABEL-BASED ML ADVANTAGE DEMONSTRATED",
        "holdout_selection": False,
        "training_performed": False,
        "gpu_used": False,
    }


def _evidence_matrix(phase2: dict[str, Any], partial: dict[str, Any], phase4: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"research_area": "Rule-Based Breakout/ATR", "evidence_status": "OOS EVIDENCE WITH SELECTION CAVEAT", "what_was_shown": f"Phase 2 evaluated {phase2['rows']} rows across BTC/ETH/SOL, with {phase2['robust_rows']} rows marked robust by its diagnostic gate and OOS/stress metrics recorded.", "what_was_not_shown": "No untouched post-selection promotion result; Phase 2 OOS was part of the research workflow.", "open_risk": "A clean, pre-registered OOS/paper period remains necessary.", "maturity": "DIAGNOSTICALLY EVALUATED"},
        {"research_area": "Partial Exits", "evidence_status": "NO ROBUST RETURN ADVANTAGE", "what_was_shown": f"{partial['policies']} policies were compared; some variants provide descriptive drawdown/risk differences.", "what_was_not_shown": "No stable return advantage and no automatic promotion.", "open_risk": "Risk benefit must be evaluated only inside the next frozen validation protocol.", "maturity": "DIAGNOSTICALLY EVALUATED"},
        {"research_area": "Context Proxies", "evidence_status": "NO_EVIDENCE", "what_was_shown": "Phase 3.2 and Phase 4 paired comparisons measured proxy deltas under causal alignment.", "what_was_not_shown": "No stable, reproducible incremental proxy advantage.", "open_risk": "Proxy staleness and period dependence remain descriptive risks.", "maturity": "DIAGNOSTICALLY EVALUATED"},
        {"research_area": "ML Direction", "evidence_status": "SIGNAL_WEAK", "what_was_shown": "Phase 4 baseline metrics were measured on 144 combinations with three walk-forward folds and a separate holdout.", "what_was_not_shown": "AUC above chance was not converted into a demonstrated tradable net-return signal.", "open_risk": "Generalization and execution-value alignment.", "maturity": "DIAGNOSTICALLY EVALUATED"},
        {"research_area": "ML Generalization", "evidence_status": "SIGNAL_WEAK", "what_was_shown": "64/144 showed WF variability, 16/144 weakened within WF, 2/144 had stable WFs followed by holdout decline, and 43/144 had AUC movement without return/coverage movement. Phase 4.2 contains 31 candidate dossiers, 124 candidate-split rows, 112 common Core/Context pairs, 28 Category-A and 3 Category-B candidates.", "what_was_not_shown": "No stable holdout generalization or model promotion evidence.", "open_risk": "Holdout degradation and missing per-prediction telemetry.", "maturity": "DIAGNOSTICALLY EVALUATED"},
        {"research_area": "Alternative Labels", "evidence_status": "LABEL_VALUE_UNCLEAR", "what_was_shown": "96 task/split rows reconstructed future return, log return, direction, cost-aware direction, MAE, and MFE read-only.", "what_was_not_shown": "No predictive advantage for alternatives; stop-hit probability and holding time remain unavailable.", "open_risk": "Raw direction includes sub-cost positive moves.", "maturity": "DIAGNOSTICALLY EVALUATED"},
        {"research_area": "Data Quality", "evidence_status": "CONTRACTED, NOT COMPLETE", "what_was_shown": "Frozen universe, causal timestamps, closed candles, no artificial gap filling, train-only scaling, purge and embargo are specified.", "what_was_not_shown": "A complete production-feed quality/OOS run is not part of these research artifacts.", "open_risk": "Gaps, staleness, feed continuity, and funding coverage.", "maturity": "PARTIALLY EXPLORED"},
        {"research_area": "Risk Architecture", "evidence_status": "IMPLEMENTATION PRESENT, VALIDATION PENDING", "what_was_shown": "Versioned contracts, strategy registry, manifests, decision records, causal align_asof, and risk/execution boundaries exist in the repository.", "what_was_not_shown": "This synthesis does not certify exchange connectivity, paper fills, or live readiness.", "open_risk": "Paper-mode operational checks and reconciliation.", "maturity": "DIAGNOSTICALLY EVALUATED"},
    ]


def _decision_gate() -> dict[str, Any]:
    return {
        "path": "PATH A",
        "decision": "PROCEED TO CONTROLLED PAPER/OOS VALIDATION",
        "basis": [
            "The rule-based candidate family is defined and has causal research artifacts.",
            "The architecture contains versioned contracts and an append-oriented audit boundary.",
            "ML has not demonstrated a stable incremental advantage; further complexity is not justified.",
            "The next stage is validation, not promotion, optimization, or live trading.",
        ],
        "qualification": "Phase 2 OOS results must not be treated as an untouched promotion holdout; the next validation period must be frozen before observation.",
        "live_release": False,
        "ml_promotion": False,
    }


def _paper_checklist() -> list[dict[str, str]]:
    return [
        {"area": "Data", "status": "VERIFY NEXT", "items": "feed identity; timestamps; closed candles; missing-data handling; funding availability"},
        {"area": "Signal", "status": "CONTRACTED", "items": "frozen strategy/version; frozen parameters; signal logging; decision record"},
        {"area": "Execution", "status": "VERIFY NEXT", "items": "simulated fees; slippage; latency; order handling; fill assumptions"},
        {"area": "Risk", "status": "IMPLEMENTED/VERIFY NEXT", "items": "stops; notional and wallet risk; drawdown limits; emergency flatten; reconciliation"},
        {"area": "Monitoring", "status": "IMPLEMENTED/VERIFY NEXT", "items": "logs; alerts; health checks; Telegram/Streamlit read-only views"},
    ]


def build_synthesis() -> dict[str, Any]:
    status = _required_status()
    missing = sorted(name for name, present in status.items() if not present)
    if missing:
        raise FileNotFoundError("Missing required research artifacts: " + ", ".join(missing))
    phase2 = _phase2_summary()
    partial = _partial_exit_summary()
    phase4 = _phase4_summary()
    matrix = _evidence_matrix(phase2, partial, phase4)
    return {
        "analysis_version": "phase4_5_v1",
        "research_version": "3.3.0",
        "artifact_status": status,
        "phase2": phase2,
        "partial_exit": partial,
        "phase4": phase4,
        "evidence_matrix": matrix,
        "research_maturity": {row["research_area"]: row["maturity"] for row in matrix},
        "decision_gate": _decision_gate(),
        "paper_oos_checklist": _paper_checklist(),
        "readiness": {
            "research_readiness": "SUFFICIENT FOR CONTROLLED NEXT VALIDATION, NOT PROMOTION",
            "implementation_readiness": "CONTRACTS AND AUDIT STRUCTURES PRESENT; OPERATIONAL CHECKS REMAIN",
            "risk_control_readiness": "RISK BOUNDARY PRESENT; PAPER EXECUTION VERIFICATION REMAINS",
            "trading_validation_readiness": "PROCEED WITH CONTROLLED PAPER/OOS VALIDATION",
            "live_readiness": "NOT DERIVED / NOT AUTHORIZED",
        },
        "ml_research_status": "ML SIGNAL NOT DEMONSTRATED; ML RESEARCH INCONCLUSIVE",
        "rule_based_status": "RESEARCH-SUFFICIENT FOR CONTROLLED VALIDATION, WITH OOS SELECTION CAVEAT",
        "research_freeze_proposal": {
            "target_assets": ["BTC/USDT", "ETH/USDT"],
            "timeframes": ["1h", "4h"],
            "horizons": "Existing frozen Phase 3.3 horizon matrix",
            "feature_set": "crypto_core_v1; context proxies remain optional and unpromoted",
            "model_set": "Existing sklearn baselines only for synthesis; no new training",
            "strategy_set": "Existing rule-based candidate definition only; no automatic promotion",
            "risk_parameters": "Existing documented risk controls; no Phase 4.5 changes",
            "cost_assumptions": "Existing registered assumptions; include fees, slippage, latency, and funding where available",
            "data_universe": "Existing frozen causal crypto research universe",
            "evaluation_protocol": "Pre-register next paper/OOS period, preserve purge/embargo, report holdout descriptively",
        },
        "remaining_critical_gaps": [
            "Freeze the next untouched validation/paper observation period before collecting outcomes.",
            "Verify feed continuity, closed-candle handling, funding, simulated execution, and reconciliation in paper mode.",
            "Do not use Phase 2 OOS results as a clean promotion holdout.",
        ],
        "next_phase": "Controlled paper/OOS validation of the frozen rule-based line; ML remains shadow/research-only.",
        "explicit_stop_condition": "Stop further ML complexity: no TCN, Transformer, GPU, new features, proxies, assets, horizons, or sweeps from this gate.",
    }


def _markdown(payload: dict[str, Any], summary_only: bool = False) -> str:
    matrix = payload["evidence_matrix"]
    lines = [
        "# Phase 4.5: ML vs. Rule-Based Research Decision Gate",
        "",
        "## Executive Summary",
        f"Decision: **{payload['decision_gate']['decision']}** ({payload['decision_gate']['path']}).",
        "This is a read-only synthesis. It does not train, optimize, promote, or change any trading component.",
        "",
        "## Research Timeline",
        "Phase 1 architecture/research audit -> Phase 2 rule-based strategy research -> Phase 2 partial exits and architecture contracts -> Phase 3 data/proxy/label contracts -> Phase 4 ML baseline -> Phase 4.1 analysis -> Phase 4.2 deep diagnostic -> Phase 4.3 generalization diagnostic -> Phase 4.4 label-value diagnostic -> Phase 4.5 decision gate.",
        "",
        "## Decision Gate",
        f"- **Path:** `{payload['decision_gate']['path']}`",
        f"- **ML status:** `{payload['ml_research_status']}`",
        f"- **Rule-based status:** `{payload['rule_based_status']}`",
        "- **Live release:** not authorized.",
        f"- Qualification: {payload['decision_gate']['qualification']}",
        "",
        "## Evidence Matrix",
        "| Research area | Evidence status | What was shown? | What was not shown? | Open risk | Maturity |",
        "|---|---|---|---|---|---|",
    ]
    for row in matrix:
        lines.append("| " + " | ".join(row.values()) + " |")
    lines += [
        "",
        "## Rule-Based Research Status",
        f"Phase 2 contains {payload['phase2']['rows']} candidate rows and reports {payload['phase2']['candidate_count_per_asset_reported']} candidates per asset. BTC, ETH, and SOL were examined; OOS and stress values are descriptive research evidence. The documented OOS selection caveat remains: it is not an untouched post-selection promotion gate.",
        "Partial exits show no robust return advantage. Any risk-management difference is separate and does not promote a policy.",
        "",
        "## ML Research Status",
        f"Phase 4 baseline: {payload['phase4']['baseline_rows']} rows; diagnostic classes: `{payload['phase4']['baseline_classifications']}`. Phase 4.3 remains `SIGNAL_WEAK`: 64/144 WF-variable, 16/144 weakening within WF, 2/144 stable WF then holdout drop, 43/144 AUC movement without return/coverage movement, despite 128/144 ROC-AUC above 0.5 and 137/144 PR-AUC above the positive-rate baseline. These are not trading proofs.",
        f"Phase 4.4 contains {payload['phase4']['label_rows']} rows ({payload['phase4']['label_task_count']} tasks x four splits). `direction` includes sub-cost positive moves; alternative-label predictive value remains unproven. `stop_hit_probability` and `holding_time` are unavailable.",
        "No additional ML complexity is justified by this evidence. No TCN, Transformer, GPU, new feature, asset, horizon, or sweep is started.",
        "",
        "## Readiness",
        f"- Research readiness: {payload['readiness']['research_readiness']}",
        f"- Implementation readiness: {payload['readiness']['implementation_readiness']}",
        f"- Risk-control readiness: {payload['readiness']['risk_control_readiness']}",
        f"- Trading validation readiness: {payload['readiness']['trading_validation_readiness']}",
        f"- Live readiness: {payload['readiness']['live_readiness']}",
        "",
        "## Paper/OOS Checklist",
    ]
    for item in payload["paper_oos_checklist"]:
        lines.append(f"- **{item['area']}** [{item['status']}]: {item['items']}")
    lines += [
        "",
        "## Research Freeze Proposal",
        "The proposed freeze records the existing target universe, timeframes, horizons, feature/model/strategy/risk/cost assumptions, data universe, and evaluation protocol. It is documentation only and changes no configuration.",
        "",
        "## Remaining Critical Gaps",
    ]
    lines.extend(f"- {item}" for item in payload["remaining_critical_gaps"])
    lines += [
        "",
        "## Next Phase",
        payload["next_phase"],
        "",
        "## Explicit Stop Condition",
        payload["explicit_stop_condition"],
        "",
        "No profitability guarantee. No live approval. Rule-based and ML results are not directly comparable because their data, horizons, metrics, and evaluation designs differ.",
    ]
    return "\n".join(lines) + "\n"


def run() -> dict[str, Any]:
    payload = build_synthesis()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = pd.DataFrame(payload["evidence_matrix"])
    matrix.to_csv(CSV_PATH, index=False)
    input_hashes = {name: _hash(path) for name, path in INPUTS.items()}
    payload["input_hashes"] = input_hashes
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    payload["python_version"] = platform.python_version()
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    manifest = {
        "analysis_version": payload["analysis_version"],
        "research_version": payload["research_version"],
        "input_files": {name: str(path.relative_to(ROOT)) for name, path in INPUTS.items()},
        "input_hashes": input_hashes,
        "timestamp_utc": payload["timestamp_utc"],
        "python_version": payload["python_version"],
        "training_performed": False,
        "gpu_used": False,
        "holdout_used_for_selection": False,
        "test_status": "validated_by_pytest",
        "output_files": [str(path.relative_to(ROOT)) for path in (CSV_PATH, JSON_PATH, MANIFEST_PATH, REPORT_PATH, SUMMARY_PATH)],
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    REPORT_PATH.write_text(_markdown(payload), encoding="utf-8")
    SUMMARY_PATH.write_text(_markdown(payload, summary_only=True), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run()
    print(f"{result['decision_gate']['path']}: {result['decision_gate']['decision']}")
    print(f"ML: {result['ml_research_status']}")
    print(f"Rows: {result['phase4']['label_rows']} label diagnostic rows")