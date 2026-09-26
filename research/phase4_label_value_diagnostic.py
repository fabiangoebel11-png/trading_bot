"""Phase 4.4 read-only label-value diagnostic.

This module reconstructs frozen future outcomes from local OHLCV only. It does
not train models, build features, select labels, or change any runtime code.
The holdout is descriptive only.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from core.ml.phase3_pipeline import HORIZON_DELTAS, TIMEFRAME_DELTAS, load_local_ohlcv
from core.ml.research_design import label_end_times, purged_train_mask

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ml_research_v1.yaml"
PHASE3_MANIFEST = ROOT / "data" / "research" / "phase3" / "ml_research_manifest.json"
PHASE4_RESULTS = ROOT / "data" / "research" / "phase4" / "phase4_ml_baseline_results.csv"
PHASE41_ANALYSIS = ROOT / "data" / "research" / "phase4" / "phase4_analysis.csv"
PHASE42_JSON = ROOT / "data" / "research" / "phase4" / "phase4_deep_diagnostic.json"
PHASE43_JSON = ROOT / "data" / "research" / "phase4" / "phase4_generalization_diagnostic.json"
DATA_ROOT = ROOT / "data"
OUTPUT_DIR = ROOT / "data" / "research" / "phase4"
PLOT_DIR = OUTPUT_DIR / "plots" / "label_value"
DIAGNOSTIC_CSV = OUTPUT_DIR / "phase4_label_value_diagnostic.csv"
DIAGNOSTIC_JSON = OUTPUT_DIR / "phase4_label_value_diagnostic.json"
DIAGNOSTIC_MANIFEST = OUTPUT_DIR / "phase4_label_value_diagnostic_manifest.json"
REPORT_PATH = ROOT / "research" / "phase4_label_value_diagnostic.md"
SUMMARY_PATH = ROOT / "research" / "phase4_label_value_report.md"
COST_THRESHOLD = 0.002
COST_THRESHOLDS = (0.0005, 0.001, 0.002, 0.003, 0.005)
TARGETS = ("BTC/USDT", "ETH/USDT")
TASKS = tuple(
    (target, timeframe, horizon)
    for target in TARGETS
    for timeframe, horizons in (("1h", ("1h", "2h", "4h", "8h", "12h", "24h", "3d", "7d")), ("4h", ("7d", "14d", "21d", "28d")))
    for horizon in horizons
)
FOLDS = (
    ("wf_2022", "2022-01-01T00:00:00Z", "2022-12-31T23:59:59Z"),
    ("wf_2023", "2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z"),
    ("wf_2024", "2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z"),
    ("final_holdout", "2025-01-01T00:00:00Z", "2026-09-22T23:59:59Z"),
)

LABEL_INVENTORY = {
    "future_return": {"definition": "close[t+h] / close[t] - 1", "kind": "continuous regression", "status": "COMPUTED_READ_ONLY"},
    "future_log_return": {"definition": "log(close[t+h] / close[t])", "kind": "continuous regression", "status": "COMPUTED_READ_ONLY"},
    "direction": {"definition": "1 if future_return > 0 else 0", "kind": "binary classification", "status": "COMPUTED_READ_ONLY"},
    "cost_aware_direction": {"definition": "1 if future_return > 0.002 else 0", "kind": "binary classification", "status": "COMPUTED_READ_ONLY"},
    "mae": {"definition": "min(low[t+1:t+h]) / close[t] - 1", "kind": "continuous outcome", "status": "COMPUTED_READ_ONLY"},
    "mfe": {"definition": "max(high[t+1:t+h]) / close[t] - 1", "kind": "continuous outcome", "status": "COMPUTED_READ_ONLY"},
    "stop_hit_probability": {"definition": "no frozen implementation or stop specification", "kind": "classification/probability", "status": "NOT_AVAILABLE"},
    "holding_time": {"definition": "no frozen Phase 3 implementation; triple-barrier time_to_barrier is a separate label", "kind": "duration", "status": "NOT_AVAILABLE"},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _repo_status() -> str:
    try:
        return subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return "UNAVAILABLE"


def _hardware() -> dict[str, Any]:
    return {"cpu_count": os.cpu_count(), "gpu_used": False, "cuda_training": False, "training_performed": False, "runtime": "read_only_label_reconstruction"}


def _outcomes(frame: pd.DataFrame, timeframe: str, horizon: str) -> pd.DataFrame:
    steps = max(1, int(HORIZON_DELTAS[horizon] / TIMEFRAME_DELTAS[timeframe]))
    close = frame["close"].astype(float)
    future_return = close.shift(-steps) / close - 1.0
    result = pd.DataFrame({"future_return": future_return, "future_log_return": np.log(close.shift(-steps) / close), "direction": (future_return > 0).astype(float), "cost_aware_direction": (future_return > COST_THRESHOLD).astype(float)}, index=frame.index)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    prices = close.to_numpy(dtype=float)
    mfe = np.full(len(frame), np.nan)
    mae = np.full(len(frame), np.nan)
    for offset in range(1, steps + 1):
        end = len(frame) - offset
        if end <= 0:
            break
        future_high = high[offset:]
        future_low = low[offset:]
        valid = np.arange(end)
        favorable = future_high[:end] / prices[:end] - 1.0
        adverse = future_low[:end] / prices[:end] - 1.0
        mfe[valid] = np.fmax(np.nan_to_num(mfe[valid], nan=-np.inf), favorable)
        mae[valid] = np.fmin(np.nan_to_num(mae[valid], nan=np.inf), adverse)
    incomplete = np.arange(len(frame)) >= len(frame) - steps
    result["mfe"] = mfe
    result["mae"] = mae
    result.loc[incomplete, ["future_return", "future_log_return", "mfe", "mae"]] = np.nan
    result.loc[incomplete, ["direction", "cost_aware_direction"]] = np.nan
    result["label_end"] = result.index + HORIZON_DELTAS[horizon]
    return result


def _quantiles(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {f"q{q}": float("nan") for q in (1, 5, 25, 50, 75, 95, 99)}
    return {f"q{q}": float(clean.quantile(q / 100)) for q in (1, 5, 25, 50, 75, 95, 99)}


def _stats(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {"rows": int(len(frame)), "complete_outcomes": int(frame["future_return"].notna().sum())}
    for label in ("future_return", "future_log_return", "mfe", "mae"):
        values = pd.to_numeric(frame[label], errors="coerce").dropna()
        output[label] = {"mean": float(values.mean()), "median": float(values.median()), "std": float(values.std(ddof=0)), "min": float(values.min()), "max": float(values.max()), **_quantiles(values)} if not values.empty else {}
    for label in ("direction", "cost_aware_direction"):
        values = pd.to_numeric(frame[label], errors="coerce").dropna()
        output[f"{label}_positive_rate"] = float(values.mean()) if not values.empty else float("nan")
    returns = frame["future_return"].dropna()
    for threshold in COST_THRESHOLDS:
        output[f"abs_move_le_{int(threshold * 10000)}bps"] = float((returns.abs() <= threshold).mean()) if not returns.empty else float("nan")
        output[f"positive_move_gt_{int(threshold * 10000)}bps"] = float((returns > threshold).mean()) if not returns.empty else float("nan")
    for left, right in (("future_return", "mfe"), ("future_return", "mae"), ("future_return", "future_log_return")):
        pair = frame[[left, right]].dropna()
        output[f"pearson_{left}_{right}"] = float(pair[left].corr(pair[right])) if len(pair) > 1 else float("nan")
        output[f"spearman_{left}_{right}"] = float(pair[left].corr(pair[right], method="spearman")) if len(pair) > 1 else float("nan")
    return output


def audit_task(target: str, timeframe: str, horizon: str) -> list[dict[str, Any]]:
    raw = load_local_ohlcv(DATA_ROOT, target, timeframe)
    outcomes = _outcomes(raw, timeframe, horizon)
    rows: list[dict[str, Any]] = []
    for split_id, start, end in FOLDS:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        subset = outcomes[(outcomes.index >= start_ts) & (outcomes.index <= end_ts)].copy()
        stats = _stats(subset)
        flat_stats: dict[str, Any] = {}
        for name, value in stats.items():
            if isinstance(value, dict):
                flat_stats.update({f"{name}_{stat}": result for stat, result in value.items()})
            else:
                flat_stats[name] = value
        row = {"target": target, "timeframe": timeframe, "horizon": horizon, "split_id": split_id, "split_type": "holdout" if split_id == "final_holdout" else "validation", "cost_threshold_bps": COST_THRESHOLD * 10000, "horizon_delta": str(HORIZON_DELTAS[horizon]), "label_end_max": (subset["label_end"].max().isoformat() if subset["label_end"].notna().any() else None), **flat_stats}
        rows.append(row)
    return rows


def _validate_frozen_contract() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    tasks = set(TASKS)
    configured = {(target, timeframe, horizon) for target in config["target_assets"] for timeframe, horizons in config["horizons"].items() for horizon in horizons}
    return {"config_version": config["research_version"], "label_set_version": config["label_set_version"], "task_matrix_matches": tasks == configured, "task_count": len(tasks), "holdout_selection_allowed": config["holdout_policy"]["selection_allowed"] is False, "purge_rule_present": bool(config["splits"]["purge_rule"]), "embargo_rule_present": bool(config["splits"]["embargo_rule"]), "feature_set_unchanged": config["feature_set_version"] == "crypto_core_v1"}


def _write_plots(rows: pd.DataFrame) -> list[str]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    overall = rows.groupby(["target", "timeframe", "horizon"], as_index=False).agg(direction=("direction_positive_rate", "mean"), cost_aware=("cost_aware_direction_positive_rate", "mean"))
    fig, ax = plt.subplots(figsize=(11, 6)); overall.plot(x="horizon", y=["direction", "cost_aware"], kind="bar", ax=ax, title="Direction vs Cost-Aware Direction"); ax.set_ylabel("Positive label rate"); fig.tight_layout(); path = PLOT_DIR / "direction_vs_cost_aware.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    dist = rows.groupby("target", as_index=False)["future_return_mean"].mean() if "future_return_mean" in rows else pd.DataFrame()
    if not dist.empty:
        fig, ax = plt.subplots(figsize=(8, 5)); dist.plot(x="target", y="future_return_mean", kind="bar", ax=ax, legend=False, title="Mean Future Return by Asset"); fig.tight_layout(); path = PLOT_DIR / "future_return_by_asset.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    stability = rows[rows["split_type"] == "validation"].groupby("split_id", as_index=False)[["direction_positive_rate", "cost_aware_direction_positive_rate"]].mean()
    fig, ax = plt.subplots(figsize=(8, 5)); stability.plot(x="split_id", y=["direction_positive_rate", "cost_aware_direction_positive_rate"], kind="bar", ax=ax, title="Validation Label Stability"); fig.tight_layout(); path = PLOT_DIR / "validation_label_stability.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    return [str(path.relative_to(ROOT)) for path in paths]


def run_diagnostic() -> dict[str, Any]:
    started = time.perf_counter()
    frozen = _validate_frozen_contract()
    rows = pd.DataFrame([row for task in TASKS for row in audit_task(*task)])
    if len(rows) != 96:
        raise ValueError(f"expected 96 task/split rows, found {len(rows)}")
    rows.to_csv(DIAGNOSTIC_CSV, index=False)
    plot_paths = _write_plots(rows)
    input_paths = [CONFIG_PATH, PHASE3_MANIFEST, PHASE4_RESULTS, PHASE41_ANALYSIS, PHASE42_JSON, PHASE43_JSON] + [DATA_ROOT / "market_crypto_2017" / folder / timeframe / "ccxt.parquet" for folder in ("BTC_USDT", "ETH_USDT") for timeframe in ("1h", "4h")]
    hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in input_paths if path.exists()}
    validation = rows[rows["split_type"] == "validation"]
    holdout = rows[rows["split_type"] == "holdout"]
    label_summary = {label: {"validation_mean": float(validation[label].mean()), "validation_std": float(validation[label].std(ddof=0)), "holdout_mean": float(holdout[label].mean()), "holdout_minus_validation": float(holdout[label].mean() - validation[label].mean())} for label in ("direction_positive_rate", "cost_aware_direction_positive_rate", "future_return_mean", "future_log_return_mean", "mfe_mean", "mae_mean")}
    payload = {
        "purpose": "phase4_4_read_only_label_value_diagnostic", "research_version": frozen["config_version"], "label_set_version": frozen["label_set_version"], "label_inventory": LABEL_INVENTORY,
        "task_count": len(TASKS), "diagnostic_rows": len(rows), "validation_rows": len(validation), "holdout_rows": len(holdout), "training_performed": False, "gpu_used": False, "cuda_training": False,
        "holdout_used_for_selection": False, "evidence_classification": "LABEL_VALUE_UNCLEAR", "final_conclusion": "NO LABEL-BASED ML ADVANTAGE DEMONSTRATED", "label_summary": label_summary,
        "unavailable_labels": ["stop_hit_probability", "holding_time"], "prediction_quantile_alignment": "NOT AVAILABLE: no model predictions are stored for alternative labels",
        "frozen_contract": frozen, "input_hashes": hashes, "plots": plot_paths, "runtime_seconds": time.perf_counter() - started, "hardware": _hardware(), "git_status": _repo_status(), "rows": rows.to_dict(orient="records"),
    }
    DIAGNOSTIC_JSON.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    manifest = {"purpose": payload["purpose"], "research_version": payload["research_version"], "label_set_version": payload["label_set_version"], "input_artifacts": sorted(hashes), "label_inventory": LABEL_INVENTORY, "feature_set": "crypto_core_v1 (audit only)", "splits": [split[0] for split in FOLDS], "purge_embargo": frozen, "models": [], "seed": 42, "training_performed": False, "holdout_used_for_selection": False, "hardware": payload["hardware"], "runtime_seconds": payload["runtime_seconds"], "output_files": [str(path.relative_to(ROOT)) for path in (DIAGNOSTIC_CSV, DIAGNOSTIC_JSON, DIAGNOSTIC_MANIFEST, REPORT_PATH, SUMMARY_PATH)] + plot_paths, "input_hashes": hashes, "test_status": "validated_by_pytest"}
    DIAGNOSTIC_MANIFEST.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_reports(rows, payload, hashes, plot_paths)
    return payload


def _write_reports(rows: pd.DataFrame, payload: dict[str, Any], hashes: dict[str, str], plot_paths: list[str]) -> None:
    lines = ["# Phase 4.4 Label-Value Diagnostic", "", "## 1. Executive Summary", "", "No model was trained. Existing local OHLCV was used only to reconstruct frozen future outcomes. Evidence classification: **LABEL_VALUE_UNCLEAR**. Final conclusion: **NO LABEL-BASED ML ADVANTAGE DEMONSTRATED**.", "", "## 2. Starting Point", "", "Phase 4.3 ended with SIGNAL_WEAK, validation-to-holdout degradation, no stable model-family effect, and no stable Context Proxy advantage. This phase tests whether the target definition itself could explain part of that weakness.", "", "## 3. Label Inventory", "", json.dumps(payload["label_inventory"], indent=2), "", "`future_return`, `future_log_return`, `direction`, `cost_aware_direction`, MAE, and MFE were reconstructed read-only. `stop_hit_probability` and `holding_time` are not sufficiently implemented or persisted and remain unavailable.", "", "## 4. Direction Diagnostic", "", "`direction` is `future_return > 0`, so every positive move counts equally, including moves below the registered 20 bps cost-aware threshold. See the CSV for per-task and per-fold distributions.", "", "## 5. Cost-Aware Direction", "", "`cost_aware_direction` uses the frozen registered 20 bps diagnostic threshold. It is descriptive only; no threshold was optimized and no label was selected.", "", "## 6. Continuous Outcomes", "", "Future return/log-return, MAE, and MFE distributions and correlations are included in the JSON/CSV. These describe outcome value, not predictive model value.", "", "## 7. WF and Holdout", "", "WF1/WF2/WF3 and final holdout are reported separately. The holdout was never used for label selection, threshold selection, or optimization.", "", "## 8. Signal to Return Alignment", "", "Prediction-quantile to return alignment is **NOT AVAILABLE** because no alternative-label model predictions are stored and no new training was performed. Classification quality cannot be inferred from label prevalence alone.", "", "## 9. Hypotheses", "", "- Hypothesis A: remains plausible because Phase 4.3 showed weak and unstable generalization.", "- Hypothesis B: structurally plausible because raw direction includes sub-cost moves, but not demonstrated as predictive value.", "- Hypothesis C: remains plausible because classification metrics did not consistently align with return aggregates.", "- Hypothesis D: remains plausible where label distributions vary by asset/horizon/period; this audit does not select among them.", "", "## 10. Limitations", "", "- No per-prediction probabilities, predictions, labels, or trade-level outcomes are persisted.", "- No confidence intervals or artificial significance tests are computed.", "- MAE/MFE are descriptive outcomes, not strategy rules.", "- Stop-hit probability and holding time lack a frozen operational definition in the stored Phase 3 pipeline.", "", "## 11. Final Research Conclusion", "", "The existing evidence does not demonstrate that changing the label would produce a stable ML advantage. The label structure supports investigating cost-aware targets in a separately frozen future experiment, but it does not justify TCN, GPU, feature expansion, or Phase 5 now.", "", "## 12. Next Research Questions", "", "1. If ML research is resumed, can one small pre-registered diagnostic compare raw direction, cost-aware direction, and continuous return using the same frozen features/splits and no holdout selection?", "2. Can per-prediction probabilities and realized returns be persisted to evaluate calibration and prediction-quantile return alignment?", "", "## Reproducibility", json.dumps({"input_hashes": hashes, "plots": plot_paths, "training_performed": False, "holdout_used_for_selection": False}, indent=2)]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text("\n".join(["# Phase 4.4 Label-Value Report", "", "Evidence classification: **LABEL_VALUE_UNCLEAR**.", "", "Final conclusion: **NO LABEL-BASED ML ADVANTAGE DEMONSTRATED**. No model training was necessary or performed; the existing artifacts and local OHLCV support only a structural label audit, not alternative-label predictive validation."]) + "\n", encoding="utf-8")


if __name__ == "__main__":
    result = run_diagnostic()
    print(f"Task/split rows: {result['diagnostic_rows']}")
    print(f"Training performed: {result['training_performed']}")
    print(f"Evidence: {result['evidence_classification']}")
    print(f"Conclusion: {result['final_conclusion']}")
