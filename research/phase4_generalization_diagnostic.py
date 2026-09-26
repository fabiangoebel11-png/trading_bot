"""Read-only Phase 4.3 generalization and signal-value diagnostic.

Consumes only frozen Phase 4/4.1/4.2 aggregate artifacts. It never trains,
selects, tunes, promotes, or changes trading/runtime behavior. The final
holdout is reported descriptively and is never used for selection.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.phase4_ml_baseline_analysis import (
    ANALYSIS_CSV,
    ANALYSIS_JSON,
    ANALYSIS_MANIFEST,
    FEATURE_SETS,
    HOLDOUT_SPLIT,
    KEYS,
    MANIFEST_PATH,
    MODELS,
    RESULTS_CSV,
    RESULTS_JSON,
    SEED,
    VALIDATION_SPLITS,
    _sha256,
    load_artifacts,
    validate_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE3_CONFIG = ROOT / "configs" / "ml_research_v1.yaml"
PHASE3_MANIFEST = ROOT / "data" / "research" / "phase3" / "ml_research_manifest.json"
PHASE42_CSV = ROOT / "data" / "research" / "phase4" / "phase4_deep_diagnostic.csv"
PHASE42_JSON = ROOT / "data" / "research" / "phase4" / "phase4_deep_diagnostic.json"
OUTPUT_DIR = ROOT / "data" / "research" / "phase4"
PLOT_DIR = OUTPUT_DIR / "plots" / "generalization"
DIAGNOSTIC_CSV = OUTPUT_DIR / "phase4_generalization_diagnostic.csv"
DIAGNOSTIC_JSON = OUTPUT_DIR / "phase4_generalization_diagnostic.json"
DIAGNOSTIC_MANIFEST = OUTPUT_DIR / "phase4_generalization_diagnostic_manifest.json"
REPORT_PATH = ROOT / "research" / "phase4_generalization_diagnostic.md"
SIGNAL_REPORT_PATH = ROOT / "research" / "phase4_signal_value_report.md"

THRESHOLDS = ("0_5", "0_55", "0_6", "0_65", "0_7")
CORE_METRICS = ("roc_auc", "pr_auc", "balanced_accuracy", "log_loss", "brier_score", "accuracy", "mean_forward_return")
SIGNAL_METRICS = CORE_METRICS + ("positive_rate",)
RETURN_METRICS = ("mean_forward_return",) + tuple(
    f"mean_forward_return_{direction}_{threshold}"
    for threshold in THRESHOLDS for direction in ("up", "down")
)
COVERAGE_METRICS = tuple(
    f"signal_coverage_{direction}_{threshold}"
    for threshold in THRESHOLDS for direction in ("up", "down")
)
ALL_METRICS = tuple(dict.fromkeys(SIGNAL_METRICS + RETURN_METRICS + COVERAGE_METRICS))
HORIZON_HOURS = {"1h": 1, "2h": 2, "4h": 4, "8h": 8, "12h": 12, "24h": 24, "3d": 72, "7d": 168, "14d": 336, "21d": 504, "28d": 672}


def _metric_column(metric: str) -> str:
    return f"metric_{metric}"


def _safe_float(value: Any) -> float:
    return float(value) if pd.notna(value) else float("nan")


def _relative_gap(validation: float, holdout: float, metric: str) -> float:
    if not np.isfinite(validation) or not np.isfinite(holdout):
        return float("nan")
    denominator = abs(validation)
    if denominator <= 1e-12:
        return float("nan")
    return (holdout - validation) / denominator


def _horizon_bucket(horizon: str) -> str:
    hours = HORIZON_HOURS[horizon]
    if hours <= 2:
        return "1-2h"
    if hours <= 8:
        return "4-8h"
    if hours <= 24:
        return "12-24h"
    if hours <= 168:
        return "3-7d"
    if hours <= 504:
        return "1-3w"
    return "longer frozen horizons"


def _aggregate(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    values = pd.to_numeric(frame[_metric_column(metric)], errors="coerce")
    return {
        "mean": _safe_float(values.mean()),
        "median": _safe_float(values.median()),
        "std": _safe_float(values.std(ddof=0)),
        "min": _safe_float(values.min()),
        "max": _safe_float(values.max()),
        "iqr": _safe_float(values.quantile(0.75) - values.quantile(0.25)),
        "count": int(values.notna().sum()),
    }


def _baseline(positive_rate: float) -> dict[str, float]:
    p = min(max(float(positive_rate), 0.0), 1.0)
    entropy = -(p * math.log(max(p, 1e-15)) + (1.0 - p) * math.log(max(1.0 - p, 1e-15)))
    return {
        "chance_roc_auc": 0.5,
        "positive_rate_pr_auc": p,
        "majority_accuracy": max(p, 1.0 - p),
        "always_positive_accuracy": p,
        "always_negative_accuracy": 1.0 - p,
        "positive_rate_brier": p * (1.0 - p),
        "positive_rate_log_loss": entropy,
        "majority_balanced_accuracy": 0.5,
    }


def _pattern(row: pd.Series) -> str:
    auc = [row.get(f"wf_{split}_roc_auc", np.nan) for split in ("2022", "2023", "2024")]
    auc = np.asarray(auc, dtype=float)
    holdout = float(row["holdout_roc_auc"])
    validation = float(row["validation_roc_auc_mean"])
    if not np.isfinite(auc).all() or not np.isfinite(holdout):
        return "INSUFFICIENT_AGGREGATES"
    if np.nanmax(auc) - np.nanmin(auc) <= 0.02 and holdout < np.nanmin(auc) - 0.03:
        return "PATTERN_1_STABLE_WF_HOLDOUT_DROP"
    if auc[0] > auc[1] > auc[2] and (auc[0] - auc[2]) >= 0.01:
        return "PATTERN_2_WITHIN_WF_WEAKENING"
    if np.nanmax(auc) - np.nanmin(auc) >= 0.03:
        return "PATTERN_3_WF_VARIABILITY"
    return_gap = float(row["gap_mean_forward_return"])
    coverage_gap = float(row["gap_signal_coverage_up_0_6"])
    if np.isfinite(return_gap) and np.isfinite(coverage_gap) and abs(return_gap) < 0.0005 and abs(coverage_gap) < 0.05 and validation >= 0.5:
        return "PATTERN_4_AUC_WITHOUT_RETURN_COVERAGE_CHANGE"
    if validation >= 0.5 and validation < 0.505:
        return "PATTERN_5_NEAR_CHANCE_AUC"
    return "MIXED_DESCRIPTIVE_PATTERN"


def _build_diagnostic(results: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key_values, group in results.groupby(KEYS, sort=True):
        key = dict(zip(KEYS, key_values))
        validation = group[group["split_type"] == "validation"].copy()
        holdout = group[group["split_type"] == "holdout"].copy()
        if len(validation) != 3 or len(holdout) != 1:
            raise ValueError(f"incomplete candidate/split group: {key}")
        row: dict[str, Any] = {**key, "horizon_bucket": _horizon_bucket(key["horizon"]), "validation_fold_count": len(validation), "holdout_record_count": len(holdout)}
        for metric in ALL_METRICS:
            column = _metric_column(metric)
            validation_values = pd.to_numeric(validation[column], errors="coerce")
            holdout_value = _safe_float(holdout.iloc[0][column])
            row[f"validation_{metric}_mean"] = _safe_float(validation_values.mean())
            row[f"validation_{metric}_median"] = _safe_float(validation_values.median())
            row[f"validation_{metric}_std"] = _safe_float(validation_values.std(ddof=0))
            row[f"holdout_{metric}"] = holdout_value
            row[f"gap_{metric}"] = holdout_value - row[f"validation_{metric}_mean"]
            row[f"relative_gap_{metric}"] = _relative_gap(row[f"validation_{metric}_mean"], holdout_value, metric)
            for split_id, split_row in validation.set_index("split_id").iterrows():
                row[f"{split_id}_{metric}"] = _safe_float(split_row[column])
        row.update({f"baseline_{name}": value for name, value in _baseline(row["validation_positive_rate_mean"]).items()})
        row["validation_auc_above_chance_fraction"] = float(np.mean([row[f"{split}_roc_auc"] > 0.5 for split in VALIDATION_SPLITS]))
        row["validation_pr_above_positive_rate_fraction"] = float(np.mean([row[f"{split}_pr_auc"] > row["validation_positive_rate_mean"] for split in VALIDATION_SPLITS]))
        row["pattern"] = _pattern(row)
        row["classification_value_flag"] = "classification_advantage" if row["validation_roc_auc_mean"] > 0.5 or row["validation_pr_auc_mean"] > row["validation_positive_rate_mean"] else "not_above_aggregate_baseline"
        row["return_value_flag"] = "return_effect_observed" if abs(row["validation_mean_forward_return_mean"]) > 0 else "no_return_effect_in_aggregate"
        rows.append(row)
    diagnostic = pd.DataFrame(rows)
    if len(diagnostic) != 144:
        raise ValueError(f"expected 144 complete candidate groups, found {len(diagnostic)}")
    stats = {
        "candidate_count": len(diagnostic),
        "validation_rows": int((results["split_type"] == "validation").sum()),
        "holdout_rows": int((results["split_type"] == "holdout").sum()),
        "pattern_counts": diagnostic["pattern"].value_counts().to_dict(),
    }
    return diagnostic, stats


def _context_pairs(results: pd.DataFrame) -> pd.DataFrame:
    pair_keys = ["target", "timeframe", "horizon", "model_id", "split_id"]
    core = results[results["feature_set_id"] == FEATURE_SETS[0]].set_index(pair_keys)
    context = results[results["feature_set_id"] == FEATURE_SETS[1]].set_index(pair_keys)
    shared = core.index.intersection(context.index)
    rows: list[dict[str, Any]] = []
    for index in shared:
        left, right = core.loc[index], context.loc[index]
        record = dict(zip(pair_keys, index))
        record["split_type"] = left["split_type"]
        for metric in CORE_METRICS + RETURN_METRICS + COVERAGE_METRICS:
            column = _metric_column(metric)
            record[f"core_{metric}"] = _safe_float(left[column])
            record[f"context_{metric}"] = _safe_float(right[column])
            record[f"delta_{metric}"] = record[f"context_{metric}"] - record[f"core_{metric}"]
        rows.append(record)
    pairs = pd.DataFrame(rows)
    if len(pairs) != 288:
        raise ValueError(f"expected 288 Core/Context pairs, found {len(pairs)}")
    return pairs


def _group_summary(frame: pd.DataFrame, group_keys: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(group_keys, sort=True):
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        record = dict(zip(group_keys, key_values))
        for metric in ("roc_auc", "pr_auc", "brier_score", "log_loss", "mean_forward_return", "signal_coverage_up_0_6"):
            column = f"{metric}_mean" if f"{metric}_mean" in group else metric
            if column not in group:
                continue
            stats = pd.to_numeric(group[column], errors="coerce")
            record[f"{metric}_mean"] = _safe_float(stats.mean())
            record[f"{metric}_median"] = _safe_float(stats.median())
            record[f"{metric}_std"] = _safe_float(stats.std(ddof=0))
            record[f"{metric}_iqr"] = _safe_float(stats.quantile(0.75) - stats.quantile(0.25))
        output.append(record)
    return output


def _context_summary(pairs: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for split_type, group in pairs.groupby("split_type", sort=True):
        summary[split_type] = {}
        for metric in ("roc_auc", "pr_auc", "brier_score", "log_loss", "mean_forward_return", "signal_coverage_up_0_6"):
            values = pd.to_numeric(group[f"delta_{metric}"], errors="coerce")
            summary[split_type][metric] = {
                "mean": _safe_float(values.mean()), "median": _safe_float(values.median()),
                "std": _safe_float(values.std(ddof=0)), "min": _safe_float(values.min()),
                "max": _safe_float(values.max()), "iqr": _safe_float(values.quantile(0.75) - values.quantile(0.25)),
                "positive_fraction": _safe_float((values > 0).mean()),
            }
    return summary


def _write_plots(diagnostic: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    wf = diagnostic.melt(id_vars=KEYS, value_vars=[f"wf_{year}_roc_auc" for year in ("2022", "2023", "2024")], var_name="split", value_name="roc_auc")
    wf["split"] = wf["split"].str.replace("_roc_auc", "", regex=False)
    fig, ax = plt.subplots(figsize=(10, 6)); wf.groupby("split")["roc_auc"].mean().plot(kind="bar", ax=ax, title="WF1/WF2/WF3 Mean ROC-AUC"); ax.axhline(0.5, color="black", linewidth=0.8); ax.set_ylabel("ROC-AUC"); fig.tight_layout(); path = PLOT_DIR / "wf_validation_holdout_roc_auc.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    for metric, filename, title in (("roc_auc", "validation_vs_holdout_roc_auc.png", "Validation vs Holdout ROC-AUC"), ("pr_auc", "validation_vs_holdout_pr_auc.png", "Validation vs Holdout PR-AUC")):
        fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(diagnostic[f"validation_{metric}_mean"], diagnostic[f"holdout_{metric}"] , alpha=0.45); low = min(ax.get_xlim()[0], ax.get_ylim()[0]); high = max(ax.get_xlim()[1], ax.get_ylim()[1]); ax.plot([low, high], [low, high], color="black", linewidth=0.8); ax.set_xlabel(f"Validation mean {metric}"); ax.set_ylabel(f"Holdout {metric}"); ax.set_title(title); ax.grid(alpha=0.2); fig.tight_layout(); path = PLOT_DIR / filename; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    for group_key, filename, title in (("horizon", "generalization_gap_by_horizon.png", "ROC-AUC Generalization Gap by Horizon"), ("target", "generalization_gap_by_asset.png", "ROC-AUC Generalization Gap by Asset"), ("model_id", "generalization_gap_by_model.png", "ROC-AUC Generalization Gap by Model")):
        frame = diagnostic.groupby(group_key, as_index=False)["gap_roc_auc"].mean(); fig, ax = plt.subplots(figsize=(9, 5)); frame.plot(x=group_key, y="gap_roc_auc", kind="bar", ax=ax, legend=False, title=title); ax.axhline(0, color="black", linewidth=0.8); ax.set_ylabel("Holdout - Validation ROC-AUC"); fig.tight_layout(); path = PLOT_DIR / filename; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    frame = pairs.groupby("split_id", as_index=False)["delta_roc_auc"].mean(); fig, ax = plt.subplots(figsize=(9, 5)); frame.plot(x="split_id", y="delta_roc_auc", kind="bar", ax=ax, legend=False, title="Core vs Context ROC-AUC Delta"); ax.axhline(0, color="black", linewidth=0.8); ax.set_ylabel("Context - Core"); fig.tight_layout(); path = PLOT_DIR / "core_vs_context_generalization.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    eligible = diagnostic[["validation_roc_auc_mean", "validation_mean_forward_return_mean"]].dropna(); fig, ax = plt.subplots(figsize=(7, 6)); ax.scatter(eligible.iloc[:, 0], eligible.iloc[:, 1], alpha=0.45); ax.set_xlabel("Validation mean ROC-AUC"); ax.set_ylabel("Validation mean forward return"); ax.set_title("Forward Return vs ROC-AUC"); ax.grid(alpha=0.2); fig.tight_layout(); path = PLOT_DIR / "forward_return_vs_roc_auc.png"; fig.savefig(path, dpi=150); plt.close(fig); paths.append(path)
    return [str(path.relative_to(ROOT)) for path in paths]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _baseline_comparison(diagnostic: pd.DataFrame) -> dict[str, Any]:
    return {
        "roc_auc_above_chance_rows": int((diagnostic["validation_roc_auc_mean"] > diagnostic["baseline_chance_roc_auc"]).sum()),
        "pr_auc_above_positive_rate_rows": int((diagnostic["validation_pr_auc_mean"] > diagnostic["baseline_positive_rate_pr_auc"]).sum()),
        "accuracy_above_majority_rows": int((diagnostic["validation_accuracy_mean"] > diagnostic["baseline_majority_accuracy"]).sum()),
        "brier_better_than_positive_rate_rows": int((diagnostic["validation_brier_score_mean"] < diagnostic["baseline_positive_rate_brier"]).sum()),
        "log_loss_better_than_positive_rate_rows": int((diagnostic["validation_log_loss_mean"] < diagnostic["baseline_positive_rate_log_loss"]).sum()),
        "note": "Comparisons use only stored fold aggregates and are descriptive; no significance test is available.",
    }


def _write_report(diagnostic: pd.DataFrame, pairs: pd.DataFrame, summaries: dict[str, Any], plot_paths: list[str], hashes: dict[str, str]) -> None:
    validation = diagnostic
    holdout_gap = diagnostic["gap_roc_auc"]
    signal_status = summaries["evidence_classification"]
    lines = [
        "# Phase 4.3 Generalization & Signal-Value Diagnostic", "",
        "Read-only analysis of frozen Phase 4, 4.1, and 4.2 aggregate artifacts. No model was trained, tuned, selected, promoted, or changed.", "",
        "## Decision", "", f"Evidence classification: **{signal_status}**.",
        "This classification describes the evidence base, not an asset, horizon, model, or candidate ranking.", "",
        "## A. Validation to Holdout", "",
        f"All {len(diagnostic)} experiment combinations have three validation folds and one final holdout. ROC-AUC gap summary: {holdout_gap.describe().round(6).to_dict()}.",
        "Absolute gap is `holdout - validation`; relative gap is divided by the absolute validation value where meaningful. Negative values indicate degradation for higher-is-better metrics. For loss metrics, a positive gap is deterioration.", "",
        "## B. WF1/WF2/WF3 vs Holdout", "", json.dumps(summaries["walk_forward_patterns"], indent=2),
        "Pattern labels are descriptive only: stable validation with a holdout drop, within-validation weakening, WF variability, AUC without return/coverage movement, and near-chance AUC.", "",
        "## C. Classification Value vs Return Value", "", json.dumps(summaries["signal_value"], indent=2),
        "ROC-AUC above 0.5 or PR-AUC above the observed positive-rate baseline does not by itself establish a tradeable return effect. The stored artifacts contain no trade-level PnL, fees, slippage, leverage, or execution data.", "",
        "## D. Simple Baselines", "", "Majority accuracy, chance ROC-AUC=0.5, positive-rate PR-AUC, positive-rate Brier, and constant positive-rate log loss are computed from stored fold positive rates only. No model was retrained and no backtest was added.", json.dumps(summaries["baseline_comparison"], indent=2), json.dumps(summaries["baseline_summary"], indent=2), "",
        "## E. Assets, Horizons, and Models", "", json.dumps(summaries["asset_summary"], indent=2), "", json.dumps(summaries["horizon_summary"], indent=2), "", json.dumps(summaries["model_summary"], indent=2), "No group is called best or preferred; differences are descriptive.", "",
        "## F. Core vs Context Proxy", "", json.dumps(summaries["context_summary"], indent=2), "Deltas are paired on identical target/timeframe/horizon/model/split keys. Positive Brier or Log Loss delta is worse; positive ROC-AUC/PR-AUC/return delta is higher. The proxy is not promoted.", "",
        "## G. Missing Data", "", "Available: fold-level aggregate classification metrics, threshold coverage, thresholded forward-return aggregates, split IDs, date ranges, sample counts, model IDs, feature-set IDs, and horizons.", "", "Missing: individual probabilities, prediction timestamps, individual labels, individual future returns, calibration buckets, reliability-curve data, and trade-level execution outcomes. Therefore confidence intervals, calibration curves, and artificial significance tests are **NOT AVAILABLE FROM STORED AGGREGATES**.", "",
        "## H. Holdout and GPU Status", "", "The holdout was used only for final OOS measurement and validation-to-holdout diagnostics. It was not used for feature, model, threshold, horizon, candidate, or strategy selection. No GPU/TCN training was performed. **NO JUSTIFIED GPU EXPERIMENT YET**: the current weakness is unresolved generalization and missing per-prediction telemetry, not a demonstrated capacity limit of sklearn baselines.", "",
        "## I. Next Research Questions", "", "1. Can a separately frozen, instrumented run capture per-prediction probability/label/return telemetry for proper calibration and signal-value analysis?", "2. Does a new chronological validation design reproduce the observed WF-to-holdout behavior without reusing this holdout for selection?", "3. Can the Core-vs-Context instability be explained descriptively across time and assets before any new feature-set experiment is considered?", "",
        "## Reproducibility", json.dumps({"input_hashes": hashes, "plots": plot_paths, "seed": SEED, "holdout_used_for_selection": False}, indent=2), "",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    SIGNAL_REPORT_PATH.write_text("\n".join(["# Phase 4.3 Signal-Value Report", "", f"Evidence classification: **{signal_status}**.", "", "Classification metrics and return/coverage metrics were analyzed separately. Aggregate evidence does not establish a sufficiently validated trading signal. See `phase4_generalization_diagnostic.md` for the complete read-only diagnostic."]) + "\n", encoding="utf-8")


def run_diagnostic() -> dict[str, Any]:
    results, manifest, result_json = load_artifacts()
    checks = validate_artifacts(results, manifest, result_json)
    diagnostic, build_stats = _build_diagnostic(results)
    pairs = _context_pairs(results)
    hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in [RESULTS_CSV, RESULTS_JSON, MANIFEST_PATH, ANALYSIS_CSV, ANALYSIS_JSON, ANALYSIS_MANIFEST, PHASE42_CSV, PHASE42_JSON, PHASE3_CONFIG, PHASE3_MANIFEST] if path.exists()}
    plot_paths = _write_plots(diagnostic, pairs)
    validation = results[results["split_type"] == "validation"]
    diagnostic.to_csv(DIAGNOSTIC_CSV, index=False)
    summaries = {
        "evidence_classification": "SIGNAL_NOT_DEMONSTRATED" if (diagnostic["holdout_roc_auc"] <= 0.53).all() and (diagnostic["gap_roc_auc"] < 0).all() else "SIGNAL_WEAK",
        "walk_forward_patterns": diagnostic["pattern"].value_counts().to_dict(),
        "signal_value": {"classification_advantage_rows": int((diagnostic["classification_value_flag"] == "classification_advantage").sum()), "return_effect_rows": int((diagnostic["return_value_flag"] == "return_effect_observed").sum())},
        "baseline_summary": _group_summary(diagnostic, ["feature_set_id", "model_id"]),
        "baseline_comparison": _baseline_comparison(diagnostic),
        "asset_summary": _group_summary(diagnostic, ["target"]),
        "horizon_summary": _group_summary(diagnostic, ["horizon_bucket"]),
        "model_summary": _group_summary(diagnostic, ["model_id"]),
        "context_summary": _context_summary(pairs),
    }
    payload = {
        "purpose": "read_only_phase4_3_generalization_signal_value_diagnostic",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(diagnostic), "diagnostic_rows": len(diagnostic), "context_pair_rows": len(pairs),
        "validation_rows": int((diagnostic["validation_fold_count"] == 3).sum()), "holdout_rows": int((diagnostic["holdout_record_count"] == 1).sum()),
        "holdout_used_for_selection": False, "training_performed": False, "gpu_training_performed": False,
        "calibration_intervals": "NOT AVAILABLE FROM STORED AGGREGATES", "evidence_classification": summaries["evidence_classification"],
        "artifact_checks": checks, "build_stats": build_stats, "summaries": _json_safe(summaries), "input_hashes": hashes, "plots": plot_paths,
    }
    DIAGNOSTIC_JSON.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    manifest_payload = {"purpose": payload["purpose"], "candidate_count": len(diagnostic), "diagnostic_rows": len(diagnostic), "context_pair_rows": len(pairs), "holdout_used_for_selection": False, "training_performed": False, "gpu_training_performed": False, "input_hashes": hashes, "plots": plot_paths}
    DIAGNOSTIC_MANIFEST.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    _write_report(diagnostic, pairs, summaries, plot_paths, hashes)
    return {"diagnostic": diagnostic, "pairs": pairs, "payload": payload}


if __name__ == "__main__":
    output = run_diagnostic()
    print(f"Combinations: {len(output['diagnostic'])}")
    print(f"Context pairs: {len(output['pairs'])}")
    print(f"Evidence: {output['payload']['evidence_classification']}")
    print(f"Plots: {len(output['payload']['plots'])}")
