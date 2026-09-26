"""Read-only Phase 4.1 analysis of frozen baseline artifacts.

This module never trains models and never reads market data. It consumes only
``data/research/phase4`` outputs and writes descriptive research diagnostics.
Holdout records are reported separately and are never used for classification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "research" / "phase4"
OUTPUT_DIR = INPUT_DIR
RESULTS_CSV = INPUT_DIR / "phase4_ml_baseline_results.csv"
RESULTS_JSON = INPUT_DIR / "phase4_ml_baseline_results.json"
MANIFEST_PATH = INPUT_DIR / "phase4_ml_baseline_manifest.json"
REPORT_PATH = INPUT_DIR / "phase4_ml_baseline_report.md"
ANALYSIS_CSV = OUTPUT_DIR / "phase4_analysis.csv"
ANALYSIS_JSON = OUTPUT_DIR / "phase4_analysis.json"
ANALYSIS_MANIFEST = OUTPUT_DIR / "phase4_analysis_manifest.json"
ANALYSIS_REPORT = ROOT / "research" / "phase4_ml_baseline_analysis.md"
PLOT_DIR = OUTPUT_DIR / "plots"
SEED = 42
FEATURE_SETS = ("crypto_core_v1", "crypto_core_v1+crypto_context_proxy_v1")
MODELS = ("logistic_regression", "hist_gradient_boosting", "random_forest")
VALIDATION_SPLITS = ("wf_2022", "wf_2023", "wf_2024")
HOLDOUT_SPLIT = "final_holdout"
THRESHOLDS = ("0_5", "0_55", "0_6", "0_65", "0_7")
KEYS = ["target", "timeframe", "horizon", "feature_set_id", "model_id"]
METRICS = [
    "roc_auc", "pr_auc", "balanced_accuracy", "log_loss", "brier_score",
    "accuracy", "mean_forward_return", "positive_rate", "sample_count",
]
QUALITY_METRICS = ["roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "log_loss"]
RETURN_METRICS = ["mean_forward_return"] + [
    f"mean_forward_return_{direction}_{threshold}"
    for threshold in THRESHOLDS for direction in ("up", "down")
]
COVERAGE_METRICS = [
    f"signal_coverage_{direction}_{threshold}"
    for threshold in THRESHOLDS for direction in ("up", "down")
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_tasks() -> set[tuple[str, str, str]]:
    tasks: set[tuple[str, str, str]] = set()
    for target in ("BTC/USDT", "ETH/USDT"):
        for horizon in ("1h", "2h", "4h", "8h", "12h", "24h", "3d", "7d"):
            tasks.add((target, "1h", horizon))
        for horizon in ("7d", "14d", "21d", "28d"):
            tasks.add((target, "4h", horizon))
    return tasks


def load_artifacts() -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    results = pd.read_csv(RESULTS_CSV)
    with RESULTS_JSON.open(encoding="utf-8") as handle:
        result_json = json.load(handle)
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if len(results) != len(result_json):
        raise ValueError("CSV/JSON record counts differ")
    return results, manifest, result_json


def validate_artifacts(results: pd.DataFrame, manifest: dict[str, Any], result_json: list[dict[str, Any]]) -> dict[str, Any]:
    validation = results[results["split_type"] == "validation"]
    holdout = results[results["split_type"] == "holdout"]
    task_set = set(zip(results["target"], results["timeframe"], results["horizon"]))
    expected_rows = 24 * 2 * 3 * 4
    checks = {
        "record_count": {"actual": len(results), "expected": expected_rows, "ok": len(results) == expected_rows},
        "validation_count": {"actual": len(validation), "expected": 432, "ok": len(validation) == 432},
        "holdout_count": {"actual": len(holdout), "expected": 144, "ok": len(holdout) == 144},
        "json_count": {"actual": len(result_json), "expected": expected_rows, "ok": len(result_json) == expected_rows},
        "unique_run_ids": {"actual": int(results["run_id"].nunique()), "expected": expected_rows, "ok": results["run_id"].is_unique},
        "task_count": {"actual": len(task_set), "expected": 24, "ok": task_set == _expected_tasks()},
        "feature_sets": {"actual": sorted(results["feature_set_id"].unique()), "expected": sorted(FEATURE_SETS), "ok": set(results["feature_set_id"]) == set(FEATURE_SETS)},
        "models": {"actual": sorted(results["model_id"].unique()), "expected": sorted(MODELS), "ok": set(results["model_id"]) == set(MODELS)},
        "validation_splits": {"actual": sorted(validation["split_id"].unique()), "expected": sorted(VALIDATION_SPLITS), "ok": set(validation["split_id"]) == set(VALIDATION_SPLITS)},
        "holdout_split": {"actual": sorted(holdout["split_id"].unique()), "expected": [HOLDOUT_SPLIT], "ok": set(holdout["split_id"]) == {HOLDOUT_SPLIT}},
        "manifest_record_count": {"actual": manifest["record_count"], "expected": expected_rows, "ok": manifest["record_count"] == expected_rows},
        "manifest_validation_count": {"actual": manifest["validation_record_count"], "expected": 432, "ok": manifest["validation_record_count"] == 432},
        "manifest_holdout_count": {"actual": manifest["holdout_record_count"], "expected": 144, "ok": manifest["holdout_record_count"] == 144},
        "config_hash_consistent": {"actual": int(results["config_hash"].nunique()), "expected": 1, "ok": results["config_hash"].nunique() == 1 and results["config_hash"].iat[0] == manifest["config_hash"]},
    }
    duplicate_keys = results.duplicated(KEYS + ["split_id"]).sum()
    checks["duplicate_experiment_keys"] = {"actual": int(duplicate_keys), "expected": 0, "ok": duplicate_keys == 0}
    if not all(item["ok"] for item in checks.values()):
        raise ValueError("Phase 4 artifact consistency check failed: " + json.dumps(checks, default=str))
    return checks


def _metric_column(metric: str) -> str:
    return f"metric_{metric}"


def _aggregate(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in METRICS + RETURN_METRICS + COVERAGE_METRICS:
        column = _metric_column(metric)
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        output[f"{prefix}_{metric}_mean"] = float(values.mean())
        output[f"{prefix}_{metric}_std"] = float(values.std(ddof=0))
        output[f"{prefix}_{metric}_min"] = float(values.min())
        output[f"{prefix}_{metric}_max"] = float(values.max())
    return output


def _classify(validation: pd.DataFrame) -> str:
    """Objective, pre-declared stability labels based only on three WF folds."""
    if len(validation) != 3:
        return "FAILED"
    auc = pd.to_numeric(validation["metric_roc_auc"], errors="coerce")
    pr_auc = pd.to_numeric(validation["metric_pr_auc"], errors="coerce")
    brier = pd.to_numeric(validation["metric_brier_score"], errors="coerce")
    coverage = pd.to_numeric(validation["metric_signal_coverage_up_0_6"], errors="coerce")
    if not all(series.notna().all() for series in (auc, pr_auc, brier, coverage)):
        return "FAILED"
    positive_auc = int((auc >= 0.5).sum())
    if (
        positive_auc == 3
        and float(auc.mean()) >= 0.505
        and float(auc.std(ddof=0)) <= 0.02
        and int((pr_auc >= 0.5).sum()) >= 2
        and float(brier.max()) <= 0.25
        and float(coverage.max() - coverage.min()) <= 0.50
    ):
        return "ROBUST CANDIDATE"
    if positive_auc >= 1 or float(auc.mean()) >= 0.5:
        return "MIXED"
    return "WEAK"


def _paired_context(results: pd.DataFrame) -> pd.DataFrame:
    validation = results[results["split_type"] == "validation"]
    pair_keys = ["target", "timeframe", "horizon", "model_id", "split_id"]
    core = validation[validation["feature_set_id"] == FEATURE_SETS[0]].set_index(pair_keys)
    context = validation[validation["feature_set_id"] == FEATURE_SETS[1]].set_index(pair_keys)
    shared = core.index.intersection(context.index)
    rows: list[dict[str, Any]] = []
    for index in shared:
        core_row = core.loc[index]
        context_row = context.loc[index]
        record = dict(zip(pair_keys, index))
        for metric in QUALITY_METRICS + ["mean_forward_return"] + COVERAGE_METRICS:
            column = _metric_column(metric)
            record[f"core_{metric}"] = float(core_row[column])
            record[f"context_{metric}"] = float(context_row[column])
            record[f"delta_{metric}"] = float(context_row[column] - core_row[column])
        rows.append(record)
    return pd.DataFrame(rows)


def build_analysis(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key_values, group in results.groupby(KEYS, sort=True):
        key = dict(zip(KEYS, key_values))
        validation = group[group["split_type"] == "validation"]
        holdout = group[group["split_type"] == "holdout"]
        row: dict[str, Any] = {**key, "classification": _classify(validation), "validation_folds": len(validation), "holdout_records": len(holdout)}
        row.update(_aggregate(validation, "validation"))
        row.update(_aggregate(holdout, "holdout"))
        for metric in QUALITY_METRICS + ["mean_forward_return"] + COVERAGE_METRICS:
            row[f"holdout_minus_validation_{metric}"] = row.get(f"holdout_{metric}_mean", np.nan) - row.get(f"validation_{metric}_mean", np.nan)
        rows.append(row)
    analysis = pd.DataFrame(rows)
    context_pairs = _paired_context(results)
    status_counts = analysis["classification"].value_counts().to_dict()
    payload = {
        "classification_definition": {
            "FAILED": "fewer than three validation folds or missing core metrics",
            "ROBUST CANDIDATE": "all three AUC >= 0.50, mean AUC >= 0.505, AUC std <= 0.02, at least two PR-AUC >= 0.50, max Brier <= 0.25, and 0.60-up coverage range <= 0.50",
            "MIXED": "at least one validation fold has AUC >= 0.50 or mean AUC >= 0.50, but robust criteria are not met",
            "WEAK": "no validation fold has AUC >= 0.50 and mean AUC < 0.50",
        },
        "classification_counts": status_counts,
        "context_pair_count": len(context_pairs),
    }
    return analysis, context_pairs, payload


def _plot_metric_by_horizon(results: pd.DataFrame, metric: str, filename: str, title: str) -> None:
    validation = results[results["split_type"] == "validation"].copy()
    validation["horizon_hours"] = validation["horizon"].map({"1h": 1, "2h": 2, "4h": 4, "8h": 8, "12h": 12, "24h": 24, "3d": 72, "7d": 168, "14d": 336, "21d": 504, "28d": 672})
    grouped = validation.groupby(["horizon_hours", "feature_set_id"], as_index=False)[_metric_column(metric)].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    for feature_set, group in grouped.groupby("feature_set_id"):
        ax.plot(group["horizon_hours"], group[_metric_column(metric)], marker="o", label=feature_set)
    ax.set_title(title)
    ax.set_xlabel("Horizon (hours)")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=150)
    plt.close(fig)


def _plot_validation_holdout(analysis: pd.DataFrame) -> None:
    grouped = analysis.groupby("feature_set_id")[["validation_roc_auc_mean", "holdout_roc_auc_mean"]].mean()
    ax = grouped.plot(kind="bar", figsize=(8, 5), rot=15, title="Validation vs Holdout ROC-AUC")
    ax.set_ylabel("ROC-AUC")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(["Validation", "Holdout"])
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "validation_vs_holdout_roc_auc.png", dpi=150)
    plt.close(fig)


def write_plots(results: pd.DataFrame, analysis: pd.DataFrame) -> list[str]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _plot_metric_by_horizon(results, "roc_auc", "validation_roc_auc_by_horizon.png", "Validation ROC-AUC by Horizon")
    _plot_metric_by_horizon(results, "pr_auc", "validation_pr_auc_by_horizon.png", "Validation PR-AUC by Horizon")
    _plot_metric_by_horizon(results, "signal_coverage_up_0_6", "validation_signal_coverage_by_horizon.png", "Validation Up-Signal Coverage by Horizon")
    _plot_validation_holdout(analysis)
    return [str(path.relative_to(ROOT)) for path in sorted(PLOT_DIR.glob("*.png"))]


def _table(frame: pd.DataFrame, columns: list[str], rows: int = 24) -> str:
    available = [column for column in columns if column in frame]
    if frame.empty:
        return "No records."
    return frame[available].head(rows).round(6).to_string(index=False)


def write_report(results: pd.DataFrame, analysis: pd.DataFrame, context_pairs: pd.DataFrame, checks: dict[str, Any], payload: dict[str, Any], plot_paths: list[str], hashes: dict[str, str]) -> None:
    validation = results[results["split_type"] == "validation"]
    holdout = results[results["split_type"] == "holdout"]
    horizon_summary = validation.groupby(["timeframe", "horizon"], as_index=False)["metric_roc_auc"].mean().sort_values(["timeframe", "horizon"])
    asset_summary = validation.groupby("target", as_index=False)[["metric_roc_auc", "metric_pr_auc", "metric_brier_score", "metric_mean_forward_return"]].mean()
    context_summary = context_pairs[["delta_roc_auc", "delta_pr_auc", "delta_brier_score", "delta_mean_forward_return"]].describe().round(6) if not context_pairs.empty else pd.DataFrame()
    holdout_delta = analysis[["holdout_minus_validation_roc_auc", "holdout_minus_validation_pr_auc", "holdout_minus_validation_brier_score", "holdout_minus_validation_mean_forward_return"]].describe().round(6)
    lines = [
        "# Phase 4.1 Frozen ML Baseline Analysis",
        "",
        "Read-only analysis of existing Phase 4 artifacts. No model was retrained, selected, promoted, or changed.",
        "",
        "## 1. Executive Summary",
        "",
        f"Analyzed {len(results)} records: {len(validation)} validation and {len(holdout)} final-holdout records. Artifact checks passed: {sum(item['ok'] for item in checks.values())}/{len(checks)}.",
        "The classification is diagnostic only. It is not a model ranking and does not authorize trading deployment.",
        "",
        "## 2. Research Design",
        "",
        "Frozen matrix: 24 BTC/ETH dataset-horizon tasks × 2 feature sets × 3 sklearn baselines × 3 validation folds plus one final holdout evaluation.",
        "Validation-only classification uses all three walk-forward folds. Holdout values are reported after, but never enter, classification or selection.",
        "",
        "## 3. Databases, Models, and Feature Sets",
        "",
        "Assets: BTC/USDT, ETH/USDT. Input timeframes: 1h and 4h. Horizons: 1h, 2h, 4h, 8h, 12h, 24h, 3d, 7d, 14d, 21d, 28d.",
        "Models: LogisticRegression, HistGradientBoostingClassifier, RandomForestClassifier.",
        "Feature sets: crypto_core_v1 and crypto_core_v1+crypto_context_proxy_v1. The context set remains an optional proxy context, not an automatically valuable input.",
        "",
        "## 4. Validation Results",
        "",
        "The complete machine-readable per-combination summary is in `data/research/phase4/phase4_analysis.csv`; the table below is a compact view.",
        _table(analysis, KEYS + ["classification", "validation_roc_auc_mean", "validation_roc_auc_std", "validation_pr_auc_mean", "validation_brier_score_mean", "validation_mean_forward_return_mean"], 30),
        "",
        "## 5. Walk-Forward Stability",
        "",
        "For each combination, the analysis exports mean/std/min/max for ROC-AUC, PR-AUC, balanced accuracy, Brier, log loss, forward return, thresholded returns, and coverage. The neutral labels are defined before interpreting results:",
        "- ROBUST CANDIDATE: all three AUC values >= 0.50, mean AUC >= 0.505, AUC std <= 0.02, at least two PR-AUC values >= 0.50, max Brier <= 0.25, and 0.60-up coverage range <= 0.50.",
        "- MIXED: at least one validation AUC >= 0.50 or mean AUC >= 0.50, but robust criteria are not met.",
        "- WEAK: no validation AUC >= 0.50 and mean AUC < 0.50.",
        "- FAILED: fewer than three validation folds or missing core metrics.",
        "No single strong walk-forward period is sufficient for ROBUST CANDIDATE.",
        "",
        "## 6. Multi-Horizon Analysis",
        "",
        "Mean validation ROC-AUC by input timeframe and horizon:",
        _table(horizon_summary, ["timeframe", "horizon", "metric_roc_auc"], 30),
        "Longer horizons, shorter horizons, and input-timeframe effects are descriptive patterns only; no horizon is selected or optimized here.",
        "",
        "## 7. BTC vs ETH",
        "",
        _table(asset_summary, ["target", "metric_roc_auc", "metric_pr_auc", "metric_brier_score", "metric_mean_forward_return"], 10),
        "Differences between BTC and ETH are reported as asset-specific behavior, not as a basis for deployment selection.",
        "",
        "## 8. Core vs Context Proxy",
        "",
        "Paired context deltas use identical asset, timeframe, horizon, model, and walk-forward keys. Positive ROC-AUC/PR-AUC/return deltas and negative Brier deltas are directionally favorable, but consistency matters more than isolated values.",
        _table(context_pairs, ["target", "timeframe", "horizon", "model_id", "split_id", "delta_roc_auc", "delta_pr_auc", "delta_brier_score", "delta_mean_forward_return"], 30),
        "Aggregate delta diagnostics:",
        context_summary.to_string() if not context_summary.empty else "No paired context records.",
        "No automatic proxy promotion is made.",
        "",
        "## 9. Classification Counts",
        "",
        json.dumps(payload["classification_counts"], indent=2),
        "",
        "## 10. Calibration and Classification vs Market Relevance",
        "",
        "Brier score and log loss are analyzed as aggregate calibration diagnostics. The artifacts do not contain per-prediction probabilities, confidence bins, or actual-positive-rate-by-confidence-bin records, so the requested 0.50-0.55/.../>0.70 calibration table cannot be reconstructed without new inference output. Thresholded coverage and forward-return columns are retained as descriptive diagnostics only.",
        "ROC-AUC/PR-AUC measure classification information. Mean forward return, thresholded forward return, and signal coverage measure a separate market-information dimension. A high AUC therefore is not interpreted as a trading signal recommendation.",
        "",
        "## 11. Validation to Holdout",
        "",
        "Holdout summary is deliberately separate. The following describes degradation/stability after the frozen validation analysis; it did not influence classification:",
        holdout_delta.to_string(),
        "",
        "## 12. Hardware",
        "",
        "Phase 4 training backend: CPU-only sklearn. The RTX 3070 was present, but CUDA was not active and GPU training was not used. GPU availability is not interpreted as a result. CPU-only inference remains the architectural requirement.",
        "",
        "## 13. Failure Modes and Limitations",
        "",
        "- Results are aggregate fold diagnostics, not a trading backtest or promotion test.",
        "- Proxy context is observational and remains proxy-only.",
        "- No per-prediction confidence distribution is stored, limiting calibration analysis.",
        "- Holdout is one chronological period and cannot establish future persistence.",
        "- Threshold metrics are descriptive and were not optimized.",
        "",
        "## 14. Research Conclusions",
        "",
        "The artifacts support a neutral assessment of temporal generalization, asset differences, horizon behavior, and proxy deltas. They do not support automatic model promotion, strategy changes, threshold changes, leverage changes, or live/paper-trading changes.",
        "",
        "## 15. Explicit Next-Step Candidates",
        "",
        "- Review the machine-readable paired deltas and stability summaries with human sign-off.",
        "- If calibration is required, design a separate pre-registered data capture step for per-prediction probabilities; do not retrofit thresholds from this report.",
        "- Keep any future experiment separate from the frozen Phase 4 artifacts and do not modify the trading runtime as part of this analysis.",
        "",
        "## Reproducibility",
        "",
        json.dumps({"seed": SEED, "input_hashes": hashes, "plots": plot_paths}, indent=2),
    ]
    ANALYSIS_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis() -> dict[str, Any]:
    results, manifest, result_json = load_artifacts()
    checks = validate_artifacts(results, manifest, result_json)
    analysis, context_pairs, payload = build_analysis(results)
    plot_paths = write_plots(results, analysis)
    hashes = {path.name: _sha256(path) for path in (RESULTS_CSV, RESULTS_JSON, MANIFEST_PATH, REPORT_PATH)}
    analysis.to_csv(ANALYSIS_CSV, index=False)
    analysis_payload = {
        "artifact_checks": checks,
        "classification": payload,
        "analysis_record_count": len(analysis),
        "context_pair_record_count": len(context_pairs),
        "analysis_rows": analysis.to_dict(orient="records"),
        "context_pairs": context_pairs.to_dict(orient="records"),
        "input_hashes": hashes,
        "plots": plot_paths,
        "holdout_used_for_selection": False,
    }
    ANALYSIS_JSON.write_text(json.dumps(analysis_payload, indent=2, default=str), encoding="utf-8")
    analysis_manifest = {
        "purpose": "read_only_phase4_1_frozen_baseline_analysis",
        "source_artifacts": {path.name: str(path.relative_to(ROOT)) for path in (RESULTS_CSV, RESULTS_JSON, MANIFEST_PATH, REPORT_PATH)},
        "input_hashes": hashes,
        "analysis_csv": str(ANALYSIS_CSV.relative_to(ROOT)),
        "analysis_json": str(ANALYSIS_JSON.relative_to(ROOT)),
        "report": str(ANALYSIS_REPORT.relative_to(ROOT)),
        "plots": plot_paths,
        "seed": SEED,
        "holdout_used_for_selection": False,
        "training_performed": False,
        "classification_source": "validation_only",
    }
    ANALYSIS_MANIFEST.write_text(json.dumps(analysis_manifest, indent=2), encoding="utf-8")
    write_report(results, analysis, context_pairs, checks, payload, plot_paths, hashes)
    return {"analysis": analysis, "context_pairs": context_pairs, "checks": checks, "payload": payload, "plots": plot_paths}


if __name__ == "__main__":
    output = run_analysis()
    print(f"Phase 4.1 analysis rows: {len(output['analysis'])}")
    print(json.dumps(output["payload"]["classification_counts"], sort_keys=True))
