"""Read-only Phase 4.2 deep diagnosis.

Consumes frozen Phase 4/4.1 artifacts only. It does not train, select, tune,
or modify any trading/runtime code. Holdout records are descriptive only.
"""
from __future__ import annotations

import hashlib
import json
import platform
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
    _classify,
    _sha256,
    load_artifacts,
    validate_artifacts,
)

ROOT = Path(__file__).resolve().parents[1]
PHASE3_CONFIG = ROOT / "configs" / "ml_research_v1.yaml"
PHASE3_MANIFEST = ROOT / "data" / "research" / "phase3" / "ml_research_manifest.json"
PHASE41_REPORT = ROOT / "research" / "phase4_ml_baseline_analysis.md"
OUTPUT_DIR = ROOT / "data" / "research" / "phase4"
PLOT_DIR = OUTPUT_DIR / "plots" / "deep_diagnostic"
DIAGNOSTIC_CSV = OUTPUT_DIR / "phase4_deep_diagnostic.csv"
DIAGNOSTIC_JSON = OUTPUT_DIR / "phase4_deep_diagnostic.json"
DIAGNOSTIC_MANIFEST = OUTPUT_DIR / "phase4_deep_diagnostic_manifest.json"
DIAGNOSTIC_REPORT = ROOT / "research" / "phase4_ml_deep_diagnostic.md"
DOSSIER_REPORT = ROOT / "research" / "phase4_candidate_dossiers.md"

METRICS = [
    "roc_auc", "pr_auc", "balanced_accuracy", "log_loss", "brier_score", "accuracy",
    "mean_forward_return", "sample_count", "positive_rate",
]
THRESHOLD_TOKENS = ("0_5", "0_55", "0_6", "0_65", "0_7")
THRESHOLD_METRICS = [
    f"signal_coverage_{direction}_{threshold}"
    for threshold in THRESHOLD_TOKENS for direction in ("up", "down")
] + [
    f"mean_forward_return_{direction}_{threshold}"
    for threshold in THRESHOLD_TOKENS for direction in ("up", "down")
]
ALL_METRICS = METRICS + THRESHOLD_METRICS
PAIR_METRICS = ["roc_auc", "pr_auc", "balanced_accuracy", "log_loss", "brier_score", "accuracy", "mean_forward_return"] + THRESHOLD_METRICS


def _metric(row: pd.Series, metric: str) -> float:
    return float(row[f"metric_{metric}"])


def _temporal_category(validation: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    auc = validation["metric_roc_auc"].astype(float).to_numpy()
    pr_auc = validation["metric_pr_auc"].astype(float).to_numpy()
    brier = validation["metric_brier_score"].astype(float).to_numpy()
    returns = validation["metric_mean_forward_return"].astype(float).to_numpy()
    positive_auc = int((auc >= 0.5).sum())
    positive_pr = int((pr_auc >= 0.5).sum())
    excess = np.maximum(auc - 0.5, 0.0)
    dominance = float(excess.max() / excess.sum()) if excess.sum() > 0 else 0.0
    weak_fold_count = int(((auc < 0.5) | (pr_auc < 0.5)).sum())
    details = {
        "positive_auc_wfs": positive_auc,
        "negative_auc_wfs": int((auc < 0.5).sum()),
        "positive_pr_auc_wfs": positive_pr,
        "negative_pr_auc_wfs": int((pr_auc < 0.5).sum()),
        "auc_range": float(auc.max() - auc.min()),
        "auc_std": float(auc.std(ddof=0)),
        "brier_range": float(brier.max() - brier.min()),
        "return_range": float(returns.max() - returns.min()),
        "positive_auc_excess_dominance": dominance,
        "worst_wf": str(validation.iloc[int(np.argmin(auc))]["split_id"]),
        "best_wf": str(validation.iloc[int(np.argmax(auc))]["split_id"]),
        "weak_fold_count": weak_fold_count,
    }
    if dominance >= 0.60 or (auc.max() >= 0.55 and auc.min() <= 0.505):
        category = "D"
    elif positive_auc == 3 and positive_pr == 3 and details["auc_range"] <= 0.03 and details["brier_range"] <= 0.03:
        category = "A"
    elif positive_auc == 3 and weak_fold_count <= 1:
        category = "B"
    else:
        category = "C"
    return category, details


def _horizon_bucket(timeframe: str, horizon: str) -> str:
    hours = {"1h": 1, "2h": 2, "4h": 4, "8h": 8, "12h": 12, "24h": 24, "3d": 72, "7d": 168, "14d": 336, "21d": 504, "28d": 672}[horizon]
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


def _paired_context(results: pd.DataFrame, candidate: pd.Series) -> pd.DataFrame:
    base = {key: candidate[key] for key in ("target", "timeframe", "horizon", "model_id")}
    subset = results[
        (results["target"] == base["target"])
        & (results["timeframe"] == base["timeframe"])
        & (results["horizon"] == base["horizon"])
        & (results["model_id"] == base["model_id"])
    ]
    core = subset[subset["feature_set_id"] == FEATURE_SETS[0]].set_index("split_id")
    context = subset[subset["feature_set_id"] == FEATURE_SETS[1]].set_index("split_id")
    rows: list[dict[str, Any]] = []
    for split_id in sorted(core.index.intersection(context.index)):
        left = core.loc[split_id]
        right = context.loc[split_id]
        record = {**base, "split_id": split_id, "split_type": left["split_type"], "candidate_feature_set": candidate["feature_set_id"]}
        for metric in PAIR_METRICS:
            record[f"core_{metric}"] = _metric(left, metric)
            record[f"context_{metric}"] = _metric(right, metric)
            record[f"delta_{metric}"] = _metric(right, metric) - _metric(left, metric)
        rows.append(record)
    return pd.DataFrame(rows)


def _candidate_rows(results: pd.DataFrame, analysis: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    candidates = analysis[analysis["classification"] == "ROBUST CANDIDATE"].copy()
    if len(candidates) != 31:
        raise ValueError(f"expected 31 Phase 4.1 candidates, found {len(candidates)}")
    diagnostic_rows: list[dict[str, Any]] = []
    dossiers: list[dict[str, Any]] = []
    for _, candidate in candidates.sort_values(KEYS).iterrows():
        selector = np.ones(len(results), dtype=bool)
        for key in KEYS:
            selector &= results[key].eq(candidate[key])
        group = results[selector]
        validation = group[group["split_id"].isin(VALIDATION_SPLITS)].sort_values("split_id")
        holdout = group[group["split_id"] == HOLDOUT_SPLIT]
        category, temporal = _temporal_category(validation)
        base = {key: candidate[key] for key in KEYS}
        for _, row in pd.concat([validation, holdout]).iterrows():
            record = {**base, "split_id": row["split_id"], "split_type": row["split_type"], "temporal_category": category, **temporal}
            for metric in ALL_METRICS:
                record[f"metric_{metric}"] = _metric(row, metric)
            diagnostic_rows.append(record)
        dossier = {**base, "phase41_classification": candidate["classification"], "temporal_category": category, "temporal_diagnosis": temporal}
        dossier["validation"] = validation[["split_id"] + [f"metric_{metric}" for metric in ALL_METRICS]].to_dict(orient="records")
        dossier["holdout"] = holdout[["split_id"] + [f"metric_{metric}" for metric in ALL_METRICS]].to_dict(orient="records")
        dossier["core_context_pairs"] = _paired_context(results, candidate).to_dict(orient="records")
        dossier["main_observation"] = (
            "all three validation folds meet the Phase 4.1 AUC criterion, but the detailed WF category "
            f"is {category}; this is descriptive and not a promotion signal."
        )
        dossier["potential_follow_up"] = "Pre-register a separate diagnostic question about temporal stability; do not alter this frozen experiment."
        dossiers.append(dossier)
    weak = analysis[analysis["classification"] == "WEAK"]
    weak_dossier: dict[str, Any] = {"count": len(weak), "rows": []}
    for _, candidate in weak.iterrows():
        selector = np.ones(len(results), dtype=bool)
        for key in KEYS:
            selector &= results[key].eq(candidate[key])
        group = results[selector].sort_values("split_id")
        weak_dossier["rows"].append({
            **{key: candidate[key] for key in KEYS},
            "classification": candidate["classification"],
            "validation": group[group["split_id"].isin(VALIDATION_SPLITS)][["split_id"] + [f"metric_{metric}" for metric in METRICS]].to_dict(orient="records"),
            "holdout": group[group["split_id"] == HOLDOUT_SPLIT][["split_id"] + [f"metric_{metric}" for metric in METRICS]].to_dict(orient="records"),
        })
    return pd.DataFrame(diagnostic_rows), dossiers, weak_dossier


def _write_plot(path: Path, title: str, ylabel: str, frame: pd.DataFrame, x: str, hue: str, value: str, rotate: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, group in frame.groupby(hue):
        ax.plot(group[x].astype(str), group[value], marker="o", label=str(name))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotate)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_plots(results: pd.DataFrame, diagnostic: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = diagnostic.drop_duplicates(KEYS).copy()
    wf = diagnostic[diagnostic["split_type"] == "validation"].copy()
    candidate_labels = wf[KEYS].astype(str).agg("|".join, axis=1)
    wf["candidate"] = candidate_labels
    wf = wf.sort_values(["candidate", "split_id"])
    fig, ax = plt.subplots(figsize=(14, 8))
    for name, group in wf.groupby("candidate"):
        ax.plot(group["split_id"], group["metric_roc_auc"], marker="o", alpha=0.7, linewidth=1)
    ax.axhline(0.5, color="black", linewidth=0.8)
    ax.set_title("WF1/WF2/WF3 ROC-AUC for Phase 4.1 Robust Candidates")
    ax.set_ylabel("ROC-AUC")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "candidate_wf_roc_auc.png", dpi=150)
    plt.close(fig)
    holdout = diagnostic[diagnostic["split_type"] == "holdout"].merge(candidates[KEYS], on=KEYS)
    val = diagnostic[diagnostic["split_type"] == "validation"].groupby(KEYS, as_index=False)["metric_roc_auc"].mean().rename(columns={"metric_roc_auc": "validation"})
    ho = holdout[[*KEYS, "metric_roc_auc"]].rename(columns={"metric_roc_auc": "holdout"})
    vh = val.merge(ho, on=KEYS)
    vh["label"] = vh[KEYS].astype(str).agg("|".join, axis=1)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.scatter(vh["validation"], vh["holdout"], alpha=0.75)
    ax.axline((0, 0), slope=1, color="black", linewidth=0.8)
    ax.set_title("Candidate Validation vs Holdout ROC-AUC")
    ax.set_xlabel("Validation mean ROC-AUC")
    ax.set_ylabel("Holdout ROC-AUC")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "candidate_validation_vs_holdout_roc_auc.png", dpi=150)
    plt.close(fig)
    for metric, filename, title in (("pr_auc", "candidate_validation_vs_holdout_pr_auc.png", "Candidate Validation vs Holdout PR-AUC"), ("brier_score", "candidate_delta_brier.png", "Candidate Core vs Context Delta Brier")):
        if metric == "brier_score":
            frame = pairs.groupby("split_id", as_index=False)["delta_brier_score"].mean()
            _write_plot(PLOT_DIR / filename, title, "Context - Core", frame, "split_id", "split_id", "delta_brier_score")
        else:
            val = diagnostic[diagnostic["split_type"] == "validation"].groupby(KEYS, as_index=False)[f"metric_{metric}"].mean()
            ho = diagnostic[diagnostic["split_type"] == "holdout"].groupby(KEYS, as_index=False)[f"metric_{metric}"].mean()
            frame = val.merge(ho, on=KEYS)
            frame["candidate"] = frame[KEYS].astype(str).agg("|".join, axis=1)
            fig, ax = plt.subplots(figsize=(12, 7))
            ax.scatter(frame[f"metric_{metric}_x"], frame[f"metric_{metric}_y"], alpha=0.75)
            ax.set_title(title)
            ax.set_xlabel(f"Validation mean {metric}")
            ax.set_ylabel(f"Holdout {metric}")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(PLOT_DIR / filename, dpi=150)
            plt.close(fig)
    pair_wf = pairs[pairs["split_id"].isin(VALIDATION_SPLITS)].groupby("split_id", as_index=False)["delta_roc_auc"].mean()
    _write_plot(PLOT_DIR / "candidate_delta_roc_auc.png", "Core vs Context Delta ROC-AUC", "Context - Core", pair_wf, "split_id", "split_id", "delta_roc_auc")
    horizon = wf.groupby("horizon", as_index=False)["metric_roc_auc"].mean()
    _write_plot(PLOT_DIR / "candidate_performance_by_horizon.png", "Candidate Validation ROC-AUC by Horizon", "ROC-AUC", horizon, "horizon", "horizon", "metric_roc_auc", 35)
    asset = wf.groupby(["target", "split_id"], as_index=False)["metric_roc_auc"].mean()
    _write_plot(PLOT_DIR / "candidate_performance_by_asset.png", "Candidate Validation ROC-AUC by Asset", "ROC-AUC", asset, "split_id", "target", "metric_roc_auc")
    model = wf.groupby(["model_id", "split_id"], as_index=False)["metric_roc_auc"].mean()
    _write_plot(PLOT_DIR / "candidate_performance_by_model.png", "Candidate Validation ROC-AUC by Model Family", "ROC-AUC", model, "split_id", "model_id", "metric_roc_auc")
    return [str(path.relative_to(ROOT)) for path in sorted(PLOT_DIR.glob("*.png"))]


def _write_dossiers(dossiers: list[dict[str, Any]]) -> None:
    lines = ["# Phase 4.2 Candidate Dossiers", "", "These 31 entries are diagnostic only. They do not rank or promote models.", ""]
    for index, dossier in enumerate(dossiers, 1):
        lines.extend([
            f"## Candidate {index}", "",
            f"- asset: {dossier['target']}", f"- timeframe: {dossier['timeframe']}", f"- horizon: {dossier['horizon']}",
            f"- model: {dossier['model_id']}", f"- feature_set: {dossier['feature_set_id']}",
            f"- Phase 4.1 classification: {dossier['phase41_classification']}", f"- Temporal diagnosis: {dossier['temporal_category']}",
            f"- Positive AUC WFs: {dossier['temporal_diagnosis']['positive_auc_wfs']}; negative AUC WFs: {dossier['temporal_diagnosis']['negative_auc_wfs']}",
            f"- Best WF: {dossier['temporal_diagnosis']['best_wf']}; worst WF: {dossier['temporal_diagnosis']['worst_wf']}",
            "", "WF diagnostics:", "", "```json", json.dumps(dossier["validation"], indent=2), "```", "",
            "Holdout diagnostics:", "", "```json", json.dumps(dossier["holdout"], indent=2), "```", "",
            f"Main observation: {dossier['main_observation']}", f"Potential follow-up: {dossier['potential_follow_up']}", "",
        ])
    DOSSIER_REPORT.write_text("\n".join(lines), encoding="utf-8")


def _write_report(results: pd.DataFrame, diagnostic: pd.DataFrame, dossiers: list[dict[str, Any]], weak_dossier: dict[str, Any], pairs: pd.DataFrame, checks: dict[str, Any], plot_paths: list[str], input_hashes: dict[str, str], temporal_counts: dict[str, int]) -> None:
    validation = diagnostic[diagnostic["split_type"] == "validation"]
    holdout = diagnostic[diagnostic["split_type"] == "holdout"]
    asset = validation.groupby("target")["metric_roc_auc"].mean().round(6).to_dict()
    horizon = validation.groupby("horizon")["metric_roc_auc"].mean().round(6).to_dict()
    model = validation.groupby("model_id")["metric_roc_auc"].mean().round(6).to_dict()
    context = pairs.groupby("split_id")[["delta_roc_auc", "delta_pr_auc", "delta_brier_score", "delta_mean_forward_return"]].mean().round(6).to_dict("index")
    candidate_holdout = holdout.merge(validation.groupby(KEYS, as_index=False)["metric_roc_auc"].mean().rename(columns={"metric_roc_auc": "validation_auc"}), on=KEYS)
    candidate_holdout["delta_auc"] = candidate_holdout["metric_roc_auc"] - candidate_holdout["validation_auc"]
    lines = [
        "# Phase 4.2 Deep Diagnostic of Frozen ML Baselines", "",
        "Read-only diagnostic. No training, tuning, feature changes, promotion, strategy, risk, paper, or live changes were performed.", "",
        "## A. Temporal Stability", "",
        f"All {len(dossiers)} Phase 4.1 ROBUST CANDIDATE combinations were expanded into WF1/WF2/WF3 dossiers. Descriptive temporal categories: {json.dumps(temporal_counts, sort_keys=True)}.",
        "A = stable across all WFs with positive AUC/PR-AUC and tight AUC/Brier ranges; B = mostly stable with one weak metric/fold; C = aggregate-positive but visibly variable; D = one WF dominates the positive AUC excess or combines a strong and weak period.",
        "These categories are diagnostic labels, not rankings or selection rules.", "",
        "## B. Asset Patterns", json.dumps(asset, indent=2), "Differences are reported by asset and do not imply that one asset is preferable.", "",
        "## C. Horizon Patterns", json.dumps(horizon, indent=2), "Horizon groups remain frozen; no horizon was selected.", "",
        "## D. Model Families", json.dumps(model, indent=2), "Model-family means are descriptive and include no hyperparameter search.", "",
        "## E. Core vs Context", json.dumps(context, indent=2), "Deltas are paired on identical asset/timeframe/horizon/model/split keys. Context remains proxy-only.", "",
        "## F. Holdout Degradation", "",
        f"Holdout rows analyzed: {len(holdout)}. The holdout was never used in temporal categorization, selection, or optimization. Candidate-level validation-to-holdout ROC-AUC diagnostics are in `phase4_deep_diagnostic.csv`; aggregate summary: {candidate_holdout['delta_auc'].describe().round(6).to_dict()}.", "",
        "## G. Weak Candidate", "",
        f"The Phase 4.1 WEAK candidate count is {weak_dossier['count']}. Its exact validation and holdout metrics are included in the machine-readable JSON under `weak_candidate`; no new decision rule was applied to it.",
        json.dumps(weak_dossier, indent=2, default=str), "",
        "## H. Probability / Calibration Data Gap", "",
        "Stored artifacts contain aggregate probabilities-derived metrics (ROC-AUC, PR-AUC, Brier, Log Loss) and threshold coverage/returns, but no per-prediction probability, timestamp, realized label, or confidence-bin counts. A future instrumented run would need timestamp, asset, timeframe, horizon, model_id, feature_set_id, probability, prediction, realized label, and realized future return per prediction. This phase did not modify the pipeline.", "",
        "## I. GPU", "",
        "Phase 4.2 used no GPU and performed no training. A later GPU phase could be justified only by a separately pre-registered question about a model family that cannot be evaluated by the frozen sklearn baselines; this diagnosis alone does not justify TCN or GPU boosting. CPU-only deployment remains required.", "",
        "## J. Next Research Questions", "",
        "1. Can per-prediction probability telemetry reproduce the observed Brier/Log-Loss behavior and resolve calibration-bin uncertainty in a separately frozen experiment?",
        "2. Do the observed temporal categories persist under a newly pre-registered chronological sample, without changing the current Phase 4 design or using the current holdout for selection?",
        "3. Does a separately specified proxy-information test explain the unstable Core-vs-Context deltas without promoting the proxy?", "",
        "## Methodological Limits", "",
        "- The final holdout is a single out-of-sample period and remains frozen.",
        "- Aggregate artifacts cannot support per-prediction confidence histograms.",
        "- These results describe information value, not a trading strategy.",
        "- No best-model, best-asset, best-horizon, or promotion statement is made.", "",
        "## Reproducibility", json.dumps({"input_hashes": input_hashes, "plots": plot_paths, "checks_passed": all(item["ok"] for item in checks.values()), "seed": SEED}, indent=2),
    ]
    DIAGNOSTIC_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic() -> dict[str, Any]:
    results, _, result_json = load_artifacts()
    checks = validate_artifacts(results, json.loads(MANIFEST_PATH.read_text(encoding="utf-8")), result_json)
    analysis = pd.read_csv(ANALYSIS_CSV)
    if len(analysis) != 144:
        raise ValueError("Phase 4.1 analysis must contain 144 combinations")
    diagnostic, dossiers, weak_dossier = _candidate_rows(results, analysis)
    candidate_rows = analysis[analysis["classification"] == "ROBUST CANDIDATE"]
    pairs = pd.concat([_paired_context(results, candidate_rows.iloc[index]) for index in range(len(dossiers))], ignore_index=True)
    temporal_counts = pd.Series([d["temporal_category"] for d in dossiers]).value_counts().to_dict()
    input_paths = [RESULTS_CSV, RESULTS_JSON, MANIFEST_PATH, ANALYSIS_CSV, ANALYSIS_JSON, ANALYSIS_MANIFEST, PHASE41_REPORT, PHASE3_CONFIG, PHASE3_MANIFEST]
    input_hashes = {str(path.relative_to(ROOT)): _sha256(path) for path in input_paths if path.exists()}
    plot_paths = write_plots(results, diagnostic, pairs)
    diagnostic.to_csv(DIAGNOSTIC_CSV, index=False)
    payload = {
        "purpose": "read_only_phase4_2_deep_diagnostic",
        "candidate_count": len(dossiers),
        "weak_candidate": weak_dossier,
        "candidate_keys": [{key: dossier[key] for key in KEYS} for dossier in dossiers],
        "temporal_category_counts": temporal_counts,
        "diagnostic_rows": len(diagnostic),
        "context_pair_rows": len(pairs),
        "holdout_rows": int((diagnostic["split_type"] == "holdout").sum()),
        "validation_only_for_candidate_diagnosis": True,
        "holdout_used_for_selection": False,
        "training_performed": False,
        "gpu_training_performed": False,
        "artifact_checks": checks,
        "input_hashes": input_hashes,
        "plots": plot_paths,
        "dossiers": dossiers,
        "core_context_pairs": pairs.to_dict(orient="records"),
    }
    DIAGNOSTIC_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    manifest = {
        "analysis_version": "4.2.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seed": SEED,
        "phase4_version": "3.3.0",
        "phase41_source": str(PHASE41_REPORT.relative_to(ROOT)),
        "frozen_research_design": str(PHASE3_MANIFEST.relative_to(ROOT)),
        "input_files": list(input_hashes),
        "input_hashes": input_hashes,
        "candidate_count": len(dossiers),
        "walk_forward_splits": list(VALIDATION_SPLITS),
        "holdout_records": int((diagnostic["split_type"] == "holdout").sum()),
        "holdout_used_for_selection": False,
        "training_performed": False,
        "gpu_training_performed": False,
        "outputs": [str(path.relative_to(ROOT)) for path in (DIAGNOSTIC_CSV, DIAGNOSTIC_JSON, DIAGNOSTIC_REPORT, DOSSIER_REPORT)],
        "plots": plot_paths,
    }
    DIAGNOSTIC_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_dossiers(dossiers)
    _write_report(results, diagnostic, dossiers, weak_dossier, pairs, checks, plot_paths, input_hashes, temporal_counts)
    return {"diagnostic": diagnostic, "pairs": pairs, "dossiers": dossiers, "weak_dossier": weak_dossier, "checks": checks, "temporal_counts": temporal_counts, "plots": plot_paths}


if __name__ == "__main__":
    output = run_diagnostic()
    print(f"Phase 4.2 diagnostic rows: {len(output['diagnostic'])}")
    print(f"Candidates: {len(output['dossiers'])}")
    print(json.dumps(output["temporal_counts"], sort_keys=True))
