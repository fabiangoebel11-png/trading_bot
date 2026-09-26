"""Phase 4 controlled baseline training and walk-forward evaluation.

This module is research-only. It trains small, fixed baseline models on the
frozen Phase 3.3 matrix and writes diagnostics below ``data/research/phase4``.
The final holdout is evaluated only after the validation protocol is complete;
no holdout value is passed to a selection or ranking function.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from core.ml.phase3_pipeline import HORIZON_DELTAS, TIMEFRAME_DELTAS, build_dataset
from core.ml.research_design import label_end_times, purged_train_mask

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "ml_research_v1.yaml"
PHASE3_MANIFEST_PATH = ROOT / "data" / "research" / "phase3" / "ml_research_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "research" / "phase4"
# Keep the manifest's repository-relative path: the audited adapter registers
# proxy sources by their canonical relative filename.
MACRO_PATH = Path("data/ml_macro_ESF_NQF_VIX_URTH_2600d.csv")
SEED = 42
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)

MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {
        "class": "LogisticRegression",
        "C": 1.0,
        "solver": "lbfgs",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": SEED,
    },
    "hist_gradient_boosting": {
        "class": "HistGradientBoostingClassifier",
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 31,
        "l2_regularization": 1.0,
        "random_state": SEED,
    },
    "random_forest": {
        "class": "RandomForestClassifier",
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "n_jobs": -1,
        "class_weight": "balanced",
        "random_state": SEED,
    },
}

FOLDS = (
    ("wf_2022", "2017-08-17T00:00:00Z", "2021-12-31T23:59:59Z", "2022-01-01T00:00:00Z", "2022-12-31T23:59:59Z"),
    ("wf_2023", "2017-08-17T00:00:00Z", "2022-12-31T23:59:59Z", "2023-01-01T00:00:00Z", "2023-12-31T23:59:59Z"),
    ("wf_2024", "2017-08-17T00:00:00Z", "2023-12-31T23:59:59Z", "2024-01-01T00:00:00Z", "2024-12-31T23:59:59Z"),
)
HOLDOUT = ("final_holdout", "2025-01-01T00:00:00Z", "2026-09-22T23:59:59Z")
OUTER_WORKERS = 1
DATASETS = tuple(
    [(target, timeframe, horizon) for target in ("BTC/USDT", "ETH/USDT") for timeframe, horizons in (("1h", ("1h", "2h", "4h", "8h", "12h", "24h", "3d", "7d")), ("4h", ("7d", "14d", "21d", "28d"))) for horizon in horizons]
)


@dataclass(frozen=True)
class SplitMasks:
    train: np.ndarray
    evaluation: np.ndarray
    train_start: str
    train_end: str
    evaluation_start: str
    evaluation_end: str
    purge: str
    embargo: str


def _memory_status() -> tuple[float, float, float]:
    """Return process RSS, total RAM, and available RAM in GB."""
    try:
        import psutil

        process = psutil.Process(os.getpid())
        memory = psutil.virtual_memory()
        return process.memory_info().rss / 2**30, memory.total / 2**30, memory.available / 2**30
    except ImportError:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("page_fault_count", wintypes.DWORD),
                            ("peak_working_set_size", ctypes.c_size_t), ("working_set_size", ctypes.c_size_t),
                            ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                            ("quota_paged_pool_usage", ctypes.c_size_t),
                            ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                            ("quota_non_paged_pool_usage", ctypes.c_size_t),
                            ("pagefile_usage", ctypes.c_size_t), ("peak_pagefile_usage", ctypes.c_size_t)]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            process_handle = ctypes.windll.kernel32.GetCurrentProcess()
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_process_memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
            get_process_memory_info.restype = wintypes.BOOL
            rss = counters.working_set_size if get_process_memory_info(process_handle, ctypes.byref(counters), counters.cb) else 0

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong),
                            ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong),
                            ("avail_extended", ctypes.c_ulonglong)]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return rss / 2**30, status.total_phys / 2**30, status.avail_phys / 2**30
    return float("nan"), float("nan"), float("nan")


def hardware_snapshot() -> dict[str, Any]:
    """Collect run metadata without making GPU or optional packages mandatory."""
    logical = os.cpu_count() or 1
    physical = logical
    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or logical
    except ImportError:
        pass
    rss, total_ram, available_ram = _memory_status()
    gpu_name = None
    gpu_vram_gb = None
    cuda_available = False
    pytorch_available = False
    try:
        import torch

        pytorch_available = True
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device = torch.cuda.get_device_properties(0)
            gpu_name = str(device.name)
            gpu_vram_gb = float(device.total_memory / 2**30)
    except ImportError:
        pass
    except Exception as error:
        gpu_name = f"detected but unavailable: {type(error).__name__}"
    if gpu_name is None:
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3, check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                name, _, vram = probe.stdout.strip().splitlines()[0].partition(",")
                gpu_name = name.strip()
                gpu_vram_gb = float(vram.strip()) / 1024 if vram.strip() else None
        except (FileNotFoundError, subprocess.SubprocessError, ValueError):
            pass
    return {
        "cpu_physical_cores": physical,
        "cpu_logical_threads": logical,
        "ram_total_gb": total_ram,
        "ram_available_at_start_gb": available_ram,
        "gpu_detected": gpu_name is not None,
        "gpu_name": gpu_name,
        "gpu_vram_gb": gpu_vram_gb,
        "pytorch_available": pytorch_available,
        "cuda_available": cuda_available,
        "gpu_used": False,
        "outer_workers": OUTER_WORKERS,
        "parallelization_used": False,
        "caching_used": True,
        "initial_process_rss_gb": rss,
    }


class _MemoryMonitor:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            rss, _, _ = _memory_status()
            if not np.isnan(rss):
                self.samples.append(rss)
            self._stop.wait(self.interval_seconds)

    def __enter__(self) -> "_MemoryMonitor":
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def peak_gb(self) -> float:
        return max(self.samples, default=float("nan"))

    @property
    def average_gb(self) -> float:
        return float(np.mean(self.samples)) if self.samples else float("nan")


def config_hash() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def _timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def build_split_masks(index: pd.DatetimeIndex, horizon: str, split_name: str) -> SplitMasks:
    """Build chronological masks with horizon-specific purge and embargo."""
    normalized = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    horizon_delta = HORIZON_DELTAS[horizon]
    if split_name == "final_holdout":
        evaluation_start, evaluation_end = HOLDOUT[1:]
        train_start, train_end = "2017-08-17T00:00:00Z", "2024-12-31T23:59:59Z"
    else:
        fold = next(item for item in FOLDS if item[0] == split_name)
        _, train_start, train_end, evaluation_start, evaluation_end = fold
    eval_start = _timestamp(evaluation_start)
    eval_end = _timestamp(evaluation_end)
    train_start_ts = _timestamp(train_start)
    train_end_ts = _timestamp(train_end)
    candidate_train = (normalized >= train_start_ts) & (normalized <= train_end_ts)
    evaluation = (normalized >= eval_start) & (normalized <= eval_end)
    label_ends = label_end_times(normalized, horizon_delta)
    purged = purged_train_mask(normalized, label_ends, eval_start, horizon_delta)
    train = candidate_train & purged
    return SplitMasks(
        train=train,
        evaluation=evaluation,
        train_start=train_start,
        train_end=train_end,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        purge=str(horizon_delta),
        embargo=str(horizon_delta),
    )


def create_model(model_id: str):
    params = dict(MODEL_PARAMS[model_id])
    params.pop("class")
    if model_id == "logistic_regression":
        return LogisticRegression(**params)
    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(**params)
    if model_id == "random_forest":
        return RandomForestClassifier(**params)
    raise ValueError(f"unknown frozen model: {model_id}")


def fit_predict_train_only(model_id: str, train_x: pd.DataFrame, train_y: pd.Series, evaluation_x: pd.DataFrame) -> np.ndarray:
    """Fit the scaler and model exclusively on one fold's training rows."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    evaluation_scaled = scaler.transform(evaluation_x)
    model = create_model(model_id)
    model.fit(train_scaled, train_y.astype(int))
    return model.predict_proba(evaluation_scaled)[:, 1]


def _safe_auc(metric: Any, y_true: pd.Series, probability: np.ndarray) -> float:
    return float(metric(y_true, probability)) if y_true.nunique() > 1 else float("nan")


def calculate_metrics(y_true: pd.Series, probability: np.ndarray, future_return: pd.Series) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    metrics: dict[str, float] = {
        "roc_auc": _safe_auc(roc_auc_score, y_true, probability),
        "pr_auc": _safe_auc(average_precision_score, y_true, probability),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "mean_forward_return": float(future_return.mean()),
        "sample_count": float(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }
    for threshold in THRESHOLDS:
        suffix = str(threshold).replace(".", "_")
        up = probability >= threshold
        down = probability <= 1.0 - threshold
        metrics[f"signal_coverage_up_{suffix}"] = float(up.mean())
        metrics[f"signal_coverage_down_{suffix}"] = float(down.mean())
        metrics[f"mean_forward_return_up_{suffix}"] = float(future_return[up].mean()) if up.any() else float("nan")
        metrics[f"mean_forward_return_down_{suffix}"] = float(-future_return[down].mean()) if down.any() else float("nan")
    return metrics


def validation_status(records: pd.DataFrame) -> str:
    """Classify a fixed configuration using validation only, without ranking."""
    validation = records[records["split_type"] == "validation"]
    if len(validation) < 3:
        return "FAILED"
    auc = validation["metric_roc_auc"].dropna()
    if len(auc) < 3:
        return "FAILED"
    good_windows = int((auc >= 0.5).sum())
    if good_windows >= 2 and float(auc.mean()) >= 0.505:
        return "ROBUST CANDIDATE"
    if good_windows >= 1:
        return "MIXED"
    return "WEAK"


def _record(
    *, target: str, timeframe: str, horizon: str, feature_set: str, model_id: str,
    split_name: str, split_type: str, dataset_id: str, feature_columns: tuple[str, ...],
    masks: SplitMasks, metrics: dict[str, float], train_rows: int, eval_rows: int,
) -> dict[str, Any]:
    raw_id = "|".join(("phase4", target, timeframe, horizon, feature_set, model_id, split_name, str(SEED)))
    run_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:20]
    return {
        "run_id": run_id,
        "research_version": "3.3.0",
        "config_hash": config_hash(),
        "dataset_id": dataset_id,
        "feature_set_id": feature_set,
        "model_id": model_id,
        "split_id": split_name,
        "split_type": split_type,
        "target": target,
        "timeframe": timeframe,
        "horizon": horizon,
        "random_seed": SEED,
        "preprocessing_version": "train_only_standard_scaler_v1",
        "feature_columns": "|".join(feature_columns),
        "label_definition": "direction_raw_v1",
        "purge_duration": masks.purge,
        "embargo_duration": masks.embargo,
        "train_start": masks.train_start,
        "train_end": masks.train_end,
        "evaluation_start": masks.evaluation_start,
        "evaluation_end": masks.evaluation_end,
        "train_rows": train_rows,
        "evaluation_rows": eval_rows,
        "holdout_used_for_selection": False,
        **{f"metric_{key}": value for key, value in metrics.items()},
    }


def evaluate_task(
    target: str,
    timeframe: str,
    horizon: str,
    feature_set: str,
    split_names: tuple[str, ...],
    dataset_cache: dict[tuple[str, str, str, str], Any] | None = None,
) -> list[dict[str, Any]]:
    macro_path = MACRO_PATH if feature_set == "crypto_core_v1+crypto_context_proxy_v1" else None
    cache = dataset_cache if dataset_cache is not None else {}
    cache_key = (target, timeframe, horizon, feature_set)
    if cache_key not in cache:
        cache[cache_key] = build_dataset(
            ROOT / "data", target, timeframe, horizon,
            context_assets=(),
            feature_set_id=feature_set,
            macro_path=macro_path,
        )
    result = cache[cache_key]
    frame = result.frame
    feature_columns = tuple(result.metadata.feature_columns)
    feature_frame = frame.loc[:, list(feature_columns)]
    output: list[dict[str, Any]] = []
    for split_name in split_names:
        masks = build_split_masks(frame.index, horizon, split_name)
        train = frame.loc[masks.train]
        evaluation = frame.loc[masks.evaluation]
        if len(train) == 0 or len(evaluation) == 0 or train["direction"].nunique() < 2 or evaluation["direction"].nunique() < 2:
            continue
        train_x = feature_frame.loc[masks.train]
        train_y = train["direction"]
        evaluation_x = feature_frame.loc[masks.evaluation]
        evaluation_y = evaluation["direction"]
        for model_id in MODEL_PARAMS:
            probability = fit_predict_train_only(model_id, train_x, train_y, evaluation_x)
            metrics = calculate_metrics(evaluation_y, probability, evaluation["future_return"])
            output.append(_record(
                target=target, timeframe=timeframe, horizon=horizon,
                feature_set=feature_set, model_id=model_id,
                split_name=split_name,
                split_type="holdout" if split_name == "final_holdout" else "validation",
                dataset_id=result.metadata.dataset_id,
                feature_columns=feature_columns, masks=masks, metrics=metrics,
                train_rows=len(train), eval_rows=len(evaluation),
            ))
    return output


def _write_report(records: pd.DataFrame, path: Path) -> None:
    validation = records[records["split_type"] == "validation"]
    holdout = records[records["split_type"] == "holdout"]
    summary = validation.groupby(["feature_set_id", "model_id"], as_index=False)[["metric_roc_auc", "metric_pr_auc", "metric_log_loss", "metric_brier_score", "metric_balanced_accuracy"]].mean()
    try:
        rendered_summary = summary.to_markdown(index=False) if not summary.empty else "No validation records."
    except ImportError:
        rendered_summary = summary.to_string(index=False) if not summary.empty else "No validation records."
    lines = [
        "# Phase 4 Controlled ML Baseline Evaluation",
        "",
        "Research-only diagnostic. No runtime, execution, strategy, risk, leverage, paper, or live code was changed.",
        "",
        "## Protocol",
        "",
        "- 24 frozen BTC/ETH tasks; 2 feature sets; 3 fixed baseline models.",
        "- WF validation: 2022, 2023, and 2024 with expanding training windows.",
        "- Purge and embargo: one complete target horizon for every task.",
        "- StandardScaler is fitted separately on each training fold only.",
        "- Thresholds 0.50/0.55/0.60/0.65/0.70 are descriptive only.",
        "- Final holdout 2025-01-01 through 2026-09-22 is evaluation-only and never enters selection.",
        "",
        "## Validation Summary",
        "",
        rendered_summary,
        "",
        "## Core vs Context",
        "",
        "The comparison is reported by feature set and walk-forward period. No overall best-model ranking or promotion decision is produced.",
        "",
        "## Holdout",
        "",
        f"Holdout records: {len(holdout)}. These values are stored separately as untouched out-of-sample diagnostics and were not used for feature/model/threshold selection.",
        "",
        "## Interpretation",
        "",
        "Results describe directional information value and temporal generalization only. They do not establish a trading strategy and do not authorize promotion.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    started = time.perf_counter()
    hardware = hardware_snapshot()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    dataset_cache: dict[tuple[str, str, str, str], Any] = {}
    memory_monitor = _MemoryMonitor()
    memory_monitor.__enter__()
    records: list[dict[str, Any]] = []
    validation_splits = tuple(item[0] for item in FOLDS)
    for target, timeframe, horizon in DATASETS:
        for feature_set in ("crypto_core_v1", "crypto_core_v1+crypto_context_proxy_v1"):
            records.extend(evaluate_task(target, timeframe, horizon, feature_set, validation_splits, dataset_cache))
    validation_frame = pd.DataFrame(records)
    if validation_frame.empty:
        raise RuntimeError("Phase 4 produced no validation records")
    statuses = validation_frame.groupby(["target", "timeframe", "horizon", "feature_set_id", "model_id"], group_keys=False).apply(validation_status, include_groups=False).rename("validation_status").reset_index()
    validation_frame = validation_frame.merge(statuses, on=["target", "timeframe", "horizon", "feature_set_id", "model_id"], how="left")
    # Holdout is deliberately executed only after validation records/statuses exist.
    holdout_records: list[dict[str, Any]] = []
    for target, timeframe, horizon in DATASETS:
        for feature_set in ("crypto_core_v1", "crypto_core_v1+crypto_context_proxy_v1"):
            holdout_records.extend(evaluate_task(target, timeframe, horizon, feature_set, ("final_holdout",), dataset_cache))
    holdout_frame = pd.DataFrame(holdout_records)
    if not holdout_frame.empty:
        holdout_frame = holdout_frame.merge(statuses, on=["target", "timeframe", "horizon", "feature_set_id", "model_id"], how="left")
    frame = pd.concat([validation_frame, holdout_frame], ignore_index=True)
    memory_monitor.__exit__(None, None, None)
    frame.to_csv(destination / "phase4_ml_baseline_results.csv", index=False)
    frame.to_json(destination / "phase4_ml_baseline_results.json", orient="records", indent=2)
    summary = frame.groupby(["feature_set_id", "model_id", "split_type"], as_index=False)[["metric_roc_auc", "metric_pr_auc", "metric_log_loss", "metric_brier_score", "metric_balanced_accuracy", "metric_accuracy", "metric_mean_forward_return"]].mean()
    summary.to_csv(destination / "phase4_model_summary.csv", index=False)
    walk_forward = frame[frame["split_type"] == "validation"].groupby(["target", "timeframe", "horizon", "feature_set_id", "model_id", "split_id"], as_index=False)[["metric_roc_auc", "metric_pr_auc", "metric_log_loss", "metric_brier_score", "metric_balanced_accuracy", "metric_accuracy", "metric_mean_forward_return"]].mean()
    walk_forward.to_csv(destination / "phase4_walk_forward_summary.csv", index=False)
    manifest = {
        "research_version": "3.3.0",
        "phase": "4",
        "purpose": "controlled_ml_information_value_and_generalization",
        "config_path": str(CONFIG_PATH.relative_to(ROOT)),
        "config_hash": config_hash(),
        "phase3_manifest": str(PHASE3_MANIFEST_PATH.relative_to(ROOT)),
        "random_seed": SEED,
        "tasks": len(DATASETS),
        "feature_sets": ["crypto_core_v1", "crypto_core_v1+crypto_context_proxy_v1"],
        "models": MODEL_PARAMS,
        "validation_splits": [item[0] for item in FOLDS],
        "holdout_split": HOLDOUT[0],
        "thresholds_descriptive_only": list(THRESHOLDS),
        "purge_rule": "one complete target horizon",
        "embargo_rule": "one complete target horizon",
        "preprocessing": "StandardScaler fit separately on each train fold only",
        "holdout_policy": {"selection_allowed": False, "training_allowed": False, "reporting_allowed": True},
        "record_count": len(frame),
        "validation_record_count": len(validation_frame),
        "holdout_record_count": len(holdout_frame),
        "selection_status_source": "validation_only",
        "hardware": {
            **hardware,
            "peak_process_rss_gb": memory_monitor.peak_gb,
            "average_process_rss_gb": memory_monitor.average_gb,
            "ram_available_at_end_gb": _memory_status()[2],
            "run_duration_seconds": time.perf_counter() - started,
            "training_time_before_seconds": None,
            "training_time_after_seconds": time.perf_counter() - started,
            "training_time_comparison_available": False,
            "model_parallelism": "RandomForest uses its configured internal threads; no outer experiment parallelism",
        },
        "outputs": [
            "phase4_ml_baseline_results.csv", "phase4_ml_baseline_results.json",
            "phase4_ml_baseline_manifest.json", "phase4_ml_baseline_report.md",
            "phase4_model_summary.csv", "phase4_walk_forward_summary.csv",
        ],
    }
    (destination / "phase4_ml_baseline_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    _write_report(frame, destination / "phase4_ml_baseline_report.md")
    return frame


if __name__ == "__main__":
    result = run_all()
    print(f"Phase 4 records: {len(result)}")