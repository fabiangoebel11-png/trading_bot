"""Controlled Phase 3.2 proxy-feature ablation evaluation.

This is a research diagnostic only. Feature groups are compared under fixed
chronological splits; the holdout is reported after the validation protocol and
is never used to select or promote a feature group.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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

from core.ml.phase3_pipeline import build_dataset

SEED = 42
TRAIN_END = pd.Timestamp("2021-12-31 23:59:59", tz="UTC")
VALIDATION_START = pd.Timestamp("2022-01-01", tz="UTC")
VALIDATION_END = pd.Timestamp("2023-12-31 23:59:59", tz="UTC")
HOLDOUT_START = pd.Timestamp("2024-01-01", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-09-01", tz="UTC")
EMBARGO_MULTIPLIER = 1

DATASETS = (
    ("BTC/USDT", "1h", "4h"),
    ("BTC/USDT", "1h", "24h"),
    ("ETH/USDT", "1h", "7d"),
    ("ETH/USDT", "4h", "14d"),
)
MODEL_NAMES = ("logistic", "hist_gradient_boosting")


@dataclass(frozen=True)
class SplitSpec:
    train_end: str
    validation_start: str
    validation_end: str
    holdout_start: str
    holdout_end: str
    purge_hours: float
    embargo_hours: float


@dataclass(frozen=True)
class EvaluationRecord:
    experiment_id: str
    dataset_id: str
    feature_set_id: str
    target: str
    timeframe: str
    horizon: str
    model_id: str
    split: str
    feature_group: str
    feature_columns: tuple[str, ...]
    train_rows: int
    validation_rows: int
    holdout_rows: int
    metrics: dict[str, float]
    preprocessing: str
    purge_embargo: str
    holdout_used_for_selection: bool
    status: str


def proxy_feature_groups(columns: list[str] | tuple[str, ...]) -> dict[str, list[str]]:
    columns = list(columns)
    base = [column for column in columns if not column.startswith("macro_")]
    groups = {
        "crypto_only": base,
        "crypto_plus_all_proxy": columns,
    }
    for name, prefix in (
        ("crypto_plus_es", "macro_es_f_proxy_"),
        ("crypto_plus_nq", "macro_nq_f_proxy_"),
        ("crypto_plus_vix", "macro_vix_proxy_"),
        ("crypto_plus_urth", "macro_urth_proxy_"),
    ):
        groups[name] = base + [column for column in columns if column.startswith(prefix)]
    return groups


def split_masks(index: pd.DatetimeIndex, horizon: pd.Timedelta) -> tuple[np.ndarray, np.ndarray, np.ndarray, SplitSpec]:
    index = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    purge = horizon
    embargo = horizon * EMBARGO_MULTIPLIER
    train_cutoff = TRAIN_END - purge - embargo
    validation_start = VALIDATION_START
    validation_end = VALIDATION_END
    train = index <= train_cutoff
    validation = (index >= validation_start) & (index <= validation_end)
    holdout = (index >= HOLDOUT_START) & (index <= HOLDOUT_END)
    spec = SplitSpec(
        TRAIN_END.isoformat(), VALIDATION_START.isoformat(), VALIDATION_END.isoformat(),
        HOLDOUT_START.isoformat(), HOLDOUT_END.isoformat(),
        purge.total_seconds() / 3600, embargo.total_seconds() / 3600,
    )
    return train, validation, holdout, spec


def _model(name: str):
    if name == "logistic":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=100, random_state=SEED)
    raise ValueError(name)


def _metrics(y_true: pd.Series, probability: np.ndarray, future_return: pd.Series) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    signed_return = np.where(prediction == y_true.to_numpy(), future_return.to_numpy(), -future_return.to_numpy())
    equity = np.cumprod(1.0 + np.nan_to_num(signed_return, nan=0.0))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    daily_scale = np.sqrt(24.0)
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)) if y_true.nunique() > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, probability)) if y_true.nunique() > 1 else float("nan"),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "hit_rate": float(np.mean(prediction == y_true.to_numpy())),
        "mean_signal_return": float(np.mean(signed_return)),
        "signal_return_std": float(np.std(signed_return)),
        "signal_sharpe_diagnostic": float(np.mean(signed_return) / np.std(signed_return) * daily_scale) if np.std(signed_return) else 0.0,
        "max_drawdown": float(drawdown.min()),
        "signal_count": float(len(prediction)),
        "signal_turnover": float(np.mean(np.abs(np.diff(prediction)))) if len(prediction) > 1 else 0.0,
    }


def _fit_predict_pair(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    X_holdout: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_holdout_scaled = scaler.transform(X_holdout)
    model = _model(model_name)
    model.fit(X_train_scaled, y_train)
    return model.predict_proba(X_validation_scaled)[:, 1], model.predict_proba(X_holdout_scaled)[:, 1]


def evaluate_dataset(target: str, timeframe: str, horizon: str, macro_path: str | Path) -> list[EvaluationRecord]:
    result = build_dataset("data", target, timeframe, horizon, macro_path=macro_path)
    frame = result.frame
    features = list(result.metadata.feature_columns)
    groups = proxy_feature_groups(features)
    horizon_delta = pd.Timedelta(hours={"1h": 1, "4h": 4, "24h": 24, "7d": 168, "14d": 336}[horizon])
    train_mask, validation_mask, holdout_mask, split = split_masks(frame.index, horizon_delta)
    records: list[EvaluationRecord] = []
    for group_name, group_columns in groups.items():
        X = frame[group_columns]
        y = frame["direction"].astype(int)
        for model_name in MODEL_NAMES:
            train_x, train_y = X.loc[train_mask], y.loc[train_mask]
            validation_x, validation_y = X.loc[validation_mask], y.loc[validation_mask]
            holdout_x, holdout_y = X.loc[holdout_mask], y.loc[holdout_mask]
            if train_y.nunique() < 2 or validation_y.nunique() < 2 or holdout_y.nunique() < 2:
                continue
            validation_probability, holdout_probability = _fit_predict_pair(model_name, train_x, train_y, validation_x, holdout_x)
            for split_name, y_eval, probabilities, returns in (
                ("validation", validation_y, validation_probability, frame.loc[validation_mask, "future_return"]),
                ("holdout", holdout_y, holdout_probability, frame.loc[holdout_mask, "future_return"]),
            ):
                experiment_seed = f"{result.metadata.dataset_id}|{group_name}|{model_name}|{split_name}|{SEED}"
                record = EvaluationRecord(
                    experiment_id=experiment_seed,
                    dataset_id=result.metadata.dataset_id,
                    feature_set_id=result.metadata.feature_set_id,
                    target=target,
                    timeframe=timeframe,
                    horizon=horizon,
                    model_id=model_name,
                    split=split_name,
                    feature_group=group_name,
                    feature_columns=tuple(group_columns),
                    train_rows=int(train_mask.sum()),
                    validation_rows=int(validation_mask.sum()),
                    holdout_rows=int(holdout_mask.sum()),
                    metrics=_metrics(y_eval, probabilities, returns),
                    preprocessing="StandardScaler fit on train only",
                    purge_embargo=json.dumps(asdict(split), sort_keys=True),
                    holdout_used_for_selection=False,
                    status="RESEARCH_DIAGNOSTIC",
                )
                records.append(record)
    return records


def classify_group(validation_records: list[EvaluationRecord], holdout_records: list[EvaluationRecord]) -> str:
    """Classify without using holdout values for selection."""
    if not validation_records or not holdout_records:
        return "NO_EVIDENCE"
    validation_delta = np.mean([record.metrics["roc_auc"] - 0.5 for record in validation_records])
    holdout_delta = np.mean([record.metrics["roc_auc"] - 0.5 for record in holdout_records])
    if validation_delta > 0.01 and holdout_delta > 0.01:
        return "POSSIBLE_SIGNAL"
    if validation_delta <= 0.01 and holdout_delta <= 0.01:
        return "NO_EVIDENCE"
    return "UNSTABLE / PERIOD_DEPENDENT"


def run_all(output_dir: str | Path = "data/research/phase3", macro_path: str | Path = "data/ml_macro_ESF_NQF_VIX_URTH_2600d.csv") -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for target, timeframe, horizon in DATASETS:
        records.extend(evaluate_dataset(target, timeframe, horizon, macro_path))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in records:
        rows.append({**asdict(record), "feature_columns": "|".join(record.feature_columns), **{f"metric_{key}": value for key, value in record.metrics.items()}})
    pd.DataFrame(rows).to_csv(destination / "proxy_feature_value.csv", index=False)
    (destination / "proxy_feature_value.json").write_text(json.dumps([asdict(record) for record in records], indent=2, default=str), encoding="utf-8")
    manifest = {
        "experiment": "phase3.2_proxy_feature_value",
        "purpose": "research_diagnostic_only",
        "seed": SEED,
        "datasets": [list(dataset) for dataset in DATASETS],
        "models": list(MODEL_NAMES),
        "feature_groups": [
            "crypto_only", "crypto_plus_all_proxy", "crypto_plus_es",
            "crypto_plus_nq", "crypto_plus_vix", "crypto_plus_urth",
        ],
        "macro_path": str(macro_path),
        "split_policy": {
            "train_end": TRAIN_END.isoformat(),
            "validation_start": VALIDATION_START.isoformat(),
            "validation_end": VALIDATION_END.isoformat(),
            "holdout_start": HOLDOUT_START.isoformat(),
            "holdout_end": HOLDOUT_END.isoformat(),
            "purge": "one target horizon",
            "embargo": "one target horizon",
        },
        "holdout_used_for_selection": False,
        "record_count": len(records),
        "outputs": ["proxy_feature_value.csv", "proxy_feature_value.json"],
    }
    (destination / "proxy_feature_value_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return records


if __name__ == "__main__":
    run_all()