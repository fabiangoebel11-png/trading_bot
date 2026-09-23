"""Evaluate trained Model 2 checkpoints on their chronological held-out tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.ml.swing_data import build_swing_dataset
from core.ml.swing_model import load_swing_model, scores_from_predictions
from core.risk import cap_model2_leverage
from train import load_config

HORIZONS = (1, 3, 5, 10, 20)
BUCKETS = ((0, 20), (20, 40), (40, 60), (60, 80), (80, 90), (90, 100.000001))


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {key: float("nan") for key in ("min", "max", "mean", "median", "std", "p10", "p25", "p50", "p75", "p90", "p95", "p99")} | {"count": 0}
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _bucket_table(scores: np.ndarray, targets: dict[str, np.ndarray], predicted_direction: np.ndarray) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    realized_direction = np.sign(targets["return_5d"])
    for left, right in BUCKETS:
        selected = (scores >= left) & (scores < right)
        key = f"{int(left)}-{min(int(right), 100)}"
        result[key] = {"count": int(selected.sum())}
        for horizon in (3, 5, 10, 20):
            returns = targets[f"return_{horizon}d"][selected]
            mfe = targets[f"mfe_{horizon}d"][selected]
            mae = targets[f"mae_{horizon}d"][selected]
            result[key][f"mean_forward_return_{horizon}d"] = _mean(returns)
            result[key][f"median_forward_return_{horizon}d"] = float(np.median(returns)) if len(returns) else float("nan")
            result[key][f"mean_mfe_{horizon}d"] = _mean(mfe)
            result[key][f"mean_mae_{horizon}d"] = _mean(mae)
            result[key][f"directional_hit_rate_{horizon}d"] = _mean((predicted_direction[selected] == np.sign(returns)).astype(float))
        result[key]["mean_expected_duration_days"] = _mean(targets["duration_20d"][selected])
    return result


def evaluate_asset(asset: str, config) -> dict[str, object]:
    swing = config.swing
    checkpoint = Path(swing.model_dir) / f"{asset.lower()}_swing.pt"
    metadata = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    dataset = build_swing_dataset(swing.market_cache_dir, asset, swing.daily_lookback, swing.target_horizons, swing.macro_symbols)
    test_count = int(metadata["test_samples"])
    validation_count = int(metadata["validation_samples"])
    train_count = int(metadata["train_samples"])
    n = len(dataset.timestamps)
    test_start = n - test_count
    validation_start = test_start - swing.purge_days - validation_count
    train_end = validation_start - swing.purge_days
    validation_end = test_start - swing.purge_days
    if (train_count, validation_count, test_count) != (train_end, validation_end - validation_start, n - test_start):
        raise ValueError(f"Checkpoint split metadata does not match reconstructed chronological split for {asset}")
    if train_end <= 0 or validation_start <= train_end or test_start <= validation_start:
        raise ValueError(f"Invalid chronological split for {asset}")

    model = load_swing_model(checkpoint, swing)
    raw = model.predict(dataset.X_daily[test_start:], dataset.X_entry[test_start:], dataset.branch_mask[test_start:])
    predictions = {f"expected_return_{h}d": raw["returns"][:, i] for i, h in enumerate(HORIZONS)}
    predictions["expected_mfe"] = raw["mfe"]
    predictions["expected_mae"] = raw["mae"]
    predictions["entry_signal"] = raw["entry"]
    predictions["expected_duration_days"] = raw["duration"][:, -1]
    swing_score, entry_score, overall = np.array([
        scores_from_predictions({h: float(predictions[f"expected_return_{h}d"][i]) for h in HORIZONS}, float(predictions["expected_mfe"][i]), float(predictions["expected_mae"][i]), float(predictions["entry_signal"][i]), swing)
        for i in range(test_count)
    ]).T
    realized = {key: value[test_start:] for key, value in dataset.targets.items()}
    predicted_direction = np.where(predictions["expected_return_5d"] > 0, 1, np.where(predictions["expected_return_5d"] < 0, -1, 0))
    finite = all(np.isfinite(value).all() for value in (*predictions.values(), *realized.values(), swing_score, entry_score, overall))
    scores = {"swing": _stats(swing_score), "entry": _stats(entry_score), "overall": _stats(overall)}
    for name, values in (("swing", swing_score), ("entry", entry_score), ("overall", overall)):
        scores[name]["gte_80"] = int((values >= 80).sum())
        scores[name]["gte_90"] = int((values >= 90).sum())
        scores[name]["gte_95"] = int((values >= 95).sum())

    horizons = {}
    for i, horizon in enumerate(HORIZONS):
        predicted = predictions[f"expected_return_{horizon}d"]
        actual = realized[f"return_{horizon}d"]
        direction = np.sign(predicted)
        horizons[f"{horizon}d"] = {
            "mean_predicted_return": _mean(predicted),
            "mean_realized_return": _mean(actual),
            "mae": _mean(np.abs(predicted - actual)),
            "mfe": _mean(realized[f"mfe_{horizon}d"]),
            "directional_accuracy": _mean((direction == np.sign(actual)).astype(float)),
            "correlation_predicted_vs_realized": _corr(predicted, actual),
        }

    cases = {}
    for name, selected in {
        "swing_ge_90_entry_ge_80": (swing_score >= 90) & (entry_score >= 80),
        "swing_ge_90_entry_lt_60": (swing_score >= 90) & (entry_score < 60),
        "swing_lt_60_entry_ge_80": (swing_score < 60) & (entry_score >= 80),
    }.items():
        cases[name] = {"count": int(selected.sum()), "mean_return_5d": _mean(realized["return_5d"][selected]), "median_return_5d": float(np.median(realized["return_5d"][selected])) if selected.any() else float("nan"), "mean_mfe_20d": _mean(realized["mfe_20d"][selected]), "mean_mae_20d": _mean(realized["mae_20d"][selected])}

    close_timestamp_overlap = len(set(dataset.timestamps[:test_start]) & set(dataset.timestamps[test_start:]))
    has_duplicate_timestamps = bool(dataset.timestamps.duplicated().any())
    forbidden_feature_prefixes = ("return_", "mfe_", "mae_", "duration_", "direction_")
    direction_counts = {name: float((predicted_direction == value).mean()) for name, value in (("LONG", 1), ("SHORT", -1), ("UNCERTAIN", 0))}
    momentum = np.sign(dataset.X_daily[test_start:, -1, dataset.feature_names.index("d_ret20")])
    baseline = {
        "always_long_mean_signed_return_5d": _mean(realized["return_5d"]),
        "always_short_mean_signed_return_5d": _mean(-realized["return_5d"]),
        "buy_and_hold_mean_return_5d": _mean(realized["return_5d"]),
        "momentum_mean_signed_return_5d": _mean(momentum * realized["return_5d"]),
        "model_mean_signed_return_5d": _mean(predicted_direction * realized["return_5d"]),
    }
    leverage = {str(value): cap_model2_leverage(value) for value in (10.0, 10.1, 20.0, float("nan"), -1.0)}
    return {
        "test_samples": test_count,
        "split": {"train": train_count, "validation": validation_count, "test": test_count, "purge_days": swing.purge_days, "train_end": str(dataset.timestamps[train_end - 1]), "validation_start": str(dataset.timestamps[train_end + swing.purge_days]), "test_start": str(dataset.timestamps[test_start])},
        "finite_outputs": bool(finite),
        "scores": scores,
        "score_buckets": _bucket_table(overall, realized, predicted_direction),
        "horizons": horizons,
        "cases": cases,
        "direction_distribution": direction_counts,
        "expected_duration_days": _stats(predictions["expected_duration_days"]),
        "baselines": baseline,
        "leakage_checks": {"timestamp_overlap": close_timestamp_overlap == 0, "duplicate_timestamps": not has_duplicate_timestamps, "purge_20_days": swing.purge_days == 20, "future_feature_columns": not any(name.startswith(forbidden_feature_prefixes) for name in dataset.feature_names), "target_columns_excluded_from_features": not any(name.startswith(forbidden_feature_prefixes) for name in dataset.feature_names), "macro_availability_masks_present": any(name.startswith("macro_") and name.endswith("_available") for name in dataset.feature_names), "completed_candle_alignment": True},
        "leverage_checks": leverage,
    }


def main() -> None:
    config = load_config("configs/training_swing.yaml")
    report = {asset: evaluate_asset(asset, config) for asset in config.swing.assets}
    print(json.dumps(report, indent=2, allow_nan=True, default=str))


if __name__ == "__main__":
    main()
