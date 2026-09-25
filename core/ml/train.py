"""End-to-end training pipeline for the trend-continuation confirmation model.

Orchestrates, per symbol:
  1. Load raw (unsmoothed) intraday OHLCV + cross-asset macro context.
  2. Build causal features and triple-barrier labels.
  3. Purged/embargoed walk-forward cross-validation of the GPU TCN classifier,
     to get an honest out-of-sample read on whether it has learned anything
     beyond noise (see ``core/ml/dataset.py`` for why plain time-split CV is
     not safe here).
  4. Fit a final model on the full history (still with a held-out tail slice
     for early stopping) and persist it under ``TrendMLConfig.model_dir`` for
     later use by ``core/ml/inference.py``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import DataConfig, TradingBotConfig, TrendMLConfig
from core.data_loader import fetch_ohlcv_history
from core.ml.dataset import IndexedSequenceView, build_row_level_dataset, make_sequences, purged_walk_forward_splits, time_decay_sample_weights
from core.ml.date_validation import chronological_purged_split, horizon_description, validate_symbol_coverage
from core.ml.features import build_feature_matrix, fetch_breadth_basket
from core.ml.providers import AssetSpec, asset_specs_from_config, canonical_cache_path
from core.ml.data import load_ohlcv_parquet
from core.ml.labeling import build_horizon_label_map, qualified_trade_labels
from core.ml.scoring import continuous_opportunity_score, score_bucket_metrics, score_trade_quality, threshold_metrics, validate_model1_prediction_outputs
from core.ml.tcn_model import TCNTrendModel
from core.strategy import compute_atr


def resolve_forecast_horizons(config: TrendMLConfig) -> tuple[int, ...]:
    """Normalize the configured forecast horizon list without breaking older single-horizon configs."""
    raw = getattr(config, "forecast_horizons", None)
    if raw:
        return tuple(int(h) for h in raw)
    return (int(config.label_horizon),)


def load_ml_ohlc(data_config: DataConfig, ml_config: TrendMLConfig) -> dict[str, pd.DataFrame]:
    """Like ``core.data_loader.load_multi_asset_data``, but at the ML pipeline's
    own timeframe/history length (now the same 1h as the rule-based strategy,
    see ``TrendMLConfig`` docstring), and keeping the ``volume`` column that
    the equal-weight portfolio loader drops."""
    if ml_config.base_timeframe not in ml_config.timeframes:
        raise ValueError(f"ML training target must be one of configured timeframes, got {ml_config.base_timeframe!r}")
    specs = [
        spec for spec in asset_specs_from_config(TradingBotConfig(data=data_config, ml=ml_config))
        if spec.timeframe == ml_config.base_timeframe and spec.name in ml_config.assets
    ]
    raw: dict[str, pd.DataFrame] = {}
    for spec in specs:
        path = canonical_cache_path(ml_config.market_cache_dir, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared ML dataset for {spec.name}: {path}")
        raw[spec.name] = load_ohlcv_parquet(path)
    if set(raw) != set(ml_config.assets):
        raise ValueError(f"Prepared ML assets do not match config: expected {ml_config.assets}, got {sorted(raw)}")
    return raw


def load_prepared_macro_matrix(config: TrendMLConfig) -> pd.DataFrame:
    """Load configured prepared daily context closes without a network fallback."""
    columns: dict[str, pd.Series] = {}
    cache_root = Path(config.macro_cache_dir)
    combined_candidates = sorted(set(cache_root.glob("ml_macro_*_2600d.csv")) | set(cache_root.parent.glob("ml_macro_*_2600d.csv")))
    combined = None
    if combined_candidates:
        combined = pd.read_csv(combined_candidates[-1], index_col=0, parse_dates=True)
    for name, definition in config.context_assets.items():
        source = str(definition.get("symbol", name))
        if source not in config.macro_symbols:
            continue
        provider = str(definition.get("provider", "yfinance"))
        spec = AssetSpec(name, source, provider, "1d", config.macro_lookback_days, source=source)
        path = canonical_cache_path(config.market_cache_dir, spec)
        if path.exists():
            series = load_ohlcv_parquet(path)["close"].copy()
            if not series.empty:
                series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
            columns[source] = series
        elif combined is not None and source in combined.columns:
            series = pd.to_numeric(combined[source], errors="coerce").copy()
            if not series.empty:
                series.index = pd.DatetimeIndex(pd.to_datetime(series.index, utc=True))
            columns[source] = series
    if not columns:
        raise FileNotFoundError("No prepared macro/context Parquet datasets matched TrendMLConfig.macro_symbols")
    frame = pd.DataFrame(columns).sort_index()
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    return frame


def _prepare_symbol_dataset(
    ohlc: pd.DataFrame,
    macro_prices: pd.DataFrame | None,
    config: TrendMLConfig,
    breadth_return: pd.Series | None = None,
    forecast_horizons: tuple[int, ...] | None = None,
):
    features = build_feature_matrix(ohlc, macro_prices, config, breadth_return)
    atr = compute_atr(ohlc["high"], ohlc["low"], ohlc["close"], config.atr_window)
    horizons = resolve_forecast_horizons(config) if forecast_horizons is None else tuple(int(h) for h in forecast_horizons)
    label_map = build_horizon_label_map(
        ohlc["close"],
        atr,
        horizons,
        config.barrier_atr_multiple,
        high=ohlc["high"],
        low=ohlc["low"],
        stop_atr_multiple=config.stop_atr_multiple,
        take_profit_atr_multiple=config.take_profit_atr_multiple,
    )
    primary_horizon = horizons[0]
    labels = label_map[primary_horizon]
    X, y, t1_pos = build_row_level_dataset(features, labels)
    quality = labels.loc[X.index]
    if forecast_horizons is None:
        return X, y, t1_pos, quality["mfe"].to_numpy(dtype=np.float32), quality["mae"].to_numpy(dtype=np.float32)
    return X, y, t1_pos, quality["mfe"].to_numpy(dtype=np.float32), quality["mae"].to_numpy(dtype=np.float32), label_map


def build_targets_by_horizon(
    X: pd.DataFrame,
    horizon_labels: dict[int, pd.DataFrame],
    horizons: tuple[int, ...],
) -> dict[int, dict[str, np.ndarray]]:
    """Convert the row-level label map to the per-horizon target structure the
    TCN uses during multi-head training."""
    targets_by_horizon: dict[int, dict[str, np.ndarray]] = {}
    for horizon in horizons:
        frame = horizon_labels[horizon].reindex(X.index)
        direction = frame["label"].fillna(0.0).to_numpy(dtype=np.float32)
        mfe = frame["mfe"].fillna(0.0).to_numpy(dtype=np.float32)
        mae = frame["mae"].fillna(0.0).to_numpy(dtype=np.float32)
        favorable = np.maximum(mfe, 0.0)
        adverse = np.maximum(-mae, 0.0)
        opportunity = np.clip(favorable / np.maximum(favorable + adverse, 1e-6), 0.0, 1.0).astype(np.float32)
        return_target = (np.sign(direction) * (favorable - adverse)).astype(np.float32)
        duration = frame["time_to_barrier"].fillna(float(horizon)).to_numpy(dtype=np.float32)
        duration = np.clip(duration / max(int(horizon), 1), 0.0, 1.0).astype(np.float32)
        targets_by_horizon[int(horizon)] = {
            "direction": direction,
            "mfe": mfe,
            "mae": mae,
            "opportunity": opportunity,
            "return": return_target,
            "duration": duration,
        }
    return targets_by_horizon


def _to_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    t1_pos: np.ndarray,
    seq_len: int,
    mfe: np.ndarray | None = None,
    mae: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    X_arr = X.to_numpy(dtype=np.float32)
    n_rows = X_arr.shape[0]
    n_seq = n_rows - seq_len + 1
    if n_seq <= 0:
        raise ValueError(f"Not enough rows ({n_rows}) for sequence_length={seq_len}.")
    X_seq = make_sequences(X_arr, seq_len)
    y_seq = y.to_numpy()[seq_len - 1 :]
    mfe_seq = np.zeros(n_rows, dtype=np.float32) if mfe is None else mfe
    mae_seq = np.zeros(n_rows, dtype=np.float32) if mae is None else mae
    # t1_pos is row-level; shift into sequence-index space (sequence i ends at row i+seq_len-1).
    t1_seq = np.clip(t1_pos[seq_len - 1 :] - (seq_len - 1), 0, n_seq - 1)
    index_seq = X.index[seq_len - 1 :]
    return X_seq, y_seq, t1_seq, index_seq, mfe_seq[seq_len - 1 :], mae_seq[seq_len - 1 :]


def evaluate_predictions(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """Metrics on purged out-of-sample test folds only. ``directional_edge`` is
    a proxy Sharpe-like ratio (mean/std of confidence * realized label
    direction) -- not a real backtested return, just a quick signal-quality
    check before any of this touches ``core/strategy.py`` position sizing."""
    pred_class = proba.argmax(axis=1) - 1
    non_flat = y_true != 0
    if non_flat.sum() > 0:
        hits = (pred_class[non_flat] == y_true[non_flat]) & (pred_class[non_flat] != 0)
        hit_rate = float(hits.mean())
    else:
        hit_rate = float("nan")

    signal = proba[:, 2] - proba[:, 0]
    proxy_returns = signal * y_true
    edge_std = float(proxy_returns.std())
    directional_edge = float(proxy_returns.mean()) / edge_std if edge_std > 0 else 0.0

    accuracy = float((pred_class == y_true).mean())
    class_values = (-1, 0, 1)
    true_counts = {str(label): int((y_true == label).sum()) for label in class_values}
    predicted_counts = {str(label): int((pred_class == label).sum()) for label in class_values}
    true_fraction = {str(label): float((y_true == label).mean()) for label in class_values}
    predicted_fraction = {str(label): float((pred_class == label).mean()) for label in class_values}
    confusion = {
        str(true_label): {str(pred_label): int(((y_true == true_label) & (pred_class == pred_label)).sum()) for pred_label in class_values}
        for true_label in class_values
    }
    majority_label = max(true_counts, key=true_counts.get)
    majority_class_baseline = float(true_counts[majority_label] / max(len(y_true), 1))

    per_class_recall: dict[str, float] = {}
    per_class_precision: dict[str, float] = {}
    per_class_f1: dict[str, float] = {}
    for label in class_values:
        label_key = str(label)
        tp = confusion[label_key][label_key]
        actual = true_counts[label_key]
        predicted = predicted_counts[label_key]
        recall = float(tp / actual) if actual else 0.0
        precision = float(tp / predicted) if predicted else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0.0 else 0.0
        per_class_recall[label_key] = recall
        per_class_precision[label_key] = precision
        per_class_f1[label_key] = f1

    observed_classes = [label_key for label_key, count in true_counts.items() if count > 0]
    balanced_accuracy = float(np.mean([per_class_recall[label_key] for label_key in observed_classes])) if observed_classes else 0.0
    macro_f1 = float(np.mean([per_class_f1[label_key] for label_key in observed_classes])) if observed_classes else 0.0
    warning = max(predicted_fraction.values(), default=0.0) >= 0.98
    return {
        "n_test": int(len(y_true)),
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "directional_hit_rate": hit_rate,
        "directional_edge": directional_edge,
        "n_directional": int(non_flat.sum()),
        "true_counts": true_counts,
        "predicted_counts": predicted_counts,
        "true_fraction": true_fraction,
        "predicted_fraction": predicted_fraction,
        "majority_class_baseline": majority_class_baseline,
        "majority_class": majority_label,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
        "collapse_warning": warning,
        "confusion_matrix": confusion,
    }


def _build_horizon_oos_rows(
    symbol: str,
    fold_name: str,
    horizon: int,
    timestamps: pd.DatetimeIndex,
    predictions: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> tuple[list[dict], dict]:
    probabilities = np.asarray(predictions["probabilities"], dtype=np.float64)
    predicted_class = probabilities.argmax(axis=1).astype(np.int64) - 1
    true_class = np.asarray(targets["direction"], dtype=np.float64).astype(np.int64)
    expected_return = np.asarray(predictions["expected_return"], dtype=np.float64)
    expected_opportunity = np.asarray(predictions["opportunity"], dtype=np.float64)
    expected_mfe = np.asarray(predictions["expected_mfe"], dtype=np.float64)
    expected_mae = np.asarray(predictions["expected_mae"], dtype=np.float64)
    expected_duration = np.asarray(predictions["expected_duration"], dtype=np.float64)
    raw_return = np.asarray(predictions["raw_return"], dtype=np.float64)
    raw_mfe = np.asarray(predictions["raw_mfe"], dtype=np.float64)
    raw_mae = np.asarray(predictions["raw_mae"], dtype=np.float64)
    raw_duration = np.asarray(predictions["raw_duration"], dtype=np.float64)
    return_scale = np.asarray(predictions["return_scale"], dtype=np.float64)
    mfe_scale = np.asarray(predictions["mfe_scale"], dtype=np.float64)
    mae_scale = np.asarray(predictions["mae_scale"], dtype=np.float64)
    duration_scale = np.asarray(predictions["duration_scale"], dtype=np.float64)
    duration_active = np.asarray(predictions["duration_active"], dtype=bool)
    realized_return = np.asarray(targets["return"], dtype=np.float64)
    realized_mfe = np.asarray(targets["mfe"], dtype=np.float64)
    realized_mae = np.asarray(targets["mae"], dtype=np.float64)
    quality = continuous_opportunity_score(expected_opportunity, expected_return, expected_mfe, expected_mae)
    metrics = evaluate_predictions(probabilities, true_class)
    metrics.update({
        "asset": symbol,
        "fold": fold_name,
        "horizon": int(horizon),
        "mean_predicted_opportunity": float(expected_opportunity.mean()),
        "std_predicted_opportunity": float(expected_opportunity.std()),
        "mean_expected_return": float(expected_return.mean()),
        "std_expected_return": float(expected_return.std()),
        "return_scale": float(return_scale[0]),
        "mfe_scale": float(mfe_scale[0]),
        "mae_scale": float(mae_scale[0]),
        "duration_scale": float(duration_scale[0]),
        "duration_active": bool(duration_active[0]),
        "score_buckets": score_bucket_metrics(quality, realized_return, realized_mfe, realized_mae),
    })
    directional_hit = ((predicted_class == true_class) & (true_class != 0) & (predicted_class != 0)).astype(np.int8)
    rows = []
    for i, timestamp in enumerate(timestamps):
        rows.append({
            "asset": symbol,
            "fold": fold_name,
            "horizon": int(horizon),
            "timestamp": timestamp.isoformat(),
            "predicted_class": int(predicted_class[i]),
            "true_class": int(true_class[i]),
            "probability_short": float(probabilities[i, 0]),
            "probability_neutral": float(probabilities[i, 1]),
            "probability_long": float(probabilities[i, 2]),
            "expected_return": float(expected_return[i]),
            "expected_opportunity": float(expected_opportunity[i]),
            "expected_mfe": float(expected_mfe[i]),
            "expected_mae": float(expected_mae[i]),
            "expected_duration": float(expected_duration[i]),
            "raw_return": float(raw_return[i]),
            "raw_mfe": float(raw_mfe[i]),
            "raw_mae": float(raw_mae[i]),
            "raw_duration": float(raw_duration[i]),
            "return_scale": float(return_scale[i]),
            "mfe_scale": float(mfe_scale[i]),
            "mae_scale": float(mae_scale[i]),
            "duration_scale": float(duration_scale[i]),
            "duration_active": bool(duration_active[i]),
            "realized_return": float(realized_return[i]),
            "realized_mfe": float(realized_mfe[i]),
            "realized_mae": float(realized_mae[i]),
            "directional_hit": int(directional_hit[i]),
            "confidence": float(probabilities[i, 2] - probabilities[i, 0]),
            "opportunity_score": float(quality["score"][i]),
        })
    return rows, metrics


def _validate_horizon_oos_export(rows: list[dict], horizons: tuple[int, ...]) -> dict:
    """Validate the new multi-horizon OOS export before it is persisted."""
    frame = pd.DataFrame(rows)
    required = {
        "asset", "fold", "horizon", "timestamp", "predicted_class", "true_class",
        "probability_short", "probability_neutral", "probability_long", "expected_return",
        "expected_opportunity", "expected_mfe", "expected_mae", "expected_duration",
        "raw_return", "raw_mfe", "raw_mae", "raw_duration",
        "return_scale", "mfe_scale", "mae_scale", "duration_scale",
        "duration_active",
        "realized_return", "realized_mfe", "realized_mae", "directional_hit", "confidence",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Horizon OOS export is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Horizon OOS export is empty")
    if frame.duplicated(["asset", "fold", "horizon", "timestamp"]).any():
        raise ValueError("Horizon OOS export contains duplicate asset/fold/horizon/timestamp rows")
    if set(frame["horizon"].astype(int)) != set(int(h) for h in horizons):
        raise ValueError("Horizon OOS export does not contain exactly the configured horizons")

    numeric_columns = [
        "probability_short", "probability_neutral", "probability_long", "expected_return",
        "expected_opportunity", "expected_mfe", "expected_mae", "expected_duration",
        "raw_return", "raw_mfe", "raw_mae", "raw_duration",
        "return_scale", "mfe_scale", "mae_scale", "duration_scale",
        "duration_active",
        "realized_return", "realized_mfe", "realized_mae", "confidence",
    ]
    numeric = frame[numeric_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Horizon OOS export contains non-finite numeric values")
    probabilities = frame[["probability_short", "probability_neutral", "probability_long"]].to_numpy(dtype=np.float64)
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("Horizon OOS export contains probabilities outside [0, 1]")
    probability_sums = probabilities.sum(axis=1)
    if not np.allclose(probability_sums, 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("Horizon OOS probabilities do not sum to 1")
    if not set(frame["predicted_class"].astype(int)).issubset({-1, 0, 1}) or not set(frame["true_class"].astype(int)).issubset({-1, 0, 1}):
        raise ValueError("Horizon OOS export contains classes outside {-1, 0, 1}")

    distinctness: dict[str, bool] = {}
    prediction_columns = [
        "probability_short", "probability_neutral", "probability_long", "expected_return",
        "expected_opportunity", "expected_mfe", "expected_mae", "expected_duration",
    ]
    for left, right in zip(horizons, horizons[1:]):
        left_frame = frame[frame["horizon"] == left].set_index(["asset", "fold", "timestamp"])[prediction_columns].sort_index()
        right_frame = frame[frame["horizon"] == right].set_index(["asset", "fold", "timestamp"])[prediction_columns].sort_index()
        aligned_left, aligned_right = left_frame.align(right_frame, join="inner", axis=0)
        distinctness[f"{left}vs{right}"] = bool(not np.allclose(aligned_left.to_numpy(), aligned_right.to_numpy(), atol=1e-7, rtol=1e-7))
    group_counts = frame.groupby(["asset", "fold", "horizon"], dropna=False).size()
    return {
        "row_count": int(len(frame)),
        "group_counts": {"|".join(map(str, key if isinstance(key, tuple) else (key,))): int(value) for key, value in group_counts.items()},
        "duplicate_keys": 0,
        "all_predictions_finite": True,
        "probabilities_in_range": True,
        "probabilities_sum_to_one": True,
        "classes_valid": True,
        "horizon_pairwise_distinct": distinctness,
    }


def train_symbol_model(
    symbol: str,
    ohlc: pd.DataFrame,
    macro_prices: pd.DataFrame | None,
    config: TrendMLConfig,
    breadth_return: pd.Series | None = None,
    diagnostic_dir: str | Path | None = None,
    diagnostic_fold: int | None = None,
    diagnostic_only: bool = False,
) -> dict:
    horizons = resolve_forecast_horizons(config)
    print(f"\n=== {symbol}: building features + triple-barrier labels (horizons={horizons}) ===")
    X, y, t1_pos, mfe, mae, horizon_labels = _prepare_symbol_dataset(ohlc, macro_prices, config, breadth_return, forecast_horizons=horizons)
    X_seq, y_seq, t1_seq, index_seq, mfe_seq, mae_seq = _to_sequences(X, y, t1_pos, config.sequence_length, mfe, mae)
    targets_by_horizon = build_targets_by_horizon(X, horizon_labels, horizons)
    targets_by_horizon_seq = {h: {name: values[config.sequence_length - 1 :] for name, values in arrays.items()} for h, arrays in targets_by_horizon.items()}
    favorable = np.maximum(mfe_seq, 0.0)
    adverse = np.maximum(-mae_seq, 0.0)
    opportunity_seq = np.clip(favorable / np.maximum(favorable + adverse, 1e-6), 0.0, 1.0).astype(np.float32)
    if config.entry_quality_mode == "payoff_volatility_compression":
        feature_names = list(X.columns)
        if "realized_vol_12" in feature_names and "realized_vol_48" in feature_names:
            vol12 = X_seq[:, -1, feature_names.index("realized_vol_12")]
            vol48 = X_seq[:, -1, feature_names.index("realized_vol_48")]
            compression = np.clip(1.0 - vol12 / np.maximum(np.abs(vol48), 1e-8), 0.0, 1.0)
            compression = np.nan_to_num(compression, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            compression = np.zeros(len(opportunity_seq), dtype=np.float32)
        opportunity_seq = (0.7 * opportunity_seq + 0.3 * compression).astype(np.float32)
    return_seq = (np.sign(y_seq) * (favorable - adverse)).astype(np.float32)
    primary_labels = horizon_labels[horizons[0]]
    duration_row = primary_labels["time_to_barrier"].reindex(X.index).to_numpy(dtype=np.float32)
    duration_seq = np.clip(duration_row / max(int(horizons[0]), 1), 0.0, 1.0).astype(np.float32)
    print(
        f"{symbol}: {X_seq.shape[0]} sequences, {X_seq.shape[2]} features, "
        f"label distribution: {pd.Series(y_seq).value_counts().to_dict()}"
    )
    horizon_stats = {}
    for horizon in horizons:
        horizon_labels_df = horizon_labels[horizon]
        label_values = horizon_labels_df["label"].fillna(0.0).to_numpy(dtype=float)
        class_counts = {str(k): int((label_values == float(k)).sum()) for k in (-1, 0, 1)}
        horizon_stats[int(horizon)] = {
            "class_counts": class_counts,
            "class_fractions": {k: float(v / max(len(label_values), 1)) for k, v in class_counts.items()},
            "duration_mean": float(horizon_labels_df["time_to_barrier"].dropna().mean()) if horizon_labels_df["time_to_barrier"].notna().any() else 0.0,
            "duration_std": float(horizon_labels_df["time_to_barrier"].dropna().std()) if horizon_labels_df["time_to_barrier"].notna().any() else 0.0,
            "mfe_mean": float(horizon_labels_df["mfe"].dropna().mean()) if horizon_labels_df["mfe"].notna().any() else 0.0,
            "mae_mean": float(horizon_labels_df["mae"].dropna().mean()) if horizon_labels_df["mae"].notna().any() else 0.0,
            "return_mean": float((np.sign(label_values) * (horizon_labels_df["mfe"].fillna(0.0).to_numpy() - horizon_labels_df["mae"].fillna(0.0).to_numpy())).mean()) if len(label_values) else 0.0,
        }
    print(f"{symbol}: horizon target stats -> {horizon_stats}")

    split = chronological_purged_split(
        index_seq,
        t1_seq,
        train_end=config.train_end,
        validation_start=config.validation_start,
        validation_end=config.validation_end,
        test_start=config.test_start,
        test_end=config.test_end,
        purge_hours=config.purge_hours,
        minimum_train_samples=config.minimum_train_samples,
        minimum_validation_samples=config.minimum_validation_samples,
        minimum_test_samples=config.minimum_test_samples,
    )
    return_center = float(np.mean(return_seq[split.train_idx])) if config.relative_return_centering else 0.0
    centered_return_seq = return_seq - return_center
    print(
        f"{symbol}: after {config.purge_hours}h purge/embargo: "
        f"train={len(split.train_idx)}, validation={len(split.validation_idx)}, test={len(split.test_idx)}"
    )
    print(
        f"{symbol}: effective train {split.train_start} -> {split.train_end}; "
        f"validation {split.validation_start} -> {split.validation_end}; "
        f"test {split.test_start} -> {split.test_end}"
    )

    # Time-decay training-loss weights (see core/ml/dataset.py:
    # time_decay_sample_weights): computed once from the full sequence index
    # so every fold's train slice is weighted consistently by the same
    # recency curve, then sliced identically to X_seq/y_seq per fold/final fit.
    sample_weight = time_decay_sample_weights(index_seq, config.sample_weight_half_life_days)

    fold_metrics = []
    horizon_oos_metrics = []
    horizon_oos_rows: list[dict] = []
    # Stitched, genuinely out-of-sample confidence for every bar covered by a
    # test fold (NaN elsewhere, e.g. the very first training-only slice before
    # any fold starts) -- this, NOT the final full-history model below, is
    # what a backtest/Monte Carlo is allowed to use to score historical bars
    # without leaking future training data into the past (see
    # ``core/ml/inference.py: load_oos_confidence`` / ``core/strategy.py:
    # generate_portfolio_signals``).
    oos_confidence = np.full(len(y_seq), np.nan, dtype=np.float64)
    walk_forward_folds = purged_walk_forward_splits(
        len(y_seq), t1_seq, config.n_splits, config.embargo_fraction
    )
    for fold_number, fold in enumerate(walk_forward_folds, start=1):
        if diagnostic_fold is not None and fold_number != diagnostic_fold:
            continue
        fold_model = TCNTrendModel(config, n_features=X_seq.shape[2], forecast_horizons=horizons)
        fold_targets = {
            h: {name: values[fold.train_idx] for name, values in arrays.items()}
            for h, arrays in targets_by_horizon_seq.items()
        }
        fold_model.fit(
            IndexedSequenceView(X_seq, fold.train_idx), y_seq[fold.train_idx],
            sample_weight=sample_weight[fold.train_idx],
            mfe_seq=mfe_seq[fold.train_idx], mae_seq=mae_seq[fold.train_idx],
            opportunity_seq=opportunity_seq[fold.train_idx], return_seq=centered_return_seq[fold.train_idx], duration_seq=duration_seq[fold.train_idx],
            targets_by_horizon=fold_targets,
            diagnostic_label=f"{symbol} wf_{fold_number}" if diagnostic_dir is not None else None,
            diagnostic_stop=diagnostic_dir is not None,
        )
        if diagnostic_dir is not None:
            diagnostic_root = Path(diagnostic_dir)
            diagnostic_root.mkdir(parents=True, exist_ok=True)
            import torch
            torch.save(
                {
                    "model_state": fold_model.state_dict(),
                    "asset": symbol,
                    "fold": fold_number,
                    "forecast_horizons": list(horizons),
                    "best_validation_loss": min((row.get("validation_loss", float("inf")) for row in fold_model.training_history), default=float("inf")),
                    "training_history": fold_model.training_history,
                    "training_config": {"max_epochs": config.max_epochs, "batch_size": config.batch_size, "learning_rate": config.learning_rate, "gradient_clip_norm": config.gradient_clip_norm},
                    "target_scales": fold_model.horizon_target_scales,
                    "loss_weights": {"direction": config.direction_loss_weight, "opportunity": config.opportunity_loss_weight, "return": config.return_loss_weight, "duration": config.duration_loss_weight, "mfe": config.mfe_loss_weight, "mae": config.mae_loss_weight},
                },
                diagnostic_root / f"{symbol.replace('/', '-')}_fold_{fold_number}_best.pt",
            )
        fold_outputs_by_horizon = fold_model.predict_trade_outputs_by_horizon(X_seq[fold.test_idx])
        primary_horizon = horizons[0]
        fold_proba = fold_outputs_by_horizon[primary_horizon]["probabilities"]
        fold_outputs = fold_outputs_by_horizon[primary_horizon]
        quality = continuous_opportunity_score(
            fold_outputs["opportunity"], fold_outputs["expected_return"], fold_outputs["expected_mfe"], fold_outputs["expected_mae"],
        )
        validate_model1_prediction_outputs(fold_outputs, quality)
        fold_return = return_seq[fold.test_idx]
        metrics = evaluate_predictions(fold_proba, y_seq[fold.test_idx])
        metrics["mean_predicted_opportunity"] = float(fold_outputs["opportunity"].mean())
        metrics["std_predicted_opportunity"] = float(fold_outputs["opportunity"].std())
        metrics["mean_expected_return"] = float(fold_outputs["expected_return"].mean())
        metrics["std_expected_return"] = float(fold_outputs["expected_return"].std())
        metrics["score_buckets"] = score_bucket_metrics(quality, fold_return, mfe_seq[fold.test_idx], mae_seq[fold.test_idx])
        metrics.update({"fold": f"wf_{fold_number}", "n_train": int(len(fold.train_idx)), "n_validation": 0, "n_test": int(len(fold.test_idx))})
        fold_metrics.append(metrics)
        oos_confidence[fold.test_idx] = fold_proba[:, 2] - fold_proba[:, 0]
        fold_name = f"wf_{fold_number}"
        fold_timestamps = index_seq[fold.test_idx]
        for horizon in horizons:
            targets = {name: values[fold.test_idx] for name, values in targets_by_horizon_seq[horizon].items()}
            rows, horizon_metrics = _build_horizon_oos_rows(
                symbol,
                fold_name,
                horizon,
                fold_timestamps,
                fold_outputs_by_horizon[horizon],
                targets,
            )
            horizon_oos_rows.extend(rows)
            horizon_oos_metrics.append(horizon_metrics)
        print(f"  purged walk-forward fold {fold_number}: {metrics}")

        if diagnostic_only:
            return {"symbol": symbol, "fold_metrics": pd.DataFrame(fold_metrics), "diagnostic_only": True}

    metrics_df = pd.DataFrame(fold_metrics)

    print(f"{symbol}: fitting final model on full history (tail slice held out for early stopping)...")
    final_model = TCNTrendModel(config, n_features=X_seq.shape[2], forecast_horizons=horizons, asset=symbol)
    fit_idx = np.concatenate([split.train_idx, split.validation_idx])
    final_return_center = float(np.mean(return_seq[fit_idx])) if config.relative_return_centering else 0.0
    final_targets = {
        h: {name: values[fit_idx] for name, values in arrays.items()}
        for h, arrays in targets_by_horizon_seq.items()
    }
    final_model.fit(IndexedSequenceView(X_seq, fit_idx), y_seq[fit_idx], sample_weight=sample_weight[fit_idx], mfe_seq=mfe_seq[fit_idx], mae_seq=mae_seq[fit_idx], opportunity_seq=opportunity_seq[fit_idx], return_seq=return_seq[fit_idx] - final_return_center, duration_seq=duration_seq[fit_idx], targets_by_horizon=final_targets)
    forecast_result = final_model.predict_forecast_result(X_seq[fit_idx], data_timestamp=index_seq[-1].isoformat())

    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "-")
    artifact_path = model_dir / f"{safe_symbol}_tcn.pt"
    meta_path = model_dir / f"{safe_symbol}_tcn_meta.json"
    oos_confidence_path = model_dir / f"{safe_symbol}_tcn_oos_confidence.csv"
    oos_metrics_path = model_dir / f"{safe_symbol}_tcn_oos_metrics.json"
    oos_predictions_path = model_dir / f"{safe_symbol}_tcn_oos_predictions.csv"

    final_model.to_cpu()

    import torch  # local import: optional dependency, only needed to persist the model

    torch.save(final_model.state_dict(), artifact_path)
    pd.Series(oos_confidence, index=index_seq, name="oos_confidence").to_frame().to_csv(oos_confidence_path)
    horizon_oos_frame = pd.DataFrame(horizon_oos_rows)
    consistency = _validate_horizon_oos_export(horizon_oos_rows, horizons)
    with oos_predictions_path.open("w", encoding="utf-8", newline="") as handle:
        horizon_oos_frame.to_csv(handle, index=False)
    oos_metrics_path.write_text(json.dumps({
        "asset": symbol,
        "model_id": config.model_id,
        "training_timeframe": config.base_timeframe,
        "forecast_horizons": list(horizons),
        "consistency_checks": consistency,
        "metrics": horizon_oos_metrics,
    }, indent=2, allow_nan=True), encoding="utf-8")
    target_scalers = {
        "return_scale": float(final_model.return_target_scale),
        "mfe_scale": float(final_model.mfe_target_scale),
        "mae_scale": float(final_model.mae_target_scale),
        "duration_scale": float(final_model.duration_target_scale),
    }
    meta = {
        "model_version": "tcn-multitask-v1",
        "model_id": config.model_id,
        "model_role": config.model_role,
        "training_timeframe": config.base_timeframe,
        "production_timeframe": config.production_timeframe,
        "context_required": config.context_required,
        "context_feature_version": config.context_feature_version,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "assets": config.assets,
        "timeframes": config.timeframes,
        "feature_columns": list(X.columns),
        "sequence_length": config.sequence_length,
        # Architecture hyperparameters actually used to train THIS checkpoint
        # -- persisted so ``core/ml/inference.py: load_symbol_model`` can
        # reconstruct the exact same layer shapes later even if
        # ``TrendMLConfig.hidden_channels``/``num_layers``/``dropout`` are
        # retuned afterwards (otherwise ``load_state_dict`` fails with a
        # shape mismatch on every previously trained checkpoint).
        "hidden_channels": config.hidden_channels,
        "num_layers": config.num_layers,
        "dropout": config.dropout,
        "walk_forward_metrics": fold_metrics,
        "validation_protocol": {
            "type": "purged_walk_forward",
            "folds": config.n_splits,
            "embargo_fraction": config.embargo_fraction,
            "holdout": {"start": config.test_start, "end": config.test_end},
        },
        # Last timestamp actually used to fit/select this artifact -- the
        # chronological holdout gate (core/ml/holdout.py, core/ml/promotion.py)
        # requires that ONLY bars strictly after this cutoff decide production
        # promotion, since anything up to and including it may have informed
        # training or hyperparameter/architecture choices.
        "training_cutoff": index_seq[-1].isoformat(),
        "label_config": {
            "horizon": config.label_horizon,
            "forecast_horizons": list(horizons),
            "primary_horizon": int(horizons[0]),
            "atr_window": config.atr_window,
            "stop_atr_multiple": config.stop_atr_multiple,
            "take_profit_atr_multiple": config.take_profit_atr_multiple,
        },
        "forecast_contract": forecast_result.as_dict(),
        "scaler": {"mean": final_model.mean_.tolist(), "std": final_model.std_.tolist()},
        "target_scalers": target_scalers,
        "horizon_target_scales": final_model.horizon_target_scales,
        "horizon_duration_active": final_model.horizon_duration_active,
        "horizon_target_stats": horizon_stats,
        "random_state": config.random_state,
        "training_period": {"start": split.train_start.isoformat(), "end": split.train_end.isoformat()},
        "validation_period": {"start": config.validation_start, "end": config.validation_end},
        "test_period": {"start": config.test_start, "end": config.test_end},
        "hyperparameters": {
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "learning_rate": config.learning_rate,
            "optimizer": config.optimizer,
            "scheduler": config.scheduler,
            "hidden_channels": config.hidden_channels,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "mfe_loss_weight": config.mfe_loss_weight,
            "mae_loss_weight": config.mae_loss_weight,
        },
        "calibration": {"method": config.calibration_method, "parameters": None},
        "entry_quality_mode": config.entry_quality_mode,
        "public_prediction_heads": ["entry_score", "opportunity", "expected_return", "expected_duration", "expected_mfe", "expected_mae"],
        "model_profile": config.model_profile,
        "relative_return_center": final_return_center,
        "score_config": {"thresholds": config.score_thresholds},
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=float))
    print(f"{symbol}: saved model -> {artifact_path}, metadata -> {meta_path}, OOS confidence -> {oos_confidence_path}, horizon OOS metrics -> {oos_metrics_path}, horizon OOS predictions -> {oos_predictions_path}")

    return {"symbol": symbol, "fold_metrics": metrics_df, "artifact_path": str(artifact_path), "forecast_contract": forecast_result.as_dict()}


def run_training_pipeline(config: TradingBotConfig) -> dict[str, dict]:
    ml_config = config.ml
    multi_ohlc = load_ml_ohlc(config.data, ml_config)
    print("\n=== DATASET VALIDATION ===")
    print(f"Target timeframe: {ml_config.base_timeframe}")
    print(f"Prediction horizon: {horizon_description(ml_config.base_timeframe, ml_config.label_horizon)}")
    validate_symbol_coverage(
        multi_ohlc,
        train_end=ml_config.train_end,
        validation_start=ml_config.validation_start,
        validation_end=ml_config.validation_end,
        test_start=ml_config.test_start,
        test_end=ml_config.test_end,
    )
    macro_prices = load_prepared_macro_matrix(ml_config) if ml_config.macro_symbols else None
    breadth_return = fetch_breadth_basket(config.data, ml_config)

    results = {}
    for symbol, ohlc in multi_ohlc.items():
        results[symbol] = train_symbol_model(symbol, ohlc, macro_prices, ml_config, breadth_return)
    return results
