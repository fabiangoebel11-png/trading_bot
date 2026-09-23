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
from core.ml.labeling import qualified_trade_labels
from core.ml.scoring import continuous_opportunity_score, score_bucket_metrics, score_trade_quality, threshold_metrics, validate_model1_prediction_outputs
from core.ml.tcn_model import TCNTrendModel
from core.strategy import compute_atr


def load_ml_ohlc(data_config: DataConfig, ml_config: TrendMLConfig) -> dict[str, pd.DataFrame]:
    """Like ``core.data_loader.load_multi_asset_data``, but at the ML pipeline's
    own timeframe/history length (now the same 1h as the rule-based strategy,
    see ``TrendMLConfig`` docstring), and keeping the ``volume`` column that
    the equal-weight portfolio loader drops."""
    if ml_config.base_timeframe != "5m":
        raise ValueError(f"ML training target must be 5m, got {ml_config.base_timeframe!r}")
    specs = [
        spec for spec in asset_specs_from_config(TradingBotConfig(data=data_config, ml=ml_config))
        if spec.timeframe == "5m" and spec.name in ml_config.assets
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
    context = {"VIX": "^VIX", "US10Y": "^TNX", "EURUSD": "EURUSD=X", "GOLD": "GC=F", "WTI": "CL=F", "DAX": "^GDAXI"}
    columns: dict[str, pd.Series] = {}
    for name, source in context.items():
        if source not in config.macro_symbols:
            continue
        spec = AssetSpec(name, source, "yfinance", "1d", config.macro_lookback_days, source=source)
        path = canonical_cache_path(config.market_cache_dir, spec)
        if path.exists():
            columns[source] = load_ohlcv_parquet(path)["close"]
    if not columns:
        raise FileNotFoundError("No prepared macro/context Parquet datasets matched TrendMLConfig.macro_symbols")
    return pd.DataFrame(columns).sort_index()


def _prepare_symbol_dataset(
    ohlc: pd.DataFrame,
    macro_prices: pd.DataFrame | None,
    config: TrendMLConfig,
    breadth_return: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    features = build_feature_matrix(ohlc, macro_prices, config, breadth_return)
    atr = compute_atr(ohlc["high"], ohlc["low"], ohlc["close"], config.atr_window)
    labels = qualified_trade_labels(
        ohlc["close"],
        atr,
        config.label_horizon,
        config.barrier_atr_multiple,
        high=ohlc["high"],
        low=ohlc["low"],
        stop_atr_multiple=config.stop_atr_multiple,
        take_profit_atr_multiple=config.take_profit_atr_multiple,
    )
    X, y, t1_pos = build_row_level_dataset(features, labels)
    quality = labels.loc[X.index]
    return X, y, t1_pos, quality["mfe"].to_numpy(dtype=np.float32), quality["mae"].to_numpy(dtype=np.float32)


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
    # Model class order is [SHORT, NO_TRADE, LONG] = [0, 1, 2]. Decode once
    # and use these same human-readable labels for every classification metric.
    pred_class = proba.argmax(axis=1) - 1
    non_flat = y_true != 0
    if non_flat.sum() > 0:
        hits = (pred_class[non_flat] == y_true[non_flat]) & (pred_class[non_flat] != 0)
        hit_rate = float(hits.mean())
    else:
        hit_rate = float("nan")

    signal = proba[:, 2] - proba[:, 0]  # retained for the proxy edge metric only
    proxy_returns = signal * y_true
    edge_std = float(proxy_returns.std())
    directional_edge = float(proxy_returns.mean()) / edge_std if edge_std > 0 else 0.0

    accuracy = float((pred_class == y_true).mean())
    class_values = (-1, 0, 1)
    confusion = {
        str(true_label): {str(pred_label): int(((y_true == true_label) & (pred_class == pred_label)).sum()) for pred_label in class_values}
        for true_label in class_values
    }
    predicted_counts = {str(label): int((pred_class == label).sum()) for label in class_values}
    predicted_fraction = {str(label): float((pred_class == label).mean()) for label in class_values}
    warning = max(predicted_fraction.values(), default=0.0) >= 0.98
    return {
        "n_test": int(len(y_true)),
        "accuracy": accuracy,
        "directional_hit_rate": hit_rate,
        "directional_edge": directional_edge,
        "n_directional": int(non_flat.sum()),
        "true_counts": {str(label): int((y_true == label).sum()) for label in class_values},
        "predicted_counts": predicted_counts,
        "true_fraction": {str(label): float((y_true == label).mean()) for label in class_values},
        "predicted_fraction": predicted_fraction,
        "collapse_warning": warning,
        "confusion_matrix": confusion,
    }


def train_symbol_model(
    symbol: str,
    ohlc: pd.DataFrame,
    macro_prices: pd.DataFrame | None,
    config: TrendMLConfig,
    breadth_return: pd.Series | None = None,
) -> dict:
    print(f"\n=== {symbol}: building features + triple-barrier labels ===")
    X, y, t1_pos, mfe, mae = _prepare_symbol_dataset(ohlc, macro_prices, config, breadth_return)
    X_seq, y_seq, t1_seq, index_seq, mfe_seq, mae_seq = _to_sequences(X, y, t1_pos, config.sequence_length, mfe, mae)
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
    duration_seq = np.clip((t1_seq - np.arange(len(t1_seq))) / max(config.label_horizon, 1), 0.0, 1.0).astype(np.float32)
    print(
        f"{symbol}: {X_seq.shape[0]} sequences, {X_seq.shape[2]} features, "
        f"label distribution: {pd.Series(y_seq).value_counts().to_dict()}"
    )

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
    # Stitched, genuinely out-of-sample confidence for every bar covered by a
    # test fold (NaN elsewhere, e.g. the very first training-only slice before
    # any fold starts) -- this, NOT the final full-history model below, is
    # what a backtest/Monte Carlo is allowed to use to score historical bars
    # without leaking future training data into the past (see
    # ``core/ml/inference.py: load_oos_confidence`` / ``core/strategy.py:
    # generate_portfolio_signals``).
    oos_confidence = np.full(len(y_seq), np.nan, dtype=np.float64)
    model = TCNTrendModel(config, n_features=X_seq.shape[2])
    model.fit(
        IndexedSequenceView(X_seq, split.train_idx), y_seq[split.train_idx],
        sample_weight=sample_weight[split.train_idx],
        mfe_seq=mfe_seq[split.train_idx], mae_seq=mae_seq[split.train_idx],
        opportunity_seq=opportunity_seq[split.train_idx], return_seq=centered_return_seq[split.train_idx], duration_seq=duration_seq[split.train_idx],
    )
    test_proba = model.predict_proba(X_seq[split.test_idx])
    test_outputs = model.predict_trade_outputs(X_seq[split.test_idx])
    quality = continuous_opportunity_score(
        test_outputs["opportunity"], test_outputs["expected_return"], test_outputs["expected_mfe"], test_outputs["expected_mae"],
    )
    validate_model1_prediction_outputs(test_outputs, quality)
    test_return = return_seq[split.test_idx]
    metrics = evaluate_predictions(test_proba, y_seq[split.test_idx])
    metrics["mean_predicted_opportunity"] = float(test_outputs["opportunity"].mean())
    metrics["std_predicted_opportunity"] = float(test_outputs["opportunity"].std())
    metrics["mean_expected_return"] = float(test_outputs["expected_return"].mean())
    metrics["std_expected_return"] = float(test_outputs["expected_return"].std())
    metrics["score_buckets"] = score_bucket_metrics(quality, test_return, mfe_seq[split.test_idx], mae_seq[split.test_idx])
    if metrics["collapse_warning"]:
        print(f"WARNING: {symbol} Model 1 predictions are direction-collapsed: {metrics['predicted_fraction']}")
    metrics.update({"fold": "fixed_test", "n_train": int(len(split.train_idx)), "n_validation": int(len(split.validation_idx))})
    fold_metrics.append(metrics)
    print(f"  fixed test: {metrics}")
    oos_confidence[split.test_idx] = test_proba[:, 2] - test_proba[:, 0]

    metrics_df = pd.DataFrame(fold_metrics)

    print(f"{symbol}: fitting final model on full history (tail slice held out for early stopping)...")
    final_model = TCNTrendModel(config, n_features=X_seq.shape[2])
    fit_idx = np.concatenate([split.train_idx, split.validation_idx])
    final_return_center = float(np.mean(return_seq[fit_idx])) if config.relative_return_centering else 0.0
    final_model.fit(IndexedSequenceView(X_seq, fit_idx), y_seq[fit_idx], sample_weight=sample_weight[fit_idx], mfe_seq=mfe_seq[fit_idx], mae_seq=mae_seq[fit_idx], opportunity_seq=opportunity_seq[fit_idx], return_seq=return_seq[fit_idx] - final_return_center, duration_seq=duration_seq[fit_idx])

    model_dir = Path(config.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    safe_symbol = symbol.replace("/", "-")
    artifact_path = model_dir / f"{safe_symbol}_tcn.pt"
    meta_path = model_dir / f"{safe_symbol}_tcn_meta.json"
    oos_confidence_path = model_dir / f"{safe_symbol}_tcn_oos_confidence.csv"

    import torch  # local import: optional dependency, only needed to persist the model

    torch.save(final_model.state_dict(), artifact_path)
    pd.Series(oos_confidence, index=index_seq, name="oos_confidence").to_frame().to_csv(oos_confidence_path)
    meta = {
        "model_version": "tcn-multitask-v1",
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
        # Last timestamp actually used to fit/select this artifact -- the
        # chronological holdout gate (core/ml/holdout.py, core/ml/promotion.py)
        # requires that ONLY bars strictly after this cutoff decide production
        # promotion, since anything up to and including it may have informed
        # training or hyperparameter/architecture choices.
        "training_cutoff": index_seq[-1].isoformat(),
        "label_config": {
            "horizon": config.label_horizon,
            "atr_window": config.atr_window,
            "stop_atr_multiple": config.stop_atr_multiple,
            "take_profit_atr_multiple": config.take_profit_atr_multiple,
        },
        "scaler": {"mean": final_model.mean_.tolist(), "std": final_model.std_.tolist()},
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
    print(f"{symbol}: saved model -> {artifact_path}, metadata -> {meta_path}, OOS confidence -> {oos_confidence_path}")

    return {"symbol": symbol, "fold_metrics": metrics_df, "artifact_path": str(artifact_path)}


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
