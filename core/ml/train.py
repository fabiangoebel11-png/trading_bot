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
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import DataConfig, TradingBotConfig, TrendMLConfig
from core.data_loader import fetch_ohlcv_history
from core.ml.dataset import build_row_level_dataset, make_sequences, purged_walk_forward_splits, time_decay_sample_weights
from core.ml.features import build_feature_matrix, fetch_breadth_basket, fetch_macro_matrix
from core.ml.labeling import triple_barrier_labels
from core.ml.tcn_model import TCNTrendModel
from core.strategy import compute_atr


def load_ml_ohlc(data_config: DataConfig, ml_config: TrendMLConfig) -> dict[str, pd.DataFrame]:
    """Like ``core.data_loader.load_multi_asset_data``, but at the ML pipeline's
    own timeframe/history length (now the same 1h as the rule-based strategy,
    see ``TrendMLConfig`` docstring), and keeping the ``volume`` column that
    the equal-weight portfolio loader drops."""
    raw = {}
    for symbol in data_config.symbols:
        history = fetch_ohlcv_history(
            symbol,
            ml_config.base_timeframe,
            ml_config.history_days,
            cache_dir=data_config.cache_dir,
            exchange_id=data_config.exchange_id,
            market_type=data_config.market_type,
        )
        raw[symbol] = history

    common_index = raw[data_config.symbols[0]].index
    for df in raw.values():
        common_index = common_index.intersection(df.index)
    return {symbol: df.loc[common_index].sort_index() for symbol, df in raw.items()}


def _prepare_symbol_dataset(
    ohlc: pd.DataFrame,
    macro_prices: pd.DataFrame | None,
    config: TrendMLConfig,
    breadth_return: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    features = build_feature_matrix(ohlc, macro_prices, config, breadth_return)
    atr = compute_atr(ohlc["high"], ohlc["low"], ohlc["close"], config.atr_window)
    labels = triple_barrier_labels(ohlc["close"], atr, config.label_horizon, config.barrier_atr_multiple)
    return build_row_level_dataset(features, labels)


def _to_sequences(
    X: pd.DataFrame, y: pd.Series, t1_pos: np.ndarray, seq_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    X_arr = X.to_numpy(dtype=np.float32)
    n_rows = X_arr.shape[0]
    n_seq = n_rows - seq_len + 1
    if n_seq <= 0:
        raise ValueError(f"Not enough rows ({n_rows}) for sequence_length={seq_len}.")
    X_seq = make_sequences(X_arr, seq_len)
    y_seq = y.to_numpy()[seq_len - 1 :]
    # t1_pos is row-level; shift into sequence-index space (sequence i ends at row i+seq_len-1).
    t1_seq = np.clip(t1_pos[seq_len - 1 :] - (seq_len - 1), 0, n_seq - 1)
    index_seq = X.index[seq_len - 1 :]
    return X_seq, y_seq, t1_seq, index_seq


def evaluate_predictions(proba: np.ndarray, y_true: np.ndarray) -> dict:
    """Metrics on purged out-of-sample test folds only. ``directional_edge`` is
    a proxy Sharpe-like ratio (mean/std of confidence * realized label
    direction) -- not a real backtested return, just a quick signal-quality
    check before any of this touches ``core/strategy.py`` position sizing."""
    signal = proba[:, 2] - proba[:, 0]  # P(up) - P(down)
    pred_dir = np.sign(signal)
    non_flat = y_true != 0
    if non_flat.sum() > 0:
        hits = (pred_dir[non_flat] == np.sign(y_true[non_flat])) & (pred_dir[non_flat] != 0)
        hit_rate = float(hits.mean())
    else:
        hit_rate = float("nan")

    proxy_returns = signal * y_true
    edge_std = float(proxy_returns.std())
    directional_edge = float(proxy_returns.mean()) / edge_std if edge_std > 0 else 0.0

    pred_class = proba.argmax(axis=1) - 1
    accuracy = float((pred_class == y_true).mean())
    return {
        "n_test": int(len(y_true)),
        "accuracy": accuracy,
        "directional_hit_rate": hit_rate,
        "directional_edge": directional_edge,
    }


def train_symbol_model(
    symbol: str,
    ohlc: pd.DataFrame,
    macro_prices: pd.DataFrame | None,
    config: TrendMLConfig,
    breadth_return: pd.Series | None = None,
) -> dict:
    print(f"\n=== {symbol}: building features + triple-barrier labels ===")
    X, y, t1_pos = _prepare_symbol_dataset(ohlc, macro_prices, config, breadth_return)
    X_seq, y_seq, t1_seq, index_seq = _to_sequences(X, y, t1_pos, config.sequence_length)
    print(
        f"{symbol}: {X_seq.shape[0]} sequences, {X_seq.shape[2]} features, "
        f"label distribution: {pd.Series(y_seq).value_counts().to_dict()}"
    )

    # Time-decay training-loss weights (see core/ml/dataset.py:
    # time_decay_sample_weights): computed once from the full sequence index
    # so every fold's train slice is weighted consistently by the same
    # recency curve, then sliced identically to X_seq/y_seq per fold/final fit.
    sample_weight = time_decay_sample_weights(index_seq, config.sample_weight_half_life_days)

    folds = purged_walk_forward_splits(len(y_seq), t1_seq, config.n_splits, config.embargo_fraction)

    fold_metrics = []
    # Stitched, genuinely out-of-sample confidence for every bar covered by a
    # test fold (NaN elsewhere, e.g. the very first training-only slice before
    # any fold starts) -- this, NOT the final full-history model below, is
    # what a backtest/Monte Carlo is allowed to use to score historical bars
    # without leaking future training data into the past (see
    # ``core/ml/inference.py: load_oos_confidence`` / ``core/strategy.py:
    # generate_portfolio_signals``).
    oos_confidence = np.full(len(y_seq), np.nan, dtype=np.float64)
    for i, fold in enumerate(folds):
        if len(fold.train_idx) < 200 or len(fold.test_idx) == 0:
            print(f"  fold {i}: skipped (too little purged training data)")
            continue
        model = TCNTrendModel(config, n_features=X_seq.shape[2])
        model.fit(X_seq[fold.train_idx], y_seq[fold.train_idx], sample_weight=sample_weight[fold.train_idx])
        proba = model.predict_proba(X_seq[fold.test_idx])
        metrics = evaluate_predictions(proba, y_seq[fold.test_idx])
        metrics["fold"] = i
        metrics["n_train"] = int(len(fold.train_idx))
        fold_metrics.append(metrics)
        print(f"  fold {i}: {metrics}")
        oos_confidence[fold.test_idx] = proba[:, 2] - proba[:, 0]

    metrics_df = pd.DataFrame(fold_metrics)

    print(f"{symbol}: fitting final model on full history (tail slice held out for early stopping)...")
    final_model = TCNTrendModel(config, n_features=X_seq.shape[2])
    final_model.fit(X_seq, y_seq, sample_weight=sample_weight)

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
        "symbol": symbol,
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
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=float))
    print(f"{symbol}: saved model -> {artifact_path}, metadata -> {meta_path}, OOS confidence -> {oos_confidence_path}")

    return {"symbol": symbol, "fold_metrics": metrics_df, "artifact_path": str(artifact_path)}


def run_training_pipeline(config: TradingBotConfig) -> dict[str, dict]:
    ml_config = config.ml
    multi_ohlc = load_ml_ohlc(config.data, ml_config)
    macro_prices = fetch_macro_matrix(ml_config) if ml_config.macro_symbols else None
    breadth_return = fetch_breadth_basket(config.data, ml_config)

    results = {}
    for symbol, ohlc in multi_ohlc.items():
        results[symbol] = train_symbol_model(symbol, ohlc, macro_prices, ml_config, breadth_return)
    return results
