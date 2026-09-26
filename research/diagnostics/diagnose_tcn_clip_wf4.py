from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ml.dataset import purged_walk_forward_splits
from core.ml.tcn_model import TCNTrendModel
from core.ml.train import _prepare_symbol_dataset, _to_sequences, build_targets_by_horizon, load_ml_ohlc
from train import load_config

SYMBOL = "BTC/USDT"
FOLD_INDEX = 3  # wf_4
HORIZONS = (1, 4, 8, 12, 24)


def _norm_no_clip(X: np.ndarray, mean_: np.ndarray, std_: np.ndarray, mask_: np.ndarray | None) -> np.ndarray:
    X_norm = np.asarray(X, dtype=np.float32, order="C", copy=True)
    np.subtract(X_norm, mean_, out=X_norm)
    np.divide(X_norm, std_, out=X_norm)
    if mask_ is not None:
        X_norm[..., mask_] = 0.0
    return X_norm


def _compute_before_after_norm(cfg, X_seq, y_seq, t1_seq):
    folds = purged_walk_forward_splits(len(y_seq), t1_seq, cfg.ml.n_splits, cfg.ml.embargo_fraction)
    fold = folds[FOLD_INDEX]
    train_X = X_seq[fold.train_idx]
    val_size = max(1, int(len(train_X) * cfg.ml.val_fraction))
    gap = min(cfg.ml.sequence_length, max(0, len(train_X) - val_size - 1))
    train_end = max(1, len(train_X) - val_size - gap)
    train_fit_X = train_X[:train_end]
    val_fit_X = train_X[train_end:]

    feature_sum = np.zeros(train_fit_X.shape[-1], dtype=np.float64)
    feature_sq_sum = np.zeros(train_fit_X.shape[-1], dtype=np.float64)
    feature_count = 0
    for start in range(0, train_end, cfg.ml.batch_size):
        block = np.asarray(train_fit_X[start : start + cfg.ml.batch_size], dtype=np.float32)
        flat_block = block.reshape(-1, block.shape[-1])
        feature_sum += flat_block.sum(axis=0, dtype=np.float64)
        feature_sq_sum += np.square(flat_block, dtype=np.float64).sum(axis=0)
        feature_count += len(flat_block)

    mean_ = feature_sum / max(feature_count, 1)
    variance = feature_sq_sum / max(feature_count, 1) - np.square(mean_)
    raw_std = np.sqrt(np.maximum(variance, 0.0))
    std_ = np.maximum(raw_std, 1e-6)
    mask_ = raw_std < 1e-6

    train_norm = _norm_no_clip(train_fit_X, mean_, std_, mask_)
    val_norm = _norm_no_clip(val_fit_X, mean_, std_, mask_)
    train_norm_clipped = np.clip(train_norm, -8.0, 8.0)
    val_norm_clipped = np.clip(val_norm, -8.0, 8.0)

    print(f"BEFORE_CLIP_TRAIN_MAX={float(np.max(np.abs(train_norm))) if train_norm.size else 0.0}")
    print(f"BEFORE_CLIP_VALID_MAX={float(np.max(np.abs(val_norm))) if val_norm.size else 0.0}")
    print(f"AFTER_CLIP_TRAIN_MAX={float(np.max(np.abs(train_norm_clipped))) if train_norm_clipped.size else 0.0}")
    print(f"AFTER_CLIP_VALID_MAX={float(np.max(np.abs(val_norm_clipped))) if val_norm_clipped.size else 0.0}")
    print(f"CLIPPED_VALUES_TRAIN={int(np.count_nonzero(np.abs(train_norm) > 8.0))}")
    print(f"CLIPPED_VALUES_VALID={int(np.count_nonzero(np.abs(val_norm) > 8.0))}")


def _build_targets(cfg, X, y, t1_pos, mfe, mae, horizon_labels):
    targets_by_horizon = build_targets_by_horizon(X, horizon_labels, HORIZONS)
    return {
        h: {name: values[cfg.ml.sequence_length - 1 :] for name, values in arrays.items()}
        for h, arrays in targets_by_horizon.items()
    }


def _run_fit_case(cfg, X_seq, y_seq, t1_seq, mfe_seq, mae_seq, targets_by_horizon_seq, fold, apply_clip: bool):
    train_idx = fold.train_idx
    train_X = X_seq[train_idx]
    train_y = y_seq[train_idx]
    train_mfe = mfe_seq[train_idx]
    train_mae = mae_seq[train_idx]
    train_targets = {h: {name: values[train_idx] for name, values in arrays.items()} for h, arrays in targets_by_horizon_seq.items()}

    model = TCNTrendModel(cfg.ml, train_X.shape[2], forecast_horizons=HORIZONS, prefer_cuda=False, requested_device="cpu", asset=SYMBOL)
    if not apply_clip:
        def _no_clip_norm(X: np.ndarray) -> np.ndarray:
            if model.mean_ is None or model.std_ is None:
                return np.asarray(X, dtype=np.float32, order="C", copy=False)
            X_norm = np.asarray(X, dtype=np.float32, order="C", copy=True)
            np.subtract(X_norm, model.mean_, out=X_norm)
            np.divide(X_norm, model.std_, out=X_norm)
            if model.constant_feature_mask_ is not None:
                X_norm[..., model.constant_feature_mask_] = 0.0
            return X_norm
        model._normalize = _no_clip_norm
        model._normalize_batch = _no_clip_norm

    stdout = io.StringIO()
    with redirect_stdout(stdout):
        model.fit(
            train_X,
            train_y,
            sample_weight=np.ones(len(train_y), dtype=np.float32),
            mfe_seq=train_mfe,
            mae_seq=train_mae,
            targets_by_horizon=train_targets,
            diagnostic_label=f"{SYMBOL} wf_4 {'NO_CLIP' if not apply_clip else 'CLIP_8'}",
            diagnostic_stop=True,
        )
    text = stdout.getvalue()

    bad = re.search(r"FIRST_BAD_EPOCH=(\d+)", text)
    val = re.search(r"validation_loss=([0-9.eE+-]+)", text)
    trunk = re.search(r"validation_trunk_max=([0-9.eE+-]+)", text)
    raw = re.search(r"validation_max_raw=([0-9.eE+-]+)", text)
    return {
        "first_bad_epoch": int(bad.group(1)) if bad else None,
        "validation_loss": float(val.group(1)) if val else float("nan"),
        "trunk_max": float(trunk.group(1)) if trunk else float("nan"),
        "raw_max": float(raw.group(1)) if raw else float("nan"),
    }


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "training.yaml")
    cfg.ml.max_epochs = 5
    cfg.ml.batch_size = 256
    cfg.ml.val_fraction = 0.15
    cfg.ml.training_device = "cpu"
    cfg.ml.inference_device = "cpu"
    cfg.ml.use_amp = False
    cfg.ml.random_state = 42
    cfg.ml.forecast_horizons = HORIZONS

    raw = load_ml_ohlc(cfg.data, cfg.ml)
    ohlc = raw[SYMBOL]
    X, y, t1_pos, mfe, mae, horizon_labels = _prepare_symbol_dataset(ohlc, None, cfg.ml, forecast_horizons=HORIZONS)
    X_seq, y_seq, t1_seq, index_seq, mfe_seq, mae_seq = _to_sequences(X, y, t1_pos, cfg.ml.sequence_length, mfe, mae)
    limit = min(len(y_seq), 2000)
    X_seq = X_seq[:limit]
    y_seq = y_seq[:limit]
    t1_seq = t1_seq[:limit]
    mfe_seq = mfe_seq[:limit]
    mae_seq = mae_seq[:limit]
    folds = purged_walk_forward_splits(len(y_seq), t1_seq, cfg.ml.n_splits, cfg.ml.embargo_fraction)
    fold = folds[FOLD_INDEX]
    targets_by_horizon_seq = _build_targets(cfg, X, y, t1_pos, mfe, mae, horizon_labels)

    _compute_before_after_norm(cfg, X_seq, y_seq, t1_seq)

    no_clip = _run_fit_case(cfg, X_seq, y_seq, t1_seq, mfe_seq, mae_seq, targets_by_horizon_seq, fold, apply_clip=False)
    clip_8 = _run_fit_case(cfg, X_seq, y_seq, t1_seq, mfe_seq, mae_seq, targets_by_horizon_seq, fold, apply_clip=True)

    print("NO_CLIP:")
    print(f"first_bad_epoch={no_clip['first_bad_epoch']}")
    print(f"validation_loss={no_clip['validation_loss']}")
    print(f"trunk_max={no_clip['trunk_max']}")
    print(f"raw_max={no_clip['raw_max']}")

    print("CLIP_8:")
    print(f"first_bad_epoch={clip_8['first_bad_epoch']}")
    print(f"validation_loss={clip_8['validation_loss']}")
    print(f"trunk_max={clip_8['trunk_max']}")
    print(f"raw_max={clip_8['raw_max']}")


if __name__ == "__main__":
    main()
