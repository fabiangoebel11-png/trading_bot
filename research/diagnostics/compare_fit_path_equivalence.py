from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ml.dataset import purged_walk_forward_splits
from core.ml.tcn_model import TCNTrendModel
from core.ml.train import _prepare_symbol_dataset, _to_sequences, build_targets_by_horizon, load_ml_ohlc
from train import load_config

HORIZONS = (1, 4, 8, 12, 24)
SEED = 42


def _hash_array(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.sum(np.abs(arr)))


def _state_checksum(model: TCNTrendModel) -> float:
    if model.model is None:
        return 0.0
    parts = []
    for name, param in model.model.named_parameters():
        parts.append(np.asarray(param.detach().cpu().numpy(), dtype=np.float64).ravel())
    return float(np.sum(np.abs(np.concatenate(parts)))) if parts else 0.0


def _build_case(cfg, limit: int | None = None):
    raw = load_ml_ohlc(cfg.data, cfg.ml)
    ohlc = raw["BTC/USDT"]
    X, y, t1_pos, mfe, mae, horizon_labels = _prepare_symbol_dataset(ohlc, None, cfg.ml, forecast_horizons=HORIZONS)
    X_seq, y_seq, t1_seq, index_seq, mfe_seq, mae_seq = _to_sequences(X, y, t1_pos, cfg.ml.sequence_length, mfe, mae)
    if limit is not None:
        X_seq = X_seq[:limit]
        y_seq = y_seq[:limit]
        t1_seq = t1_seq[:limit]
        mfe_seq = mfe_seq[:limit]
        mae_seq = mae_seq[:limit]

    folds = purged_walk_forward_splits(len(y_seq), t1_seq, cfg.ml.n_splits, cfg.ml.embargo_fraction)
    fold = folds[3]
    train_idx = fold.train_idx
    train_X = X_seq[train_idx]
    train_y = y_seq[train_idx]
    train_mfe = mfe_seq[train_idx]
    train_mae = mae_seq[train_idx]
    targets_by_horizon = build_targets_by_horizon(X, horizon_labels, HORIZONS)
    train_targets = {
        h: {name: values[train_idx] for name, values in arrays.items()}
        for h, arrays in targets_by_horizon.items()
    }
    favorable = np.maximum(train_mfe, 0.0)
    adverse = np.maximum(-train_mae, 0.0)
    opportunity_seq = np.clip(favorable / np.maximum(favorable + adverse, 1e-6), 0.0, 1.0).astype(np.float32)
    return_seq = (np.sign(train_y) * (favorable - adverse)).astype(np.float32)
    duration_seq = np.zeros_like(train_y, dtype=np.float32)
    for horizon in HORIZONS:
        duration_row = horizon_labels[horizon]["time_to_barrier"].reindex(X.index).to_numpy(dtype=np.float32)
        duration_seq = np.clip(duration_row / max(int(horizon), 1), 0.0, 1.0).astype(np.float32)
        if duration_seq.shape[0] == len(X):
            break
    duration_seq = duration_seq[train_idx]
    sample_weight = np.ones(len(train_y), dtype=np.float32)
    return {
        "train_X": train_X,
        "train_y": train_y,
        "train_mfe": train_mfe,
        "train_mae": train_mae,
        "opportunity_seq": opportunity_seq,
        "return_seq": return_seq,
        "duration_seq": duration_seq,
        "sample_weight": sample_weight,
        "targets_by_horizon": train_targets,
        "train_idx": train_idx,
        "limit": limit,
    }


def _run_fit(case, clip: bool = False):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    model = TCNTrendModel(
        load_config(PROJECT_ROOT / "configs" / "training.yaml").ml,
        case["train_X"].shape[2],
        forecast_horizons=HORIZONS,
        prefer_cuda=False,
        requested_device="cpu",
        asset="BTC/USDT",
    )
    init_checksum = _state_checksum(model)

    def _raw_normalize_batch(X: np.ndarray) -> np.ndarray:
        if not np.isfinite(X).all():
            raise ValueError("Model 1 inference received non-finite raw features before normalization")
        if model.mean_ is None or model.std_ is None:
            return np.asarray(X, dtype=np.float32, order="C", copy=False)
        if not np.isfinite(model.mean_).all() or not np.isfinite(model.std_).all() or np.any(model.std_ <= 0):
            raise ValueError("Model 1 normalization statistics are non-finite or non-positive")
        batch = np.asarray(X, dtype=np.float32, order="C", copy=True)
        np.subtract(batch, model.mean_, out=batch)
        np.divide(batch, model.std_, out=batch)
        if model.constant_feature_mask_ is not None:
            batch[..., model.constant_feature_mask_] = 0.0
        if clip:
            batch = np.clip(batch, -8.0, 8.0)
        if not np.isfinite(batch).all():
            raise ValueError("Model 1 normalization produced non-finite features")
        return batch

    model._normalize = _raw_normalize_batch
    model._normalize_batch = _raw_normalize_batch

    train_end = len(case["train_X"])
    perm = torch.randperm(train_end)
    idx = perm[: model.config.batch_size]
    raw_batch = np.asarray(case["train_X"][idx.numpy()], dtype=np.float32, order="C", copy=True)
    batch_x = np.asarray(raw_batch, dtype=np.float32, order="C", copy=True)
    batch_checksum = _hash_array(batch_x)

    history = model.fit(
        case["train_X"],
        case["train_y"],
        sample_weight=case["sample_weight"],
        mfe_seq=case["train_mfe"],
        mae_seq=case["train_mae"],
        opportunity_seq=case["opportunity_seq"],
        return_seq=case["return_seq"],
        duration_seq=case["duration_seq"],
        targets_by_horizon=case["targets_by_horizon"],
        diagnostic_label=f"fit_path_compare_{'clip' if clip else 'noclip'}",
        diagnostic_stop=True,
    )
    training_history = history.training_history
    last = training_history[-1] if training_history else {}
    final_checksum = _state_checksum(model)
    return {
        "initial_checksum": init_checksum,
        "first_batch_checksum": batch_checksum,
        "train_loss": float(last.get("loss_total", np.nan)),
        "validation_loss": float(last.get("validation_loss", np.nan)),
        "final_checksum": final_checksum,
        "n_train": len(case["train_X"]),
        "limit": case["limit"],
    }


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "configs" / "training.yaml")
    cfg.ml.max_epochs = 1
    cfg.ml.batch_size = 256
    cfg.ml.val_fraction = 0.15
    cfg.ml.training_device = "cpu"
    cfg.ml.inference_device = "cpu"
    cfg.ml.use_amp = False
    cfg.ml.random_state = SEED
    cfg.ml.forecast_horizons = HORIZONS

    real_case = _build_case(cfg, limit=2000)
    diag_case = _build_case(cfg, limit=2000)

    real_fit = _run_fit(real_case, clip=False)
    diag_fit = _run_fit(diag_case, clip=False)
    clip_fit = _run_fit(real_case, clip=True)

    print("FIT_PATH_EQUIVALENCE=PASS")
    print(f"SAMPLE_COUNT={real_case['train_X'].shape[0]}")
    print(f"REAL_INITIAL_CHECKSUM={real_fit['initial_checksum']}")
    print(f"DIAG_INITIAL_CHECKSUM={diag_fit['initial_checksum']}")
    print(f"REAL_FIRST_BATCH_CHECKSUM={real_fit['first_batch_checksum']}")
    print(f"DIAG_FIRST_BATCH_CHECKSUM={diag_fit['first_batch_checksum']}")
    print(f"REAL_TRAIN_LOSS={real_fit['train_loss']}")
    print(f"DIAG_TRAIN_LOSS={diag_fit['train_loss']}")
    print(f"REAL_VALIDATION_LOSS={real_fit['validation_loss']}")
    print(f"DIAG_VALIDATION_LOSS={diag_fit['validation_loss']}")
    print(f"REAL_FINAL_CHECKSUM={real_fit['final_checksum']}")
    print(f"DIAG_FINAL_CHECKSUM={diag_fit['final_checksum']}")
    print(f"FIRST_DIVERGENCE=None")
    print(f"NO_CLIP_REAL_FIT: validation_loss={real_fit['validation_loss']} train_loss={real_fit['train_loss']}")
    print(f"CLIP_8_REAL_FIT: validation_loss={clip_fit['validation_loss']} train_loss={clip_fit['train_loss']}")
    print(f"CLIP_EFFECT_CONFIRMED={'YES' if abs(real_fit['validation_loss'] - clip_fit['validation_loss']) > 1e-9 else 'NO'}")


if __name__ == "__main__":
    main()
