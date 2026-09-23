from __future__ import annotations

import time

import numpy as np
import torch

from core.ml.tcn_model import TCNTrendModel
from core.ml.train import _prepare_symbol_dataset, _to_sequences, load_ml_ohlc
from train import load_config


def test_training_benchmark_smoke_stays_within_safe_vram_and_completes() -> None:
    if not torch.cuda.is_available():
        return

    cfg = load_config('configs/training.yaml')
    cfg.ml.max_epochs = 1
    cfg.ml.sequence_length = 128
    cfg.ml.hidden_channels = 96
    cfg.ml.num_layers = 6
    cfg.ml.dropout = 0.35
    cfg.ml.batch_size = 512
    cfg.ml.val_fraction = 0.15

    raw = load_ml_ohlc(cfg.data, cfg.ml)
    ohlc = raw["BTC/USDT"]
    X, y, t1_pos, mfe, mae = _prepare_symbol_dataset(ohlc, None, cfg.ml)
    X_seq, y_seq, t1_seq, index_seq, mfe_seq, mae_seq = _to_sequences(X, y, t1_pos, cfg.ml.sequence_length, mfe, mae)

    n = min(4000, len(y_seq))
    X_seq = X_seq[:n]
    y_seq = y_seq[:n]
    mfe_seq = mfe_seq[:n]
    mae_seq = mae_seq[:n]

    model = TCNTrendModel(cfg.ml, X_seq.shape[2])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.fit(X_seq, y_seq, mfe_seq=mfe_seq, mae_seq=mae_seq)
    elapsed = time.perf_counter() - t0

    samples_per_sec = n / elapsed
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    assert elapsed > 0.0
    assert samples_per_sec > 750.0
    assert peak_vram_mb < 7000.0
