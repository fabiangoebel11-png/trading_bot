"""Regularized dilated causal CNN (TCN) trend-continuation classifier, trained
on the RTX 3070 eGPU.

Chosen over a plain LSTM (the existing ``core/ml/lstm_model.py``, retired
reversion gate) for three reasons: dilated causal convolutions have a fixed,
inspectable receptive field (no vanishing-gradient risk over long sequences),
train faster per epoch on a GPU (fully parallel over the time axis, unlike a
recurrent scan), and weight-normalized conv filters are easy to regularize
with dropout + weight decay -- exactly the "don't memorize the noise" goal.

3-class softmax output (down / flat / up, matching the triple-barrier label),
trained with label smoothing + AdamW weight decay + gradient clipping + early
stopping on a held-out validation slice -- every one of these is an explicit
overfitting guard, not incidental.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from core.config import TrendMLConfig
from core.ml.device import device_summary, get_device

try:
    import torch
    from torch import nn
    from torch.nn.utils.parametrizations import weight_norm
except ImportError as exc:  # pragma: no cover - exercised only without the "ml" extra
    torch = None
    nn = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for the TCN trend model. Install it with "
            "`uv sync --extra ml` (uses the RTX 3070 eGPU via CUDA if available)."
        ) from _IMPORT_ERROR


if nn is not None:

    class _Chomp1d(nn.Module):
        """Removes the trailing padding a same-padded dilated conv leaves on the
        right, which is what makes the convolution strictly causal."""

        def __init__(self, chomp_size: int) -> None:
            super().__init__()
            self.chomp_size = chomp_size

        def forward(self, x):  # noqa: ANN001 - torch tensor
            return x[:, :, : -self.chomp_size] if self.chomp_size > 0 else x

    class _TemporalBlock(nn.Module):
        def __init__(self, n_in: int, n_out: int, kernel_size: int, dilation: int, dropout: float) -> None:
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.net = nn.Sequential(
                weight_norm(nn.Conv1d(n_in, n_out, kernel_size, padding=padding, dilation=dilation)),
                _Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout),
                weight_norm(nn.Conv1d(n_out, n_out, kernel_size, padding=padding, dilation=dilation)),
                _Chomp1d(padding),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
            self.relu = nn.ReLU()

        def forward(self, x):  # noqa: ANN001 - torch tensor
            out = self.net(x)
            res = x if self.downsample is None else self.downsample(x)
            return self.relu(out + res)

    class _TCN(nn.Module):
        def __init__(self, n_features: int, hidden_channels: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            blocks = []
            for i in range(num_layers):
                in_ch = n_features if i == 0 else hidden_channels
                blocks.append(_TemporalBlock(in_ch, hidden_channels, kernel_size=3, dilation=2**i, dropout=dropout))
            self.network = nn.Sequential(*blocks)
            self.head = nn.Linear(hidden_channels, 3)

        def forward(self, x):  # noqa: ANN001 - x: (batch, seq_len, features)
            x = x.transpose(1, 2)  # -> (batch, features, seq_len) for Conv1d
            y = self.network(x)
            last_step = y[:, :, -1]  # causal: last timestep already saw the full receptive field
            return self.head(last_step)


class TCNTrendModel:
    """Sklearn-ish wrapper: ``fit(X_seq, y_seq)`` / ``predict_proba(X_seq)`` on
    already-built (n, seq_len, n_features) float32 sequence arrays (see
    ``core/ml/dataset.py:make_sequences``), plus feature standardization fit
    only on the training split."""

    def __init__(self, config: TrendMLConfig, n_features: int) -> None:
        _require_torch()
        self.config = config
        self.n_features = n_features
        self.device = get_device()
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.model: _TCN | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self._is_fitted = False
        print(f"[TCNTrendModel] device={self.device} ({device_summary(self.device)}), amp={self.use_amp}")

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        val_fraction: float | None = None,
    ) -> "TCNTrendModel":
        """``y_seq`` must contain values in {-1, 0, 1}. The tail
        ``val_fraction`` of the (chronologically ordered) input is held out,
        with a ``sequence_length``-wide gap before it, for early stopping --
        never used for gradient updates."""
        cfg = self.config
        n = X_seq.shape[0]
        if n < 50 or len(np.unique(y_seq)) < 2:
            self._is_fitted = False
            return self

        val_fraction = cfg.val_fraction if val_fraction is None else val_fraction
        val_size = max(1, int(n * val_fraction))
        gap = min(cfg.sequence_length, max(0, n - val_size - 1))
        train_end = max(1, n - val_size - gap)

        train_X, train_y = X_seq[:train_end], y_seq[:train_end]
        val_X, val_y = X_seq[train_end + gap :], y_seq[train_end + gap :]
        if len(val_X) == 0:
            val_X, val_y = train_X[-val_size:], train_y[-val_size:]

        # Fit standardization on the training rows only (last timestep of each
        # sequence would double-count overlaps; using all rows of all training
        # sequences is a reasonable, mildly conservative approximation).
        flat_train = train_X.reshape(-1, train_X.shape[-1])
        self.mean_ = flat_train.mean(axis=0)
        self.std_ = flat_train.std(axis=0)
        self.std_[self.std_ == 0] = 1.0

        train_X_norm = self._normalize(train_X).astype(np.float32)
        val_X_norm = self._normalize(val_X).astype(np.float32)
        train_y_cls = (train_y + 1).astype(np.int64)  # {-1,0,1} -> {0,1,2}
        val_y_cls = (val_y + 1).astype(np.int64)

        self.model = _TCN(self.n_features, cfg.hidden_channels, cfg.num_layers, cfg.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)

        x_train_t = torch.tensor(train_X_norm)
        y_train_t = torch.tensor(train_y_cls)
        x_val_t = torch.tensor(val_X_norm, device=self.device)
        y_val_t = torch.tensor(val_y_cls, device=self.device)

        best_val_loss = float("inf")
        best_state = None
        patience_left = cfg.early_stopping_patience
        batch_size = cfg.batch_size

        for _epoch in range(cfg.max_epochs):
            self.model.train()
            permutation = torch.randperm(x_train_t.shape[0])
            for start in range(0, x_train_t.shape[0], batch_size):
                idx = permutation[start : start + batch_size]
                batch_x = x_train_t[idx].to(self.device, non_blocking=True)
                batch_y = y_train_t[idx].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(batch_x)
                    loss = loss_fn(logits, batch_y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            self.model.eval()
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                val_logits = self.model(x_val_t)
                val_loss = loss_fn(val_logits, y_val_t).item()

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_left = cfg.early_stopping_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self._is_fitted = True
        return self

    def predict_proba(self, X_seq: np.ndarray) -> np.ndarray:
        """Returns (n, 3) class probabilities in order [down, flat, up]."""
        if not self._is_fitted or X_seq.shape[0] == 0:
            return np.full((X_seq.shape[0], 3), 1.0 / 3.0, dtype=np.float32)

        X_norm = self._normalize(X_seq).astype(np.float32)
        x_tensor = torch.tensor(X_norm)
        batch_size = self.config.batch_size

        self.model.eval()
        outputs = []
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            for start in range(0, x_tensor.shape[0], batch_size):
                batch = x_tensor[start : start + batch_size].to(self.device, non_blocking=True)
                logits = self.model(batch)
                outputs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        return np.concatenate(outputs)

    def state_dict(self) -> dict:
        return {
            "model_state": self.model.state_dict() if self.model is not None else None,
            "mean": self.mean_,
            "std": self.std_,
            "n_features": self.n_features,
            "is_fitted": self._is_fitted,
        }

    def load_state_dict(self, state: dict) -> "TCNTrendModel":
        cfg = self.config
        self.mean_ = state["mean"]
        self.std_ = state["std"]
        self.n_features = state["n_features"]
        self._is_fitted = state["is_fitted"]
        if state["model_state"] is not None:
            self.model = _TCN(self.n_features, cfg.hidden_channels, cfg.num_layers, cfg.dropout).to(self.device)
            self.model.load_state_dict(state["model_state"])
        return self
