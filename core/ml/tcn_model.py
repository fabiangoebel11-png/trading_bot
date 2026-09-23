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

Sized for the 1h-native architecture (Session 2026-09-22, "nativ auf 1h
vereinheitlicht"): ``TrendMLConfig`` defaults now use 6 dilated layers
(dilations 1,2,4,...,32 -> ~253h/~10.5-day causal receptive field) and a
128-bar (~5.3-day) input sequence, plus a learned causal attention-pooling
head (below) instead of reading out only the very last timestep -- deeper
context matters more once a "bar" means an hour instead of 5 minutes, and
training compute is a non-issue on the RTX 3070.
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
            # Causal attention pooling over the full sequence instead of only
            # reading out the last timestep: every position in ``y`` already
            # only encodes information up to and including that position (the
            # TCN's causal padding/chomp guarantees this), so a weighted
            # average over all of them is still strictly causal -- it just
            # lets the model learn *which* past regime/lookback horizon
            # (recent breakout vs. multi-day trend context) is most
            # informative for the current prediction instead of hard-coding
            # "only the very last bar matters".
            self.attn = nn.Linear(hidden_channels, 1)
            self.head = nn.Linear(hidden_channels, 3)
            self.opportunity_head = nn.Linear(hidden_channels, 1)
            self.return_head = nn.Linear(hidden_channels, 1)
            self.duration_head = nn.Linear(hidden_channels, 1)
            self.mfe_head = nn.Linear(hidden_channels, 1)
            self.mae_head = nn.Linear(hidden_channels, 1)

        def forward(self, x):  # noqa: ANN001 - x: (batch, seq_len, features)
            x = x.transpose(1, 2)  # -> (batch, features, seq_len) for Conv1d
            y = self.network(x).transpose(1, 2)  # -> (batch, seq_len, hidden)
            attn_scores = self.attn(y).squeeze(-1)  # (batch, seq_len)
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
            pooled = (y * attn_weights).sum(dim=1)  # (batch, hidden)
            return self.head(pooled)

        def forward_outputs(self, x):  # noqa: ANN001 - torch tensor
            x = x.transpose(1, 2)
            y = self.network(x).transpose(1, 2)
            attn_weights = torch.softmax(self.attn(y).squeeze(-1), dim=1).unsqueeze(-1)
            pooled = (y * attn_weights).sum(dim=1)
            return {
                "logits": self.head(pooled),
                "opportunity_logit": self.opportunity_head(pooled).squeeze(-1),
                "expected_return": self.return_head(pooled).squeeze(-1),
                "expected_duration": self.duration_head(pooled).squeeze(-1),
                "expected_mfe": self.mfe_head(pooled).squeeze(-1),
                "expected_mae": self.mae_head(pooled).squeeze(-1),
            }


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
        self.training_history: list[dict[str, float]] = []
        print(f"[TCNTrendModel] device={self.device} ({device_summary(self.device)}), amp={self.use_amp}")

    def _normalize_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted normalization to one contiguous batch without
        materializing a full dataset-wide normalized array.

        The previous implementation created a fresh normalized copy for each
        batch and then wrapped it in ``torch.tensor(...)`` again. The hot loop
        here keeps the data in float32, normalizes the batch in-place on a
        contiguous copy, and transfers it to CUDA with ``torch.from_numpy`` to
        avoid a second duplicate allocation before the kernel launch.
        """
        if not np.isfinite(X).all():
            raise ValueError("Model 1 inference received non-finite raw features before normalization")
        if self.mean_ is None or self.std_ is None:
            return np.asarray(X, dtype=np.float32, order="C", copy=False)
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.std_).all() or np.any(self.std_ <= 0):
            raise ValueError("Model 1 normalization statistics are non-finite or non-positive")
        batch = np.asarray(X, dtype=np.float32, order="C", copy=True)
        np.subtract(batch, self.mean_, out=batch)
        np.divide(batch, self.std_, out=batch)
        if not np.isfinite(batch).all():
            raise ValueError("Model 1 normalization produced non-finite features")
        return batch

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        if not np.isfinite(X).all():
            raise ValueError("Model 1 inference received non-finite raw features before normalization")
        if self.mean_ is None or self.std_ is None:
            return np.asarray(X, dtype=np.float32, order="C", copy=False)
        if not np.isfinite(self.mean_).all() or not np.isfinite(self.std_).all() or np.any(self.std_ <= 0):
            raise ValueError("Model 1 normalization statistics are non-finite or non-positive")
        X_norm = np.asarray(X, dtype=np.float32, order="C", copy=True)
        np.subtract(X_norm, self.mean_, out=X_norm)
        np.divide(X_norm, self.std_, out=X_norm)
        if not np.isfinite(X_norm).all():
            raise ValueError("Model 1 normalization produced non-finite features")
        return X_norm

    @staticmethod
    def _validate_outputs(outputs: dict[str, torch.Tensor], stage: str) -> None:
        for name, value in outputs.items():
            if not torch.isfinite(value).all().item():
                finite = int(torch.isfinite(value).sum().item())
                total = int(value.numel())
                raise ValueError(f"Model 1 {stage} head {name!r} is non-finite ({finite}/{total} finite)")

    def _evaluate_validation_loss(
        self,
        X: np.ndarray,
        y_cls: np.ndarray,
        mfe: np.ndarray,
        mae: np.ndarray,
        opportunity: np.ndarray | None = None,
        expected_return: np.ndarray | None = None,
        duration: np.ndarray | None = None,
    ) -> float:
        """Evaluate validation loss in bounded batches with CPU accumulation."""
        val_loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing, reduction="sum")
        regression_loss_fn = nn.SmoothL1Loss(reduction="sum")
        opportunity_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
        opportunity = np.zeros(len(X), dtype=np.float32) if opportunity is None else opportunity
        expected_return = np.zeros(len(X), dtype=np.float32) if expected_return is None else expected_return
        duration = np.zeros(len(X), dtype=np.float32) if duration is None else duration
        total_loss = 0.0
        total_rows = 0
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(X), self.config.batch_size):
                stop = start + self.config.batch_size
                batch_x = torch.from_numpy(self._normalize_batch(X[start:stop])).to(self.device, non_blocking=self.device.type == "cuda")
                batch_y = torch.from_numpy(y_cls[start:stop].astype(np.int64, copy=False)).to(self.device, non_blocking=self.device.type == "cuda")
                batch_mfe = torch.from_numpy(np.asarray(mfe[start:stop], dtype=np.float32, order="C", copy=False) / self.config.excursion_target_scale).to(self.device, non_blocking=self.device.type == "cuda")
                batch_mae = torch.from_numpy(np.asarray(mae[start:stop], dtype=np.float32, order="C", copy=False) / self.config.excursion_target_scale).to(self.device, non_blocking=self.device.type == "cuda")
                batch_opportunity = torch.from_numpy(np.asarray(opportunity[start:stop], dtype=np.float32, order="C", copy=False)).to(self.device, non_blocking=self.device.type == "cuda")
                batch_return = torch.from_numpy(np.asarray(expected_return[start:stop], dtype=np.float32, order="C", copy=False) / self.config.return_target_scale).to(self.device, non_blocking=self.device.type == "cuda")
                batch_duration = torch.from_numpy(np.asarray(duration[start:stop], dtype=np.float32, order="C", copy=False)).to(self.device, non_blocking=self.device.type == "cuda")
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model.forward_outputs(batch_x)
                    self._validate_outputs(outputs, "validation raw")
                    loss = self.config.direction_loss_weight * val_loss_fn(outputs["logits"], batch_y)
                    if "opportunity_logit" in outputs:
                        if self.config.entry_quality_mode == "payoff_volatility_compression":
                            loss = loss + self.config.entry_quality_loss_weight * regression_loss_fn(torch.sigmoid(outputs["opportunity_logit"]), batch_opportunity)
                        else:
                            loss = loss + self.config.opportunity_loss_weight * opportunity_loss_fn(outputs["opportunity_logit"], batch_opportunity)
                    if "expected_return" in outputs:
                        loss = loss + self.config.return_loss_weight * regression_loss_fn(outputs["expected_return"], batch_return)
                    if "expected_duration" in outputs:
                        loss = loss + self.config.duration_loss_weight * regression_loss_fn(outputs["expected_duration"], batch_duration)
                    loss = loss + self.config.mfe_loss_weight * regression_loss_fn(outputs["expected_mfe"], batch_mfe)
                    loss = loss + self.config.mae_loss_weight * regression_loss_fn(outputs["expected_mae"], batch_mae)
                total_loss += float(loss.float().cpu())
                total_rows += len(batch_x)
        return total_loss / max(total_rows, 1)

    def fit(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        val_fraction: float | None = None,
        sample_weight: np.ndarray | None = None,
        mfe_seq: np.ndarray | None = None,
        mae_seq: np.ndarray | None = None,
        opportunity_seq: np.ndarray | None = None,
        return_seq: np.ndarray | None = None,
        duration_seq: np.ndarray | None = None,
    ) -> "TCNTrendModel":
        """``y_seq`` must contain values in {-1, 0, 1}. The tail
        ``val_fraction`` of the (chronologically ordered) input is held out,
        with a ``sequence_length``-wide gap before it, for early stopping --
        never used for gradient updates. ``sample_weight`` (same length as
        ``X_seq``/``y_seq``, e.g. from ``core.ml.dataset.
        time_decay_sample_weights``) weights each row's contribution to the
        *training* loss only -- the held-out early-stopping validation loss
        stays unweighted, since it should reflect plain, undistorted recent
        performance, not a re-weighted proxy of it.
        """
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

        if sample_weight is None:
            train_w = np.ones(train_end, dtype=np.float32)
        else:
            train_w = sample_weight[:train_end].astype(np.float32)

        mfe_seq = np.zeros(n, dtype=np.float32) if mfe_seq is None else np.asarray(mfe_seq, dtype=np.float32)
        mae_seq = np.zeros(n, dtype=np.float32) if mae_seq is None else np.asarray(mae_seq, dtype=np.float32)
        train_mfe, val_mfe = mfe_seq[:train_end], mfe_seq[train_end + gap :]
        train_mae, val_mae = mae_seq[:train_end], mae_seq[train_end + gap :]

        # Fit standardization on the training rows only (last timestep of each
        # sequence would double-count overlaps; using all rows of all training
        # sequences is a reasonable, mildly conservative approximation).
        opportunity_seq = np.zeros(n, dtype=np.float32) if opportunity_seq is None else np.asarray(opportunity_seq, dtype=np.float32)
        return_seq = np.zeros(n, dtype=np.float32) if return_seq is None else np.asarray(return_seq, dtype=np.float32)
        duration_seq = np.zeros(n, dtype=np.float32) if duration_seq is None else np.asarray(duration_seq, dtype=np.float32)
        train_opportunity, val_opportunity = opportunity_seq[:train_end], opportunity_seq[train_end + gap :]
        train_return, val_return = return_seq[:train_end], return_seq[train_end + gap :]
        train_duration, val_duration = duration_seq[:train_end], duration_seq[train_end + gap :]
        feature_sum = np.zeros(train_X.shape[-1], dtype=np.float64)
        feature_sq_sum = np.zeros(train_X.shape[-1], dtype=np.float64)
        feature_count = 0
        for start in range(0, train_end, cfg.batch_size):
            block = np.asarray(train_X[start : start + cfg.batch_size], dtype=np.float32)
            flat_block = block.reshape(-1, block.shape[-1])
            feature_sum += flat_block.sum(axis=0, dtype=np.float64)
            feature_sq_sum += np.square(flat_block, dtype=np.float64).sum(axis=0)
            feature_count += len(flat_block)
        self.mean_ = feature_sum / max(feature_count, 1)
        variance = feature_sq_sum / max(feature_count, 1) - np.square(self.mean_)
        self.std_ = np.maximum(np.sqrt(np.maximum(variance, 0.0)), 1e-6)

        val_X_norm = val_X
        train_y_cls = (train_y + 1).astype(np.int64)  # {-1,0,1} -> {0,1,2}
        val_y_cls = (val_y + 1).astype(np.int64)

        self.model = _TCN(self.n_features, cfg.hidden_channels, cfg.num_layers, cfg.dropout).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        # reduction="none" so each row's loss can be scaled by its time-decay
        # sample weight before averaging; the validation criterion below stays
        # a plain (unweighted) mean.
        class_weight = None
        if cfg.class_weight_mode == "balanced":
            counts = np.bincount(train_y_cls, minlength=3).astype(np.float32)
            class_weight = np.where(counts > 0, len(train_y_cls) / (3.0 * counts), 0.0)
            class_weight = torch.tensor(class_weight, dtype=torch.float32, device=self.device)
        train_loss_fn = nn.CrossEntropyLoss(
            weight=class_weight, label_smoothing=cfg.label_smoothing, reduction="none"
        )
        val_loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        regression_loss_fn = nn.SmoothL1Loss(reduction="none")
        opportunity_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)

        y_train_t = torch.from_numpy(train_y_cls.astype(np.int64, copy=False))
        mfe_train_t = torch.from_numpy(np.asarray(train_mfe, dtype=np.float32, order="C", copy=False) / cfg.excursion_target_scale)
        mae_train_t = torch.from_numpy(np.asarray(train_mae, dtype=np.float32, order="C", copy=False) / cfg.excursion_target_scale)
        opportunity_train_t = torch.from_numpy(np.asarray(train_opportunity, dtype=np.float32, order="C", copy=False))
        return_train_t = torch.from_numpy(np.asarray(train_return, dtype=np.float32, order="C", copy=False) / cfg.return_target_scale)
        duration_train_t = torch.from_numpy(np.asarray(train_duration, dtype=np.float32, order="C", copy=False))
        w_train_t = torch.from_numpy(np.asarray(train_w, dtype=np.float32, order="C", copy=False))

        best_val_loss = float("inf")
        best_state = None
        patience_left = cfg.early_stopping_patience
        batch_size = cfg.batch_size
        self.training_history = []

        for _epoch in range(cfg.max_epochs):
            self.model.train()
            permutation = torch.randperm(train_end)
            component_sums = {name: 0.0 for name in ("opportunity", "return", "duration", "mfe", "mae", "direction", "total")}
            batches = 0
            for start in range(0, train_end, batch_size):
                idx = permutation[start : start + batch_size]
                batch_x = torch.from_numpy(self._normalize_batch(train_X[idx.numpy()])).to(self.device, non_blocking=self.device.type == "cuda")
                batch_y = y_train_t[idx].to(self.device, non_blocking=self.device.type == "cuda")
                batch_w = w_train_t[idx].to(self.device, non_blocking=self.device.type == "cuda")

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model.forward_outputs(batch_x)
                    self._validate_outputs(outputs, "training raw")
                    direction_loss = train_loss_fn(outputs["logits"], batch_y)
                    if cfg.class_weight_mode == "balanced_focal":
                        unweighted_direction_loss = nn.functional.cross_entropy(
                            outputs["logits"], batch_y, label_smoothing=cfg.label_smoothing, reduction="none"
                        )
                        direction_probability = torch.exp(-unweighted_direction_loss.detach()).clamp_min(1e-6)
                        direction_loss = ((1.0 - direction_probability) ** cfg.focal_gamma) * unweighted_direction_loss
                        if class_weight is not None:
                            direction_loss = direction_loss * class_weight[batch_y]
                    if cfg.entry_quality_mode == "payoff_volatility_compression":
                        opportunity_loss = regression_loss_fn(torch.sigmoid(outputs["opportunity_logit"]), opportunity_train_t[idx].to(self.device))
                        opportunity_weight = cfg.entry_quality_loss_weight
                    else:
                        opportunity_loss = opportunity_loss_fn(outputs["opportunity_logit"], opportunity_train_t[idx].to(self.device))
                        opportunity_weight = cfg.opportunity_loss_weight
                    return_loss = regression_loss_fn(outputs["expected_return"], return_train_t[idx].to(self.device))
                    duration_loss = regression_loss_fn(outputs["expected_duration"], duration_train_t[idx].to(self.device))
                    mfe_loss = regression_loss_fn(outputs["expected_mfe"], mfe_train_t[idx].to(self.device))
                    mae_loss = regression_loss_fn(outputs["expected_mae"], mae_train_t[idx].to(self.device))
                    total_loss = (cfg.direction_loss_weight * direction_loss + opportunity_weight * opportunity_loss + cfg.return_loss_weight * return_loss + cfg.duration_loss_weight * duration_loss + cfg.mfe_loss_weight * mfe_loss + cfg.mae_loss_weight * mae_loss)
                    loss = (total_loss * batch_w).sum() / batch_w.sum().clamp_min(1e-8)
                component_sums["opportunity"] += float((opportunity_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                component_sums["return"] += float((return_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                component_sums["duration"] += float((duration_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                component_sums["mfe"] += float((mfe_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                component_sums["mae"] += float((mae_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                component_sums["direction"] += float((direction_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                component_sums["total"] += float(loss.detach().float().cpu())
                batches += 1
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            val_loss = self._evaluate_validation_loss(val_X_norm, val_y_cls, val_mfe, val_mae, val_opportunity, val_return, val_duration)
            history_row = {f"loss_{name}": value / max(batches, 1) for name, value in component_sums.items()}
            history_row["validation_loss"] = float(val_loss)
            self.training_history.append(history_row)
            print("[Model1] " + " ".join(f"{key}={value:.6f}" for key, value in history_row.items()))

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
        """Returns classifier probabilities in order [down, flat, up].

        The auxiliary signed-return head remains available through
        ``predict_trade_outputs``. It must not replace the trained classifier
        during decoding because that made a continuous market drift look like
        a collapsed class prediction.
        """
        if not self._is_fitted or X_seq.shape[0] == 0:
            return np.full((X_seq.shape[0], 3), 1.0 / 3.0, dtype=np.float32)

        X_norm = self._normalize(X_seq)
        x_tensor = torch.from_numpy(np.asarray(X_norm, dtype=np.float32, order="C", copy=False))
        batch_size = self.config.batch_size

        self.model.eval()
        outputs = []
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            for start in range(0, x_tensor.shape[0], batch_size):
                batch = x_tensor[start : start + batch_size].to(self.device, non_blocking=self.device.type == "cuda")
                raw = self.model.forward_outputs(batch)
                self._validate_outputs(raw, "probability raw")
                probabilities = torch.softmax(raw["logits"].float(), dim=-1)
                if not torch.isfinite(probabilities).all().item():
                    raise ValueError("Model 1 probability conversion produced non-finite probabilities")
                outputs.append(probabilities.cpu().numpy())
        return np.concatenate(outputs)

    def predict_trade_outputs(self, X_seq: np.ndarray) -> dict[str, np.ndarray]:
        """Return probabilities, expected MFE, and expected MAE by name."""
        if not self._is_fitted or X_seq.shape[0] == 0:
            n = X_seq.shape[0]
            return {
                "probabilities": np.full((n, 3), 1.0 / 3.0, dtype=np.float32),
                "opportunity": np.full(n, 0.5, dtype=np.float32),
                "expected_return": np.zeros(n, dtype=np.float32),
                "expected_duration": np.zeros(n, dtype=np.float32),
                "expected_mfe": np.zeros(n, dtype=np.float32),
                "expected_mae": np.zeros(n, dtype=np.float32),
            }
        X_norm = self._normalize(X_seq)
        x_tensor = torch.from_numpy(np.asarray(X_norm, dtype=np.float32, order="C", copy=False))
        probabilities, opportunities, returns, durations, mfes, maes = [], [], [], [], [], []
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            for start in range(0, len(x_tensor), self.config.batch_size):
                batch = x_tensor[start : start + self.config.batch_size].to(self.device, non_blocking=self.device.type == "cuda")
                outputs = self.model.forward_outputs(batch)
                self._validate_outputs(outputs, "prediction raw")
                probabilities.append(torch.softmax(outputs["logits"].float(), dim=-1).cpu().numpy())
                opportunities.append(torch.sigmoid(outputs["opportunity_logit"].float()).cpu().numpy())
                returns.append((outputs["expected_return"].float() * self.config.return_target_scale).cpu().numpy())
                durations.append(torch.sigmoid(outputs["expected_duration"].float()).cpu().numpy())
                mfes.append((outputs["expected_mfe"].float() * self.config.excursion_target_scale).cpu().numpy())
                maes.append((outputs["expected_mae"].float() * self.config.excursion_target_scale).cpu().numpy())
        result = {
            "probabilities": np.concatenate(probabilities),
            "opportunity": np.concatenate(opportunities),
            "expected_return": np.concatenate(returns),
            "expected_duration": np.concatenate(durations),
            "expected_mfe": np.concatenate(mfes),
            "expected_mae": np.concatenate(maes),
        }
        result["entry_score"] = result["opportunity"] * 100.0
        for name, value in result.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Model 1 prediction conversion produced non-finite output: {name}")
        return result

    def state_dict(self) -> dict:
        return {
            "model_state": self.model.state_dict() if self.model is not None else None,
            "mean": self.mean_,
            "std": self.std_,
            "n_features": self.n_features,
            "is_fitted": self._is_fitted,
            "training_history": self.training_history,
        }

    def load_state_dict(self, state: dict) -> "TCNTrendModel":
        cfg = self.config
        self.mean_ = state["mean"]
        self.std_ = state["std"]
        self.n_features = state["n_features"]
        self._is_fitted = state["is_fitted"]
        self.training_history = state.get("training_history", [])
        if state["model_state"] is not None:
            self.model = _TCN(self.n_features, cfg.hidden_channels, cfg.num_layers, cfg.dropout).to(self.device)
            # Older checkpoints contain only the classifier head. Loading them
            # non-strictly preserves backward compatibility; new regression
            # heads are then available for retraining.
            self.model.load_state_dict(state["model_state"], strict=False)
        return self
