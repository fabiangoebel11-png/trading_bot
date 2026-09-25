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
from core.foundation_models import BaseForecastModel
from core.ml.device import device_summary, resolve_device

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

    class _HorizonHead(nn.Module):
        def __init__(self, hidden_channels: int) -> None:
            super().__init__()
            self.direction_head = nn.Linear(hidden_channels, 3)
            self.opportunity_head = nn.Linear(hidden_channels, 1)
            self.return_head = nn.Linear(hidden_channels, 1)
            self.duration_head = nn.Linear(hidden_channels, 1)
            self.mfe_head = nn.Linear(hidden_channels, 1)
            self.mae_head = nn.Linear(hidden_channels, 1)

    class _TCN(nn.Module):
        def __init__(self, n_features: int, hidden_channels: int, num_layers: int, dropout: float, forecast_horizons: tuple[int, ...] = ()) -> None:
            super().__init__()
            blocks = []
            for i in range(num_layers):
                in_ch = n_features if i == 0 else hidden_channels
                blocks.append(_TemporalBlock(in_ch, hidden_channels, kernel_size=3, dilation=2**i, dropout=dropout))
            self.network = nn.Sequential(*blocks)
            self.attn = nn.Linear(hidden_channels, 1)
            self.forecast_horizons = tuple(int(h) for h in forecast_horizons)
            self.heads = nn.ModuleDict({str(h): _HorizonHead(hidden_channels) for h in self.forecast_horizons})
            if not self.forecast_horizons:
                self.heads[str(0)] = _HorizonHead(hidden_channels)

        def forward(self, x):  # noqa: ANN001 - x: (batch, seq_len, features)
            x = x.transpose(1, 2)
            y = self.network(x).transpose(1, 2)
            attn_scores = self.attn(y).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
            pooled = (y * attn_weights).sum(dim=1)
            return {str(h): head.direction_head(pooled) for h, head in self.heads.items()}

        def forward_outputs(self, x):  # noqa: ANN001 - torch tensor
            x = x.transpose(1, 2)
            y = self.network(x).transpose(1, 2)
            attn_weights = torch.softmax(self.attn(y).squeeze(-1), dim=1).unsqueeze(-1)
            pooled = (y * attn_weights).sum(dim=1)
            outputs: dict[int, dict[str, torch.Tensor]] = {}
            for horizon_key, head in self.heads.items():
                horizon = int(horizon_key)
                outputs[horizon] = {
                    "logits": head.direction_head(pooled),
                    "opportunity_logit": head.opportunity_head(pooled).squeeze(-1),
                    "expected_opportunity": head.opportunity_head(pooled).squeeze(-1),
                    "expected_return": head.return_head(pooled).squeeze(-1),
                    # Duration targets are non-negative but standardized by a
                    # positive per-horizon scale, so Softplus preserves the
                    # required [0, inf) range without imposing an upper bound.
                    "expected_duration": torch.nn.functional.softplus(head.duration_head(pooled).squeeze(-1)),
                    "expected_mfe": head.mfe_head(pooled).squeeze(-1),
                    "expected_mae": head.mae_head(pooled).squeeze(-1),
                }
            return outputs


class TCNTrendModel:
    """Sklearn-ish wrapper: ``fit(X_seq, y_seq)`` / ``predict_proba(X_seq)`` on
    already-built (n, seq_len, n_features) float32 sequence arrays (see
    ``core/ml/dataset.py:make_sequences``), plus feature standardization fit
    only on the training split."""

    def __init__(self, config: TrendMLConfig, n_features: int, forecast_horizons: tuple[int, ...] | None = None, prefer_cuda: bool = True, requested_device: str | None = None, asset: str | None = None) -> None:
        _require_torch()
        self.config = config
        self.asset = asset or getattr(config, "asset", config.model_id)
        self.n_features = n_features
        requested = requested_device
        if requested is None:
            requested = getattr(config, "training_device", None)
            if requested is None:
                requested = "cuda" if prefer_cuda else "cpu"
            elif prefer_cuda is False and requested == "cuda":
                requested = "cpu"
        elif prefer_cuda is False and requested == "cuda":
            requested = "cpu"
        self.device = resolve_device(requested, prefer_cuda=prefer_cuda)
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.model: _TCN | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.constant_feature_mask_: np.ndarray | None = None
        self.return_target_scale = float(getattr(config, "return_target_scale", 1.0) or 1.0)
        self.mfe_target_scale = float(getattr(config, "excursion_target_scale", 1.0) or 1.0)
        self.mae_target_scale = float(getattr(config, "excursion_target_scale", 1.0) or 1.0)
        self.duration_target_scale = 1.0
        self.duration_head_active = True
        self.horizon_target_scales: dict[int, dict[str, float]] = {}
        self.horizon_class_weights: dict[int, dict[int, float]] = {}
        self.horizon_duration_active: dict[int, bool] = {}
        self.class_weights_: dict[int, float] = { -1: 1.0, 0: 1.0, 1: 1.0 }
        self.forecast_horizons = tuple(int(h) for h in (forecast_horizons if forecast_horizons is not None else getattr(config, "forecast_horizons", (int(config.label_horizon),))))
        self._is_fitted = False
        self.training_history: list[dict[str, float]] = []
        print(f"[TCNTrendModel] device={self.device} ({device_summary(self.device)}), amp={self.use_amp}, horizons={self.forecast_horizons}")

    @staticmethod
    def _compute_target_scale(values: np.ndarray, *, floor: float = 1e-3) -> float:
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float(floor)
        scale = float(np.percentile(np.abs(finite), 95.0))
        if not np.isfinite(scale) or scale <= floor:
            scale = float(np.std(finite)) if finite.size > 1 else float(floor)
        return max(scale, float(floor))

    @staticmethod
    def _compute_class_weights(labels: np.ndarray, *, max_weight: float = 5.0, min_weight: float = 0.25) -> dict[int, float]:
        labels = np.asarray(labels, dtype=int).ravel()
        if labels.size == 0:
            return {-1: 1.0, 0: 1.0, 1: 1.0}
        counts = np.bincount(labels + 1, minlength=3).astype(np.float64)
        valid = counts > 0
        if not valid.any():
            return {-1: 1.0, 0: 1.0, 1: 1.0}
        mean_count = float(counts[valid].mean())
        raw = np.divide(mean_count, counts, out=np.zeros_like(counts, dtype=np.float64), where=valid)
        clipped = np.clip(raw, min_weight, max_weight)
        return { -1: float(clipped[0]), 0: float(clipped[1]), 1: float(clipped[2]) }

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
        if self.constant_feature_mask_ is not None:
            batch[..., self.constant_feature_mask_] = 0.0
        batch = np.clip(batch, -8.0, 8.0)
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
        if self.constant_feature_mask_ is not None:
            X_norm[..., self.constant_feature_mask_] = 0.0
        X_norm = np.clip(X_norm, -8.0, 8.0)
        if not np.isfinite(X_norm).all():
            raise ValueError("Model 1 normalization produced non-finite features")
        return X_norm

    @staticmethod
    @staticmethod
    def _finite_summary(values: torch.Tensor) -> tuple[float, float, float]:
        finite = torch.isfinite(values)
        if finite.any().item():
            finite_values = values[finite]
            return float(finite_values.min().item()), float(finite_values.max().item()), float(finite_values.mean().item())
        return float("nan"), float("nan"), float("nan")

    def _assert_model_finite(self, stage: str, epoch: int | None = None, batch: int | None = None, loss: dict[str, float] | None = None, grad_norm: float | None = None) -> None:
        if self.model is None:
            return
        bad_parameter = None
        bad_min = bad_max = bad_mean = None
        for name, param in self.model.named_parameters():
            param_is_finite = torch.isfinite(param).all().item()
            grad_is_finite = param.grad is None or torch.isfinite(param.grad).all().item()
            if not param_is_finite or not grad_is_finite:
                bad_parameter = name
                bad_min, bad_max, bad_mean = self._finite_summary(param)
                break
        if bad_parameter is not None:
            raise ValueError(
                "TRAINING NUMERIC FAILURE: non-finite parameter detected "
                f"epoch={epoch} batch={batch} stage={stage} "
                f"loss={loss} grad_norm={grad_norm} parameter={bad_parameter} "
                f"min={bad_min} max={bad_max} mean={bad_mean} "
                f"amp={self.use_amp} device={self.device}"
            )

    def _assert_loss_finite(self, losses: dict[str, torch.Tensor], stage: str, epoch: int | None = None, batch: int | None = None, grad_norm: float | None = None) -> None:
        for name, value in losses.items():
            if value is None:
                continue
            if not torch.isfinite(value).all().item():
                vmin, vmax, vmean = self._finite_summary(value)
                raise ValueError(
                    "TRAINING NUMERIC FAILURE: non-finite loss detected "
                    f"epoch={epoch} batch={batch} stage={stage} head={name} "
                    f"loss={float(value.detach().float().cpu().item()) if value.numel() == 1 else 'non_scalar'} "
                    f"grad_norm={grad_norm} min={vmin} max={vmax} mean={vmean} "
                    f"amp={self.use_amp} device={self.device}"
                )

    @staticmethod
    def _describe_target_stats(name: str, values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            print(f"[TCN target stats] {name}: all_non_finite")
            return
        q = np.quantile(finite, [0.01, 0.5, 0.99])
        print(f"[TCN target stats] {name}: min={finite.min():.6g} max={finite.max():.6g} mean={finite.mean():.6g} std={finite.std():.6g} p1={q[0]:.6g} p50={q[1]:.6g} p99={q[2]:.6g}")

    def _resolve_head_outputs(self, outputs: dict[str, torch.Tensor] | dict[int, dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not isinstance(outputs, dict):
            raise TypeError(f"Expected output mapping, got {type(outputs).__name__}")
        if outputs and all(isinstance(value, dict) for value in outputs.values()):
            primary_horizon = self.forecast_horizons[0] if self.forecast_horizons else next(iter(outputs))
            if primary_horizon in outputs:
                return outputs[primary_horizon]
            return next(iter(outputs.values()))
        return outputs

    def _validate_outputs(self, outputs: dict[str, torch.Tensor] | dict[int, dict[str, torch.Tensor]], stage: str) -> None:
        if isinstance(outputs, dict) and outputs and all(isinstance(v, dict) for v in outputs.values()):
            for horizon, inner in outputs.items():
                self._validate_outputs(inner, f"{stage}[horizon={horizon}]")
            return
        for name, value in outputs.items():
            if not torch.isfinite(value).all().item():
                finite = int(torch.isfinite(value).sum().item())
                total = int(value.numel())
                raise ValueError(f"Model 1 {stage} head {name!r} is non-finite ({finite}/{total} finite)")

    def _apply_early_stopping(
        self,
        *,
        current_state: dict | None,
        best_val_loss: float,
        val_loss: float,
        best_state: dict | None,
        epochs_without_improvement: int,
    ) -> tuple[float, dict | None, int, bool, bool]:
        if best_state is None or val_loss < best_val_loss - 1e-8:
            updated_best = True
            best_val_loss = float(val_loss)
            best_state = copy.deepcopy(current_state) if current_state is not None else None
            epochs_without_improvement = 0
        else:
            updated_best = False
            epochs_without_improvement += 1
        should_stop = epochs_without_improvement >= self.config.early_stopping_patience
        return best_val_loss, best_state, epochs_without_improvement, updated_best, should_stop

    def _evaluate_validation_loss(
        self,
        X: np.ndarray,
        y_cls: np.ndarray,
        mfe: np.ndarray,
        mae: np.ndarray,
        opportunity: np.ndarray | None = None,
        expected_return: np.ndarray | None = None,
        duration: np.ndarray | None = None,
        horizon: int | None = None,
        collect_diagnostics: bool = False,
    ) -> float:
        """Evaluate validation loss in bounded batches with per-sample normalization.

        The loss is intentionally averaged over samples and over the active head
        weights so that the validation metric remains comparable across different
        batch sizes and target scales instead of exploding with raw target units.
        """
        val_loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing, reduction="none")
        regression_loss_fn = nn.SmoothL1Loss(reduction="none")
        opportunity_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        opportunity = np.zeros(len(X), dtype=np.float32) if opportunity is None else opportunity
        expected_return = np.zeros(len(X), dtype=np.float32) if expected_return is None else expected_return
        duration = np.zeros(len(X), dtype=np.float32) if duration is None else duration
        head_losses: dict[str, float] = {}
        raw_ranges: dict[str, list[float]] = {}
        trunk_max = 0.0
        trunk_std_sum = 0.0
        trunk_batches = 0
        total_rows = 0
        evaluation_horizon = int(horizon) if horizon is not None else (self.forecast_horizons[0] if self.forecast_horizons else 0)
        scales = self.horizon_target_scales.get(evaluation_horizon, {})
        return_scale = float(scales.get("return", self.return_target_scale))
        mfe_scale = float(scales.get("mfe", self.mfe_target_scale))
        mae_scale = float(scales.get("mae", self.mae_target_scale))
        duration_scale = float(scales.get("duration", self.duration_target_scale))
        duration_active = self.horizon_duration_active.get(evaluation_horizon, self.duration_head_active)
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(X), self.config.batch_size):
                stop = start + self.config.batch_size
                batch_x = torch.from_numpy(self._normalize_batch(X[start:stop])).to(self.device, non_blocking=self.device.type == "cuda")
                batch_y_values = np.asarray(y_cls[start:stop], dtype=np.float64)
                if np.nanmin(batch_y_values) >= 0.0 and np.nanmax(batch_y_values) <= 2.0:
                    batch_y_encoded = batch_y_values.astype(np.int64, copy=False)
                else:
                    batch_y_encoded = (batch_y_values + 1.0).astype(np.int64, copy=False)
                batch_y = torch.from_numpy(batch_y_encoded).to(self.device, non_blocking=self.device.type == "cuda")
                batch_mfe = torch.from_numpy(np.asarray(mfe[start:stop], dtype=np.float32, order="C", copy=False) / mfe_scale).to(self.device, non_blocking=self.device.type == "cuda")
                batch_mae = torch.from_numpy(np.asarray(mae[start:stop], dtype=np.float32, order="C", copy=False) / mae_scale).to(self.device, non_blocking=self.device.type == "cuda")
                batch_opportunity = torch.from_numpy(np.asarray(opportunity[start:stop], dtype=np.float32, order="C", copy=False)).to(self.device, non_blocking=self.device.type == "cuda")
                batch_return = torch.from_numpy(np.asarray(expected_return[start:stop], dtype=np.float32, order="C", copy=False) / return_scale).to(self.device, non_blocking=self.device.type == "cuda")
                batch_duration = torch.from_numpy(np.asarray(duration[start:stop], dtype=np.float32, order="C", copy=False) / duration_scale).to(self.device, non_blocking=self.device.type == "cuda")
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model.forward_outputs(batch_x)
                    self._validate_outputs(outputs, "validation raw")
                    if collect_diagnostics:
                        trunk = self.model.network(batch_x.transpose(1, 2)).transpose(1, 2)
                        trunk_max = max(trunk_max, float(trunk.detach().float().abs().max().cpu()))
                        trunk_std_sum += float(trunk.detach().float().std().cpu())
                        trunk_batches += 1
                    if horizon is None:
                        head_out = self._resolve_head_outputs(outputs)
                    else:
                        if not isinstance(outputs, dict) or evaluation_horizon not in outputs:
                            raise ValueError(f"Validation output is missing horizon={evaluation_horizon}")
                        head_out = outputs[evaluation_horizon]
                    direction_logits = head_out["logits"].reshape(len(batch_x), -1)
                    if collect_diagnostics:
                        for name, value in (("direction_logits", direction_logits), ("opportunity_logit", head_out.get("opportunity_logit", head_out.get("expected_opportunity"))), ("raw_return", head_out.get("expected_return")), ("raw_mfe", head_out.get("expected_mfe")), ("raw_mae", head_out.get("expected_mae")), ("raw_duration", head_out.get("expected_duration"))):
                            if value is not None:
                                values = value.detach().float()
                                current = raw_ranges.setdefault(name, [float("inf"), float("-inf")])
                                current[0] = min(current[0], float(values.min().cpu()))
                                current[1] = max(current[1], float(values.max().cpu()))
                    direction_loss = val_loss_fn(direction_logits, batch_y).mean()
                    head_losses["direction"] = head_losses.get("direction", 0.0) + float(direction_loss.detach().cpu()) * len(batch_x)
                    opportunity_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                    opportunity_logit = head_out.get("opportunity_logit", head_out.get("expected_opportunity"))
                    if opportunity_logit is not None:
                        opportunity_logit = torch.reshape(opportunity_logit, (-1,))
                        batch_opportunity = torch.reshape(batch_opportunity, (-1,))
                        if opportunity_logit.numel() == batch_opportunity.numel():
                            if self.config.entry_quality_mode == "payoff_volatility_compression":
                                opportunity_loss = regression_loss_fn(torch.sigmoid(opportunity_logit), batch_opportunity).mean()
                            else:
                                opportunity_loss = opportunity_loss_fn(opportunity_logit, batch_opportunity).mean()
                        head_losses["opportunity"] = head_losses.get("opportunity", 0.0) + float(opportunity_loss.detach().cpu()) * len(batch_x) if isinstance(opportunity_loss, torch.Tensor) else head_losses.get("opportunity", 0.0) + float(opportunity_loss) * len(batch_x)
                    return_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                    if "expected_return" in head_out:
                        return_pred = torch.reshape(head_out["expected_return"], (-1,))
                        batch_return = torch.reshape(batch_return, (-1,))
                        if return_pred.numel() == batch_return.numel():
                            return_loss = regression_loss_fn(return_pred, batch_return).mean()
                        head_losses["return"] = head_losses.get("return", 0.0) + float(return_loss.detach().cpu()) * len(batch_x) if isinstance(return_loss, torch.Tensor) else head_losses.get("return", 0.0) + float(return_loss) * len(batch_x)
                    duration_loss = torch.zeros((), device=self.device, dtype=torch.float32)
                    if duration_active and "expected_duration" in head_out:
                        duration_pred = torch.reshape(head_out["expected_duration"], (-1,))
                        batch_duration = torch.reshape(batch_duration, (-1,))
                        if duration_pred.numel() == batch_duration.numel():
                            duration_loss = regression_loss_fn(duration_pred, batch_duration).mean()
                        head_losses["duration"] = head_losses.get("duration", 0.0) + float(duration_loss.detach().cpu()) * len(batch_x) if isinstance(duration_loss, torch.Tensor) else head_losses.get("duration", 0.0) + float(duration_loss) * len(batch_x)
                    mfe_pred = torch.reshape(head_out["expected_mfe"], (-1,))
                    mae_pred = torch.reshape(head_out["expected_mae"], (-1,))
                    batch_mfe = torch.reshape(batch_mfe, (-1,))
                    batch_mae = torch.reshape(batch_mae, (-1,))
                    mfe_loss = regression_loss_fn(mfe_pred, batch_mfe).mean() if mfe_pred.numel() == batch_mfe.numel() else torch.zeros((), device=self.device, dtype=torch.float32)
                    mae_loss = regression_loss_fn(mae_pred, batch_mae).mean() if mae_pred.numel() == batch_mae.numel() else torch.zeros((), device=self.device, dtype=torch.float32)
                    head_losses["mfe"] = head_losses.get("mfe", 0.0) + float(mfe_loss.detach().cpu()) * len(batch_x) if isinstance(mfe_loss, torch.Tensor) else head_losses.get("mfe", 0.0) + float(mfe_loss) * len(batch_x)
                    head_losses["mae"] = head_losses.get("mae", 0.0) + float(mae_loss.detach().cpu()) * len(batch_x) if isinstance(mae_loss, torch.Tensor) else head_losses.get("mae", 0.0) + float(mae_loss) * len(batch_x)
                total_rows += len(batch_x)
        if total_rows == 0:
            return 0.0
        normalized_heads = {name: value / total_rows for name, value in head_losses.items()}
        head_weights = {
            "direction": self.config.direction_loss_weight,
            "opportunity": self.config.opportunity_loss_weight,
            "return": self.config.return_loss_weight,
            "duration": self.config.duration_loss_weight,
            "mfe": self.config.mfe_loss_weight,
            "mae": self.config.mae_loss_weight,
        }
        if not duration_active:
            normalized_heads.pop("duration", None)
        total_weight = sum(weight for name, weight in head_weights.items() if name in normalized_heads)
        if total_weight <= 0.0:
            return float(sum(normalized_heads.values()) / max(len(normalized_heads), 1))
        weighted_loss = 0.0
        for name, weight in head_weights.items():
            if name in normalized_heads:
                weighted_loss += weight * normalized_heads[name]
        result = weighted_loss / total_weight
        if collect_diagnostics:
            return result, {"head_losses": normalized_heads, "raw_ranges": raw_ranges, "trunk_max": trunk_max, "trunk_std": trunk_std_sum / max(trunk_batches, 1)}
        return result

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
        targets_by_horizon: dict[int, dict[str, np.ndarray]] | None = None,
        diagnostic_label: str | None = None,
        diagnostic_stop: bool = False,
    ) -> "TCNTrendModel":
        """Fit either a legacy single-horizon target set or the real multi-horizon
        training contract keyed by forecast horizon."""
        cfg = self.config
        n = X_seq.shape[0]
        if targets_by_horizon is None:
            horizon_targets = {int(h): {
                "direction": np.asarray(y_seq, dtype=np.float32),
                "mfe": np.asarray(mfe_seq if mfe_seq is not None else np.zeros(n, dtype=np.float32), dtype=np.float32),
                "mae": np.asarray(mae_seq if mae_seq is not None else np.zeros(n, dtype=np.float32), dtype=np.float32),
                "opportunity": np.asarray(opportunity_seq if opportunity_seq is not None else np.zeros(n, dtype=np.float32), dtype=np.float32),
                "return": np.asarray(return_seq if return_seq is not None else np.zeros(n, dtype=np.float32), dtype=np.float32),
                "duration": np.asarray(duration_seq if duration_seq is not None else np.zeros(n, dtype=np.float32), dtype=np.float32),
            } for h in (self.forecast_horizons or (int(getattr(cfg, "label_horizon", 1)),))}
        else:
            horizon_targets = {int(h): {k: np.asarray(v, dtype=np.float32) if k in {"mfe", "mae", "opportunity", "return", "duration"} else np.asarray(v, dtype=np.float32) for k, v in values.items()} for h, values in targets_by_horizon.items()}

        if n < 50 or len(np.unique(np.asarray(list(horizon_targets.values())[0]["direction"]))) < 2:
            self._is_fitted = False
            return self

        self.forecast_horizons = tuple(sorted(int(h) for h in horizon_targets))
        self.horizon_duration_active = {}
        self.horizon_target_scales = {}
        self.horizon_class_weights = {}

        val_fraction = cfg.val_fraction if val_fraction is None else val_fraction
        val_size = max(1, int(n * val_fraction))
        gap = min(cfg.sequence_length, max(0, n - val_size - 1))
        train_end = max(1, n - val_size - gap)
        train_X = X_seq[:train_end]
        if sample_weight is None:
            train_w = np.ones(train_end, dtype=np.float32)
        else:
            train_w = sample_weight[:train_end].astype(np.float32)
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
        raw_std = np.sqrt(np.maximum(variance, 0.0))
        self.constant_feature_mask_ = raw_std < 1e-6
        self.std_ = np.maximum(raw_std, 1e-6)

        for horizon, arrays in horizon_targets.items():
            direction = np.asarray(arrays["direction"], dtype=np.float32).ravel()
            mfe = np.asarray(arrays.get("mfe", np.zeros_like(direction, dtype=np.float32)), dtype=np.float32)
            mae = np.asarray(arrays.get("mae", np.zeros_like(direction, dtype=np.float32)), dtype=np.float32)
            opportunity = np.asarray(arrays.get("opportunity", np.zeros_like(direction, dtype=np.float32)), dtype=np.float32)
            ret = np.asarray(arrays.get("return", np.zeros_like(direction, dtype=np.float32)), dtype=np.float32)
            duration = np.asarray(arrays.get("duration", np.zeros_like(direction, dtype=np.float32)), dtype=np.float32)
            duration_values = duration[:train_end]
            duration_std = float(np.std(duration_values)) if duration_values.size > 1 else 0.0
            active = duration_values.size > 0 and duration_std > 1e-8
            self.horizon_duration_active[int(horizon)] = active
            self.horizon_target_scales[int(horizon)] = {
                "return": max(float(self._compute_target_scale(ret[:train_end])), 1e-3),
                "mfe": max(float(self._compute_target_scale(mfe[:train_end])), 1e-3),
                "mae": max(float(self._compute_target_scale(mae[:train_end])), 1e-3),
                "duration": (1.0 if not active else max(duration_std, 1e-3)),
            }
            self.horizon_class_weights[int(horizon)] = self._compute_class_weights(direction[:train_end])
            self._describe_target_stats(f"direction[{horizon}]", direction)
            self._describe_target_stats(f"return[{horizon}]", ret)
            self._describe_target_stats(f"duration[{horizon}]", duration)
            self._describe_target_stats(f"mfe[{horizon}]", mfe)
            self._describe_target_stats(f"mae[{horizon}]", mae)
            self._describe_target_stats(f"opportunity[{horizon}]", opportunity)

        primary_horizon = self.forecast_horizons[0]
        self.return_target_scale = self.horizon_target_scales[primary_horizon]["return"]
        self.mfe_target_scale = self.horizon_target_scales[primary_horizon]["mfe"]
        self.mae_target_scale = self.horizon_target_scales[primary_horizon]["mae"]
        self.duration_target_scale = self.horizon_target_scales[primary_horizon]["duration"]
        self.duration_head_active = self.horizon_duration_active.get(primary_horizon, False)
        self.class_weights_ = self.horizon_class_weights.get(primary_horizon, {-1: 1.0, 0: 1.0, 1: 1.0})

        self.model = _TCN(self.n_features, cfg.hidden_channels, cfg.num_layers, cfg.dropout, forecast_horizons=self.forecast_horizons).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.use_amp)
        best_val_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0
        self.training_history = []

        for epoch in range(cfg.max_epochs):
            self.model.train()
            permutation = torch.randperm(train_end)
            epoch_component_sums: dict[str, float] = {}
            horizon_loss_values: dict[int, float] = {}
            epoch_grad_norm_before_clip = 0.0
            epoch_grad_norm_after_clip = 0.0
            epoch_output_ranges: dict[str, list[float]] = {}
            for start in range(0, train_end, cfg.batch_size):
                idx = permutation[start : start + cfg.batch_size]
                batch_x = torch.from_numpy(self._normalize_batch(train_X[idx.numpy()])).to(self.device, non_blocking=self.device.type == "cuda")
                batch_w = torch.from_numpy(np.asarray(train_w[idx.numpy()], dtype=np.float32, order="C", copy=False)).to(self.device, non_blocking=self.device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model.forward_outputs(batch_x)
                    for horizon, horizon_out in outputs.items():
                        for name in ("logits", "opportunity_logit", "expected_return", "expected_duration", "expected_mfe", "expected_mae"):
                            if name in horizon_out:
                                values = horizon_out[name].detach().float()
                                key = f"h{horizon}_{name}"
                                current = epoch_output_ranges.setdefault(key, [float("inf"), float("-inf")])
                                current[0] = min(current[0], float(values.min().cpu()))
                                current[1] = max(current[1], float(values.max().cpu()))
                    horizon_losses = []
                    for horizon in self.forecast_horizons:
                        arrays = horizon_targets[horizon]
                        direction = np.asarray(arrays["direction"][idx.numpy()], dtype=np.float32)
                        y_t = torch.from_numpy((direction + 1).astype(np.int64)).to(self.device, non_blocking=self.device.type == "cuda")
                        class_weight = torch.tensor([
                            self.horizon_class_weights[horizon][-1],
                            self.horizon_class_weights[horizon][0],
                            self.horizon_class_weights[horizon][1],
                        ], dtype=torch.float32, device=self.device)
                        direction_loss_fn = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=cfg.label_smoothing, reduction="none")
                        horizon_out = outputs[horizon]
                        direction_loss = direction_loss_fn(horizon_out["logits"], y_t)
                        opportunity = torch.from_numpy(np.asarray(arrays.get("opportunity", np.zeros_like(direction, dtype=np.float32))[idx.numpy()], dtype=np.float32, order="C", copy=False)).to(self.device, non_blocking=self.device.type == "cuda")
                        ret = torch.from_numpy(np.asarray(arrays.get("return", np.zeros_like(direction, dtype=np.float32))[idx.numpy()], dtype=np.float32, order="C", copy=False) / self.horizon_target_scales[horizon]["return"]).to(self.device, non_blocking=self.device.type == "cuda")
                        mfe = torch.from_numpy(np.asarray(arrays.get("mfe", np.zeros_like(direction, dtype=np.float32))[idx.numpy()], dtype=np.float32, order="C", copy=False) / self.horizon_target_scales[horizon]["mfe"]).to(self.device, non_blocking=self.device.type == "cuda")
                        mae = torch.from_numpy(np.asarray(arrays.get("mae", np.zeros_like(direction, dtype=np.float32))[idx.numpy()], dtype=np.float32, order="C", copy=False) / self.horizon_target_scales[horizon]["mae"]).to(self.device, non_blocking=self.device.type == "cuda")
                        duration = torch.from_numpy(np.asarray(arrays.get("duration", np.zeros_like(direction, dtype=np.float32))[idx.numpy()], dtype=np.float32, order="C", copy=False) / self.horizon_target_scales[horizon]["duration"]).to(self.device, non_blocking=self.device.type == "cuda") if self.horizon_duration_active.get(horizon, False) else torch.zeros_like(ret)
                        opportunity_logit = torch.reshape(horizon_out.get("opportunity_logit", horizon_out.get("expected_opportunity")), (-1,))
                        opportunity = torch.reshape(opportunity, (-1,))
                        if opportunity_logit.numel() == opportunity.numel():
                            opportunity_loss = nn.BCEWithLogitsLoss(reduction="none")(opportunity_logit.float(), opportunity.float())
                        else:
                            opportunity_loss = torch.zeros_like(opportunity_logit, dtype=torch.float32)
                        return_pred = torch.reshape(horizon_out["expected_return"], (-1,))
                        ret = torch.reshape(ret, (-1,))
                        if return_pred.numel() == ret.numel():
                            return_loss = nn.SmoothL1Loss(reduction="none")(return_pred.float(), ret.float())
                        else:
                            return_loss = torch.zeros_like(return_pred, dtype=torch.float32)
                        if self.horizon_duration_active.get(horizon, False):
                            duration_pred = torch.reshape(horizon_out["expected_duration"], (-1,)).float()
                            duration_target = torch.reshape(duration, (-1,)).float()
                            duration_loss = nn.SmoothL1Loss(reduction="none")(duration_pred, duration_target) if duration_pred.numel() == duration_target.numel() else torch.zeros_like(duration_pred, dtype=torch.float32)
                        else:
                            duration_loss = torch.zeros_like(return_loss, dtype=torch.float32)
                        mfe_pred = torch.reshape(horizon_out["expected_mfe"], (-1,)).float()
                        mae_pred = torch.reshape(horizon_out["expected_mae"], (-1,)).float()
                        mfe_target = torch.reshape(mfe, (-1,)).float()
                        mae_target = torch.reshape(mae, (-1,)).float()
                        mfe_loss = nn.SmoothL1Loss(reduction="none")(mfe_pred, mfe_target) if mfe_pred.numel() == mfe_target.numel() else torch.zeros_like(mfe_pred, dtype=torch.float32)
                        mae_loss = nn.SmoothL1Loss(reduction="none")(mae_pred, mae_target) if mae_pred.numel() == mae_target.numel() else torch.zeros_like(mae_pred, dtype=torch.float32)
                        horizon_loss = (
                            cfg.direction_loss_weight * direction_loss +
                            cfg.opportunity_loss_weight * opportunity_loss +
                            cfg.return_loss_weight * return_loss +
                            (cfg.duration_loss_weight * duration_loss if self.horizon_duration_active.get(horizon, False) else 0.0) +
                            cfg.mfe_loss_weight * mfe_loss +
                            cfg.mae_loss_weight * mae_loss
                        )
                        horizon_loss_value = (horizon_loss * batch_w).sum() / batch_w.sum().clamp_min(1e-8)
                        horizon_losses.append(horizon_loss_value)
                        epoch_component_sums[f"{horizon}_direction"] = epoch_component_sums.get(f"{horizon}_direction", 0.0) + float((direction_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                        epoch_component_sums[f"{horizon}_return"] = epoch_component_sums.get(f"{horizon}_return", 0.0) + float((return_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                        if self.horizon_duration_active.get(horizon, False):
                            epoch_component_sums[f"{horizon}_duration"] = epoch_component_sums.get(f"{horizon}_duration", 0.0) + float((duration_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                        epoch_component_sums[f"{horizon}_mfe"] = epoch_component_sums.get(f"{horizon}_mfe", 0.0) + float((mfe_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                        epoch_component_sums[f"{horizon}_mae"] = epoch_component_sums.get(f"{horizon}_mae", 0.0) + float((mae_loss * batch_w).sum().detach().cpu()) / float(batch_w.sum().detach().cpu())
                    total_loss = sum(horizon_losses) / max(len(horizon_losses), 1)
                    horizon_loss_values[epoch] = float(total_loss.detach().cpu())
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.gradient_clip_norm)
                epoch_grad_norm_before_clip = max(epoch_grad_norm_before_clip, float(grad_norm.detach().cpu().item()))
                post_clip_norm = torch.sqrt(
                    sum(
                        torch.sum(parameter.grad.detach().float() ** 2)
                        for parameter in self.model.parameters()
                        if parameter.grad is not None
                    )
                )
                epoch_grad_norm_after_clip = max(epoch_grad_norm_after_clip, float(post_clip_norm.detach().cpu().item()))
                self._assert_model_finite("after_backward", epoch=epoch, batch=start, grad_norm=float(grad_norm.detach().cpu().item()))
                optimizer.step()
                self._assert_model_finite("after_optimizer_step", epoch=epoch, batch=start)

            val_loss = 0.0
            val_batches = 0
            validation_diagnostics: dict[int, dict] = {}
            for horizon in self.forecast_horizons:
                arrays = horizon_targets[horizon]
                val_start = train_end + gap
                y_val = np.asarray(arrays["direction"][val_start:], dtype=np.float32)
                if len(y_val) == 0:
                    continue
                val_batches += 1
                y_val_cls = (y_val + 1).astype(np.int64)
                validation_result = self._evaluate_validation_loss(
                    X_seq[val_start:],
                    y_val_cls,
                    np.asarray(arrays.get("mfe", np.zeros_like(y_val, dtype=np.float32)), dtype=np.float32),
                    np.asarray(arrays.get("mae", np.zeros_like(y_val, dtype=np.float32)), dtype=np.float32),
                    np.asarray(arrays.get("opportunity", np.zeros_like(y_val, dtype=np.float32)), dtype=np.float32),
                    np.asarray(arrays.get("return", np.zeros_like(y_val, dtype=np.float32)), dtype=np.float32),
                    np.asarray(arrays.get("duration", np.zeros_like(y_val, dtype=np.float32)), dtype=np.float32),
                    horizon=horizon,
                    collect_diagnostics=diagnostic_label is not None,
                )
                if diagnostic_label is not None:
                    horizon_loss, horizon_diagnostics = validation_result
                    validation_diagnostics[horizon] = horizon_diagnostics
                else:
                    horizon_loss = validation_result
                val_loss += horizon_loss
            val_loss = val_loss / max(val_batches, 1)
            history_row = {
                "epoch": float(epoch),
                "validation_loss": float(val_loss),
                "loss_total": float(sum(horizon_loss_values.values()) / max(len(horizon_loss_values), 1)),
                "max_gradient_norm_before_clip": epoch_grad_norm_before_clip,
                "max_gradient_norm_after_clip": epoch_grad_norm_after_clip,
                "output_ranges": epoch_output_ranges,
            }
            history_row.update({k: v / max(len(self.forecast_horizons), 1) for k, v in epoch_component_sums.items()})
            self.training_history.append(history_row)
            if diagnostic_label is not None:
                parameter_norms = {}
                for name, parameter in self.model.named_parameters():
                    group = "trunk" if name.startswith("network") or name.startswith("attn") else next((head for head in ("direction", "opportunity", "return", "mfe", "mae", "duration") if head in name), "trunk")
                    parameter_norms.setdefault(group, []).append(parameter.detach().float().norm().cpu())
                parameter_norms = {group: float(torch.linalg.vector_norm(torch.stack(values))) for group, values in parameter_norms.items()}
                max_raw = max((max(abs(value) for value in ranges) for diagnostic in validation_diagnostics.values() for ranges in diagnostic["raw_ranges"].values()), default=0.0)
                max_trunk = max((diagnostic["trunk_max"] for diagnostic in validation_diagnostics.values()), default=0.0)
                print(f"DIAGNOSTIC {diagnostic_label} epoch={epoch + 1} train_loss={history_row['loss_total']:.6f} validation_loss={val_loss:.6f} validation_max_raw={max_raw:.6f} validation_trunk_max={max_trunk:.6f} parameter_norms={parameter_norms}")
                for horizon, diagnostic in validation_diagnostics.items():
                    print(f"DIAGNOSTIC {diagnostic_label} epoch={epoch + 1} horizon={horizon} validation_heads={diagnostic['head_losses']} raw_ranges={diagnostic['raw_ranges']} trunk_max={diagnostic['trunk_max']:.6f} trunk_std={diagnostic['trunk_std']:.6f}")
                if diagnostic_stop and (val_loss > 100.0 or max_raw > 1000.0 or max_trunk > 1000.0):
                    print(f"FIRST_BAD_EPOCH={epoch + 1} DIAGNOSTIC_STOP=YES")
                    break
            print(
                "[Model1] " + " ".join(
                    f"{key}={value:.6f}"
                    for key, value in history_row.items()
                    if isinstance(value, (int, float))
                )
            )

            best_val_loss, best_state, epochs_without_improvement, best_state_updated, should_stop = self._apply_early_stopping(
                current_state=self.model.state_dict(),
                best_val_loss=best_val_loss,
                val_loss=float(val_loss),
                best_state=best_state,
                epochs_without_improvement=epochs_without_improvement,
            )
            if diagnostic_label is not None:
                print(
                    f"DIAGNOSTIC {diagnostic_label} epoch={epoch + 1} "
                    f"best_val_loss={best_val_loss:.6f} best_state_updated={best_state_updated} "
                    f"patience={epochs_without_improvement}/{cfg.early_stopping_patience}"
                )
            if should_stop:
                print(
                    f"EARLY_STOP epoch={epoch + 1} patience={cfg.early_stopping_patience} "
                    f"no_improvement={epochs_without_improvement} best_val_loss={best_val_loss:.6f}"
                )
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

        self._assert_model_finite("before_predict_proba")

        X_norm = self._normalize(X_seq)
        x_tensor = torch.from_numpy(np.asarray(X_norm, dtype=np.float32, order="C", copy=False))
        batch_size = self.config.batch_size

        self.model.eval()
        outputs = []
        primary_horizon = self.forecast_horizons[0] if self.forecast_horizons else 0
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            for start in range(0, x_tensor.shape[0], batch_size):
                batch = x_tensor[start : start + batch_size].to(self.device, non_blocking=self.device.type == "cuda")
                raw = self.model.forward_outputs(batch)
                self._validate_outputs(raw, "probability raw")
                head_out = self._resolve_head_outputs(raw)
                probabilities = torch.softmax(head_out["logits"].float(), dim=-1)
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
        primary_horizon = self.forecast_horizons[0] if self.forecast_horizons else 0
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            for start in range(0, len(x_tensor), self.config.batch_size):
                batch = x_tensor[start : start + self.config.batch_size].to(self.device, non_blocking=self.device.type == "cuda")
                outputs = self.model.forward_outputs(batch)
                self._validate_outputs(outputs, "prediction raw")
                head_out = self._resolve_head_outputs(outputs)
                opportunity_logit = head_out.get("opportunity_logit", head_out.get("expected_opportunity"))
                probabilities.append(torch.softmax(head_out["logits"].float(), dim=-1).cpu().numpy())
                opportunities.append(torch.sigmoid(opportunity_logit.float()).cpu().numpy())
                returns.append((head_out["expected_return"].float() * self.return_target_scale).cpu().numpy())
                durations.append((head_out["expected_duration"].float() * self.duration_target_scale).cpu().numpy())
                mfes.append((head_out["expected_mfe"].float() * self.mfe_target_scale).cpu().numpy())
                maes.append((head_out["expected_mae"].float() * self.mae_target_scale).cpu().numpy())
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

    def predict_trade_outputs_by_horizon(self, X_seq: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
        """Return denormalized prediction heads for every fitted horizon.

        This is intentionally separate from ``predict_trade_outputs`` so the
        legacy primary-horizon inference contract remains unchanged.
        """
        horizons = self.forecast_horizons or (int(getattr(self.config, "label_horizon", 1)),)
        n = int(X_seq.shape[0])
        if not self._is_fitted or n == 0:
            return {
                int(horizon): {
                    "probabilities": np.full((n, 3), 1.0 / 3.0, dtype=np.float32),
                    "opportunity": np.full(n, 0.5, dtype=np.float32),
                    "expected_return": np.zeros(n, dtype=np.float32),
                    "expected_duration": np.zeros(n, dtype=np.float32),
                    "expected_mfe": np.zeros(n, dtype=np.float32),
                    "expected_mae": np.zeros(n, dtype=np.float32),
                }
                for horizon in horizons
            }

        X_norm = self._normalize(X_seq)
        x_tensor = torch.from_numpy(np.asarray(X_norm, dtype=np.float32, order="C", copy=False))
        collected = {
            int(horizon): {name: [] for name in ("probabilities", "opportunity", "raw_return", "raw_duration", "raw_mfe", "raw_mae", "expected_return", "expected_duration", "expected_mfe", "expected_mae", "return_scale", "duration_scale", "mfe_scale", "mae_scale", "duration_active")}
            for horizon in horizons
        }
        self.model.eval()
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            for start in range(0, len(x_tensor), self.config.batch_size):
                batch = x_tensor[start : start + self.config.batch_size].to(self.device, non_blocking=self.device.type == "cuda")
                outputs = self.model.forward_outputs(batch)
                self._validate_outputs(outputs, "horizon prediction raw")
                for horizon in horizons:
                    head_out = outputs[int(horizon)]
                    scales = self.horizon_target_scales.get(int(horizon), {})
                    return_scale = float(scales.get("return", self.return_target_scale))
                    duration_scale = float(scales.get("duration", self.duration_target_scale))
                    mfe_scale = float(scales.get("mfe", self.mfe_target_scale))
                    mae_scale = float(scales.get("mae", self.mae_target_scale))
                    opportunity_logit = head_out.get("opportunity_logit", head_out.get("expected_opportunity"))
                    collected[int(horizon)]["probabilities"].append(torch.softmax(head_out["logits"].float(), dim=-1).cpu().numpy())
                    collected[int(horizon)]["opportunity"].append(torch.sigmoid(opportunity_logit.float()).cpu().numpy())
                    raw_return = head_out["expected_return"].float()
                    raw_duration = head_out["expected_duration"].float()
                    raw_mfe = head_out["expected_mfe"].float()
                    raw_mae = head_out["expected_mae"].float()
                    collected[int(horizon)]["raw_return"].append(raw_return.cpu().numpy())
                    collected[int(horizon)]["raw_duration"].append(raw_duration.cpu().numpy())
                    collected[int(horizon)]["raw_mfe"].append(raw_mfe.cpu().numpy())
                    collected[int(horizon)]["raw_mae"].append(raw_mae.cpu().numpy())
                    collected[int(horizon)]["expected_return"].append((raw_return * return_scale).cpu().numpy())
                    collected[int(horizon)]["expected_duration"].append((raw_duration * duration_scale).cpu().numpy())
                    collected[int(horizon)]["expected_mfe"].append((raw_mfe * mfe_scale).cpu().numpy())
                    collected[int(horizon)]["expected_mae"].append((raw_mae * mae_scale).cpu().numpy())
                    batch_size = int(raw_return.shape[0])
                    for name, scale in (("return_scale", return_scale), ("duration_scale", duration_scale), ("mfe_scale", mfe_scale), ("mae_scale", mae_scale)):
                        collected[int(horizon)][name].append(np.full(batch_size, scale, dtype=np.float32))
                    collected[int(horizon)]["duration_active"].append(np.full(batch_size, self.horizon_duration_active.get(int(horizon), False), dtype=bool))

        result: dict[int, dict[str, np.ndarray]] = {}
        for horizon in horizons:
            result[int(horizon)] = {name: np.concatenate(values) for name, values in collected[int(horizon)].items()}
            for name, value in result[int(horizon)].items():
                if not np.isfinite(value).all():
                    raise ValueError(f"Model 1 horizon prediction conversion produced non-finite output: horizon={horizon}, name={name}")
        return result

    def predict_forecast_result(self, X_seq: np.ndarray, *, data_timestamp: str | None = None) -> BaseForecastModel:
        """Emit the actual structured forecast contract used by the higher-level decision stack."""
        horizons = tuple(int(h) for h in self.forecast_horizons)
        if X_seq is None or X_seq.shape[0] == 0:
            expected_returns = {h: 0.0 for h in horizons}
            direction_probabilities = {h: {"LONG": 0.0, "SHORT": 0.0, "NO_TRADE": 1.0} for h in horizons}
            model = BaseForecastModel(
                model_id=self.config.model_id,
                asset=self.asset,
                mode=self.config.model_role,
                native_timeframe=self.config.base_timeframe,
                forecast_horizons=horizons,
                model_version=self.config.model_id,
            )
            return model.as_result(expected_returns=expected_returns, direction_probabilities=direction_probabilities, predicted_directions={h: "NO_TRADE" for h in horizons}, expected_opportunities={h: 0.0 for h in horizons}, expected_mfe={h: 0.0 for h in horizons}, expected_mae={h: 0.0 for h in horizons}, expected_durations={}, score=None, data_timestamp=data_timestamp)

        outputs_by_horizon = self.predict_trade_outputs_by_horizon(X_seq)
        expected_returns = {}
        direction_probabilities = {}
        predicted_directions = {}
        expected_opportunities = {}
        expected_mfe = {}
        expected_mae = {}
        expected_durations = {}
        for horizon in horizons:
            outputs = outputs_by_horizon[horizon]
            probabilities = outputs["probabilities"]
            mean_probabilities = probabilities.mean(axis=0)
            direction_labels = ("SHORT", "NO_TRADE", "LONG")
            predicted_index = int(np.argmax(mean_probabilities))
            expected_returns[horizon] = float(np.mean(outputs["expected_return"]))
            direction_probabilities[horizon] = {
                "LONG": float(mean_probabilities[2]),
                "SHORT": float(mean_probabilities[0]),
                "NO_TRADE": float(mean_probabilities[1]),
            }
            predicted_directions[horizon] = direction_labels[predicted_index]
            expected_opportunities[horizon] = float(np.mean(outputs["opportunity"]))
            expected_mfe[horizon] = float(np.mean(outputs["expected_mfe"]))
            expected_mae[horizon] = float(np.mean(outputs["expected_mae"]))
            if self.horizon_duration_active.get(horizon, False):
                expected_durations[horizon] = float(np.mean(outputs["expected_duration"]))
        model = BaseForecastModel(
            model_id=self.config.model_id,
            asset=self.asset,
            mode=self.config.model_role,
            native_timeframe=self.config.base_timeframe,
            forecast_horizons=horizons,
            model_version=self.config.model_id,
        )
        return model.as_result(expected_returns=expected_returns, direction_probabilities=direction_probabilities, predicted_directions=predicted_directions, expected_opportunities=expected_opportunities, expected_mfe=expected_mfe, expected_mae=expected_mae, expected_durations=expected_durations, score=float(np.mean(list(expected_opportunities.values()))), data_timestamp=data_timestamp)

    def to_cpu(self) -> "TCNTrendModel":
        if self.model is not None:
            self.model = self.model.cpu()
        self.device = torch.device("cpu")
        self.use_amp = False
        return self

    def cpu(self) -> "TCNTrendModel":
        return self.to_cpu()

    def state_dict(self) -> dict:
        return {
            "model_state": self.model.state_dict() if self.model is not None else None,
            "mean": self.mean_,
            "std": self.std_,
            "constant_feature_mask": self.constant_feature_mask_,
            "n_features": self.n_features,
            "is_fitted": self._is_fitted,
            "training_history": self.training_history,
            "return_target_scale": self.return_target_scale,
            "mfe_target_scale": self.mfe_target_scale,
            "mae_target_scale": self.mae_target_scale,
            "duration_target_scale": self.duration_target_scale,
            "class_weights": self.class_weights_,
            "forecast_horizons": list(self.forecast_horizons),
            "horizon_target_scales": self.horizon_target_scales,
            "horizon_duration_active": self.horizon_duration_active,
            "horizon_class_weights": self.horizon_class_weights,
        }

    def load_state_dict(self, state: dict) -> "TCNTrendModel":
        cfg = self.config
        self.mean_ = state["mean"]
        self.std_ = state["std"]
        self.constant_feature_mask_ = state.get("constant_feature_mask")
        if self.constant_feature_mask_ is None and self.std_ is not None:
            self.constant_feature_mask_ = np.asarray(self.std_ <= 1e-6, dtype=bool)
        self.n_features = state["n_features"]
        self._is_fitted = state["is_fitted"]
        self.training_history = state.get("training_history", [])
        self.return_target_scale = float(state.get("return_target_scale", getattr(self.config, "return_target_scale", 1.0) or 1.0))
        self.mfe_target_scale = float(state.get("mfe_target_scale", getattr(self.config, "excursion_target_scale", 1.0) or 1.0))
        self.mae_target_scale = float(state.get("mae_target_scale", getattr(self.config, "excursion_target_scale", 1.0) or 1.0))
        self.duration_target_scale = float(state.get("duration_target_scale", 1.0))
        self.class_weights_ = state.get("class_weights", {-1: 1.0, 0: 1.0, 1: 1.0})
        self.forecast_horizons = tuple(int(h) for h in state.get("forecast_horizons", self.forecast_horizons))
        self.horizon_target_scales = {
            int(h): {name: float(value) for name, value in scales.items()}
            for h, scales in state.get("horizon_target_scales", {}).items()
        }
        self.horizon_duration_active = {int(h): bool(active) for h, active in state.get("horizon_duration_active", {}).items()}
        self.horizon_class_weights = {
            int(h): {int(label): float(value) for label, value in weights.items()}
            for h, weights in state.get("horizon_class_weights", {}).items()
        }
        if self.forecast_horizons and self.horizon_target_scales:
            primary_horizon = self.forecast_horizons[0]
            primary_scales = self.horizon_target_scales.get(primary_horizon, {})
            self.return_target_scale = float(primary_scales.get("return", self.return_target_scale))
            self.mfe_target_scale = float(primary_scales.get("mfe", self.mfe_target_scale))
            self.mae_target_scale = float(primary_scales.get("mae", self.mae_target_scale))
            self.duration_target_scale = float(primary_scales.get("duration", self.duration_target_scale))
            self.duration_head_active = self.horizon_duration_active.get(primary_horizon, self.duration_head_active)
        if state["model_state"] is not None:
            self.model = _TCN(self.n_features, cfg.hidden_channels, cfg.num_layers, cfg.dropout, forecast_horizons=self.forecast_horizons).to(self.device)
            # Older checkpoints contain only the classifier head. Loading them
            # non-strictly preserves backward compatibility; new regression
            # heads are then available for retraining.
            self.model.load_state_dict(state["model_state"], strict=False)
        return self
