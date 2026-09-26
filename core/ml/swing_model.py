"""Trainable daily-primary multi-resolution Model 2."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.config import SwingMLConfig
from core.market_state import PredictionSnapshot
from core.risk import cap_model2_leverage
from core.ml.device import get_device

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    torch = None
    nn = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


HORIZONS = (1, 3, 5, 10, 20)


def _require_torch() -> None:
    if torch is None:
        raise ImportError("Model 2 requires PyTorch; install the project ML extra") from _IMPORT_ERROR


def bounded_score(value: float) -> float:
    return float(np.clip(np.nan_to_num(value, nan=0.0), 0.0, 100.0))


def combine_scores(swing_score: float, entry_score: float, swing_weight: float = 0.70, entry_weight: float = 0.30) -> float:
    total = swing_weight + entry_weight
    if total <= 0:
        raise ValueError("score weights must sum to a positive value")
    return bounded_score((swing_weight * swing_score + entry_weight * entry_score) / total)


def horizon_opportunity_score(expected_return: float, expected_mfe: float, expected_mae: float, entry_signal: float, config: SwingMLConfig) -> float:
    """Score one swing forecast from its own horizon-specific trade quality.

    Each swing horizon already exposes a dedicated expected return plus its
    favorable and adverse excursion tails. The score therefore combines:
    - directional confidence: expected return relative to the horizon's MFE/MAE
    - payoff quality: favorable vs adverse excursion balance
    - entry conviction: the model's directional entry signal

    This keeps the score transparent and horizon-specific without altering the
    underlying Model-2 architecture or requiring retraining.
    """
    favorable = max(float(expected_mfe), 0.0)
    adverse = max(-float(expected_mae), 0.0)
    directional_confidence = np.clip(abs(float(expected_return)) / max(favorable + adverse + 1e-8, 1e-8), 0.0, 1.0)
    payoff = favorable / max(favorable + adverse, 1e-8)
    entry_strength = np.clip((float(entry_signal) + 1.0) / 2.0, 0.0, 1.0)
    score = 100.0 * (0.55 * directional_confidence + 0.25 * payoff + 0.20 * entry_strength)
    return bounded_score(score)


def scores_from_predictions(expected_returns: dict[int, float], expected_mfe: float, expected_mae: float, entry_signal: float, config: SwingMLConfig) -> tuple[float, float, float]:
    return_signal = float(np.mean([expected_returns.get(h, 0.0) for h in HORIZONS]))
    quality = return_signal + 0.5 * expected_mfe + 0.5 * expected_mae
    swing = bounded_score(50.0 + 2500.0 * quality)
    entry = bounded_score(50.0 + 50.0 * entry_signal)
    return swing, entry, combine_scores(swing, entry, config.swing_score_weight, config.entry_score_weight)


def swing_snapshot(asset: str, timestamp, predictions: dict[str, float], config: SwingMLConfig) -> PredictionSnapshot:
    returns = {h: float(predictions.get(f"expected_return_{h}d", 0.0)) for h in HORIZONS}
    swing, entry, overall = scores_from_predictions(
        returns, float(predictions.get("expected_mfe", 0.0)), float(predictions.get("expected_mae", 0.0)), float(predictions.get("entry_signal", 0.0)), config,
    )
    expected_return = returns[5]
    return PredictionSnapshot(
        score=overall,
        direction="LONG" if expected_return > 0 else "SHORT" if expected_return < 0 else "UNCERTAIN",
        expected_return=expected_return,
        expected_duration_bars=float(predictions.get("expected_duration_days", 0.0)) * 288.0,
        expected_mfe=float(predictions.get("expected_mfe", 0.0)),
        expected_mae=float(predictions.get("expected_mae", 0.0)),
        timestamp=timestamp,
        model_type="swing",
        asset=asset,
        swing_score=swing,
        entry_score=entry,
        swing_opportunity_score=overall,
        expected_duration_days=float(predictions.get("expected_duration_days", 0.0)),
    )


if nn is not None:

    class _SwingNet(nn.Module):
        def __init__(self, daily_size: int, entry_size: int, hidden: list[int], dropout: float) -> None:
            super().__init__()
            dims = [daily_size + entry_size] + hidden
            layers: list[nn.Module] = []
            for left, right in zip(dims, dims[1:]):
                layers.extend([nn.Linear(left, right), nn.ReLU(), nn.Dropout(dropout)])
            self.body = nn.Sequential(*layers)
            last = dims[-1]
            self.returns = nn.Linear(last, len(HORIZONS))
            self.mfe = nn.Linear(last, 1)
            self.mae = nn.Linear(last, 1)
            self.duration = nn.Linear(last, len(HORIZONS))
            self.entry = nn.Linear(last, 1)

        def forward(self, daily, entry, mask):
            x = torch.cat([daily.flatten(1), entry * mask], dim=1)
            hidden = self.body(x)
            return {"returns": self.returns(hidden), "mfe": self.mfe(hidden).squeeze(-1), "mae": self.mae(hidden).squeeze(-1), "duration": self.duration(hidden), "entry": self.entry(hidden).squeeze(-1)}


class SwingModel:
    def __init__(self, config: SwingMLConfig, daily_shape: tuple[int, int], entry_size: int) -> None:
        _require_torch()
        self.config = config
        self.daily_shape = daily_shape
        self.entry_size = entry_size
        self.device = get_device()
        self.model = _SwingNet(int(np.prod(daily_shape)), entry_size, config.hidden_dimensions, config.dropout).to(self.device)
        self.daily_mean: np.ndarray | None = None
        self.daily_std: np.ndarray | None = None
        self.entry_mean: np.ndarray | None = None
        self.entry_std: np.ndarray | None = None
        self.training_history: list[dict[str, float]] = []

    def _normalize(self, daily: np.ndarray, entry: np.ndarray, fit: bool = False) -> tuple[np.ndarray, np.ndarray]:
        daily_flat = daily.reshape(len(daily), -1).astype(np.float32)
        entry = entry.astype(np.float32)
        if fit:
            self.daily_mean, self.daily_std = daily_flat.mean(0), np.maximum(daily_flat.std(0), 1e-6)
            self.entry_mean, self.entry_std = entry.mean(0), np.maximum(entry.std(0), 1e-6)
        assert self.daily_mean is not None and self.daily_std is not None and self.entry_mean is not None and self.entry_std is not None
        return ((daily_flat - self.daily_mean) / self.daily_std).reshape(daily.shape), (entry - self.entry_mean) / self.entry_std

    def fit(self, daily: np.ndarray, entry: np.ndarray, mask: np.ndarray, targets: dict[str, np.ndarray], epochs: int | None = None) -> None:
        daily, entry = self._normalize(daily, entry, fit=True)
        target_returns = np.column_stack([targets[f"return_{h}d"] for h in HORIZONS]).astype(np.float32)
        target_duration = np.column_stack([targets[f"duration_{h}d"] for h in HORIZONS]).astype(np.float32)
        target_mfe = targets["mfe_20d"].astype(np.float32)
        target_mae = targets["mae_20d"].astype(np.float32)
        target_entry = np.clip(target_returns[:, 0] * 100.0, -1.0, 1.0)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        self.model.train()
        self.training_history = []
        batch_size = max(1, self.config.batch_size)
        for epoch in range(epochs if epochs is not None else self.config.epochs):
            order = np.random.permutation(len(daily))
            losses = []
            for start in range(0, len(order), batch_size):
                idx = order[start:start + batch_size]
                tensors = [torch.as_tensor(value[idx], device=self.device) for value in (daily, entry, mask, target_returns, target_duration, target_mfe, target_mae, target_entry)]
                expanded_mask = tensors[2].repeat_interleave(tensors[1].shape[1] // tensors[2].shape[1], dim=1)
                output = self.model(tensors[0], tensors[1], expanded_mask)
                loss = ((output["returns"] - tensors[3]) ** 2).mean() + 0.1 * ((output["duration"] - tensors[4]) ** 2).mean() + 0.2 * ((output["mfe"] - tensors[5]) ** 2).mean() + 0.2 * ((output["mae"] - tensors[6]) ** 2).mean() + 0.1 * ((output["entry"] - tensors[7]) ** 2).mean()
                losses.append(float(loss.detach().cpu()))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            self.training_history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses)) if losses else float("nan")})

    def predict(self, daily: np.ndarray, entry: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        daily, entry = self._normalize(daily, entry)
        self.model.eval()
        with torch.no_grad():
            entry_tensor = torch.as_tensor(entry, device=self.device)
            mask_tensor = torch.as_tensor(mask, device=self.device).repeat_interleave(entry_tensor.shape[1] // mask.shape[1], dim=1)
            output = self.model(torch.as_tensor(daily, device=self.device), entry_tensor, mask_tensor)
        return {key: value.detach().cpu().numpy() for key, value in output.items()}

    def save(self, path: str | Path, metadata: dict[str, object]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "daily_mean": self.daily_mean, "daily_std": self.daily_std, "entry_mean": self.entry_mean, "entry_std": self.entry_std, "daily_shape": self.daily_shape, "entry_size": self.entry_size, "training_history": self.training_history}, path)
        path.with_suffix(".json").write_text(json.dumps({**metadata, "training_date": datetime.now(timezone.utc).isoformat(), "model_type": "swing"}, indent=2, default=str), encoding="utf-8")


def load_swing_model(path: str | Path, config: SwingMLConfig) -> SwingModel:
    _require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = SwingModel(config, tuple(payload["daily_shape"]), int(payload["entry_size"]))
    model.model.load_state_dict(payload["state_dict"])
    model.daily_mean, model.daily_std = payload["daily_mean"], payload["daily_std"]
    model.entry_mean, model.entry_std = payload["entry_mean"], payload["entry_std"]
    model.training_history = payload.get("training_history", [])
    return model
