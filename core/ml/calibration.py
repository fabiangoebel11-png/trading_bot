"""Optional post-training probability calibration."""
from __future__ import annotations

import numpy as np


class TemperatureCalibrator:
    """Fit one temperature on validation logits only."""

    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureCalibrator":
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if logits.ndim != 2 or len(logits) != len(labels):
            raise ValueError("logits must have shape (n_samples, n_classes) and match labels")
        if not np.isfinite(logits).all():
            raise ValueError("Temperature calibration requires finite validation logits")
        if len(labels) == 0 or np.any(labels < 0) or np.any(labels >= logits.shape[1]):
            raise ValueError("Temperature calibration labels are empty or out of range")
        candidates = np.exp(np.linspace(np.log(0.05), np.log(20.0), 200))
        losses = []
        for temperature in candidates:
            shifted = logits / temperature
            shifted -= shifted.max(axis=1, keepdims=True)
            log_prob = shifted - np.logaddexp.reduce(shifted, axis=1, keepdims=True)
            losses.append(float(-log_prob[np.arange(len(labels)), labels].mean()))
        self.temperature = float(candidates[int(np.argmin(losses))])
        return self

    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float)
        if not np.isfinite(logits).all() or not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Temperature calibration requires finite logits and a positive finite temperature")
        logits = logits / self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(probabilities).all():
            raise ValueError("Temperature calibration produced non-finite probabilities")
        return probabilities

    def to_dict(self) -> dict[str, float | str]:
        return {"method": "temperature", "temperature": self.temperature}


def calibrate_validation_logits(
    logits: np.ndarray, labels: np.ndarray, method: str | None = "temperature"
) -> TemperatureCalibrator | None:
    """Fit calibration on validation logits only; never accepts test data."""
    if method in (None, "none"):
        return None
    if method != "temperature":
        raise ValueError(f"Unsupported calibration method: {method}")
    return TemperatureCalibrator().fit(logits, labels)
