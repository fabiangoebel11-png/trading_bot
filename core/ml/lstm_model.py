"""Optional PyTorch LSTM mean-reversion classifier, intended to be trained on the
RTX 3070 eGPU. Only imported/instantiated when ``MLConfig.model_type == "lstm"``;
``torch`` is an optional dependency (``uv sync --extra ml``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import MLConfig
from core.ml.device import device_summary, get_device

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - exercised only without the "ml" extra
    torch = None
    nn = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for model_type='lstm'. Install it with "
            "`uv sync --extra ml` (uses the RTX 3070 eGPU via CUDA if available)."
        ) from _IMPORT_ERROR


class _LSTMClassifier(nn.Module if nn is not None else object):
    def __init__(self, n_features: int, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):  # noqa: ANN001 - torch tensor
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def _make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Turn a (n, features) matrix into (n - seq_len + 1, seq_len, features) windows."""
    n = X.shape[0]
    if n < seq_len:
        return np.empty((0, seq_len, X.shape[1]))
    return np.stack([X[i : i + seq_len] for i in range(n - seq_len + 1)])


class LSTMReversionModel:
    def __init__(self, ml_config: MLConfig) -> None:
        _require_torch()
        self.config = ml_config
        self.device = get_device()
        self.use_amp = ml_config.lstm_use_amp and self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.model: _LSTMClassifier | None = None
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None
        self._is_fitted = False
        print(f"[LSTMReversionModel] device={self.device} ({device_summary(self.device)}), amp={self.use_amp}")

    def _normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        return (X - self.mean_) / self.std_.replace(0, 1)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMReversionModel":
        seq_len = self.config.lstm_sequence_length
        if y.nunique() < 2 or len(X) < seq_len + 1:
            self._is_fitted = False
            return self

        self.mean_ = X.mean()
        self.std_ = X.std()
        X_norm = self._normalize(X).to_numpy(dtype=np.float32)
        y_arr = y.to_numpy(dtype=np.float32)

        sequences = _make_sequences(X_norm, seq_len)
        targets = y_arr[seq_len - 1 :]

        self.model = _LSTMClassifier(n_features=X.shape[1], hidden_size=self.config.lstm_hidden_size)
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lstm_learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        # Pinned host memory + non_blocking transfer overlaps H2D copy with compute over
        # the OCuLink link; irrelevant on CPU, where pin_memory() is a no-op.
        x_tensor = torch.tensor(sequences).pin_memory() if self.device.type == "cuda" else torch.tensor(sequences)
        y_tensor = torch.tensor(targets).pin_memory() if self.device.type == "cuda" else torch.tensor(targets)
        batch_size = self.config.lstm_batch_size

        self.model.train()
        for _ in range(self.config.lstm_epochs):
            permutation = torch.randperm(x_tensor.shape[0])
            for start in range(0, x_tensor.shape[0], batch_size):
                idx = permutation[start : start + batch_size]
                batch_x = x_tensor[idx].to(self.device, non_blocking=True)
                batch_y = y_tensor[idx].to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits = self.model(batch_x)
                    loss = loss_fn(logits, batch_y)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        seq_len = self.config.lstm_sequence_length
        if not self._is_fitted or len(X) < seq_len:
            return pd.Series(0.5, index=X.index)

        X_norm = self._normalize(X).to_numpy(dtype=np.float32)
        sequences = _make_sequences(X_norm, seq_len)
        x_tensor = torch.tensor(sequences)
        batch_size = self.config.lstm_batch_size

        self.model.eval()
        outputs = []
        with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            for start in range(0, x_tensor.shape[0], batch_size):
                batch = x_tensor[start : start + batch_size].to(self.device, non_blocking=True)
                logits = self.model(batch)
                outputs.append(torch.sigmoid(logits).float().cpu().numpy())
        proba = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)

        # First (seq_len - 1) rows have no full lookback window yet -> neutral probability.
        result = np.full(len(X), 0.5, dtype=np.float32)
        result[seq_len - 1 :] = proba
        return pd.Series(result, index=X.index)
