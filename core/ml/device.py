"""Shared CUDA/device utilities for the optional PyTorch models.

Centralizes device selection so every GPU-capable module (LSTM training/
inference, future deep-learning extensions) consistently prefers the local
RTX 3070 eGPU when available and falls back to CPU otherwise.
"""
from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover - only hit without the "ml" extra installed
    torch = None


def torch_available() -> bool:
    return torch is not None


def get_device(prefer_cuda: bool = True):
    """Return the best available torch device, enabling cuDNN autotuning for the
    fixed-shape LSTM batches used here (safe win on a single dedicated GPU)."""
    if torch is None:
        raise ImportError("PyTorch is not installed. Run `uv sync --extra ml`.")
    if prefer_cuda:
        try:
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                return torch.device("cuda")
        except Exception:  # noqa: BLE001 - broken/missing CUDA must degrade to CPU
            pass
    return torch.device("cpu")


def device_summary(device) -> str:
    if torch is None:
        return "torch not installed"
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        return f"CUDA: {name} ({total_mem_gb:.1f} GB)"
    return "CPU"
