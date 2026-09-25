"""Shared CUDA/device utilities for the optional PyTorch models.

Centralizes device selection so every GPU-capable module (LSTM training/
inference, future deep-learning extensions) consistently prefers the local
RTX 3070 eGPU when available and falls back to CPU otherwise.
"""
from __future__ import annotations

import os

try:
    import torch
except ImportError:  # pragma: no cover - only hit without the "ml" extra installed
    torch = None


def torch_available() -> bool:
    return torch is not None


def _normalize_requested_device(requested_device: str | None) -> str | None:
    if requested_device is None:
        return None
    normalized = str(requested_device).strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in {"cpu", "cuda"}:
        return normalized
    raise ValueError(f"Unsupported device mode: {requested_device!r}. Expected one of: cpu, cuda, auto.")


def resolve_device(requested_device: str | None = None, *, prefer_cuda: bool = True, force_cpu: bool | None = None):
    """Resolve the active torch device with explicit CPU/CUDA policy semantics.

    - training_device='cuda' requires real CUDA availability and fails hard
      instead of silently falling back to CPU.
    - training_device='cpu' or inference_device='cpu' forces CPU.
    - auto/default behavior prefers CUDA when available and otherwise falls back
      to CPU.
    """
    if torch is None:
        raise ImportError("PyTorch is not installed. Run `uv sync --extra ml`.")
    if force_cpu is None:
        force_cpu = os.getenv("TRADING_BOT_FORCE_CPU", "").strip().lower() in {"1", "true", "yes", "on"}

    requested = _normalize_requested_device(requested_device)
    if force_cpu:
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training_device='cuda' but CUDA is not available in this environment.")
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    if prefer_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def get_device(prefer_cuda: bool = True, *, force_cpu: bool | None = None, requested_device: str | None = None):
    """Backward-compatible device selection helper."""
    return resolve_device(requested_device=requested_device, prefer_cuda=prefer_cuda, force_cpu=force_cpu)


def device_summary(device) -> str:
    if torch is None:
        return "torch not installed"
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        return f"CUDA: {name} ({total_mem_gb:.1f} GB)"
    return "CPU"
