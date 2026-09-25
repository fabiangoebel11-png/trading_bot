import numpy as np
import pytest

from core.config import TradingBotConfig, TrendMLConfig
from core.ml.device import resolve_device
from core.ml.tcn_model import TCNTrendModel, _TCN


def test_tcn_model_can_be_forced_to_cpu() -> None:
    config = TrendMLConfig(
        model_id="device_cpu_test",
        assets=["BTC/USDT"],
        batch_size=32,
    )

    model = TCNTrendModel(config, n_features=4, prefer_cuda=False)
    assert model.device.type == "cpu"

    model.to_cpu()
    assert model.device.type == "cpu"


def test_default_runtime_device_policy_is_cuda_for_training_and_cpu_for_inference() -> None:
    config = TradingBotConfig()
    assert config.training_device == "cuda"
    assert config.inference_device == "cpu"


def test_resolve_device_cuda_requires_real_cuda() -> None:
    import torch

    original = torch.cuda.is_available
    try:
        torch.cuda.is_available = lambda: False
        with pytest.raises(RuntimeError, match="CUDA is required"):
            resolve_device("cuda")
    finally:
        torch.cuda.is_available = original


def test_tcn_model_accepts_explicit_cpu_device() -> None:
    config = TrendMLConfig(
        model_id="device_cpu_test_2",
        assets=["BTC/USDT"],
        batch_size=32,
        training_device="cpu",
        inference_device="cpu",
    )

    model = TCNTrendModel(config, n_features=4, requested_device="cpu")
    assert model.device.type == "cpu"


def test_training_defaults_are_conservative_for_numeric_stability() -> None:
    cfg = TrendMLConfig()
    assert cfg.use_amp is False
    assert cfg.learning_rate == pytest.approx(3e-4)
    assert cfg.batch_size == 2048
    assert cfg.gradient_clip_norm == pytest.approx(1.0)


def test_tcn_model_rejects_non_finite_parameters_before_prediction() -> None:
    config = TrendMLConfig(model_id="device_nan_guard", assets=["BTC/USDT"], batch_size=32, use_amp=False)
    model = TCNTrendModel(config, n_features=4, requested_device="cpu")
    model.model = _TCN(4, 8, 2, 0.1, forecast_horizons=(72,))
    model._is_fitted = True
    with np.errstate(all="ignore"):
        model.model.heads["72"].direction_head.weight.data.fill_(np.nan)

    with pytest.raises(ValueError, match="non-finite"):
        model.predict_proba(np.ones((4, 8, 4), dtype=np.float32))
