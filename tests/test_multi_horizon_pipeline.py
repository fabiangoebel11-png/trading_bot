import numpy as np
import pandas as pd
import torch

from core.config import TrendMLConfig
from core.ml.labeling import build_horizon_label_map
from core.ml.tcn_model import TCNTrendModel, _TCN
from core.ml.train import build_targets_by_horizon


HORIZONS = (1, 4, 8, 12, 24)


def test_config_exposes_explicit_forecast_horizons() -> None:
    cfg = TrendMLConfig()
    assert tuple(cfg.forecast_horizons) == (cfg.label_horizon,)
    assert cfg.primary_horizon == cfg.label_horizon


def test_build_horizon_label_map_returns_per_horizon_contract() -> None:
    index = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
    close = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
    atr = pd.Series(np.full(len(index), 1.0), index=index)

    labels = build_horizon_label_map(
        close,
        atr,
        (12, 24),
        atr_multiple=2.0,
        high=close,
        low=close,
    )

    assert set(labels) == {12, 24}
    assert list(labels[12].columns) == ["label", "t1", "mfe", "mae", "barrier_return", "time_to_barrier", "tradeable", "quality_ratio"]
    assert list(labels[24].columns) == ["label", "t1", "mfe", "mae", "barrier_return", "time_to_barrier", "tradeable", "quality_ratio"]


def test_duration_targets_are_normalized_nonnegative_per_horizon() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    features = pd.DataFrame({"feature": [1.0, 2.0, 3.0]}, index=index)
    labels = {
        4: pd.DataFrame({"label": [1, 0, -1], "mfe": [0.1, 0.0, 0.1], "mae": [-0.1, 0.0, -0.1], "time_to_barrier": [1.0, 2.0, 4.0]}, index=index),
        8: pd.DataFrame({"label": [1, 0, -1], "mfe": [0.1, 0.0, 0.1], "mae": [-0.1, 0.0, -0.1], "time_to_barrier": [1.0, 4.0, 8.0]}, index=index),
    }

    targets = build_targets_by_horizon(features, labels, (4, 8))

    assert np.all((targets[4]["duration"] >= 0.0) & (targets[4]["duration"] <= 1.0))
    assert np.allclose(targets[4]["duration"], [0.25, 0.5, 1.0])
    assert np.allclose(targets[8]["duration"], [0.125, 0.5, 1.0])


def _deterministic_model() -> TCNTrendModel:
    config = TrendMLConfig()
    config.batch_size = 2
    config.use_amp = False
    model = TCNTrendModel(config, n_features=2, forecast_horizons=HORIZONS, prefer_cuda=False, asset="BTC/USDT")
    model.horizon_duration_active = {horizon: True for horizon in HORIZONS}
    class DeterministicHeads(torch.nn.Module):
        def forward_outputs(self, batch: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
            outputs = {}
            for index, horizon in enumerate(HORIZONS):
                logits = torch.full((len(batch), 3), -10.0, device=batch.device)
                logits[:, index % 3] = 10.0
                value = torch.full((len(batch),), float(horizon), device=batch.device)
                outputs[horizon] = {
                    "logits": logits,
                    "opportunity_logit": torch.full((len(batch),), float(index - 2), device=batch.device),
                    "expected_return": value,
                    "expected_duration": value + 0.5,
                    "expected_mfe": value + 1.0,
                    "expected_mae": -value - 1.0,
                }
            return outputs

    model.model = DeterministicHeads()
    model.mean_ = np.zeros(2, dtype=np.float64)
    model.std_ = np.ones(2, dtype=np.float64)
    model._is_fitted = True
    return model


def test_public_forecast_uses_each_distinct_horizon_head() -> None:
    result = _deterministic_model().predict_forecast_result(np.zeros((2, 3, 2), dtype=np.float32)).as_dict()

    assert result["asset"] == "BTC/USDT"
    assert result["forecast_horizons"] == list(HORIZONS)
    assert result["expected_returns"] == {horizon: float(horizon) for horizon in HORIZONS}
    assert set(result["expected_opportunities"]) == {str(horizon) for horizon in HORIZONS}
    assert len(set(result["expected_opportunities"].values())) == len(HORIZONS)
    assert result["expected_mfe"] == {str(horizon): float(horizon) + 1.0 for horizon in HORIZONS}
    assert result["expected_mae"] == {str(horizon): -float(horizon) - 1.0 for horizon in HORIZONS}
    assert set(result["expected_durations"]) == {str(horizon) for horizon in HORIZONS}
    assert len(set(result["expected_durations"].values())) == len(HORIZONS)
    assert all(value > 0.0 for value in result["expected_durations"].values())
    assert set(result["predicted_directions"].values()) == {"SHORT", "NO_TRADE", "LONG"}


def test_horizon_scales_survive_state_dict_roundtrip() -> None:
    config = TrendMLConfig()
    config.hidden_channels = 4
    config.num_layers = 1
    config.dropout = 0.0
    model = TCNTrendModel(config, n_features=2, forecast_horizons=HORIZONS, prefer_cuda=False, asset="ETH/USDT")
    model.model = _TCN(2, 4, 1, 0.0, forecast_horizons=HORIZONS)
    model.horizon_target_scales = {
        horizon: {"return": horizon + 0.1, "mfe": horizon + 0.2, "mae": horizon + 0.3, "duration": horizon + 0.4}
        for horizon in HORIZONS
    }
    model.horizon_duration_active = {horizon: horizon != 4 for horizon in HORIZONS}
    model._is_fitted = True
    state = model.state_dict()

    restored = TCNTrendModel(config, n_features=2, forecast_horizons=(1,), prefer_cuda=False, asset="ETH/USDT")
    restored.load_state_dict(state)

    assert restored.forecast_horizons == HORIZONS
    assert restored.horizon_target_scales == model.horizon_target_scales
    assert restored.horizon_duration_active == model.horizon_duration_active
    assert set(restored.model.forecast_horizons) == set(HORIZONS)


def test_oos_decode_uses_fold_local_return_scale() -> None:
    class IdenticalRawHead(torch.nn.Module):
        def forward_outputs(self, batch: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
            outputs = {}
            for horizon in HORIZONS:
                size = len(batch)
                outputs[horizon] = {
                    "logits": torch.zeros((size, 3), device=batch.device),
                    "opportunity_logit": torch.zeros(size, device=batch.device),
                    "expected_return": torch.full((size,), 2.0, device=batch.device),
                    "expected_duration": torch.zeros(size, device=batch.device),
                    "expected_mfe": torch.ones(size, device=batch.device),
                    "expected_mae": -torch.ones(size, device=batch.device),
                }
            return outputs

    def build_fold_model(return_scale: float) -> TCNTrendModel:
        config = TrendMLConfig()
        config.batch_size = 2
        config.use_amp = False
        fold_model = TCNTrendModel(config, n_features=2, forecast_horizons=HORIZONS, prefer_cuda=False)
        fold_model.model = IdenticalRawHead()
        fold_model.mean_ = np.zeros(2, dtype=np.float64)
        fold_model.std_ = np.ones(2, dtype=np.float64)
        fold_model.horizon_target_scales = {
            horizon: {"return": return_scale, "mfe": 1.0, "mae": 1.0, "duration": 1.0}
            for horizon in HORIZONS
        }
        fold_model.horizon_duration_active = {horizon: True for horizon in HORIZONS}
        fold_model._is_fitted = True
        return fold_model

    inputs = np.zeros((2, 3, 2), dtype=np.float32)
    first_fold = build_fold_model(1.0).predict_trade_outputs_by_horizon(inputs)
    second_fold = build_fold_model(3.0).predict_trade_outputs_by_horizon(inputs)

    assert np.all(first_fold[1]["expected_return"] == 2.0)
    assert np.all(second_fold[1]["expected_return"] == 6.0)
    assert not np.array_equal(first_fold[1]["expected_return"], second_fold[1]["expected_return"])


def test_validation_loss_uses_explicit_horizon_head_and_scales() -> None:
    config = TrendMLConfig()
    config.batch_size = 2
    config.use_amp = False
    model = TCNTrendModel(config, n_features=2, forecast_horizons=HORIZONS, prefer_cuda=False)
    model.mean_ = np.zeros(2, dtype=np.float64)
    model.std_ = np.ones(2, dtype=np.float64)
    model.horizon_target_scales = {
        horizon: {"return": float(horizon), "mfe": 1.0, "mae": 1.0, "duration": 1.0}
        for horizon in HORIZONS
    }
    model.horizon_duration_active = {horizon: False for horizon in HORIZONS}

    class DifferentHeads(torch.nn.Module):
        def forward_outputs(self, batch: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
            outputs = {}
            for horizon in HORIZONS:
                value = torch.full((len(batch),), float(horizon), device=batch.device)
                outputs[horizon] = {
                    "logits": torch.zeros((len(batch), 3), device=batch.device),
                    "opportunity_logit": torch.zeros(len(batch), device=batch.device),
                    "expected_return": value,
                    "expected_mfe": value,
                    "expected_mae": value,
                    "expected_duration": value,
                }
            return outputs

    model.model = DifferentHeads()
    X = np.zeros((2, 1, 2), dtype=np.float32)
    labels = np.zeros(2, dtype=np.int64)
    zeros = np.zeros(2, dtype=np.float32)

    loss_1h = model._evaluate_validation_loss(X, labels, zeros, zeros, zeros, zeros, zeros, horizon=1)
    loss_24h = model._evaluate_validation_loss(X, labels, zeros, zeros, zeros, zeros, zeros, horizon=24)

    assert loss_1h != loss_24h


def test_duration_decoding_matches_linear_training_target_scale() -> None:
    model = _deterministic_model()
    model.horizon_target_scales = {
        horizon: {"return": 1.0, "mfe": 1.0, "mae": 1.0, "duration": 2.5}
        for horizon in HORIZONS
    }
    outputs = model.predict_trade_outputs_by_horizon(np.zeros((2, 3, 2), dtype=np.float32))

    assert np.allclose(outputs[1]["expected_duration"], outputs[1]["raw_duration"] * 2.5)


def test_negative_duration_logit_decodes_to_nonnegative_scaled_duration() -> None:
    config = TrendMLConfig()
    config.hidden_channels = 4
    config.num_layers = 1
    config.dropout = 0.0
    config.batch_size = 2
    config.use_amp = False
    model = TCNTrendModel(config, n_features=2, forecast_horizons=HORIZONS, prefer_cuda=False)
    model.model = _TCN(2, 4, 1, 0.0, forecast_horizons=HORIZONS)
    for head in model.model.heads.values():
        head.duration_head.weight.data.zero_()
        head.duration_head.bias.data.fill_(-4.0)
    model.mean_ = np.zeros(2, dtype=np.float64)
    model.std_ = np.ones(2, dtype=np.float64)
    model.horizon_target_scales = {
        horizon: {"return": 1.0, "mfe": 1.0, "mae": 1.0, "duration": 2.5}
        for horizon in HORIZONS
    }
    model.horizon_duration_active = {horizon: True for horizon in HORIZONS}
    model._is_fitted = True

    outputs = model.predict_trade_outputs_by_horizon(np.zeros((2, 3, 2), dtype=np.float32))
    expected = float(torch.nn.functional.softplus(torch.tensor(-4.0)) * 2.5)

    assert np.all(outputs[1]["raw_duration"] >= 0.0)
    assert np.all(outputs[1]["expected_duration"] >= 0.0)
    assert np.allclose(outputs[1]["expected_duration"], expected)


def test_empty_forecast_contract_contains_all_horizons_without_primary_fallback() -> None:
    result = _deterministic_model().predict_forecast_result(np.empty((0, 3, 2), dtype=np.float32)).as_dict()

    assert result["forecast_horizons"] == list(HORIZONS)
    assert set(result["expected_returns"]) == set(HORIZONS)
    assert set(result["direction_probabilities"]) == {str(horizon) for horizon in HORIZONS}
    assert result["predicted_directions"] == {str(horizon): "NO_TRADE" for horizon in HORIZONS}
