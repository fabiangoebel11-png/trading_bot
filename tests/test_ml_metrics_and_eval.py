from __future__ import annotations

import numpy as np
import pandas as pd
from core.config import TrendMLConfig
import pytest

from core.config import TradingBotConfig
from core.ml.labeling import build_horizon_label_map
from core.ml.scoring import score_trade_quality
from core.ml.train import evaluate_predictions
from core.ml.tcn_model import TCNTrendModel


def test_class_mapping_and_metrics_are_consistent() -> None:
    # Class indices [0, 1, 2] map to labels [-1, 0, 1].
    proba = np.eye(3, dtype=float)
    labels = np.array([-1, 0, 1])
    metrics = evaluate_predictions(proba, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["directional_hit_rate"] == 1.0


def test_directional_hit_rate_ignores_no_trade_labels() -> None:
    proba = np.array([
        [0.9, 0.05, 0.05],  # correctly predicts short
        [0.05, 0.9, 0.05],  # no-trade prediction
        [0.05, 0.05, 0.9],  # correctly predicts long
    ])
    labels = np.array([-1, 0, 1])
    metrics = evaluate_predictions(proba, labels)
    assert metrics["accuracy"] == 1.0
    assert metrics["directional_hit_rate"] == 1.0


def test_accuracy_and_directional_rate_use_same_decoded_predictions() -> None:
    # True labels: [-1, 0, 1, -1, 1]
    # Decoded predictions: [-1, 0, 1, 1, -1]
    proba = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.05, 0.05, 0.9],
        [0.9, 0.05, 0.05],
    ])
    metrics = evaluate_predictions(proba, np.array([-1, 0, 1, -1, 1]))
    assert metrics["accuracy"] == 3 / 5
    assert metrics["directional_hit_rate"] == 2 / 4
    assert metrics["n_directional"] == 4
    assert metrics["true_counts"] == {"-1": 2, "0": 1, "1": 2}
    assert metrics["predicted_counts"] == {"-1": 2, "0": 1, "1": 2}


def test_trade_quality_scores_do_not_honor_high_no_trade_probability() -> None:
    quality = score_trade_quality(
        np.array([[0.02, 0.95, 0.03]]),
        np.array([0.01]),
        np.array([0.01]),
    )
    assert quality["direction"][0] == "NO_TRADE"
    assert quality["score"][0] < 60.0


def test_evaluate_predictions_reports_balanced_accuracy_and_majority_baseline() -> None:
    proba = np.array([
        [0.8, 0.1, 0.1],
        [0.8, 0.1, 0.1],
        [0.1, 0.1, 0.8],
        [0.1, 0.1, 0.8],
    ], dtype=float)
    y_true = np.array([-1, -1, 1, 1], dtype=int)
    metrics = evaluate_predictions(proba, y_true)
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["majority_class_baseline"] == pytest.approx(0.5)
    assert metrics["per_class_recall"] == {"-1": 1.0, "0": 0.0, "1": 1.0}


def test_duration_targets_have_nonzero_variance_across_horizon() -> None:
    index = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
    close = pd.Series(
        np.array([
            100.0, 101.0, 103.5, 105.0, 108.0, 99.0, 101.0, 104.0,
            106.0, 108.0, 110.0, 102.0, 100.5, 102.0, 103.8, 108.0,
            96.0, 98.0, 101.0, 104.0, 107.0, 109.5, 98.0, 100.0,
            102.5, 105.5, 101.0, 99.5, 103.0, 107.0, 110.0, 100.0,
            101.0, 103.0, 105.0, 108.0, 110.5, 102.0, 100.0, 103.5,
        ], dtype=float),
        index=index,
    )
    atr = pd.Series(np.full(len(index), 1.0), index=index)
    labels = build_horizon_label_map(close, atr, (4, 12), 2.0, high=close, low=close)
    for horizon in (4, 12):
        duration = labels[horizon]["time_to_barrier"].dropna().to_numpy(dtype=float)
        assert duration.size > 0
        assert np.unique(duration).size > 1
        assert duration.std() > 0.0


def test_training_scaling_and_class_weights_are_train_only() -> None:
    torch = pytest.importorskip("torch")
    from core.ml.tcn_model import TCNTrendModel

    config = TradingBotConfig().ml
    config.batch_size = 4
    config.use_amp = False
    config.hidden_channels = 4
    config.num_layers = 1
    config.dropout = 0.0
    wrapper = TCNTrendModel(config, n_features=2)
    train_y = np.array([-1, -1, 0, 0, 0, 1, 1, 1], dtype=int)
    weights = wrapper._compute_class_weights(train_y)
    assert set(weights) == {-1, 0, 1}
    assert weights[0] < weights[-1] or weights[0] < weights[1]
    assert max(weights.values()) <= 10.0


def test_batched_validation_loss_matches_direct_loss() -> None:
    torch = pytest.importorskip("torch")
    from core.ml.tcn_model import TCNTrendModel

    config = TradingBotConfig().ml
    config.batch_size = 3
    config.use_amp = False
    config.hidden_channels = 4
    config.num_layers = 1
    config.dropout = 0.0
    wrapper = TCNTrendModel(config, n_features=2)

    class DummyModel(torch.nn.Module):
        def forward_outputs(self, inputs):
            first = inputs[:, 0, 0]
            logits = torch.stack((first, -first, torch.zeros_like(first)), dim=1)
            return {"logits": logits, "expected_mfe": first, "expected_mae": -first}

    wrapper.model = DummyModel().to(wrapper.device)
    X = np.arange(10 * 2, dtype=np.float32).reshape(10, 1, 2) / 10.0
    y = np.arange(10, dtype=np.int64) % 3
    mfe = X[:, 0, 0].astype(np.float32)
    mae = -mfe
    batched = wrapper._evaluate_validation_loss(X, y, mfe, mae)

    with torch.no_grad():
        outputs = wrapper.model.forward_outputs(torch.tensor(X, device=wrapper.device))
        ce = torch.nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing, reduction="none"
        )(outputs["logits"], torch.tensor(y, device=wrapper.device))
        reg = torch.nn.SmoothL1Loss(reduction="none")
        direction_term = config.direction_loss_weight * ce.mean()
        mfe_term = config.mfe_loss_weight * reg(outputs["expected_mfe"], torch.tensor(mfe, device=wrapper.device)).mean()
        mae_term = config.mae_loss_weight * reg(outputs["expected_mae"], torch.tensor(mae, device=wrapper.device)).mean()
        weights = config.direction_loss_weight + config.mfe_loss_weight + config.mae_loss_weight
        direct = (direction_term + mfe_term + mae_term) / weights

    assert float(batched) == pytest.approx(float(direct.cpu()), rel=1e-6)


def test_validation_loss_is_batch_size_invariant_for_same_samples() -> None:
    torch = pytest.importorskip("torch")
    from core.ml.tcn_model import TCNTrendModel

    config = TradingBotConfig().ml
    config.batch_size = 4
    config.use_amp = False
    config.hidden_channels = 4
    config.num_layers = 1
    config.dropout = 0.0
    wrapper = TCNTrendModel(config, n_features=2)

    class DummyModel(torch.nn.Module):
        def forward_outputs(self, inputs):
            first = inputs[:, 0, 0]
            logits = torch.stack((first, -first, torch.zeros_like(first)), dim=1)
            return {"logits": logits, "expected_mfe": first, "expected_mae": -first}

    wrapper.model = DummyModel().to(wrapper.device)
    X = np.linspace(0.0, 1.0, num=24, dtype=np.float32).reshape(12, 2, 1)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    mfe = np.linspace(0.1, 0.6, num=12, dtype=np.float32)
    mae = np.linspace(0.2, 0.7, num=12, dtype=np.float32)

    loss_full = wrapper._evaluate_validation_loss(X, y, mfe, mae)
    loss_half = wrapper._evaluate_validation_loss(X[:6], y[:6], mfe[:6], mae[:6])
    loss_two = wrapper._evaluate_validation_loss(X[6:], y[6:], mfe[6:], mae[6:])
    assert loss_full == pytest.approx((loss_half + loss_two) / 2.0, rel=1e-5)


def test_constant_duration_target_is_skipped_in_loss() -> None:
    torch = pytest.importorskip("torch")
    from core.ml.tcn_model import TCNTrendModel

    config = TradingBotConfig().ml
    config.batch_size = 4
    config.use_amp = False
    config.hidden_channels = 4
    config.num_layers = 1
    config.dropout = 0.0
    config.max_epochs = 2
    config.val_fraction = 0.2

    X = np.random.default_rng(0).normal(size=(80, 8, 2)).astype(np.float32)
    y = np.array(
        [0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0,
         1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1,
         0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0,
         1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1,
         0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0,
         1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1], dtype=np.int64
    )
    wrapper = TCNTrendModel(config, n_features=2)
    wrapper.fit(
        X,
        y,
        sample_weight=np.ones(len(X), dtype=np.float32),
        mfe_seq=np.abs(np.linspace(0.01, 0.05, num=len(X), dtype=np.float32)),
        mae_seq=np.abs(np.linspace(0.01, 0.04, num=len(X), dtype=np.float32)),
        opportunity_seq=np.linspace(0.2, 0.8, num=len(X), dtype=np.float32),
        return_seq=np.linspace(-0.02, 0.02, num=len(X), dtype=np.float32),
        duration_seq=np.ones(len(X), dtype=np.float32),
    )

    assert wrapper.duration_target_scale == pytest.approx(1.0)
    assert all(row.get("loss_duration", 0.0) == 0.0 for row in wrapper.training_history)


def test_near_constant_features_are_zeroed_after_fold_normalization() -> None:
    config = TrendMLConfig()
    config.hidden_channels = 4
    config.num_layers = 1
    config.dropout = 0.0
    config.batch_size = 2
    config.use_amp = False
    model = TCNTrendModel(config, n_features=2, prefer_cuda=False)
    model.mean_ = np.array([5.0, 0.0])
    model.std_ = np.array([1e-6, 2.0])
    model.constant_feature_mask_ = np.array([True, False])

    values = np.array([[[5.0, 2.0], [6.0, 4.0]]], dtype=np.float32)
    normalized = model._normalize(values)

    assert np.all(normalized[..., 0] == 0.0)
    assert np.allclose(normalized[..., 1], [[1.0, 2.0]])
    assert np.isfinite(normalized).all()


def test_tcn_horizon_heads_have_the_expected_multi_horizon_contract() -> None:
    torch = pytest.importorskip("torch")
    from core.ml.tcn_model import _TCN

    model = _TCN(3, 8, 1, 0.0, forecast_horizons=(1, 4, 8, 12, 24))
    x = torch.randn(2, 16, 3)
    outputs = model.forward_outputs(x)

    assert set(outputs) == {1, 4, 8, 12, 24}
    assert all(set(outputs[h]) >= {"logits", "expected_return", "expected_opportunity", "expected_mfe", "expected_mae", "expected_duration"} for h in outputs)
    for h in outputs:
        assert outputs[h]["logits"].shape == (2, 3)
        assert outputs[h]["expected_return"].shape == (2,)
        assert outputs[h]["expected_opportunity"].shape == (2,)
        assert outputs[h]["expected_mfe"].shape == (2,)
        assert outputs[h]["expected_mae"].shape == (2,)
        assert outputs[h]["expected_duration"].shape == (2,)


def test_multi_horizon_targets_are_independent_across_heads() -> None:
    torch = pytest.importorskip("torch")
    from core.ml.tcn_model import _TCN

    model = _TCN(3, 8, 1, 0.0, forecast_horizons=(1, 24))
    x = torch.randn(8, 16, 3)
    outputs = model.forward_outputs(x)
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")

    h1_target = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1], dtype=torch.long)
    h24_target = torch.tensor([1, 2, 1, 0, 2, 1, 0, 2], dtype=torch.long)
    h24_target_changed = torch.tensor([2, 2, 1, 0, 2, 1, 0, 1], dtype=torch.long)

    h1_loss = loss_fn(outputs[1]["logits"], h1_target)
    h24_loss = loss_fn(outputs[24]["logits"], h24_target)
    h24_loss_changed = loss_fn(outputs[24]["logits"], h24_target_changed)

    assert abs(float(h24_loss_changed - h24_loss)) > 1e-6
    assert abs(float(h1_loss - loss_fn(outputs[1]["logits"], h1_target))) < 1e-8


def test_horizon_duration_flags_are_set_per_horizon() -> None:
    torch = pytest.importorskip("torch")
    from core.ml.tcn_model import TCNTrendModel, _TCN

    config = TradingBotConfig().ml
    config.batch_size = 4
    config.use_amp = False
    config.hidden_channels = 4
    config.num_layers = 1
    config.dropout = 0.0
    config.val_fraction = 0.2

    wrapper = TCNTrendModel(config, n_features=2, forecast_horizons=(1, 4, 8, 12, 24))
    wrapper.model = _TCN(2, 4, 1, 0.0, forecast_horizons=(1, 4, 8, 12, 24)).to(wrapper.device)
    wrapper.model.eval()
    wrapper.horizon_duration_active = {1: False, 4: True, 8: True, 12: True, 24: True}
    wrapper.duration_head_active = True
    wrapper.return_target_scale = 1.0
    wrapper.mfe_target_scale = 1.0
    wrapper.mae_target_scale = 1.0
    wrapper.duration_target_scale = 1.0

    x = np.random.default_rng(0).normal(size=(8, 4, 2)).astype(np.float32)
    y = np.array([-1, 0, 1, -1, 0, 1, 0, -1], dtype=np.float32)
    val_loss = wrapper._evaluate_validation_loss(
        x,
        (y + 1).astype(np.int64),
        np.abs(np.linspace(0.01, 0.05, num=len(x), dtype=np.float32)),
        np.abs(np.linspace(0.01, 0.04, num=len(x), dtype=np.float32)),
        np.linspace(0.2, 0.8, num=len(x), dtype=np.float32),
        np.linspace(-0.02, 0.02, num=len(x), dtype=np.float32),
        np.linspace(0.5, 1.5, num=len(x), dtype=np.float32),
    )

    assert np.isfinite(val_loss)
    assert wrapper.horizon_duration_active[1] is False
    assert wrapper.horizon_duration_active[4] is True
    assert wrapper.horizon_duration_active[8] is True
    assert wrapper.horizon_duration_active[12] is True
    assert wrapper.horizon_duration_active[24] is True
