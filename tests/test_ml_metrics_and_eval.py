from __future__ import annotations

import numpy as np
import pytest

from core.config import TradingBotConfig
from core.ml.scoring import score_trade_quality
from core.ml.train import evaluate_predictions


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
            label_smoothing=config.label_smoothing, reduction="sum"
        )(outputs["logits"], torch.tensor(y, device=wrapper.device))
        reg = torch.nn.SmoothL1Loss(reduction="sum")
        direct = (ce + reg(outputs["expected_mfe"], torch.tensor(mfe, device=wrapper.device)) + reg(outputs["expected_mae"], torch.tensor(mae, device=wrapper.device))) / len(X)
    assert batched == pytest.approx(float(direct.cpu()), rel=1e-6)
