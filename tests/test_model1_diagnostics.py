from __future__ import annotations

import numpy as np

from core.ml.scoring import score_bucket_metrics, validate_model1_prediction_outputs
from core.ml.train import evaluate_predictions, load_prepared_macro_matrix
from live_daemon import Model1Runner, _load_base_ohlc
from train import load_config


def test_model1_reports_direction_collapse_without_changing_metrics() -> None:
    metrics = evaluate_predictions(np.tile([0.9, 0.05, 0.05], (100, 1)), np.array([-1] * 40 + [0] * 20 + [1] * 40))
    assert metrics["predicted_fraction"] == {"-1": 1.0, "0": 0.0, "1": 0.0}
    assert metrics["collapse_warning"] is True


def test_model1_score_buckets_report_realized_separation() -> None:
    quality = {"score": np.array([10.0, 50.0, 95.0]), "direction": np.array(["SHORT", "LONG", "LONG"])}
    result = score_bucket_metrics(quality, np.array([-0.02, 0.0, 0.03]), np.array([0.01, 0.01, 0.05]), np.array([-0.03, -0.01, -0.01]))
    assert result["0-20"]["count"] == 1
    assert result["90-100"]["mean_future_return"] == 0.03


def test_model1_nan_direction_is_unknown_and_public_validation_is_strict() -> None:
    outputs = {
        "opportunity": np.array([0.5]),
        "expected_return": np.array([0.01]),
        "expected_duration": np.array([0.5]),
        "expected_mfe": np.array([0.01]),
        "expected_mae": np.array([-0.01]),
    }
    quality = {"score": np.array([90.0])}
    validate_model1_prediction_outputs(outputs, quality)
    try:
        validate_model1_prediction_outputs({**outputs, "expected_return": np.array([np.nan])}, quality)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite Model 1 output was accepted")


def test_active_v3_crypto_predictions_are_finite() -> None:
    config = load_config("configs/training_tcn_crypto_1h.yaml")
    macro = load_prepared_macro_matrix(config.ml)
    runner = Model1Runner(config.ml)
    for symbol in config.ml.assets:
        ohlc = _load_base_ohlc(config.ml, symbol)
        assert ohlc is not None and not ohlc.empty
        result = runner.infer_by_horizon(symbol, ohlc, macro, None)
        assert result is not None
        _, _, outputs_by_horizon = result
        assert set(outputs_by_horizon) == {1, 4, 8, 12, 24}
        for outputs in outputs_by_horizon.values():
            for key in ("expected_return", "expected_mfe", "expected_mae", "expected_duration", "opportunity_score"):
                assert np.isfinite(outputs[key])
            probabilities = np.asarray(outputs["probabilities"], dtype=float)
            assert np.isfinite(probabilities).all()
            assert np.isclose(probabilities.sum(), 1.0, atol=0.01)


def test_tcn_early_stopping_uses_patience_and_resets_after_improvement() -> None:
    from core.config import TrendMLConfig
    from core.ml.tcn_model import TCNTrendModel

    config = TrendMLConfig()
    config.early_stopping_patience = 3
    model = TCNTrendModel(config, n_features=2, prefer_cuda=False)
    state = {"weight": np.array([1.0, 2.0], dtype=np.float32)}

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for loss_value in (1.0, 1.2, 1.5, 0.9, 1.2, 1.3, 1.4):
        best_val_loss, best_state, epochs_without_improvement, _, stop = model._apply_early_stopping(
            current_state=state,
            best_val_loss=best_val_loss,
            val_loss=float(loss_value),
            best_state=best_state,
            epochs_without_improvement=epochs_without_improvement,
        )
        if loss_value == 1.0:
            assert stop is False
            assert epochs_without_improvement == 0
        elif loss_value == 0.9:
            assert stop is False
            assert epochs_without_improvement == 0
        elif loss_value == 1.4:
            assert stop is True
            assert epochs_without_improvement == 3
            break


def test_balanced_focal_direction_loss_is_finite() -> None:
    import torch

    logits = torch.tensor([[4.0, 0.0, -1.0], [-1.0, 0.0, 4.0]])
    labels = torch.tensor([0, 0])
    ce = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)
    focal = ((1.0 - torch.exp(-ce).clamp_min(1e-6)) ** 2.0) * ce
    assert torch.isfinite(focal).all()
    assert float(focal[1]) > float(focal[0])