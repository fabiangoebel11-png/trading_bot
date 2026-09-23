from __future__ import annotations

import numpy as np
import pandas as pd

from core.ml.data import load_ohlcv_parquet
from core.ml.inference import load_symbol_model
from core.ml.scoring import score_bucket_metrics, validate_model1_prediction_outputs
from core.ml.train import _prepare_symbol_dataset, _to_sequences, evaluate_predictions, load_prepared_macro_matrix
from core.ml.providers import AssetSpec, canonical_cache_path
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


def test_model1_real_cached_predictions_are_finite() -> None:
    config = load_config("configs/training.yaml")
    config.ml.model_dir = "data/models"
    macro = load_prepared_macro_matrix(config.ml)
    breadth = pd.read_csv(
        "data/ml_breadth_BNB-USDT_XRP-USDT_ADA-USDT_DOGE-USDT_LINK-USDT_5m_2500d.csv",
        index_col=0,
        parse_dates=True,
    )["breadth_return"]
    for symbol in ("NASDAQ100_PROXY", "SP500_PROXY"):
        spec = AssetSpec(symbol, "QQQ" if symbol == "NASDAQ100_PROXY" else "SPY", "twelve_data", "5m", 1095)
        ohlc = load_ohlcv_parquet(canonical_cache_path(config.ml.market_cache_dir, spec))
        X, y, t1_pos, mfe, mae = _prepare_symbol_dataset(ohlc, macro, config.ml, breadth)
        X_seq, y_seq, _, _, _, _ = _to_sequences(X, y, t1_pos, config.ml.sequence_length, mfe, mae)
        model, _ = load_symbol_model(symbol, config.ml)
        outputs = model.predict_trade_outputs(X_seq[-32:])
        for value in outputs.values():
            assert np.isfinite(value).all()
        assert np.isfinite(outputs["opportunity"]).all()
        assert np.isfinite(outputs["expected_return"]).all()


def test_balanced_focal_direction_loss_is_finite() -> None:
    import torch

    logits = torch.tensor([[4.0, 0.0, -1.0], [-1.0, 0.0, 4.0]])
    labels = torch.tensor([0, 0])
    ce = torch.nn.CrossEntropyLoss(reduction="none")(logits, labels)
    focal = ((1.0 - torch.exp(-ce).clamp_min(1e-6)) ** 2.0) * ce
    assert torch.isfinite(focal).all()
    assert float(focal[1]) > float(focal[0])