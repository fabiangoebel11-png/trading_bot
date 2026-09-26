import numpy as np
import pandas as pd

from research.phase4_ml_baseline_evaluation import (
    MODEL_PARAMS,
    build_split_masks,
    calculate_metrics,
    create_model,
    fit_predict_train_only,
)


def test_train_only_scaling_is_insensitive_to_future_feature_mutation() -> None:
    train_x = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0]})
    train_y = pd.Series([0, 0, 1, 1])
    evaluation_x = pd.DataFrame({"x": [0.5, 1.5]})
    first = fit_predict_train_only("logistic_regression", train_x, train_y, evaluation_x)
    mutated_future = evaluation_x.copy()
    mutated_future.loc[:, "x"] = [5000.0, -5000.0]
    second = fit_predict_train_only("logistic_regression", train_x, train_y, mutated_future)
    assert np.allclose(first, fit_predict_train_only("logistic_regression", train_x, train_y, evaluation_x))
    assert not np.allclose(first, second)


def test_split_masks_use_horizon_specific_purge_and_embargo() -> None:
    index = pd.date_range("2020-01-01", "2026-09-22", freq="h", tz="UTC")
    for horizon, expected in (("4h", "0 days 04:00:00"), ("24h", "1 days 00:00:00"), ("7d", "7 days 00:00:00"), ("14d", "14 days 00:00:00")):
        masks = build_split_masks(index, horizon, "wf_2022")
        assert masks.purge == expected
        assert masks.embargo == expected
        assert not np.any(masks.train & masks.evaluation)


def test_models_and_seed_are_frozen() -> None:
    assert set(MODEL_PARAMS) == {"logistic_regression", "hist_gradient_boosting", "random_forest"}
    first = create_model("random_forest")
    second = create_model("random_forest")
    assert first.get_params()["random_state"] == 42
    assert first.get_params() == second.get_params()


def test_core_and_context_columns_are_separate() -> None:
    core = ["btc_usdt_return_1", "btc_usdt_rsi_14"]
    context = core + ["macro_es_f_proxy_return_1", "macro_nq_f_proxy_return_1", "macro_vix_proxy_return_1", "macro_urth_proxy_return_1"]
    assert not any(column.startswith("macro_") for column in core)
    assert set(context) - set(core) == {
        "macro_es_f_proxy_return_1", "macro_nq_f_proxy_return_1",
        "macro_vix_proxy_return_1", "macro_urth_proxy_return_1",
    }


def test_future_return_mutation_changes_only_information_diagnostic() -> None:
    y = pd.Series([0, 1, 0, 1])
    probability = np.array([0.2, 0.8, 0.4, 0.7])
    original = calculate_metrics(y, probability, pd.Series([0.01, 0.02, -0.01, 0.03]))
    mutated = calculate_metrics(y, probability, pd.Series([100.0, -100.0, 100.0, -100.0]))
    assert original["roc_auc"] == mutated["roc_auc"]
    assert original["brier_score"] == mutated["brier_score"]
    assert original["mean_forward_return"] != mutated["mean_forward_return"]