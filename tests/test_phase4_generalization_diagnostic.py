import json

import pandas as pd

from research.phase4_generalization_diagnostic import (
    DIAGNOSTIC_CSV,
    DIAGNOSTIC_JSON,
    DIAGNOSTIC_MANIFEST,
    KEYS,
    RESULTS_CSV,
    _baseline,
    _build_diagnostic,
    _context_pairs,
    load_artifacts,
)


def test_all_frozen_experiment_groups_have_complete_splits_and_correct_gaps() -> None:
    results, manifest, result_json = load_artifacts()
    diagnostic, _ = _build_diagnostic(results)
    assert len(diagnostic) == 144
    assert diagnostic.groupby(KEYS).size().eq(1).all()
    assert set(diagnostic["horizon"]) == {"1h", "2h", "4h", "8h", "12h", "24h", "3d", "7d", "14d", "21d", "28d"}
    assert diagnostic["validation_fold_count"].eq(3).all()
    assert diagnostic["holdout_record_count"].eq(1).all()
    assert (diagnostic["gap_roc_auc"] - (diagnostic["holdout_roc_auc"] - diagnostic["validation_roc_auc_mean"])).abs().max() < 1e-12
    assert (diagnostic["gap_brier_score"] - (diagnostic["holdout_brier_score"] - diagnostic["validation_brier_score_mean"])).abs().max() < 1e-12


def test_simple_baselines_are_derived_without_training() -> None:
    baseline = _baseline(0.4)
    assert baseline["chance_roc_auc"] == 0.5
    assert baseline["positive_rate_pr_auc"] == 0.4
    assert baseline["majority_accuracy"] == 0.6
    assert baseline["positive_rate_brier"] == 0.24
    assert baseline["always_positive_accuracy"] == 0.4
    assert baseline["always_negative_accuracy"] == 0.6


def test_context_pairs_are_complete_and_unique() -> None:
    results, _, _ = load_artifacts()
    pairs = _context_pairs(results)
    assert len(pairs) == 288
    assert set(pairs["split_type"]) == {"validation", "holdout"}
    assert pairs.groupby(["target", "timeframe", "horizon", "model_id", "split_id"]).size().eq(1).all()
    assert pairs["delta_roc_auc"].notna().all()


def test_outputs_keep_holdout_descriptive_and_manifest_frozen_flags() -> None:
    assert RESULTS_CSV.exists()
    assert DIAGNOSTIC_CSV.exists()
    assert DIAGNOSTIC_JSON.exists()
    assert DIAGNOSTIC_MANIFEST.exists()
    payload = json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(DIAGNOSTIC_MANIFEST.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 144
    assert payload["context_pair_rows"] == 288
    assert payload["holdout_used_for_selection"] is False
    assert payload["training_performed"] is False
    assert payload["gpu_training_performed"] is False
    assert payload["calibration_intervals"] == "NOT AVAILABLE FROM STORED AGGREGATES"
    assert manifest["holdout_used_for_selection"] is False
    assert manifest["training_performed"] is False


def test_baseline_comparison_is_present_in_json() -> None:
    payload = json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8"))
    comparison = payload["summaries"]["baseline_comparison"]
    assert set(comparison) >= {
        "roc_auc_above_chance_rows",
        "pr_auc_above_positive_rate_rows",
        "accuracy_above_majority_rows",
        "brier_better_than_positive_rate_rows",
        "log_loss_better_than_positive_rate_rows",
    }
