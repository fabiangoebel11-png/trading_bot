import hashlib
import json

import pandas as pd

from research.phase4_ml_baseline_analysis import (
    ANALYSIS_CSV,
    ANALYSIS_JSON,
    MANIFEST_PATH,
    RESULTS_CSV,
    RESULTS_JSON,
    build_analysis,
    load_artifacts,
    validate_artifacts,
)


def test_frozen_phase4_artifacts_are_complete_and_unique() -> None:
    results, manifest, result_json = load_artifacts()
    checks = validate_artifacts(results, manifest, result_json)
    assert all(check["ok"] for check in checks.values())
    assert len(results) == 576
    assert results["run_id"].is_unique


def test_analysis_separates_validation_and_holdout() -> None:
    results, _, _ = load_artifacts()
    analysis, context_pairs, payload = build_analysis(results)
    assert len(analysis) == 144
    assert len(context_pairs) == 216
    assert set(analysis["classification"]) <= {"ROBUST CANDIDATE", "MIXED", "WEAK", "FAILED"}
    assert payload["classification_counts"].get("FAILED", 0) == 0
    assert (analysis["validation_folds"] == 3).all()
    assert (analysis["holdout_records"] == 1).all()


def test_analysis_outputs_are_reproducible_and_match_sources() -> None:
    results = pd.read_csv(RESULTS_CSV)
    with RESULTS_JSON.open(encoding="utf-8") as handle:
        result_json = json.load(handle)
    assert len(results) == len(result_json) == 576
    assert hashlib.sha256(RESULTS_CSV.read_bytes()).hexdigest()
    assert hashlib.sha256(RESULTS_JSON.read_bytes()).hexdigest()
    assert hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    if ANALYSIS_CSV.exists() and ANALYSIS_JSON.exists():
        analysis = pd.read_csv(ANALYSIS_CSV)
        with ANALYSIS_JSON.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        assert len(analysis) == payload["analysis_record_count"] == 144
        assert payload["holdout_used_for_selection"] is False
