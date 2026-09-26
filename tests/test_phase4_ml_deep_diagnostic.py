import json

import pandas as pd

from research.phase4_ml_deep_diagnostic import (
    DIAGNOSTIC_CSV,
    DIAGNOSTIC_JSON,
    DIAGNOSTIC_MANIFEST,
    _candidate_rows,
    _paired_context,
    load_artifacts,
)
from research.phase4_ml_baseline_analysis import ANALYSIS_CSV


def test_all_phase41_candidates_have_complete_walk_forward_diagnostics() -> None:
    results, _, _ = load_artifacts()
    analysis = pd.read_csv(ANALYSIS_CSV)
    diagnostic, dossiers, weak = _candidate_rows(results, analysis)
    assert len(dossiers) == 31
    assert len(diagnostic) == 31 * 4
    assert diagnostic.groupby(["target", "timeframe", "horizon", "feature_set_id", "model_id"]).size().eq(4).all()
    assert set(diagnostic["split_id"]) == {"wf_2022", "wf_2023", "wf_2024", "final_holdout"}
    assert weak["count"] == 1


def test_core_context_pairs_match_same_experiment_keys() -> None:
    results, _, _ = load_artifacts()
    analysis = pd.read_csv(ANALYSIS_CSV)
    candidates = analysis[analysis["classification"] == "ROBUST CANDIDATE"]
    pairs = pd.concat([_paired_context(results, row) for _, row in candidates.iterrows()], ignore_index=True)
    assert len(pairs) == 31 * 4
    assert set(pairs["split_id"]) == {"wf_2022", "wf_2023", "wf_2024", "final_holdout"}
    assert pairs[["target", "timeframe", "horizon", "model_id", "split_id", "candidate_feature_set"]].duplicated().sum() == 0


def test_deep_diagnostic_outputs_are_present_and_holdout_is_not_selection_input() -> None:
    assert DIAGNOSTIC_CSV.exists()
    assert DIAGNOSTIC_JSON.exists()
    assert DIAGNOSTIC_MANIFEST.exists()
    payload = json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(DIAGNOSTIC_MANIFEST.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 31
    assert payload["holdout_used_for_selection"] is False
    assert payload["training_performed"] is False
    assert manifest["candidate_count"] == 31
    assert manifest["holdout_used_for_selection"] is False
