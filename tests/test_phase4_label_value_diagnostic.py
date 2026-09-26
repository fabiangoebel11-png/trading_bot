import json

import numpy as np
import pandas as pd

from core.ml.research_design import label_end_times, purged_train_mask
from research.phase4_label_value_diagnostic import (
    CONFIG_PATH,
    DIAGNOSTIC_CSV,
    DIAGNOSTIC_JSON,
    DIAGNOSTIC_MANIFEST,
    PHASE3_MANIFEST,
    _outcomes,
    _validate_frozen_contract,
)


def test_label_definitions_and_cost_threshold_are_exact() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100, 100, 100, 100, 100, 100],
            "high": [100, 100.5, 101, 100, 100, 100],
            "low": [100, 100, 100, 99, 100, 100],
            "close": [100, 100.1, 100.3, 99.9, 100, 100],
            "volume": [1] * 6,
        },
        index=index,
    )
    outcomes = _outcomes(frame, "1h", "2h")
    expected_return = frame["close"].shift(-2) / frame["close"] - 1
    assert np.allclose(outcomes["future_return"].dropna(), expected_return.dropna())
    assert (outcomes.loc[expected_return.notna(), "direction"] == (expected_return.dropna() > 0).astype(float)).all()
    assert (outcomes.loc[expected_return.notna(), "cost_aware_direction"] == (expected_return.dropna() > 0.002).astype(float)).all()
    assert outcomes["label_end"].iloc[0] == index[0] + pd.Timedelta(hours=2)


def test_frozen_contract_and_purge_embargo_are_unchanged() -> None:
    contract = _validate_frozen_contract()
    assert contract["task_matrix_matches"] is True
    assert contract["task_count"] == 24
    assert contract["holdout_selection_allowed"] is True
    assert contract["purge_rule_present"] is True
    assert contract["embargo_rule_present"] is True
    sample_times = pd.date_range("2022-01-01", periods=5, freq="1h", tz="UTC")
    label_ends = label_end_times(sample_times, pd.Timedelta(hours=2))
    mask = purged_train_mask(sample_times, label_ends, pd.Timestamp("2022-01-01T04:00:00Z"), pd.Timedelta(hours=2))
    assert mask.tolist() == [True, True, False, False, False]
    assert CONFIG_PATH.exists()
    assert PHASE3_MANIFEST.exists()


def test_phase44_artifacts_are_complete_and_holdout_is_descriptive() -> None:
    assert DIAGNOSTIC_CSV.exists()
    assert DIAGNOSTIC_JSON.exists()
    assert DIAGNOSTIC_MANIFEST.exists()
    diagnostic = pd.read_csv(DIAGNOSTIC_CSV)
    payload = json.loads(DIAGNOSTIC_JSON.read_text(encoding="utf-8"))
    manifest = json.loads(DIAGNOSTIC_MANIFEST.read_text(encoding="utf-8"))
    assert len(diagnostic) == 96
    assert diagnostic.groupby(["target", "timeframe", "horizon"]).size().eq(4).all()
    assert set(diagnostic["split_id"]) == {"wf_2022", "wf_2023", "wf_2024", "final_holdout"}
    assert payload["training_performed"] is False
    assert payload["gpu_used"] is False
    assert payload["holdout_used_for_selection"] is False
    assert payload["evidence_classification"] == "LABEL_VALUE_UNCLEAR"
    assert payload["final_conclusion"] == "NO LABEL-BASED ML ADVANTAGE DEMONSTRATED"
    assert set(payload["unavailable_labels"]) == {"stop_hit_probability", "holding_time"}
    assert manifest["training_performed"] is False
    assert manifest["holdout_used_for_selection"] is False


def test_label_reconstruction_is_reproducible() -> None:
    first = pd.read_csv(DIAGNOSTIC_CSV)
    second = pd.read_csv(DIAGNOSTIC_CSV)
    pd.testing.assert_frame_equal(first, second)
