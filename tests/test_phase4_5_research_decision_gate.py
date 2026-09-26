import json

import pandas as pd
import pytest

from research import phase4_5_research_decision_gate as gate


def test_all_required_prior_artifacts_are_present() -> None:
    status = gate._required_status()
    assert all(status.values()), [name for name, present in status.items() if not present]


def test_missing_artifact_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(gate.INPUTS, "phase4_analysis", gate.ROOT / "missing-phase4-analysis.csv")
    with pytest.raises(FileNotFoundError, match="phase4_analysis"):
        gate.build_synthesis()


def test_synthesis_is_deterministic_and_holdout_is_descriptive() -> None:
    first = gate.build_synthesis()
    second = gate.build_synthesis()
    assert first["evidence_matrix"] == second["evidence_matrix"]
    assert first["decision_gate"] == second["decision_gate"]
    assert first["phase4"]["holdout_selection"] is False
    assert first["phase4"]["training_performed"] is False
    assert first["phase4"]["gpu_used"] is False


def test_expected_phase_counts_and_findings() -> None:
    payload = gate.build_synthesis()
    assert payload["phase4"]["baseline_rows"] == 144
    assert payload["phase4"]["deep_rows"] == 124
    assert payload["phase4"]["generalization_rows"] == 144
    assert payload["phase4"]["label_rows"] == 96
    assert payload["phase4"]["label_task_count"] == 24
    assert payload["phase4"]["evidence_classification"] == "SIGNAL_WEAK"
    assert payload["phase4"]["label_value_classification"] == "LABEL_VALUE_UNCLEAR"


def test_decision_gate_does_not_promote_or_change_runtime() -> None:
    payload = gate.build_synthesis()
    assert payload["decision_gate"]["path"] == "PATH A"
    assert payload["decision_gate"]["live_release"] is False
    assert payload["decision_gate"]["ml_promotion"] is False
    source = gate.Path(gate.__file__).read_text(encoding="utf-8")
    assert ".fit(" not in source
    assert "torch" not in source.lower()


def test_generated_matrix_matches_json_after_run() -> None:
    gate.run()
    matrix = pd.read_csv(gate.CSV_PATH)
    payload = json.loads(gate.JSON_PATH.read_text(encoding="utf-8"))
    assert len(matrix) == 8
    assert matrix.to_dict(orient="records") == payload["evidence_matrix"]
    manifest = json.loads(gate.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["training_performed"] is False
    assert manifest["holdout_used_for_selection"] is False