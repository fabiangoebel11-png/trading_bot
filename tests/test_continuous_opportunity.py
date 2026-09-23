from __future__ import annotations

import numpy as np

from core.ml.scoring import continuous_opportunity_score


def test_continuous_score_is_defined_and_direction_is_separate() -> None:
    quality = continuous_opportunity_score(
        opportunity=np.array([0.25, 0.70, 0.95]),
        expected_return=np.array([-0.001, 0.0, 0.004]),
        expected_favorable=np.array([0.002, 0.004, 0.012]),
        expected_adverse=np.array([-0.003, -0.002, -0.003]),
    )
    assert quality["score"].shape == (3,)
    assert np.all((quality["score"] >= 0) & (quality["score"] <= 100))
    assert quality["direction"].tolist() == ["SHORT", "UNCERTAIN", "LONG"]
    assert quality["score"][2] > quality["score"][0]


def test_score_is_monotonic_for_stronger_opportunities() -> None:
    weak = continuous_opportunity_score(np.array([0.3]), np.array([0.001]), np.array([0.002]), np.array([-0.004]))
    strong = continuous_opportunity_score(np.array([0.9]), np.array([0.008]), np.array([0.012]), np.array([-0.002]))
    assert strong["score"][0] > weak["score"][0]
