from __future__ import annotations

import pandas as pd

from research.partial_exit_holdout import _lookahead_validation, registered_policies
from research.partial_exit_simulator import PartialExitEngine


def test_registered_holdout_policies_are_exact():
    policies = {policy.policy_id: policy for policy in registered_policies()}

    assert tuple(policies) == (
        "control_full_position",
        "scaleout_a_25_25_50",
        "scaleout_b_50_50",
    )
    assert policies["scaleout_a_25_25_50"].allocations == (0.25, 0.25, 0.50)
    assert policies["scaleout_a_25_25_50"].take_profit.values == (1.0, 2.0)
    assert policies["scaleout_a_25_25_50"].activate_trailing_after_stage == 2
    assert policies["scaleout_b_50_50"].allocations == (0.50, 0.50)
    assert policies["scaleout_b_50_50"].take_profit.values == (1.5,)
    assert policies["scaleout_b_50_50"].activate_trailing_after_stage == 1


def test_future_extremes_do_not_determine_earlier_partial_exit():
    index = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    base = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [100.0, 100.0, 102.0, 101.0, 101.0, 101.0],
            "low": [100.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 101.0, 101.0, 100.0, 100.0],
            "atr": [1.0] * 6,
            "signal_position": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            "realized_vol": [0.1] * 6,
        },
        index=index,
    )
    altered = base.copy()
    altered.loc[index[5], "high"] = 10_000.0
    altered.loc[index[5], "low"] = 0.01
    policy = registered_policies()[1]

    base_result, base_trades = PartialExitEngine(0.00035, 0.0002).execute(base, policy, 2.5)
    altered_result, altered_trades = PartialExitEngine(0.00035, 0.0002).execute(altered, policy, 2.5)

    assert base_trades.to_dict("records") == altered_trades.to_dict("records")
    assert base_result.iloc[:5].equals(altered_result.iloc[:5])


def test_runtime_lookahead_probe_passes():
    assert _lookahead_validation()["status"] == "PASSED"