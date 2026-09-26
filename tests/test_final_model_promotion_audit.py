import pandas as pd

from research.final_model_promotion_audit import _tcn_status, gate_positions


def test_tcn_task_status_rejects_mismatched_oos_cadence() -> None:
    status, reason, confidence = _tcn_status("BTC/USDT", "1h")
    assert status == "MODEL_UNAVAILABLE"
    assert "cadence" in reason
    assert confidence is None


def test_model_gate_keeps_only_directionally_confirmed_rule_entry() -> None:
    index = pd.date_range("2026-01-01", periods=4, freq="1h", tz="UTC")
    rule = pd.DataFrame({"position": [0.0, 1.0, 1.0, 0.0]}, index=index)
    forecast = pd.DataFrame({"direction": ["SHORT"]}, index=index[1:2])
    assert gate_positions(rule, forecast)["position"].tolist() == [0.0, 0.0, 0.0, 0.0]