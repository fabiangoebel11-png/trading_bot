from __future__ import annotations

from datetime import datetime, timezone

from core.market_state import DataHealth, FeedStatus, MarketState, PredictionSnapshot
from core.signal_orchestrator import AlertGate, AlertGateConfig


def _market() -> MarketState:
    now = datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)
    return MarketState("QQQ", now, True, "REGULAR", now, now, "NASDAQ", "UTC", True, False, 0.5)


def _health() -> DataHealth:
    now = datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)
    return DataHealth(now, now, 0.0, True, True, FeedStatus.AVAILABLE, True, True, True, True, True)


def _prediction(model: str, score: float, direction: str, expected_return: float, **kwargs) -> PredictionSnapshot:
    return PredictionSnapshot(score, direction, expected_return, 48.0, 0.02, -0.01, datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc), model, "QQQ", **kwargs)


def test_alert_gate_requires_swing_and_entry_alignment() -> None:
    swing = _prediction("swing", 92.0, "LONG", 0.02, swing_score=92.0, entry_score=55.0, swing_opportunity_score=92.0, expected_duration_days=10.0)
    intraday = _prediction("intraday", 75.0, "LONG", 0.01)
    assert AlertGate().evaluate(swing, intraday, _market(), _health()).allowed

    disagreement = _prediction("intraday", 75.0, "SHORT", -0.01)
    assert AlertGate().evaluate(swing, disagreement, _market(), _health()).allowed


def test_alert_gate_uses_strict_swing_threshold_and_fails_closed() -> None:
    swing = _prediction("swing", 80.0, "LONG", 0.02, swing_score=80.0, swing_opportunity_score=80.0)
    intraday = _prediction("intraday", 75.0, "LONG", 0.01)
    assert not AlertGate(AlertGateConfig()).evaluate(swing, intraday, _market(), _health()).allowed
