from __future__ import annotations

from datetime import datetime, timezone

from core.market_state import (
    DataHealth, FeedStatus, MarketStateService, PredictionSnapshot, decide_alert, snapshot_dict
)


def _health(service, market, *, fresh=True, context=True):
    last = datetime(2026, 9, 22, 14, 55, tzinfo=timezone.utc) if fresh else datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc)
    return service.health(market, last_candle=last, expected_candle_time=last, now=datetime(2026, 9, 22, 14, 55, tzinfo=timezone.utc), context_available=context, context_freshness_ok=context)


def _prediction(score: float) -> PredictionSnapshot:
    return PredictionSnapshot(score, "LONG", 0.01, 48, 0.02, -0.005, datetime(2026, 9, 22, 14, 55, tzinfo=timezone.utc))


def test_us_sessions_weekend_holiday_and_dst() -> None:
    service = MarketStateService()
    regular = service.state_at("QQQ", datetime(2026, 3, 9, 14, 0, tzinfo=timezone.utc))
    pre = service.state_at("SPY", datetime(2026, 3, 9, 12, 0, tzinfo=timezone.utc))
    weekend = service.state_at("QQQ", datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc))
    holiday = service.state_at("SPY", datetime(2026, 7, 3, 15, 0, tzinfo=timezone.utc))
    assert regular.session_type == "REGULAR"
    assert pre.session_type == "PRE_MARKET"
    assert weekend.market_open is False
    assert holiday.is_holiday is True


def test_crypto_is_24_7() -> None:
    state = MarketStateService().state_at("BTC/USDT", datetime(2026, 1, 1, 3, tzinfo=timezone.utc))
    assert state.market_open and state.session_type == "24_7"


def test_alert_gate_preserves_score_and_fails_closed() -> None:
    service = MarketStateService()
    market = service.state_at("QQQ", datetime(2026, 9, 22, 15, 0, tzinfo=timezone.utc))
    assert decide_alert(_prediction(94), market, _health(service, market), threshold=90).allowed
    closed = service.state_at("QQQ", datetime(2026, 9, 22, 22, 0, tzinfo=timezone.utc))
    decision = decide_alert(_prediction(94), closed, _health(service, closed), threshold=90)
    assert not decision.allowed and decision.score == 94
    assert not decide_alert(_prediction(75), market, _health(service, market), threshold=90).allowed
    assert not decide_alert(_prediction(94), market, _health(service, market, fresh=False), threshold=90).allowed
    assert not decide_alert(_prediction(94), market, _health(service, market, context=False), threshold=90).allowed


def test_state_serialization_is_gui_ready() -> None:
    service = MarketStateService()
    market = service.state_at("BTC/USDT", datetime(2026, 9, 22, 15, 0, tzinfo=timezone.utc))
    health = _health(service, market)
    payload = snapshot_dict(_prediction(87), market, health, decide_alert(_prediction(87), market, health))
    assert payload["prediction"]["score"] == 87
    assert payload["market"]["session_type"] == "24_7"
    assert payload["health"]["feed_healthy"] is True


def test_feed_health_missing_and_stale_statuses() -> None:
    service = MarketStateService()
    market = service.state_at("BTC/USDT", datetime(2026, 9, 22, 15, 0, tzinfo=timezone.utc))
    missing = service.health(market, last_candle=None, expected_candle_time=None)
    stale = service.health(market, last_candle=datetime(2026, 9, 22, 14, 0, tzinfo=timezone.utc), expected_candle_time=None, now=datetime(2026, 9, 22, 15, 0, tzinfo=timezone.utc))
    assert missing.primary_status == FeedStatus.MISSING
    assert stale.primary_status == FeedStatus.STALE


def test_prediction_duration_is_decoupled_from_leverage() -> None:
    prediction = _prediction(87)
    assert prediction.expected_duration_bars == 48
    assert prediction.expected_duration_minutes == 240.0
