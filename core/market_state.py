"""Historical/live market state, data health, and alert gating.

This module is deliberately read-only: it consumes timestamps and cached-feed
metadata and never fetches data or places orders.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, datetime, time, timedelta
from enum import Enum
from typing import Any, Protocol
from zoneinfo import ZoneInfo


class FeedStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    CLOSED = "CLOSED"
    STALE = "STALE"
    MISSING = "MISSING"
    INVALID = "INVALID"


@dataclass(frozen=True)
class MarketState:
    asset: str
    timestamp: datetime
    market_open: bool
    session_type: str
    next_open: datetime | None
    next_close: datetime | None
    exchange_or_calendar: str
    timezone: str
    is_trading_day: bool
    is_holiday: bool
    session_progress: float | None


@dataclass(frozen=True)
class DataHealth:
    last_candle: datetime | None
    expected_candle_time: datetime | None
    data_age_seconds: float | None
    freshness_ok: bool
    feed_healthy: bool
    primary_status: FeedStatus
    context_available: bool
    context_freshness_ok: bool
    context_status: FeedStatus
    feature_ready: bool
    warmup_ready: bool
    reason: str = ""


@dataclass(frozen=True)
class PredictionSnapshot:
    score: float
    direction: str
    expected_return: float
    expected_duration_bars: float
    expected_mfe: float
    expected_mae: float
    timestamp: datetime
    model_type: str = "intraday"
    asset: str = ""
    swing_score: float | None = None
    entry_score: float | None = None
    swing_opportunity_score: float | None = None
    expected_duration_days: float | None = None

    @property
    def expected_duration_minutes(self) -> float:
        return self.expected_duration_bars * 5.0


@dataclass(frozen=True)
class AlertDecision:
    allowed: bool
    reason: str
    score: float
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RiskEngineInput(Protocol):
    def evaluate(self, prediction: PredictionSnapshot, market: MarketState, health: DataHealth) -> Any: ...


def _observed(year: int, month: int, day: int) -> date:
    value = date(year, month, day)
    if value.weekday() == 5:
        return value - timedelta(days=1)
    if value.weekday() == 6:
        return value + timedelta(days=1)
    return value


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _us_holidays(year: int) -> set[date]:
    holidays = {
        _observed(year, 1, 1), _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
        _last_weekday(year, 5, 0), _observed(year, 7, 4), _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 11, 3, 4), _observed(year, 12, 25),
    }
    if year >= 2022:
        holidays.add(_observed(year, 6, 19))
    # Good Friday: sufficient for NYSE/Nasdaq session reconstruction.
    easter = _easter(year)
    holidays.add(easter - timedelta(days=2))
    return holidays


def _easter(year: int) -> date:
    a = year % 19; b = year // 100; c = year % 100; d = b // 4; e = b % 4
    f = (b + 8) // 25; g = (b - f + 1) // 3; h = (19 * a + b - d - g + 15) % 30
    i = c // 4; k = c % 4; l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def _is_us_holiday(day: date) -> bool:
    return day in _us_holidays(day.year)


def _next_session_day(day: date, market: str) -> date:
    candidate = day + timedelta(days=1)
    while candidate.weekday() >= 5 or (market == "us_equity" and _is_us_holiday(candidate)):
        candidate += timedelta(days=1)
    return candidate


def _us_state(asset: str, timestamp: datetime) -> MarketState:
    tz = ZoneInfo("America/New_York")
    local = timestamp.astimezone(tz)
    day = local.date()
    holiday = day.weekday() >= 5 or _is_us_holiday(day)
    regular_open = datetime.combine(day, time(9, 30), tzinfo=tz)
    regular_close = datetime.combine(day, time(16), tzinfo=tz)
    pre_open = datetime.combine(day, time(4), tzinfo=tz)
    after_close = datetime.combine(day, time(20), tzinfo=tz)
    if holiday:
        nxt = _next_session_day(day, "us_equity")
        return MarketState(asset, timestamp, False, "CLOSED", datetime.combine(nxt, time(4), tzinfo=tz), None, "NYSE/Nasdaq", str(tz), False, True, None)
    if local < pre_open:
        return MarketState(asset, timestamp, False, "CLOSED", pre_open, None, "NYSE/Nasdaq", str(tz), True, False, None)
    if local < regular_open:
        return MarketState(asset, timestamp, True, "PRE_MARKET", regular_open, regular_close, "NYSE/Nasdaq", str(tz), True, False, 0.0)
    if local < regular_close:
        progress = (local - regular_open).total_seconds() / (regular_close - regular_open).total_seconds()
        return MarketState(asset, timestamp, True, "REGULAR", regular_open, regular_close, "NYSE/Nasdaq", str(tz), True, False, progress)
    if local < after_close:
        return MarketState(asset, timestamp, True, "AFTER_HOURS", datetime.combine(_next_session_day(day, "us_equity"), time(4), tzinfo=tz), None, "NYSE/Nasdaq", str(tz), True, False, 1.0)
    nxt = _next_session_day(day, "us_equity")
    return MarketState(asset, timestamp, False, "CLOSED", datetime.combine(nxt, time(4), tzinfo=tz), None, "NYSE/Nasdaq", str(tz), True, False, None)


def _crypto_state(asset: str, timestamp: datetime) -> MarketState:
    return MarketState(asset, timestamp, True, "24_7", timestamp, timestamp, "24/7", "UTC", True, False, None)


def _dax_state(asset: str, timestamp: datetime) -> MarketState:
    tz = ZoneInfo("Europe/Berlin")
    local = timestamp.astimezone(tz)
    holiday = local.weekday() >= 5 or local.date() in {_easter(local.year) - timedelta(days=2), _easter(local.year) + timedelta(days=1), date(local.year, 5, 1), date(local.year, 10, 3), date(local.year, 12, 25), date(local.year, 12, 26)}
    opening = datetime.combine(local.date(), time(9), tzinfo=tz)
    closing = datetime.combine(local.date(), time(17, 30), tzinfo=tz)
    if holiday or local < opening or local >= closing:
        return MarketState(asset, timestamp, False, "CLOSED", opening if local < opening and not holiday else None, None, "Xetra", str(tz), not holiday, holiday, None)
    progress = (local - opening).total_seconds() / (closing - opening).total_seconds()
    return MarketState(asset, timestamp, True, "REGULAR", opening, closing, "Xetra", str(tz), True, False, progress)


class MarketStateService:
    def state_at(self, asset: str, timestamp: datetime) -> MarketState:
        timestamp = timestamp.astimezone(ZoneInfo("UTC")) if timestamp.tzinfo else timestamp.replace(tzinfo=ZoneInfo("UTC"))
        if asset in {"QQQ", "SPY", "NASDAQ100_PROXY", "SP500_PROXY"}:
            return _us_state(asset, timestamp)
        if asset in {"DAX", "^GDAXI"}:
            return _dax_state(asset, timestamp)
        return _crypto_state(asset, timestamp)

    @staticmethod
    def _equity_closed_session_is_current(market: MarketState, last_candle: datetime) -> bool:
        if market.exchange_or_calendar != "NYSE/Nasdaq" or market.market_open:
            return False
        last_local = last_candle.astimezone(ZoneInfo("America/New_York")) if last_candle.tzinfo else last_candle.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
        if market.next_open is None:
            return True
        next_open_local = market.next_open.astimezone(ZoneInfo("America/New_York")) if market.next_open.tzinfo else market.next_open.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
        return last_local <= next_open_local

    def health(
        self,
        market: MarketState,
        *,
        last_candle: datetime | None,
        expected_candle_time: datetime | None,
        bar_seconds: int = 300,
        context_available: bool = True,
        context_freshness_ok: bool = True,
        feature_ready: bool = True,
        warmup_ready: bool = True,
        now: datetime | None = None,
    ) -> DataHealth:
        if last_candle is None:
            return DataHealth(None, expected_candle_time, None, False, False, FeedStatus.MISSING, context_available, context_freshness_ok, context_available and context_freshness_ok, feature_ready, warmup_ready, "missing primary candle")
        now = now or datetime.now(ZoneInfo("UTC"))
        last = last_candle.astimezone(ZoneInfo("UTC")) if last_candle.tzinfo else last_candle.replace(tzinfo=ZoneInfo("UTC"))
        age = max(0.0, (now - last).total_seconds())
        if self._equity_closed_session_is_current(market, last):
            fresh = True
            primary = FeedStatus.AVAILABLE
        else:
            fresh = age <= bar_seconds * 1.5
            primary = FeedStatus.AVAILABLE if fresh else FeedStatus.STALE
        healthy = fresh and feature_ready and warmup_ready
        return DataHealth(last, expected_candle_time, age, fresh, healthy, primary, context_available, context_freshness_ok, context_available and context_freshness_ok, feature_ready, warmup_ready, "" if healthy else "stale or incomplete prerequisites")


def decide_alert(prediction: PredictionSnapshot, market: MarketState, health: DataHealth, *, threshold: float = 90.0) -> AlertDecision:
    if prediction.score < threshold:
        return AlertDecision(False, "score below threshold", prediction.score, threshold)
    if not market.market_open or market.session_type not in {"REGULAR", "24_7"}:
        return AlertDecision(False, f"market session not alertable: {market.session_type}", prediction.score, threshold)
    if not health.feed_healthy or not health.freshness_ok:
        return AlertDecision(False, "primary data stale or unhealthy", prediction.score, threshold)
    if not health.context_available or not health.context_freshness_ok:
        return AlertDecision(False, "context unavailable or stale", prediction.score, threshold)
    return AlertDecision(True, "all alert prerequisites satisfied", prediction.score, threshold)


def snapshot_dict(prediction: PredictionSnapshot, market: MarketState, health: DataHealth, decision: AlertDecision) -> dict[str, Any]:
    return {"prediction": asdict(prediction), "market": asdict(market), "health": asdict(health), "alert": decision.as_dict()}
