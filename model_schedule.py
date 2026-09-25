"""Central closed-candle cadence contract for the decision assistant.

This policy does not authorize a model. It tells callers when a cached input is
fresh enough to evaluate and prevents GUI refreshes from becoming inference
triggers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


GUI_REFRESH_SECONDS = 15


@dataclass(frozen=True)
class TaskSchedule:
    asset: str
    timeframe: str
    market_hours: str
    candle_close_rule: str
    data_refresh: str
    model_inference: str
    strategy_refresh: str
    gui_refresh_seconds: int
    expected_latency: timedelta
    stale_threshold: timedelta
    inference_mode: str

    def is_stale(self, timestamp: str | pd.Timestamp, now: pd.Timestamp | None = None) -> bool:
        observed = pd.Timestamp(timestamp)
        if observed.tzinfo is None:
            observed = observed.tz_localize("UTC")
        reference = now or pd.Timestamp.now(tz="UTC")
        if reference.tzinfo is None:
            reference = reference.tz_localize("UTC")
        return reference - observed > self.stale_threshold


_CRYPTO = {
    "15m": (timedelta(minutes=2), timedelta(minutes=20)),
    "1h": (timedelta(minutes=5), timedelta(minutes=75)),
    "4h": (timedelta(minutes=10), timedelta(hours=4, minutes=30)),
    "1d": (timedelta(minutes=15), timedelta(hours=25)),
}
_EQUITY = {
    "15m": (timedelta(minutes=5), timedelta(minutes=30)),
    "1h": (timedelta(minutes=10), timedelta(minutes=90)),
    "4h": (timedelta(minutes=15), timedelta(hours=5)),
    "1d": (timedelta(minutes=30), timedelta(hours=30)),
}


def get_task_schedule(asset: str, timeframe: str) -> TaskSchedule:
    """Return the only supported cadence; reject unsupported tasks explicitly."""
    crypto = asset.endswith("/USDT")
    timings = (_CRYPTO if crypto else _EQUITY).get(timeframe)
    if timings is None:
        raise ValueError(f"No closed-candle cadence is defined for {asset} {timeframe}")
    latency, stale = timings
    market_hours = "24/7" if crypto else "US equity regular session; no new official signal while closed"
    cadence = timeframe
    return TaskSchedule(
        asset=asset,
        timeframe=timeframe,
        market_hours=market_hours,
        candle_close_rule=f"Evaluate only after the {cadence} candle is closed and provider latency has elapsed.",
        data_refresh=f"Poll for a new closed {cadence} candle; expected provider latency <= {latency}.",
        model_inference="Once per new closed candle; never on GUI refresh.",
        strategy_refresh="Immediately after required model inference for that same closed candle.",
        gui_refresh_seconds=GUI_REFRESH_SECONDS,
        expected_latency=latency,
        stale_threshold=stale,
        inference_mode="OBSERVE_ONLY_PENDING_OOS_PROMOTION",
    )