"""Shared forecast and context contracts for production model preparation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

CONTEXT_ASSETS = ("GOLD", "VIX", "US10Y", "EURUSD", "WTI", "DAX", "ES", "NQ")

@dataclass(frozen=True)
class ContextSeriesStatus:
    asset: str
    status: str
    source_timestamp: str | None
    updated_at: str
    age_seconds: float | None
    reason: str = ""

@dataclass(frozen=True)
class ContextSnapshot:
    status: str
    timestamp: str
    features: dict[str, float]
    series: tuple[ContextSeriesStatus, ...]
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

@dataclass(frozen=True)
class ForecastOutput:
    model_id: str
    model_version: str
    asset: str
    asset_class: str
    timeframe: str
    horizon_bars: tuple[int, ...]
    input_timestamp: str
    inference_timestamp: str
    data_age_seconds: float | None
    direction_probability_long: float | None
    direction_probability_short: float | None
    expected_return: tuple[float | None, ...]
    lower_quantile: tuple[float | None, ...]
    median_quantile: tuple[float | None, ...]
    upper_quantile: tuple[float | None, ...]
    expected_volatility: float | None
    forecast_uncertainty: str
    forecast_score: float | None
    status: str
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def context_feature_frame(context: dict[str, pd.DataFrame], target_index: pd.DatetimeIndex, *, max_age: pd.Timedelta = pd.Timedelta(days=3)) -> tuple[pd.DataFrame, ContextSnapshot]:
    """Build causal context returns/trend/vol/z-score/relative-position features."""
    target = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True))
    now = pd.Timestamp.now(tz="UTC")
    features = pd.DataFrame(index=target)
    statuses: list[ContextSeriesStatus] = []
    for name in CONTEXT_ASSETS:
        frame = context.get(name)
        if frame is None or frame.empty or "close" not in frame:
            statuses.append(ContextSeriesStatus(name, "MISSING", None, now.isoformat(), None, "no cached context series"))
            continue
        close = pd.to_numeric(frame["close"], errors="coerce").dropna().sort_index()
        close.index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
        if close.empty:
            statuses.append(ContextSeriesStatus(name, "MISSING", None, now.isoformat(), None, "empty context series"))
            continue
        aligned = close.reindex(target, method="ffill")
        ret = np.log(aligned / aligned.shift(1))
        daily_return = close.pct_change(5).reindex(target, method="ffill")
        vol = ret.rolling(20).std()
        trend = aligned.ewm(span=20, adjust=False).mean() / aligned.ewm(span=100, adjust=False).mean() - 1.0
        mean = aligned.rolling(60).mean()
        std = aligned.rolling(60).std().replace(0, np.nan)
        zscore = (aligned - mean) / std
        relative = aligned / aligned.rolling(60).max() - 1.0
        for suffix, series in (("ret1", ret), ("ret5", daily_return), ("vol20", vol), ("trend", trend), ("zscore60", zscore), ("relative60", relative)):
            features[f"context_{name}_{suffix}"] = series.to_numpy()
        source = close.index[-1]
        age = max(0.0, (now - source).total_seconds())
        status = "OK" if age <= max_age.total_seconds() else "STALE"
        statuses.append(ContextSeriesStatus(name, status, source.isoformat(), now.isoformat(), age, "" if status == "OK" else "context exceeds staleness limit"))
    missing = [item for item in statuses if item.status == "MISSING"]
    overall = "OK" if len(statuses) == len(CONTEXT_ASSETS) and not missing and all(item.status == "OK" for item in statuses) else "STALE" if not missing else "MISSING"
    timestamp = target[-1].isoformat() if len(target) else now.isoformat()
    values = {} if features.empty else {key: float(value) for key, value in features.iloc[-1].dropna().items()}
    reason = "" if overall == "OK" else "one or more required context series unavailable/stale"
    return features, ContextSnapshot(overall, timestamp, values, tuple(statuses), reason)
