"""Historical-data normalization and Parquet persistence for ML research."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
TIMEFRAME_DELTAS = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}


def validate_ohlcv(
    frame: pd.DataFrame,
    *,
    timeframe: str | None = None,
    min_rows: int = 1,
) -> dict[str, object]:
    """Return quality diagnostics and raise on fatal OHLCV violations."""
    data = normalize_ohlcv(frame)
    diagnostics: dict[str, object] = {
        "rows": len(data),
        "timezone": str(data.index.tz),
        "missing_values": int(data.isna().sum().sum()),
        "zero_volume": int((data["volume"] == 0).sum()),
        "duplicate_timestamps": int(data.index.duplicated().sum()),
    }
    if len(data) < min_rows:
        raise ValueError(f"Insufficient OHLCV history: {len(data)} rows, need {min_rows}")
    if data["close"].isna().any() or data[["open", "high", "low"]].isna().any().any():
        raise ValueError("OHLCV contains invalid or missing price rows")
    if timeframe and timeframe in {"5m", "15m", "1h", "4h", "1d"} and len(data) > 1:
        expected = pd.Timedelta(timeframe)
        gaps = data.index.to_series().diff().dropna()
        diagnostics["gap_count"] = int((gaps > expected * 1.5).sum())
    return diagnostics


def quality_report(
    frame: pd.DataFrame,
    *,
    asset: str,
    timeframe: str,
    provider: str,
    cache_path: str,
    market_type: str,
    session_timezone: str,
    proxy_of: str | None = None,
    source: str | None = None,
    requested_start: str | None = None,
    requested_end: str | None = None,
    provider_status: str | None = None,
) -> dict[str, object]:
    """Build a preparation report without filling gaps across market closures."""
    raw = frame.copy()
    raw.columns = [str(column).lower() for column in raw.columns]
    raw_prices = raw[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    raw_volume = pd.to_numeric(raw.get("volume", pd.Series(index=raw.index, dtype=float)), errors="coerce")
    raw_invalid = (
        (raw_prices[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (raw_prices["high"] < raw_prices[["open", "close"]].max(axis=1))
        | (raw_prices["low"] > raw_prices[["open", "close"]].min(axis=1))
    )
    data = normalize_ohlcv(frame)
    expected = 0
    gap_count = 0
    if len(data) > 1 and timeframe in TIMEFRAME_DELTAS:
        deltas = data.index.to_series().diff().dropna()
        expected = int(((data.index[-1] - data.index[0]) / pd.Timedelta(TIMEFRAME_DELTAS[timeframe])) + 1)
        gap_count = int((deltas > pd.Timedelta(TIMEFRAME_DELTAS[timeframe]) * 1.5).sum())
    session_aware = market_type in {"equity_index", "rate"}
    volume_semantics = "meaningful" if market_type in {"crypto", "swap"} else "unavailable"
    coverage_ratio = 0.0
    if requested_start and requested_end and not data.empty:
        req_start = pd.Timestamp(requested_start)
        req_end = pd.Timestamp(requested_end)
        span = max((req_end - req_start).total_seconds(), 1.0)
        covered = max(0.0, (min(data.index.max(), req_end) - max(data.index.min(), req_start)).total_seconds())
        coverage_ratio = min(1.0, covered / span)
    status = provider_status or ("unavailable" if data.empty else ("success" if coverage_ratio >= 0.995 else "partial"))
    return {
        "asset": asset,
        "timeframe": timeframe,
        "start": data.index.min().isoformat() if not data.empty else None,
        "end": data.index.max().isoformat() if not data.empty else None,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "coverage_ratio": coverage_ratio,
        "candles": int(len(data)),
        "expected_candles": expected,
        "missing_ratio": 0.0 if session_aware else (float(max(expected - len(data), 0) / expected) if expected else 0.0),
        "duplicate_timestamps": int(frame.index.duplicated().sum()),
        "invalid_ohlc": int(raw_invalid.sum()),
        "invalid_volume": int((raw_volume < 0).sum()),
        "zero_volume": int((data["volume"] == 0).sum()),
        "volume_semantics": volume_semantics,
        "timezone": str(data.index.tz),
        "session_timezone": session_timezone,
        "gap_count": gap_count,
        "market_session_issue": "regular_sessions_not_counted_as_missing" if session_aware else "24_7_gap_check",
        "provider": provider,
        "source": source,
        "cache_path": cache_path,
        "proxy_of": proxy_of,
        "market_type": market_type,
        "status": status,
    }


def normalize_ohlcv(frame: pd.DataFrame, *, timezone: str = "UTC") -> pd.DataFrame:
    """Normalize OHLCV without filling across market closures."""
    data = frame.copy()
    data.columns = [str(column).lower() for column in data.columns]
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {sorted(missing)}")
    index = pd.to_datetime(data.index, utc=True).tz_convert(timezone)
    data.index = index
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    numeric = list(REQUIRED_COLUMNS)
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    invalid = (
        (data["open"] <= 0)
        | (data["high"] <= 0)
        | (data["low"] <= 0)
        | (data["close"] <= 0)
        | (data["high"] < data[["open", "close"]].max(axis=1))
        | (data["low"] > data[["open", "close"]].min(axis=1))
    )
    data.loc[invalid, numeric] = np.nan
    data["volume"] = data["volume"].clip(lower=0)
    return data.loc[:, list(REQUIRED_COLUMNS)]


def save_ohlcv_parquet(frame: pd.DataFrame, path: str | Path) -> Path:
    """Normalize and persist a historical dataset as Parquet."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalize_ohlcv(frame).to_parquet(destination)
    return destination


def load_ohlcv_parquet(path: str | Path) -> pd.DataFrame:
    """Load and re-apply normalization to a Parquet dataset."""
    return normalize_ohlcv(pd.read_parquet(path))
