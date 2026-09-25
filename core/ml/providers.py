"""Provider abstraction for historical ML data.

Providers are read-only. They do not expose exchange order methods.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Protocol
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import time

import pandas as pd
from dotenv import load_dotenv

from core.data_loader import fetch_ohlcv_history
from core.ml.data import load_ohlcv_parquet, normalize_ohlcv, save_ohlcv_parquet

load_dotenv()

TWELVE_DATA_MAX_INTRADAY_DAYS = 365
_CACHE_MAX_AGE = {
    "5m": pd.Timedelta(minutes=15),
    "15m": pd.Timedelta(minutes=35),
    "1h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=8),
    "1d": pd.Timedelta(hours=36),
}


def _history_days_for(config, definition: dict[str, object], timeframe: str) -> int:
    overrides = definition.get("history_days_by_timeframe", {})
    if isinstance(overrides, dict) and timeframe in overrides:
        return int(overrides[timeframe])
    return int(config.ml.timeframe_history_days[timeframe])


@dataclass(frozen=True)
class AssetSpec:
    name: str
    symbol: str
    provider: str
    timeframe: str
    history_days: int
    source: str | None = None
    market_type: str = "unknown"
    allow_history_shortfall: bool = True
    proxy_of: str | None = None
    session_timezone: str = "UTC"
    logical_market: str | None = None
    volume_semantics: str = "meaningful"
    optional: bool = False
    history_start: str | None = None


def asset_specs_from_config(config) -> list[AssetSpec]:
    """Expand primary and contextual config entries into preparation requests."""
    specs: list[AssetSpec] = []
    primary = config.ml.assets + config.ml.optional_assets
    for name in primary:
        definition = config.ml.market_assets.get(name)
        if definition is None:
            raise ValueError(f"No market_assets definition for configured asset {name}")
        for timeframe in config.ml.timeframes:
            if name in config.ml.optional_assets and not bool(definition.get("required", False)):
                provider = str(definition.get("provider_by_timeframe", {}).get(timeframe, definition["provider"]))
            else:
                provider = str(definition.get("provider_by_timeframe", {}).get(timeframe, definition["provider"]))
            if timeframe in {"15m", "1h", "4h"} and definition.get("derive_from"):
                provider = "local_derived"
            symbol = str(definition.get("provider_symbols", {}).get(provider, definition["symbol"]))
            specs.append(
                AssetSpec(
                    name=name,
                    symbol=symbol,
                    provider=provider,
                    timeframe=timeframe,
                    history_days=_history_days_for(config, definition, timeframe),
                    market_type=str(definition.get("market_type", "unknown")),
                    allow_history_shortfall=config.ml.allow_history_shortfall,
                    proxy_of=definition.get("proxy_of"),
                    session_timezone=str(definition.get("session_timezone", "UTC")),
                    logical_market=definition.get("logical_market"),
                    volume_semantics=str(definition.get("volume_semantics", "meaningful")),
                    optional=name in config.ml.optional_assets,
                    source=str(definition.get("provider_symbols", {}).get(provider, definition["symbol"])),
                    history_start=definition.get("history_start"),
                )
            )
    for name, definition in config.ml.context_assets.items():
        for timeframe in config.ml.context_timeframes:
            specs.append(
                AssetSpec(
                    name=name,
                    symbol=str(definition["symbol"]),
                    provider=str(definition["provider"]),
                    timeframe=timeframe,
                    history_days=int(config.ml.timeframe_history_days[timeframe]),
                    source=str(definition["symbol"]),
                    market_type=str(definition.get("market_type", "context")),
                    allow_history_shortfall=config.ml.allow_history_shortfall,
                    proxy_of=definition.get("proxy_of"),
                    session_timezone=str(definition.get("session_timezone", "UTC")),
                    optional=False,
                    history_start=definition.get("history_start"),
                )
            )
    return specs


class HistoricalProvider(Protocol):
    def load(self, asset: AssetSpec) -> pd.DataFrame: ...


class ParquetProvider:
    def load(self, asset: AssetSpec) -> pd.DataFrame:
        if not asset.source:
            raise ValueError(f"Parquet source is required for {asset.name}")
        return load_ohlcv_parquet(asset.source)


class CCXTHistoricalProvider:
    def __init__(self, cache_dir: str, exchange_id: str = "binance", market_type: str = "swap", max_cache_age: pd.Timedelta = pd.Timedelta(days=2)) -> None:
        self.cache_dir = cache_dir
        self.exchange_id = exchange_id
        self.market_type = market_type
        self.last_report: dict[str, object] = {}
        self.seed_frame: pd.DataFrame | None = None
        self.max_cache_age = max_cache_age

    def load(self, asset: AssetSpec) -> pd.DataFrame:
        cache_path = canonical_cache_path(self.cache_dir, asset)
        stale_canonical = self.seed_frame
        if cache_path.exists():
            cached = load_ohlcv_parquet(cache_path)
            requested_start = pd.Timestamp(asset.history_start, tz="UTC") if asset.history_start else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=asset.history_days)
            if (
                not cached.empty
                and cached.index.max() >= pd.Timestamp.now(tz="UTC") - self.max_cache_age
                and cached.index.min() <= requested_start + pd.Timedelta(days=2)
            ):
                self.last_report = {"status": "success", "requested_start": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=asset.history_days)).isoformat(), "requested_end": pd.Timestamp.now(tz="UTC").isoformat()}
                return cached
            stale_canonical = cached
        legacy = sorted(Path(self.cache_dir).glob(
            f"ohlcv_{asset.symbol.replace('/', '-').replace(' ', '')}_{asset.timeframe}_*d.csv"
        ))
        loaded = fetch_ohlcv_history(
            asset.symbol,
            asset.timeframe,
            asset.history_days,
            cache_dir=self.cache_dir,
            exchange_id=self.exchange_id,
            market_type=self.market_type,
            return_report=True,
            seed_path=legacy[-1] if legacy else None,
            seed_frame=stale_canonical,
            start_date=asset.history_start,
        )
        frame, report = loaded
        self.last_report = report
        return normalize_ohlcv(frame)


class EODHDHistoricalProvider:
    """Read-only EODHD intraday provider with bounded date chunks."""

    def __init__(self, api_key: str | None = None, base_url: str = "https://eodhd.com/api") -> None:
        self.api_key = api_key or os.getenv("EODHD_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.last_report: dict[str, object] = {}

    def load(self, asset: AssetSpec) -> pd.DataFrame:
        if not self.api_key:
            raise RuntimeError("EODHD_API_KEY is required for the configured EODHD provider")
        if asset.timeframe not in {"5m", "15m", "1h", "4h"}:
            raise ValueError("EODHDHistoricalProvider is intended for intraday timeframes")
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(days=asset.history_days)
        step = pd.Timedelta(days=30 if asset.timeframe in {"5m", "15m"} else 180)
        rows: dict[int, dict] = {}
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + step, end)
            params = urlencode({"api_token": self.api_key, "interval": asset.timeframe, "from": int(cursor.timestamp()), "to": int(chunk_end.timestamp()), "fmt": "json"})
            request = Request(f"{self.base_url}/intraday/{asset.symbol}?{params}", headers={"User-Agent": "trading-bot-research"})
            with urlopen(request, timeout=30) as response:  # noqa: S310 - configured HTTPS endpoint
                payload = json.loads(response.read().decode("utf-8"))
            if isinstance(payload, dict) and payload.get("error"):
                raise RuntimeError(f"EODHD request failed for {asset.name}: {payload['error']}")
            for item in payload:
                timestamp = int(item.get("timestamp", item.get("datetime", 0)))
                if timestamp:
                    rows[timestamp] = {
                        "timestamp": pd.to_datetime(timestamp, unit="s", utc=True),
                        "open": item.get("open"), "high": item.get("high"),
                        "low": item.get("low"), "close": item.get("close"),
                        "volume": item.get("volume", 0.0),
                    }
            cursor = chunk_end
        frame = pd.DataFrame(rows.values()).set_index("timestamp").sort_index()
        self.last_report = {"requested_start": start.isoformat(), "requested_end": end.isoformat(), "provider": "eodhd", "symbol": asset.symbol, "actual_start": frame.index.min().isoformat() if not frame.empty else None, "actual_end": frame.index.max().isoformat() if not frame.empty else None, "candles": len(frame)}
        return normalize_ohlcv(frame)


class TwelveDataHistoricalProvider:
    """Chunked Twelve Data REST provider shared by historical and future polling."""

    _INTERVALS = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1day"}

    @classmethod
    def estimate_requests(cls, history_days: int, timeframe: str) -> int:
        interval = cls._INTERVALS[timeframe]
        if timeframe == "5m":
            history_days = min(history_days, TWELVE_DATA_MAX_INTRADAY_DAYS)
        points_per_day = int(pd.Timedelta(days=1) / pd.Timedelta(interval))
        chunk_days = max(1, min(365, 4500 // max(points_per_day, 1)))
        return (history_days + chunk_days - 1) // chunk_days

    def __init__(self, settings: dict[str, object] | None = None, api_key: str | None = None, base_url: str = "https://api.twelvedata.com") -> None:
        settings = settings or {}
        self.api_key_env = str(settings.get("api_key_env", "TWELVE_DATA_API_KEY"))
        self.api_key = api_key or os.getenv(self.api_key_env)
        self.base_url = base_url.rstrip("/")
        self.max_requests_per_day = int(settings.get("max_requests_per_day", 800))
        self.max_requests_per_minute = int(settings.get("max_requests_per_minute", 8))
        self.safe_requests_per_minute = min(
            self.max_requests_per_minute,
            int(settings.get("safe_requests_per_minute", max(1, self.max_requests_per_minute - 1))),
        )
        self.max_retries = int(settings.get("max_retries", 3))
        self.request_timeout_s = float(settings.get("request_timeout_s", 30))
        self.requests_used = 0
        self.request_times: list[float] = []
        self.last_report: dict[str, object] = {}
        self.partial_path: Path | None = None

    def _wait_for_rate_limit(self) -> None:
        now = time.monotonic()
        self.request_times = [value for value in self.request_times if now - value < 60.0]
        if len(self.request_times) >= self.safe_requests_per_minute:
            time.sleep(max(0.0, 60.0 - (now - self.request_times[0])))
            self.request_times = []

    def _request(self, params: dict[str, object]) -> list[dict]:
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} is required for the configured Twelve Data provider")
        if self.requests_used >= self.max_requests_per_day:
            raise RuntimeError("Twelve Data daily request budget exhausted")
        params = {**params, "apikey": self.api_key, "format": "JSON"}
        for attempt in range(self.max_retries + 1):
            try:
                if self.requests_used >= self.max_requests_per_day:
                    raise RuntimeError("Twelve Data daily request budget exhausted")
                self._wait_for_rate_limit()
                # Count every HTTP attempt, including retries, against the
                # daily quota before sending it.
                self.requests_used += 1
                request = Request(f"{self.base_url}/time_series?{urlencode(params)}", headers={"User-Agent": "trading-bot-research"})
                with urlopen(request, timeout=self.request_timeout_s) as response:  # noqa: S310 - fixed HTTPS API endpoint
                    payload = json.loads(response.read().decode("utf-8"))
                self.request_times.append(time.monotonic())
                if isinstance(payload, dict) and payload.get("status") == "error":
                    raise RuntimeError(str(payload.get("message", "Twelve Data request failed")))
                return list(payload.get("values", [])) if isinstance(payload, dict) else []
            except HTTPError as exc:
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 2.0**attempt)
                if exc.code != 429 or attempt >= self.max_retries:
                    raise RuntimeError(f"Twelve Data HTTP {exc.code} after {attempt + 1} attempts") from exc
                time.sleep(delay)
            except Exception as exc:  # noqa: BLE001 - bounded provider retry boundary
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Twelve Data request failed after {self.max_retries} retries: {exc}") from exc
                time.sleep(min(30.0, 2.0**attempt))
        return []

    def load(self, asset: AssetSpec) -> pd.DataFrame:
        interval = self._INTERVALS.get(asset.timeframe)
        if interval is None:
            raise ValueError(f"Unsupported Twelve Data timeframe: {asset.timeframe}")
        end = pd.Timestamp.now(tz="UTC")
        effective_days = min(asset.history_days, TWELVE_DATA_MAX_INTRADAY_DAYS) if asset.timeframe == "5m" else asset.history_days
        start = end - pd.Timedelta(days=effective_days)
        points_per_day = int(pd.Timedelta(days=1) / pd.Timedelta(interval))
        chunk_days = max(1, min(365, 4500 // max(points_per_day, 1)))
        rows: dict[pd.Timestamp, dict[str, object]] = {}
        cursor = start
        chunks = 0
        if self.partial_path and self.partial_path.exists():
            cached = load_ohlcv_parquet(self.partial_path)
            if not cached.empty:
                rows = cached.to_dict(orient="index")
                cursor = max(cursor, cached.index.max() + pd.Timedelta(interval))
        while cursor < end:
            chunk_end = min(end, cursor + pd.Timedelta(days=chunk_days))
            if self.requests_used >= self.max_requests_per_day:
                break
            try:
                values = self._request({"symbol": asset.symbol, "interval": interval, "start_date": cursor.strftime("%Y-%m-%d %H:%M:%S"), "end_date": chunk_end.strftime("%Y-%m-%d %H:%M:%S"), "outputsize": 5000, "timezone": "UTC"})
            except RuntimeError as exc:
                if not self.api_key:
                    raise
                self.last_report = {"provider": "twelve_data", "source_symbol": asset.symbol, "requested_start": start.isoformat(), "requested_end": end.isoformat(), "requests_used": self.requests_used, "chunks": chunks, "status": "partial", "warning": str(exc)}
                break
            chunks += 1
            for item in values:
                timestamp = pd.to_datetime(item["datetime"], utc=True)
                rows[timestamp] = {"open": item.get("open"), "high": item.get("high"), "low": item.get("low"), "close": item.get("close"), "volume": item.get("volume", 0.0)}
            if self.partial_path and rows:
                save_ohlcv_parquet(pd.DataFrame.from_dict(rows, orient="index"), self.partial_path)
            cursor = chunk_end
        frame = pd.DataFrame.from_dict(rows, orient="index").sort_index()
        frame = normalize_ohlcv(frame) if not frame.empty else pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], tz="UTC"))
        points_per_day = int(pd.Timedelta(days=1) / pd.Timedelta(interval))
        self.last_report = {"provider": "twelve_data", "source_symbol": asset.symbol, "requested_start": start.isoformat(), "requested_end": end.isoformat(), "actual_start": frame.index.min().isoformat() if not frame.empty else None, "actual_end": frame.index.max().isoformat() if not frame.empty else None, "candles": len(frame), "expected_candles": effective_days * points_per_day, "requests_used": self.requests_used, "chunks": chunks, "status": "partial" if frame.empty or frame.index.max() < end - pd.Timedelta(days=2) else "success"}
        return frame

    def latest_completed(self, symbol: str, interval: str = "5min") -> pd.DataFrame:
        values = self._request({"symbol": symbol, "interval": interval, "outputsize": 2, "timezone": "UTC"})
        frame = pd.DataFrame(values)
        if frame.empty:
            return frame
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
        duration = pd.Timedelta(interval)
        now = pd.Timestamp.now(tz="UTC")
        frame = frame[frame["datetime"] + duration <= now]
        frame = frame.set_index("datetime").rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
        return normalize_ohlcv(frame)


class YFinanceHistoricalProvider:
    def __init__(self) -> None:
        self.last_report: dict[str, object] = {}

    def load(self, asset: AssetSpec) -> pd.DataFrame:
        if not asset.source:
            raise ValueError(f"YFinance ticker is required as source for {asset.name}")
        if asset.timeframe in {"5m", "15m", "1h", "4h"} and asset.history_days > 60:
            raise RuntimeError(
                f"yfinance does not reliably provide a {asset.history_days}-day {asset.timeframe} history "
                f"for {asset.name}; configure a dedicated intraday provider or an explicit local Parquet source."
            )
        import yfinance as yf

        frame = yf.download(asset.source, period=f"{asset.history_days}d", interval=asset.timeframe, progress=False)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
        normalized = normalize_ohlcv(frame)
        self.last_report = {"provider": "yfinance", "symbol": asset.symbol, "actual_start": normalized.index.min().isoformat() if not normalized.empty else None, "actual_end": normalized.index.max().isoformat() if not normalized.empty else None, "candles": len(normalized)}
        return normalized


def provider_for(name: str, *, cache_dir: str, exchange_id: str = "binance", market_type: str = "swap", twelve_data_settings: dict[str, object] | None = None, max_cache_age: pd.Timedelta = pd.Timedelta(days=2)) -> HistoricalProvider:
    if name == "parquet":
        return ParquetProvider()
    if name == "ccxt":
        return CCXTHistoricalProvider(cache_dir, exchange_id, market_type, max_cache_age)
    if name == "eodhd":
        return EODHDHistoricalProvider()
    if name == "twelve_data":
        return TwelveDataHistoricalProvider(twelve_data_settings)
    if name == "yfinance":
        return YFinanceHistoricalProvider()
    raise ValueError(f"Unknown historical provider: {name}")


def load_asset_specs(
    specs: list[AssetSpec],
    *,
    cache_dir: str,
    exchange_id: str = "binance",
    market_type: str = "swap",
) -> dict[str, pd.DataFrame]:
    """Load configured assets through their selected read-only providers."""
    loaded: dict[str, pd.DataFrame] = {}
    for spec in specs:
        provider = provider_for(
            spec.provider,
            cache_dir=cache_dir,
            exchange_id=exchange_id,
            market_type=market_type,
        )
        loaded[spec.name] = provider.load(spec)
    return loaded


def canonical_cache_path(cache_root: str | Path, asset: AssetSpec) -> Path:
    """Deterministic cache path containing asset, timeframe, and provider."""
    safe_asset = asset.name.replace("/", "_").replace(" ", "_").replace("&", "and")
    return Path(cache_root) / safe_asset / asset.timeframe / f"{asset.provider}.parquet"


def prepare_asset_specs(
    specs: list[AssetSpec],
    *,
    cache_root: str | Path,
    cache_dir: str,
    exchange_id: str = "binance",
    default_market_type: str = "swap",
    twelve_data_settings: dict[str, object] | None = None,
    force_refresh: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    """Load, validate, and persist configured assets without proxy substitution."""
    from core.ml.data import quality_report, save_ohlcv_parquet

    loaded: dict[str, pd.DataFrame] = {}
    reports: list[dict[str, object]] = []
    for spec in specs:
        path = canonical_cache_path(cache_root, spec)
        try:
            provider = None if spec.provider == "local_derived" else provider_for(
                spec.provider,
                cache_dir=cache_dir,
                exchange_id=exchange_id,
                market_type=spec.market_type or default_market_type,
                twelve_data_settings=twelve_data_settings,
                max_cache_age=_CACHE_MAX_AGE.get(spec.timeframe, pd.Timedelta(hours=2)),
            )
            canonical_is_fresh = False
            if path.exists():
                frame = load_ohlcv_parquet(path)
                requested_start = pd.Timestamp(spec.history_start, tz="UTC") if spec.history_start else pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=spec.history_days)
                canonical_is_fresh = not force_refresh and (
                    not frame.empty
                    and frame.index.max() >= pd.Timestamp.now(tz="UTC") - _CACHE_MAX_AGE.get(spec.timeframe, pd.Timedelta(hours=2))
                    and frame.index.min() <= requested_start + pd.Timedelta(days=2)
                )
            if not canonical_is_fresh:
                if spec.provider == "local_derived":
                    base_key = f"{spec.name}:5m"
                    base = loaded.get(base_key)
                    if base is None:
                        base_paths = [
                            canonical_cache_path(cache_root, AssetSpec(spec.name, spec.symbol, provider_name, "5m", spec.history_days, source=spec.source))
                            for provider_name in ("twelve_data", "yfinance", "ccxt")
                        ]
                        base_path = next((candidate for candidate in base_paths if candidate.exists()), None)
                        if base_path is None:
                            raise RuntimeError(f"Cannot derive {spec.timeframe}: missing 5m source for {spec.name}")
                        base = load_ohlcv_parquet(base_path)
                    rule = {"15m": "15min", "1h": "1h", "4h": "4h"}.get(spec.timeframe)
                    if rule is None:
                        raise ValueError(f"Unsupported locally derived timeframe: {spec.timeframe}")
                    frame = base.resample(rule, label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
                    provider_report = {"status": "success", "provider": "local-derived", "requested_start": base.index.min().isoformat() if not base.empty else None, "requested_end": base.index.max().isoformat() if not base.empty else None}
                    save_ohlcv_parquet(frame, path)
                else:
                    if isinstance(provider, TwelveDataHistoricalProvider):
                        provider.partial_path = path.with_suffix(".partial.parquet")
                    if isinstance(provider, CCXTHistoricalProvider) and path.exists():
                        provider.seed_frame = frame
                    frame = provider.load(spec)
                    provider_report = getattr(provider, "last_report", {})
                    if provider_report.get("status") == "partial":
                        save_ohlcv_parquet(frame, path.with_suffix(".partial.parquet"))
                    else:
                        save_ohlcv_parquet(frame, path)
        except RuntimeError as exc:
            if not spec.allow_history_shortfall:
                raise
            reports.append({"asset": spec.name, "timeframe": spec.timeframe, "provider": spec.provider, "status": "unavailable", "optional": spec.optional, "warning": str(exc)})
            continue
        loaded[f"{spec.name}:{spec.timeframe}"] = frame
        provider_report = getattr(provider, "last_report", {})
        requested_start = provider_report.get("requested_start") or (
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=spec.history_days)
        ).isoformat()
        requested_end = provider_report.get("requested_end") or pd.Timestamp.now(tz="UTC").isoformat()
        source_report = next(
            (
                report
                for report in reports
                if report.get("asset") == spec.name and report.get("timeframe") == "5m"
            ),
            None,
        )
        derived_status = None
        if spec.provider == "local_derived" and source_report is not None:
            source_start = pd.Timestamp(source_report["start"]) if source_report.get("start") else None
            source_end = pd.Timestamp(source_report["end"]) if source_report.get("end") else None
            source_span_days = (source_end - source_start).total_seconds() / 86400.0 if source_start is not None and source_end is not None else 0.0
            source_is_recent = source_end is not None and source_end >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)
            if source_report.get("status") in {"success", "complete", "DERIVED_COMPLETE"} or (source_span_days >= 730.0 and source_is_recent):
                derived_status = "DERIVED_COMPLETE"
            else:
                derived_status = "partial"
        report = quality_report(
                frame,
                asset=spec.name,
                timeframe=spec.timeframe,
                provider=spec.provider,
                cache_path=str(path),
                market_type=spec.market_type,
                session_timezone=spec.session_timezone,
                proxy_of=spec.proxy_of,
                source=spec.source,
                requested_start=requested_start,
                requested_end=requested_end,
                provider_status=derived_status or provider_report.get("status"),
            )
        report["optional"] = spec.optional
        reports.append(report)
    return loaded, reports
