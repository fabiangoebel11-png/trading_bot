"""CCXT-based OHLCV data loading with disk caching and resampling.

Fetches paginated OHLCV history from Binance USDT-M futures (or any CCXT
exchange), caches it as CSV under ``data/`` and resamples to the target
intraday timeframe used by the strategy (e.g. 5m -> 10m, since Binance does
not natively offer a 10m bucket).
"""
from __future__ import annotations

import os
import json
import time
import warnings
from pathlib import Path

import ccxt
import pandas as pd

from core.config import DataConfig

warnings.filterwarnings("ignore")

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def _timeframe_ms(exchange, timeframe: str) -> int:
    return int(exchange.parse_timeframe(timeframe) * 1000)


def _coverage_status(actual_start, actual_end, requested_start, requested_end) -> tuple[float, str]:
    if actual_start is None or actual_end is None:
        return 0.0, "unavailable"
    actual_start = pd.Timestamp(actual_start)
    actual_end = pd.Timestamp(actual_end)
    requested_start = pd.Timestamp(requested_start)
    requested_end = pd.Timestamp(requested_end)
    if actual_start.tzinfo is None:
        actual_start = actual_start.tz_localize("UTC")
    if actual_end.tzinfo is None:
        actual_end = actual_end.tz_localize("UTC")
    if requested_start.tzinfo is None:
        requested_start = requested_start.tz_localize("UTC")
    if requested_end.tzinfo is None:
        requested_end = requested_end.tz_localize("UTC")
    requested_seconds = max((requested_end - requested_start).total_seconds(), 1.0)
    covered_seconds = max(0.0, (min(actual_end, requested_end) - max(actual_start, requested_start)).total_seconds())
    coverage = min(1.0, covered_seconds / requested_seconds)
    return coverage, "success" if actual_end >= requested_end - pd.Timedelta(days=2) else "partial"


def _cache_path(cache_dir: str, symbol: str, timeframe: str, history_days: int) -> Path:
    safe_symbol = symbol.replace("/", "-")
    return Path(cache_dir) / f"ohlcv_{safe_symbol}_{timeframe}_{history_days}d.csv"


def fetch_ohlcv_history(
    symbol: str,
    timeframe: str,
    history_days: int,
    cache_dir: str = "data",
    exchange_id: str = "binance",
    market_type: str = "swap",
    use_cache: bool = True,
    max_retries: int = 3,
    return_report: bool = False,
    seed_path: str | Path | None = None,
    seed_frame: pd.DataFrame | None = None,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, object]]:
    """Fetch OHLCV with resumable pagination and explicit coverage status."""
    path = _cache_path(cache_dir, symbol, timeframe, history_days)
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "options": {"defaultType": market_type}})

    timeframe_ms = _timeframe_ms(exchange, timeframe)
    requested_end = pd.Timestamp.utcnow().tz_convert("UTC")
    requested_start = (
        pd.Timestamp(start_date, tz="UTC")
        if start_date is not None
        else requested_end - pd.Timedelta(days=history_days)
    )
    target_candles = int((requested_end - requested_start).total_seconds() * 1000 / timeframe_ms)

    cached_seed: dict[int, list[float]] = {}
    seed_file = Path(seed_path) if seed_path else path
    if seed_frame is not None:
        cached = seed_frame.copy()
        cached.index = pd.to_datetime(cached.index, utc=True)
        for timestamp, row in zip(cached.index, cached.to_numpy().tolist()):
            cached_seed[int(timestamp.timestamp() * 1000)] = [int(timestamp.timestamp() * 1000), *row]
    elif seed_file.exists():
        cached = pd.read_csv(seed_file, index_col=0, parse_dates=True)
        cached.index = pd.to_datetime(cached.index, utc=True)
        if not cached.empty and cached.index.max() >= requested_end - pd.Timedelta(days=2):
            report = {"status": "success", "coverage_ratio": 1.0, "requested_start": requested_start.isoformat(), "requested_end": requested_end.isoformat(), "actual_start": cached.index.min().isoformat(), "actual_end": cached.index.max().isoformat(), "provider": "ccxt"}
            return (cached, report) if return_report else cached
        for timestamp, row in zip(cached.index, cached.to_numpy().tolist()):
            cached_seed[int(timestamp.timestamp() * 1000)] = [int(timestamp.timestamp() * 1000), *row]

    partial_path = path.with_suffix(path.suffix + ".partial.csv")
    existing: dict[int, list[float]] = cached_seed
    if partial_path.exists():
        partial = pd.read_csv(partial_path)
        for row in partial.to_numpy().tolist():
            existing[int(row[0])] = row

    print(f"📥 Lade {target_candles} {timeframe}-Kerzen für {symbol} via CCXT-Pagination...")
    all_ohlcv = existing
    since = int(requested_start.timestamp() * 1000)
    if all_ohlcv:
        earliest_cached = min(all_ohlcv)
        latest_cached = max(all_ohlcv)
        # A short recent cache (the observed 45-day 1h failure) is missing the
        # historical prefix, so restart at the requested beginning. Otherwise
        # continue after the cached tail for normal incremental updates.
        since = since if earliest_cached > since + timeframe_ms else latest_cached + timeframe_ms
    consecutive_failures = 0

    while since <= int(requested_end.timestamp() * 1000) and len(all_ohlcv) < target_candles:
        try:
            limit = min(1000, target_candles - len(all_ohlcv))
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break
            before = len(all_ohlcv)
            for row in batch:
                all_ohlcv[int(row[0])] = row
            last_timestamp = max(int(row[0]) for row in batch)
            if len(all_ohlcv) == before:
                break
            since = max(since + timeframe_ms, last_timestamp + timeframe_ms)
            consecutive_failures = 0
            partial_frame = pd.DataFrame(sorted(all_ohlcv.values()), columns=OHLCV_COLUMNS)
            partial_frame.to_csv(partial_path, index=False)
            print(f"  - {symbol}: {len(all_ohlcv)}/{target_candles} Kerzen geladen...")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as exc:  # noqa: BLE001 - network layer, log and stop pagination
            consecutive_failures += 1
            if consecutive_failures > max_retries:
                print(f"  - Fehler beim Laden von {symbol} nach {max_retries} Retries: {exc}")
                break
            delay = min(60.0, 2.0 ** (consecutive_failures - 1))
            print(f"  - transienter Fehler bei {symbol}, Retry {consecutive_failures}/{max_retries} in {delay:.1f}s: {exc}")
            time.sleep(delay)

    df = pd.DataFrame(sorted(all_ohlcv.values()), columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    actual_start = df.index.min() if not df.empty else None
    actual_end = df.index.max() if not df.empty else None
    coverage, status = _coverage_status(actual_start, actual_end, requested_start, requested_end)
    report = {"status": status, "coverage_ratio": coverage, "requested_start": requested_start.isoformat(), "requested_end": requested_end.isoformat(), "actual_start": actual_start.isoformat() if actual_start is not None else None, "actual_end": actual_end.isoformat() if actual_end is not None else None, "provider": "ccxt", "partial_path": str(partial_path) if status != "success" else None}
    if use_cache and status == "success":
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(path)
        if partial_path.exists():
            partial_path.unlink()

    return (df, report) if return_report else df


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV candles to a coarser timeframe (e.g. 5m -> 10min)."""
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def drop_incomplete_last_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Drop the last row if it is a still-forming (not yet closed) candle.

    CCXT's ``fetch_ohlcv`` includes the currently-open candle as its last
    element on most exchanges -- its high/low/close keep changing until the
    candle actually closes. The backtester only ever sees fully closed,
    static historical candles, so feeding the live poller's in-progress
    candle into the exact same signal logic would make live decisions based
    on data the backtest never had (and would keep flip-flopping every poll
    as the open candle updates), a live/backtest mismatch rather than a
    genuine future-leak but with the same practical effect of invalidating
    the backtested edge.
    """
    if df.empty:
        return df
    candle_duration = pd.Timedelta(timeframe)
    last_close_time = df.index[-1] + candle_duration
    now = pd.Timestamp.utcnow().tz_localize(None)
    return df.iloc[:-1] if now < last_close_time else df


def load_multi_asset_data(config: DataConfig) -> dict[str, pd.DataFrame]:
    """Load, resample and align OHLC data for every symbol in ``config.symbols``.

    Returns one DataFrame per symbol (columns: open, high, low, close), all
    reindexed onto the **union** (outer join) of their timestamps -- a
    "dynamic universe": the combined history reaches back as far as the
    *oldest* symbol's data allows, instead of being clipped to the youngest
    symbol's listing date. Symbols with no data yet at a given timestamp
    (e.g. SOL/USDT before its ~2020-08 listing) get explicit NaN rows there
    rather than being silently dropped -- ``core/strategy.py`` and
    ``core/backtester.py`` are responsible for treating those NaNs as "not
    yet tradable" (flat/excluded from weighting), not for masking them here.
    """
    raw = {}
    for symbol in config.symbols:
        history = fetch_ohlcv_history(
            symbol,
            config.base_timeframe,
            config.history_days,
            cache_dir=config.cache_dir,
            exchange_id=config.exchange_id,
            market_type=config.market_type,
        )
        raw[symbol] = resample_ohlcv(history, config.resample_to) if config.resample_to else history

    union_index = raw[config.symbols[0]].index
    for df in raw.values():
        union_index = union_index.union(df.index)
    union_index = union_index.sort_values()

    aligned = {
        symbol: df.reindex(union_index)[["open", "high", "low", "close"]].sort_index()
        for symbol, df in raw.items()
    }

    timeframe_label = config.resample_to or config.base_timeframe
    print(
        f"✅ Dynamisches Universum (Outer-Join): {len(union_index)} {timeframe_label}-Kerzen "
        f"von {union_index.min()} bis {union_index.max()} über {len(config.symbols)} Symbole."
    )
    for symbol, df in aligned.items():
        first_valid = df["close"].first_valid_index()
        print(f"   - {symbol}: erste verfügbare Kerze {first_valid}")
    return aligned
