"""CCXT-based OHLCV data loading with disk caching and resampling.

Fetches paginated OHLCV history from Binance USDT-M futures (or any CCXT
exchange), caches it as CSV under ``data/`` and resamples to the target
intraday timeframe used by the strategy (e.g. 5m -> 10m, since Binance does
not natively offer a 10m bucket).
"""
from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import ccxt
import pandas as pd

from core.config import DataConfig

warnings.filterwarnings("ignore")

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


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
) -> pd.DataFrame:
    """Fetch full OHLCV history via CCXT pagination, with local CSV caching."""
    path = _cache_path(cache_dir, symbol, timeframe, history_days)
    if use_cache and path.exists():
        print(f"⚡ Lade OHLCV-Cache: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "options": {"defaultType": market_type}})

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    target_candles = int((history_days * 24 * 60 * 60 * 1000) / timeframe_ms)

    print(f"📥 Lade {target_candles} {timeframe}-Kerzen für {symbol} via CCXT-Pagination...")
    all_ohlcv: list[list[float]] = []
    since = exchange.milliseconds() - target_candles * timeframe_ms

    while len(all_ohlcv) < target_candles:
        try:
            limit = min(1000, target_candles - len(all_ohlcv))
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break
            since = batch[-1][0] + 1
            all_ohlcv.extend(batch)
            print(f"  - {symbol}: {len(all_ohlcv)}/{target_candles} Kerzen geladen...")
            time.sleep(exchange.rateLimit / 1000)
        except Exception as exc:  # noqa: BLE001 - network layer, log and stop pagination
            print(f"  - Fehler beim Laden von {symbol}: {exc}")
            break

    df = pd.DataFrame(all_ohlcv, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(path)

    return df


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
