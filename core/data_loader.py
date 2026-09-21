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


def load_pair_data(config: DataConfig) -> pd.DataFrame:
    """Load, resample and align close prices for the configured pair."""
    raw_a = fetch_ohlcv_history(
        config.symbol_a,
        config.base_timeframe,
        config.history_days,
        cache_dir=config.cache_dir,
        exchange_id=config.exchange_id,
        market_type=config.market_type,
    )
    raw_b = fetch_ohlcv_history(
        config.symbol_b,
        config.base_timeframe,
        config.history_days,
        cache_dir=config.cache_dir,
        exchange_id=config.exchange_id,
        market_type=config.market_type,
    )

    res_a = resample_ohlcv(raw_a, config.resample_to) if config.resample_to else raw_a
    res_b = resample_ohlcv(raw_b, config.resample_to) if config.resample_to else raw_b

    df = pd.DataFrame(
        {
            config.symbol_a: res_a["close"],
            config.symbol_b: res_b["close"],
            f"{config.symbol_a}_high": res_a["high"],
            f"{config.symbol_a}_low": res_a["low"],
            f"{config.symbol_b}_high": res_b["high"],
            f"{config.symbol_b}_low": res_b["low"],
        }
    ).dropna()

    print(f"✅ Synchronisiert: {len(df)} gemeinsame {config.resample_to or config.base_timeframe}-Kerzen.")
    return df
