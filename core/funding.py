"""Historical funding-rate integration for perpetual-swap holding costs.

Perpetual futures pay/charge a funding rate roughly every 8h between longs and
shorts. A directional trend position (long or short) is never funding-neutral,
so realistic holding costs must be modeled explicitly rather than assumed away.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

from core.config import DataConfig, FundingConfig


def _cache_path(cache_dir: str, symbol: str, history_days: int) -> Path:
    safe_symbol = symbol.replace("/", "-")
    return Path(cache_dir) / f"funding_{safe_symbol}_{history_days}d.csv"


def fetch_funding_rate_history(
    symbol: str,
    history_days: int,
    cache_dir: str = "data",
    exchange_id: str = "binance",
    market_type: str = "swap",
    use_cache: bool = True,
) -> pd.Series:
    """Fetch the historical funding-rate series for one perpetual symbol via CCXT
    pagination, cached as CSV. Returns a Series of funding rates indexed by the
    (roughly 8-hourly) funding event timestamp."""
    path = _cache_path(cache_dir, symbol, history_days)
    if use_cache and path.exists():
        print(f"⚡ Lade Funding-Rate-Cache: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)["funding_rate"]

    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "options": {"defaultType": market_type}})

    since = exchange.milliseconds() - history_days * 24 * 60 * 60 * 1000
    all_rates: list[dict] = []

    print(f"📥 Lade Funding-Rate-Historie für {symbol}...")
    while True:
        try:
            batch = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
            if not batch:
                break
            all_rates.extend(batch)
            since = batch[-1]["timestamp"] + 1
            print(f"  - {symbol}: {len(all_rates)} Funding-Events geladen...")
            time.sleep(exchange.rateLimit / 1000)
            if len(batch) < 1000:
                break
        except Exception as exc:  # noqa: BLE001 - network layer, log and stop pagination
            print(f"  - Fehler beim Laden der Funding-Rate für {symbol}: {exc}")
            break

    if not all_rates:
        return pd.Series(dtype=float, name="funding_rate")

    df = pd.DataFrame(all_rates)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    series = df.set_index("timestamp")["fundingRate"].rename("funding_rate").sort_index()
    series = series[~series.index.duplicated(keep="last")]

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        series.to_frame().to_csv(path)

    return series


def align_funding_to_index(
    funding_rate: pd.Series, target_index: pd.DatetimeIndex
) -> tuple[pd.Series, pd.Series]:
    """Return (rate_aligned, is_funding_event) on ``target_index``.

    ``rate_aligned`` forward-fills the last known funding rate onto every intraday
    bar (causal: only ever reflects rates already published in the past).
    ``is_funding_event`` flags the first bar at/after each real funding timestamp,
    i.e. the only bars where a funding payment actually occurs.
    """
    if funding_rate.empty:
        return (
            pd.Series(0.0, index=target_index),
            pd.Series(False, index=target_index),
        )

    funding_sorted = funding_rate.sort_index()
    union_index = funding_sorted.index.union(target_index)
    rate_aligned = funding_sorted.reindex(union_index).ffill().reindex(target_index).fillna(0.0)

    event_mask = np.zeros(len(target_index), dtype=bool)
    positions = target_index.searchsorted(funding_sorted.index, side="left")
    positions = positions[positions < len(target_index)]
    event_mask[positions] = True

    return rate_aligned, pd.Series(event_mask, index=target_index)


def load_portfolio_funding(config: DataConfig, funding_config: FundingConfig) -> dict[str, pd.Series]:
    """Fetch the funding-rate history for every symbol in the trend-following book."""
    return {
        symbol: fetch_funding_rate_history(
            symbol,
            config.history_days,
            cache_dir=funding_config.cache_dir,
            exchange_id=config.exchange_id,
            market_type=config.market_type,
        )
        for symbol in config.symbols
    }


def apply_funding_costs(result: pd.DataFrame, funding_rate: pd.Series | None, config) -> pd.DataFrame:
    """Deduct realistic funding PnL from an already-computed single-symbol backtest
    result. A directional (long/short) position is not funding-neutral like a
    market-neutral pair: holding a long pays the funding rate when positive,
    holding a short receives it. Cost is only applied on bars flagged as an
    actual funding event, scaled by the leverage held at that moment.
    """
    if not config.funding.enabled or funding_rate is None or funding_rate.empty:
        result = result.copy()
        result["funding_cost"] = 0.0
        return result

    rate_aligned, funding_event = align_funding_to_index(funding_rate.dropna(), result.index)
    position = result["execution_position"]
    leverage = result["leverage"]

    # Long (position=+1) pays a positive funding rate; short (position=-1) receives it.
    funding_return = -position * leverage * rate_aligned * funding_event.astype(float)

    out = result.copy()
    out["funding_cost"] = -funding_return  # positive number = cost paid
    out["strategy_return"] = out["strategy_return"] + funding_return
    out["cumulative_strategy"] = (1 + out["strategy_return"]).cumprod()
    return out

