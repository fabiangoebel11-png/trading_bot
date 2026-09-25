"""Causal feature engineering for the trend-continuation confirmation model.

Supersedes the retired mean-reversion feature builder that used to live here
(it depended on ``core.config.MLConfig``, which no longer exists -- see
``core/hybrid_strategy.py`` for that dead code path). Every feature below is
computed strictly from data available at or before bar ``t`` -- no
``.shift(-n)`` anywhere in this file (that's only ever used for labels, in
``core/ml/labeling.py``, which is explicit about it).

Two feature groups are combined:
  1. Technical, single-asset: multi-lag returns, realized volatility, ATR%,
     EMA/Donchian distances (reusing ``core.strategy``'s exact rolling-window
     logic so the ML features are consistent with the rule-based signals),
     RSI, volume z-score, and cyclical time-of-day/day-of-week encodings.
  2. Cross-asset macro context: S&P 500 / Nasdaq 100 / MSCI World / VIX daily
     closes (yfinance), turned into daily return/volatility/z-score features,
     shifted by a reporting lag and forward-filled onto the intraday index --
     the same causality pattern as ``core/macro.py``'s regime filter, reused
     here so an intraday bar can only ever see a macro data point that had
     already closed in the past.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import DataConfig, TrendMLConfig
from core.data_loader import fetch_ohlcv_history
from core.macro import _download_close
from core.strategy import compute_atr, compute_donchian_channels


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_technical_features(ohlc: pd.DataFrame, config: TrendMLConfig) -> pd.DataFrame:
    """``ohlc`` must have columns: open, high, low, close, volume."""
    close, high, low = ohlc["close"], ohlc["high"], ohlc["low"]
    atr = compute_atr(high, low, close, config.atr_window)
    atr_safe = atr.replace(0, np.nan)

    out = pd.DataFrame(index=ohlc.index)
    log_ret = np.log(close / close.shift(1))
    for lag in (1, 3, 6, 12, 24, 48):
        out[f"log_ret_{lag}"] = np.log(close / close.shift(lag))
    for window in (12, 48, 288):
        out[f"realized_vol_{window}"] = log_ret.rolling(window).std()

    out["atr_pct"] = atr / close

    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=100, adjust=False).mean()
    out["ema_fast_dist"] = (close - ema_fast) / atr_safe
    out["ema_slow_dist"] = (close - ema_slow) / atr_safe
    out["ema_regime"] = np.sign(ema_fast - ema_slow)

    upper_entry, lower_entry, upper_exit, lower_exit = compute_donchian_channels(high, low, 55, 20)
    out["donchian_upper_dist"] = (close - upper_entry) / atr_safe
    out["donchian_lower_dist"] = (close - lower_entry) / atr_safe
    out["donchian_exit_upper_dist"] = (close - upper_exit) / atr_safe
    out["donchian_exit_lower_dist"] = (close - lower_exit) / atr_safe

    out["rsi_14"] = compute_rsi(close, 14)

    if "volume" in ohlc.columns:
        vol = ohlc["volume"]
        vol_mean = vol.rolling(288).mean()
        vol_std = vol.rolling(288).std().replace(0, np.nan)
        out["volume_zscore"] = (vol - vol_mean) / vol_std

    hour = ohlc.index.hour + ohlc.index.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = ohlc.index.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return out


def _resample_causal(ohlc: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1h OHLC to a coarser bar (e.g. 4h/1d), labeling each row by
    its CLOSE time (``label='right', closed='right'``) instead of pandas'
    default bar-*start* labeling. This is the causality-critical bit: a 4h/1d
    bar's high/low/close only actually exist once that bar has finished
    forming, so indexing it by the close time means the plain ffill in
    ``align_macro_to_intraday`` (reused below with zero reporting lag) can
    never let an hourly bar see a coarser bar before it has actually closed."""
    pandas_rule = {"5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1D"}.get(rule, rule)
    return (
        ohlc.resample(pandas_rule, label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )


def build_higher_timeframe_features(ohlc: pd.DataFrame, rule: str, config: TrendMLConfig) -> pd.DataFrame:
    """Trend/regime context from a coarser resample of the SAME 1h OHLC
    (e.g. 4h, 1d) -- a classic multi-timeframe CTA technique ("confirm the
    hourly breakout against the daily trend") added as extra context, not a
    new tradeable signal. Deliberately reuses the exact same feature formulas
    and window lengths as ``build_technical_features`` (EMA 20/100 regime,
    Donchian 55/20 distance, ATR(14)%, RSI(14)) -- only the *resolution*
    changes (e.g. "EMA 100" on a daily resample is a ~100-day trend filter),
    so this introduces no new tunable numbers, only new views of already-
    vetted ones. Column names are prefixed with the rule (e.g. ``htf_4h_``)
    so the model can tell timeframes apart; still indexed at each coarse
    bar's own close time -- causal alignment onto the 1h index happens in
    ``build_feature_matrix`` via the same ``align_macro_to_intraday`` ffill
    pattern used for macro data (with zero reporting lag, since there is no
    real-world publication delay for a self-resampled candle)."""
    coarse = _resample_causal(ohlc, rule)
    close, high, low = coarse["close"], coarse["high"], coarse["low"]
    atr = compute_atr(high, low, close, 14)
    atr_safe = atr.replace(0, np.nan)

    prefix = f"htf_{rule}_"
    out = pd.DataFrame(index=coarse.index)
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=100, adjust=False).mean()
    out[f"{prefix}ema_regime"] = np.sign(ema_fast - ema_slow)
    out[f"{prefix}ema_fast_dist"] = (close - ema_fast) / atr_safe
    out[f"{prefix}atr_pct"] = atr / close

    upper_entry, lower_entry, _, _ = compute_donchian_channels(high, low, 55, 20)
    out[f"{prefix}donchian_upper_dist"] = (close - upper_entry) / atr_safe
    out[f"{prefix}donchian_lower_dist"] = (close - lower_entry) / atr_safe
    out[f"{prefix}rsi_14"] = compute_rsi(close, 14)
    return out


def _macro_cache_path(config: TrendMLConfig) -> Path:
    tag = "_".join(s.replace("^", "").replace("=", "").replace("/", "-") for s in config.macro_symbols)
    return Path(config.macro_cache_dir) / f"ml_macro_{tag}_{config.macro_lookback_days}d.csv"


def fetch_macro_matrix(config: TrendMLConfig, use_cache: bool = True) -> pd.DataFrame:
    """Daily close prices for every configured macro symbol, cached to CSV like
    ``core/macro.py``'s regime filter (same yfinance access pattern, reused)."""
    path = _macro_cache_path(config)
    if use_cache and path.exists():
        print(f"⚡ Lade ML-Makro-Cache: {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)

    print(f"🌐 Lade ML-Makro-Daten {config.macro_symbols} ({config.macro_lookback_days} Tage)...")
    columns = {}
    for symbol in config.macro_symbols:
        series = _download_close(symbol, config.macro_lookback_days)
        if not series.empty:
            columns[symbol] = series
        else:
            print(f"  - Keine Daten für {symbol}, wird übersprungen.")

    data = pd.DataFrame(columns).ffill().dropna(how="all")
    if use_cache and not data.empty:
        os.makedirs(config.macro_cache_dir, exist_ok=True)
        data.to_csv(path)
    return data


def build_macro_features(macro_prices: pd.DataFrame, config: TrendMLConfig) -> pd.DataFrame:
    """Daily return/volatility/z-score per macro symbol -- still on the daily
    index here; causal alignment onto the intraday index happens separately in
    ``align_macro_to_intraday`` so the lag/ffill logic lives in one place."""
    out = pd.DataFrame(index=macro_prices.index)
    for symbol in macro_prices.columns:
        price = macro_prices[symbol]
        ret = np.log(price / price.shift(1))
        safe_name = symbol.replace("^", "").replace("=", "").replace("/", "-")
        out[f"{safe_name}_ret"] = ret
        out[f"{safe_name}_vol20"] = ret.rolling(20).std()
        roll_mean = price.rolling(60).mean()
        roll_std = price.rolling(60).std().replace(0, np.nan)
        out[f"{safe_name}_zscore60"] = (price - roll_mean) / roll_std
    return out


def align_macro_to_intraday(
    macro_features_daily: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    reporting_lag_days: int,
    age_column: str = "macro_data_age_hours",
) -> pd.DataFrame:
    """Shift daily macro features forward by the reporting lag, then
    forward-fill onto the intraday index -- an intraday bar only ever sees a
    macro data point that had already closed in the past (identical causality
    pattern to ``core.macro._align_daily_to_index``).

    This deliberately avoids building a full ``macro_index.union(target_index)``
    intermediate, because for a 5m target the union creates a large temporary
    DataFrame that is then forward-filled again. The macro series is daily and
    therefore only needs to be aligned at the daily level and then broadcast to
    all intraday timestamps in that day. This keeps the semantics causal while
    reducing the temporary memory footprint substantially.

    Also appends ``age_column`` (default ``macro_data_age_hours``): hours
    elapsed since the last *actual* (non-ffilled) observation. Crypto trades
    24/7 but the underlying equity-index futures/VIX data only update on US
    trading days (Problem C: weekend gap) -- plain ffill alone silently
    presents Saturday's stale Friday-close numbers as if they were fresh.
    Rather than inventing a new weekend-specific feature/special-case, this
    single continuous staleness signal lets the model itself learn how much
    to trust/downweight the ffilled block: it is small and near-constant on a
    normal weekday (just the reporting lag), and grows to ~48-72h over a
    weekend/holiday, unifying "weekend", "holiday" and any other data-gap
    case under one quantitative, causally-computed feature instead of several
    brittle calendar-rule special cases. This helper is reused for the
    higher-timeframe features below (``age_column`` given a distinct name per
    source there) since both share the exact same causal shift+ffill pattern."""
    # Normalize both sides before any reindexing. Mixing a naive DatetimeIndex
    # with a timezone-aware one can make pandas construct an object-backed index,
    # which breaks timedelta arithmetic and can also produce positional rather
    # than timestamp alignment.
    macro = macro_features_daily.copy()
    macro.index = pd.DatetimeIndex(pd.to_datetime(macro.index, utc=True))
    target = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True)).sort_values()

    if macro.empty or target.empty:
        aligned = pd.DataFrame(index=target)
        aligned[age_column] = np.nan
        return aligned

    lagged = macro.sort_index().shift(reporting_lag_days)
    target_days = target.normalize()
    unique_days = pd.DatetimeIndex(target_days.unique())
    # Only align at the daily grain and then broadcast to intraday timestamps.
    daily_aligned = lagged.reindex(unique_days).ffill()

    aligned = pd.DataFrame(index=target)
    for column in lagged.columns:
        aligned[column] = daily_aligned[column].reindex(target_days).to_numpy()

    valid_dates = lagged.index[lagged.notna().any(axis=1)]
    if valid_dates.empty:
        aligned[age_column] = np.nan
    else:
        observed_dates = pd.DatetimeIndex(pd.to_datetime(valid_dates, utc=True))
        search_pos = observed_dates.searchsorted(target_days, side="right") - 1
        first_valid_mask = target_days >= observed_dates[0]
        search_pos = np.where(first_valid_mask, search_pos, -1)
        search_pos = np.clip(search_pos, 0, len(observed_dates) - 1)
        last_observed = observed_dates.take(search_pos)
        age_hours = np.full(len(target_days), np.nan, dtype=float)
        valid_idx = np.flatnonzero(first_valid_mask)
        age_hours[valid_idx] = (target_days[valid_idx] - last_observed[valid_idx]).total_seconds() / 3600.0
        aligned[age_column] = age_hours

    aligned.index = target
    return aligned


def _breadth_cache_path(data_config: DataConfig, config: TrendMLConfig) -> Path:
    tag = "_".join(s.replace("/", "-") for s in config.breadth_symbols)
    return Path(data_config.cache_dir) / f"ml_breadth_{tag}_{config.base_timeframe}_{config.history_days}d.csv"


def fetch_breadth_basket(data_config: DataConfig, config: TrendMLConfig, use_cache: bool = True) -> pd.Series:
    """Equal-weight log-return index of a basket of liquid crypto majors
    *outside* the traded BTC/ETH/SOL universe -- a proxy for systemic
    "rest of the crypto market" noise (TOTAL2/TOTAL3-style breadth), since
    those indices aren't directly fetchable via ccxt/yfinance. Any basket
    symbol that happens to also be in the traded universe is dropped, so this
    can never leak a traded coin back in as its own "external" context.
    Cached like the OHLCV loader (same directory, same pagination path)."""
    if not config.breadth_enabled or not config.breadth_symbols:
        return pd.Series(dtype=float)

    cache_path = _breadth_cache_path(data_config, config)
    if use_cache and cache_path.exists():
        print(f"⚡ Lade Breadth-Basket-Cache: {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)["breadth_return"]

    closes = {}
    for symbol in config.breadth_symbols:
        if symbol in data_config.symbols:
            continue  # never let a traded symbol double as its own "external" breadth proxy
        history = fetch_ohlcv_history(
            symbol,
            config.base_timeframe,
            config.history_days,
            cache_dir=data_config.cache_dir,
            exchange_id=data_config.exchange_id,
            market_type=data_config.market_type,
        )
        if not history.empty:
            closes[symbol] = history["close"]

    if not closes:
        return pd.Series(dtype=float)

    price_df = pd.DataFrame(closes).sort_index().ffill().dropna(how="all")
    log_returns = np.log(price_df / price_df.shift(1))
    breadth_return = log_returns.mean(axis=1).rename("breadth_return")  # equal-weight basket, not raw per-coin prices

    if use_cache:
        os.makedirs(data_config.cache_dir, exist_ok=True)
        breadth_return.to_frame().to_csv(cache_path)
    return breadth_return


def build_breadth_features(close: pd.Series, breadth_return: pd.Series, config: TrendMLConfig) -> pd.DataFrame:
    """A small number of *decorrelating* cross-coin systemic features, derived
    from the breadth basket rather than its raw price/return series (see
    ``TrendMLConfig.breadth_symbols`` docstring for the multicollinearity
    rationale): the basket's own short-term return and vol/z-score regime, plus
    this symbol's rolling correlation with it (a regime indicator -- high when
    the whole market is moving together/systemic, low when this coin is
    behaving idiosyncratically)."""
    if breadth_return.empty:
        return pd.DataFrame(index=close.index)

    breadth = breadth_return.copy()
    breadth.index = pd.DatetimeIndex(pd.to_datetime(breadth.index, utc=True))
    close_index = pd.DatetimeIndex(pd.to_datetime(close.index, utc=True))
    aligned = breadth.sort_index().reindex(close_index).ffill()
    aligned.index = close_index
    out = pd.DataFrame(index=close.index)
    out["breadth_ret_1"] = aligned
    out["breadth_ret_6"] = aligned.rolling(6).sum()
    out["breadth_vol"] = aligned.rolling(config.breadth_vol_window).std()

    roll_mean = aligned.rolling(config.breadth_zscore_window).mean()
    roll_std = aligned.rolling(config.breadth_zscore_window).std().replace(0, np.nan)
    out["breadth_zscore"] = (aligned - roll_mean) / roll_std

    symbol_close = close.copy()
    symbol_close.index = close_index
    symbol_log_ret = np.log(symbol_close / symbol_close.shift(1))
    out["breadth_corr"] = symbol_log_ret.rolling(config.breadth_corr_window).corr(aligned)
    return out


def align_time_indexed_to_intraday(
    source: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    age_column: str,
) -> pd.DataFrame:
    """Align a time-indexed feature table onto an intraday index using the last
    observed prior value, without reusing the daily macro helper that broadcasts
    by calendar day. This preserves higher-timeframe sematics for 15m/1h feature
    rows while keeping the same causal forward-fill rule."""
    data = source.copy()
    data.index = pd.DatetimeIndex(pd.to_datetime(data.index, utc=True))
    target = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True)).sort_values()
    if data.empty or target.empty:
        aligned = pd.DataFrame(index=target)
        aligned[age_column] = np.nan
        return aligned

    union = data.index.union(target)
    aligned = data.reindex(union).ffill().loc[target].copy()

    observed = data.index.sort_values()
    search_pos = np.searchsorted(observed, target, side="right") - 1
    search_pos = np.clip(search_pos, 0, len(observed) - 1)
    last_observed = observed.take(search_pos)
    age_hours = (target - last_observed).total_seconds() / 3600.0
    aligned[age_column] = age_hours
    return aligned


def build_feature_matrix(
    ohlc: pd.DataFrame,
    macro_prices: pd.DataFrame | None,
    config: TrendMLConfig,
    breadth_return: pd.Series | None = None,
) -> pd.DataFrame:
    """Full causal feature matrix for one symbol: technical features, plus
    (optional) cross-asset macro context, plus (optional) cross-coin breadth
    context, all aligned onto the same intraday index."""
    features = build_technical_features(ohlc, config)
    if macro_prices is not None and not macro_prices.empty:
        macro_daily = build_macro_features(macro_prices, config)
        macro_aligned = align_macro_to_intraday(macro_daily, ohlc.index, config.macro_reporting_lag_days)
        features = features.join(macro_aligned)
    if breadth_return is not None and not breadth_return.empty:
        breadth_features = build_breadth_features(ohlc["close"], breadth_return, config)
        features = features.join(breadth_features)
    for rule in config.higher_timeframes or []:
        htf_features = build_higher_timeframe_features(ohlc, rule, config)
        if htf_features.empty:
            continue
        htf_aligned = align_time_indexed_to_intraday(
            htf_features,
            ohlc.index,
            age_column=f"htf_{rule}_data_age_hours",
        )
        features = features.join(htf_aligned)
    return features
