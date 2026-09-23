"""Daily-primary dataset construction for the independent Model 2 family."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.ml.data import load_ohlcv_parquet
from core.ml.providers import AssetSpec, canonical_cache_path


@dataclass(frozen=True)
class SwingDataset:
    X_daily: np.ndarray
    X_entry: np.ndarray
    branch_mask: np.ndarray
    targets: dict[str, np.ndarray]
    timestamps: pd.DatetimeIndex
    feature_names: list[str]
    metadata: dict[str, object]


def _load(cache_dir: str, asset: str, timeframe: str) -> pd.DataFrame | None:
    name = {"QQQ": "NASDAQ100_PROXY", "SPY": "SP500_PROXY"}.get(asset, asset)
    for provider in ("yfinance", "twelve_data", "local_derived"):
        path = canonical_cache_path(cache_dir, AssetSpec(name, asset, provider, timeframe, 14600))
        if path.exists():
            return load_ohlcv_parquet(path)
    return None


def _load_context(cache_dir: str, name: str, symbol: str) -> pd.DataFrame | None:
    path = canonical_cache_path(cache_dir, AssetSpec(name, symbol, "yfinance", "1d", 14600))
    return load_ohlcv_parquet(path) if path.exists() else None


def discover_swing_history(cache_dir: str, asset: str) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for timeframe in ("1d", "1h", "5m"):
        frame = _load(cache_dir, asset, timeframe)
        report[timeframe] = {
            "earliest": frame.index.min().isoformat() if frame is not None and not frame.empty else None,
            "latest": frame.index.max().isoformat() if frame is not None and not frame.empty else None,
            "rows": int(len(frame)) if frame is not None else 0,
        }
    return report


def _technical(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    close = frame["close"].astype(float)
    high, low, volume = frame["high"], frame["low"], frame["volume"]
    ret = close.pct_change()
    ema20, ema50 = close.ewm(span=20, adjust=False).mean(), close.ewm(span=50, adjust=False).mean()
    delta = close.diff()
    gain, loss = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.fillna(pd.Series(np.where(gain > 0, 100.0, 50.0), index=frame.index))
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    out = pd.DataFrame(index=frame.index)
    out[f"{prefix}ret1"], out[f"{prefix}ret5"], out[f"{prefix}ret20"] = ret, close.pct_change(5), close.pct_change(20)
    out[f"{prefix}ema20_dist"], out[f"{prefix}ema50_dist"] = close / ema20 - 1, close / ema50 - 1
    out[f"{prefix}trend"], out[f"{prefix}rsi"] = ema20 / ema50 - 1, rsi / 100
    out[f"{prefix}atr_pct"], out[f"{prefix}vol20"] = atr / close, ret.rolling(20).std()
    out[f"{prefix}relvol"] = volume / volume.rolling(20).mean()
    out[f"{prefix}high_dist"], out[f"{prefix}low_dist"] = close / close.rolling(20).max() - 1, close / close.rolling(20).min() - 1
    return out


def _technical_columns(prefix: str) -> list[str]:
    return [f"{prefix}{name}" for name in ("ret1", "ret5", "ret20", "ema20_dist", "ema50_dist", "trend", "rsi", "atr_pct", "vol20", "relvol", "high_dist", "low_dist")]


def _daily_targets(daily: pd.DataFrame, horizons: list[int]) -> dict[str, np.ndarray]:
    close, high, low = daily["close"], daily["high"], daily["low"]
    result: dict[str, np.ndarray] = {}
    for horizon in horizons:
        result[f"return_{horizon}d"] = (close.shift(-horizon) / close - 1).to_numpy(np.float32)
        result[f"direction_{horizon}d"] = np.sign(result[f"return_{horizon}d"]).astype(np.float32)
        highs = pd.concat([high.shift(-i) for i in range(1, horizon + 1)], axis=1).max(axis=1)
        lows = pd.concat([low.shift(-i) for i in range(1, horizon + 1)], axis=1).min(axis=1)
        result[f"mfe_{horizon}d"] = (highs / close - 1).to_numpy(np.float32)
        result[f"mae_{horizon}d"] = (lows / close - 1).to_numpy(np.float32)
        result[f"duration_{horizon}d"] = np.full(len(daily), float(horizon), dtype=np.float32)
    return result


def _resample_intraday(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.resample("1h", label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def build_swing_dataset(cache_dir: str, asset: str, daily_lookback: int, horizons: list[int], macro_symbols: list[str] | None = None) -> SwingDataset:
    daily = _load(cache_dir, asset, "1d")
    if daily is None or daily.empty:
        raise FileNotFoundError(f"Missing daily history for {asset}")
    hourly, five = _load(cache_dir, asset, "1h"), _load(cache_dir, asset, "5m")
    daily = daily.sort_index()
    daily_features = _technical(daily, "d_")
    macro_features = pd.DataFrame(index=daily.index)
    for name, symbol in {"VIX": "^VIX", "US10Y": "^TNX", "EURUSD": "EURUSD=X", "GOLD": "GC=F", "WTI": "CL=F", "DAX": "^GDAXI"}.items():
        if macro_symbols is not None and symbol not in macro_symbols:
            continue
        context = _load_context(cache_dir, name, symbol)
        if context is None or context.empty:
            macro_features[f"macro_{name}_available"] = 0.0
            continue
        price = context["close"].astype(float).sort_index()
        aligned = price.reindex(daily.index).ffill().shift(1)
        macro_features[f"macro_{name}_ret"] = aligned.pct_change().fillna(0.0)
        macro_features[f"macro_{name}_vol20"] = aligned.pct_change().rolling(20).std().fillna(0.0)
        macro_features[f"macro_{name}_available"] = aligned.notna().astype(float)
    daily_features = pd.concat([daily_features, macro_features], axis=1)
    entry_parts, masks = [], []
    for frame, prefix in ((hourly, "h_"), (five, "m_")):
        if frame is None or frame.empty:
            entry_parts.append(pd.DataFrame(0.0, index=daily.index, columns=_technical_columns(prefix)))
            masks.append(pd.Series(0.0, index=daily.index))
            continue
        frame_features = _technical(frame, prefix)
        local_day = frame_features.index.tz_convert("America/New_York").date
        grouped = frame_features.assign(_day=local_day).groupby("_day").last()
        days = daily.index.tz_convert("America/New_York").date
        aligned = grouped.reindex(days).set_axis(daily.index)
        entry_parts.append(aligned)
        masks.append(aligned.notna().any(axis=1).astype(float))
    entry = pd.concat(entry_parts, axis=1).fillna(0.0)
    branch_mask = np.column_stack([masks[0].to_numpy(), masks[1].to_numpy()]).astype(np.float32)
    targets_full = _daily_targets(daily, horizons)
    daily_valid = daily_features.notna().all(axis=1)
    for target in targets_full.values():
        daily_valid &= np.isfinite(target)
    daily_valid.iloc[: max(daily_lookback - 1, 0)] = False
    daily_valid.iloc[-max(horizons):] = False
    daily_indices = np.flatnonzero(daily_valid.to_numpy())
    if len(daily_indices) == 0:
        raise ValueError(f"No usable daily samples for {asset}")
    daily_arr = daily_features.fillna(0.0).to_numpy(np.float32)

    # Keep the long daily history, but use each completed 5m bar as a distinct
    # recent entry decision. The daily lookup is strictly ``side='left'`` so an
    # intraday bar never sees that session's not-yet-completed daily candle.
    sample_daily_pos = daily_indices
    sample_entry = entry.to_numpy(np.float32)[daily_indices]
    sample_mask = branch_mask[daily_indices]
    sample_timestamps = daily.index[daily_indices]
    if five is not None and not five.empty:
        five_features = _technical(five.sort_index(), "m_").dropna()
        derived_hourly = _resample_intraday(five.sort_index())
        hour_features = _technical(derived_hourly, "h_").dropna()
        if len(five_features) and len(hour_features):
            daily_dates = daily.index.tz_convert("America/New_York").normalize()
            five_dates = five_features.index.tz_convert("America/New_York").normalize()
            daily_pos = daily_dates.searchsorted(five_dates, side="left") - 1
            hour_pos = hour_features.index.searchsorted(five_features.index, side="right") - 1
            eligible = (daily_pos >= daily_lookback - 1) & (daily_pos < len(daily) - max(horizons)) & (hour_pos >= 0)
            eligible &= np.array([bool(daily_valid.iloc[pos]) for pos in np.maximum(daily_pos, 0)])
            recent_dates = five_features.index[eligible]
            if len(recent_dates):
                recent_daily_pos = daily_pos[eligible]
                recent_hour_pos = hour_pos[eligible]
                recent_daily = np.stack([daily_arr[pos - daily_lookback + 1:pos + 1] for pos in recent_daily_pos])
                recent_hour = hour_features.iloc[recent_hour_pos].to_numpy(np.float32)
                recent_five = five_features.iloc[np.flatnonzero(eligible)].to_numpy(np.float32)
                recent_entry = np.concatenate([recent_hour, recent_five], axis=1)
                recent_targets = {key: value[recent_daily_pos] for key, value in targets_full.items()}
                daily_cutoff = five_features.index.min()
                keep_daily = sample_timestamps < daily_cutoff
                sample_daily_pos = np.concatenate([sample_daily_pos[keep_daily], recent_daily_pos])
                sample_entry = np.concatenate([sample_entry[keep_daily], recent_entry], axis=0)
                sample_mask = np.concatenate([sample_mask[keep_daily], np.ones((len(recent_entry), 2), dtype=np.float32)], axis=0)
                sample_timestamps = sample_timestamps[keep_daily].append(recent_dates)
                targets_full = {key: np.concatenate([value[daily_indices[keep_daily]], recent_targets[key]]) for key, value in targets_full.items()}

    targets = {key: value[:len(sample_timestamps)] for key, value in targets_full.items()}
    intraday_frames = [f for f in (hourly, five) if f is not None and not f.empty]
    return SwingDataset(np.stack([daily_arr[pos - daily_lookback + 1:pos + 1] for pos in sample_daily_pos]), sample_entry, sample_mask, targets, sample_timestamps, list(daily_features.columns) + list(entry.columns), {
        "entry_branch_widths": [len(_technical_columns("h_")), len(_technical_columns("m_"))],
        "asset": asset, "daily_data_start": daily.index.min().isoformat(), "daily_data_end": daily.index.max().isoformat(),
        "intraday_data_start": min((f.index.min() for f in intraday_frames), default=None), "intraday_data_end": max((f.index.max() for f in intraday_frames), default=None),
        "daily_only_samples": int((sample_mask.sum(axis=1) == 0).sum()), "full_multitimeframe_samples": int((sample_mask.sum(axis=1) == 2).sum()),
    })