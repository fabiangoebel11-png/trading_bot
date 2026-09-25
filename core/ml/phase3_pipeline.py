"""Reproducible, target-specific Phase 3 data and feature pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from core.architecture import align_asof, content_hash
from core.ml.data import normalize_ohlcv
from core.ml.macro_adapter import align_macro_proxies
from sklearn.preprocessing import StandardScaler

TIMEFRAME_DELTAS = {
    "5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1), "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4), "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12), "1d": pd.Timedelta(days=1),
}
HORIZON_DELTAS = {
    "1h": pd.Timedelta(hours=1), "2h": pd.Timedelta(hours=2),
    "4h": pd.Timedelta(hours=4), "8h": pd.Timedelta(hours=8),
    "12h": pd.Timedelta(hours=12), "24h": pd.Timedelta(hours=24),
    "48h": pd.Timedelta(hours=48), "3d": pd.Timedelta(days=3),
    "7d": pd.Timedelta(days=7), "14d": pd.Timedelta(days=14),
    "21d": pd.Timedelta(days=21), "28d": pd.Timedelta(days=28),
}
ASSET_FOLDERS = {"BTC/USDT": "BTC_USDT", "ETH/USDT": "ETH_USDT", "SOL/USDT": "SOL_USDT"}


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    dataset_version: str
    target: str
    context_assets: tuple[str, ...]
    observation_timeframe: str
    prediction_horizon: str
    feature_set_id: str
    data_universe_id: str
    row_count: int
    start: str | None
    end: str | None
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    dropped_rows: int
    missing_data: dict[str, int]
    status: str
    content_hash: str


@dataclass(frozen=True)
class DatasetResult:
    frame: pd.DataFrame
    metadata: DatasetMetadata
    provenance: dict[str, dict[str, Any]]
    macro_statuses: tuple[Any, ...] = ()


def load_feature_set(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def asset_path(data_root: str | Path, asset: str, timeframe: str) -> Path:
    folder = ASSET_FOLDERS.get(asset)
    if folder is None:
        raise FileNotFoundError(f"INSUFFICIENT_DATA: no local crypto mapping for {asset}")
    return Path(data_root) / "market_crypto_2017" / folder / timeframe / "ccxt.parquet"


def load_local_ohlcv(data_root: str | Path, asset: str, timeframe: str) -> pd.DataFrame:
    path = asset_path(data_root, asset, timeframe)
    if not path.exists():
        raise FileNotFoundError(f"INSUFFICIENT_DATA: {asset} {timeframe} at {path}")
    return normalize_ohlcv(pd.read_parquet(path))


def inspect_ohlcv_quality(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    """Return diagnostics without filling gaps or hiding market closures."""
    if timeframe not in TIMEFRAME_DELTAS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    raw_index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    normalized = normalize_ohlcv(frame)
    deltas = raw_index.to_series().sort_values().diff().dropna()
    expected = TIMEFRAME_DELTAS[timeframe]
    prices = normalized[["open", "high", "low", "close"]]
    return {
        "rows": len(frame),
        "start": raw_index.min().isoformat() if len(raw_index) else None,
        "end": raw_index.max().isoformat() if len(raw_index) else None,
        "timezone": str(raw_index.tz),
        "duplicate_timestamps": int(raw_index.duplicated().sum()),
        "unsorted": not raw_index.is_monotonic_increasing,
        "gap_count": int((deltas > expected * 1.5).sum()),
        "negative_or_zero_prices": int((prices <= 0).any(axis=1).sum()),
        "invalid_ohlc": int(((prices["high"] < prices[["open", "close"]].max(axis=1)) | (prices["low"] > prices[["open", "close"]].min(axis=1))).sum()),
        "negative_volume": int((normalized["volume"] < 0).sum()),
        "zero_volume": int((normalized["volume"] == 0).sum()),
        "median_interval": deltas.median().isoformat() if len(deltas) else None,
    }


def fit_train_only_scaler(features: pd.DataFrame, train_index: pd.Index) -> tuple[StandardScaler, pd.DataFrame]:
    """Fit normalization only on train rows, then transform all supplied rows."""
    if len(train_index) == 0:
        raise ValueError("training split cannot be empty")
    scaler = StandardScaler()
    scaler.fit(features.loc[train_index])
    transformed = pd.DataFrame(scaler.transform(features), index=features.index, columns=features.columns)
    return scaler, transformed


def closed_candles(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe not in TIMEFRAME_DELTAS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    result = frame.copy()
    result.index = pd.DatetimeIndex(result.index + TIMEFRAME_DELTAS[timeframe])
    return result


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_local_features(ohlcv: pd.DataFrame, asset_prefix: str = "") -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    close = ohlcv["close"]
    log_return = np.log(close / close.shift(1))
    features: dict[str, pd.Series] = {}
    provenance: dict[str, dict[str, Any]] = {}

    def add(name: str, value: pd.Series, lookback: int | None = None) -> None:
        column = f"{asset_prefix}{name}"
        features[column] = value
        provenance[column] = {"source_asset": asset_prefix.rstrip("_") or "TARGET", "source_timeframe": "input", "calculation": name, "lookback": lookback, "availability_rule": "closed_candle"}

    for period in (1, 3, 6, 12, 24):
        add(f"return_{period}", close.pct_change(period), period)
        add(f"log_return_{period}", log_return.rolling(period).sum(), period)
    for period in (6, 12, 24):
        add(f"momentum_{period}", close / close.shift(period) - 1, period)
    for period in (20, 50):
        ma = close.rolling(period).mean()
        add(f"sma_distance_{period}", close / ma - 1, period)
        add(f"ema_distance_{period}", close / close.ewm(span=period, adjust=False).mean() - 1, period)
    add("rsi_14", _rsi(close), 14)
    for period in (12, 24, 48):
        add(f"realized_vol_{period}", log_return.rolling(period).std(), period)
        add(f"range_{period}", (ohlcv["high"].rolling(period).max() - ohlcv["low"].rolling(period).min()) / close, period)
    add("relative_volume_24", ohlcv["volume"] / ohlcv["volume"].rolling(24).mean(), 24)
    for period in (20, 55):
        high = ohlcv["high"].rolling(period).max()
        low = ohlcv["low"].rolling(period).min()
        add(f"range_position_{period}", (close - low) / (high - low).replace(0, np.nan), period)
        add(f"breakout_distance_{period}", close / high - 1, period)
    return pd.DataFrame(features, index=ohlcv.index), provenance


def build_context_features(context: dict[str, pd.DataFrame], target_index: pd.DatetimeIndex, target: str) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    output = pd.DataFrame(index=target_index)
    provenance: dict[str, dict[str, Any]] = {}
    for asset, frame in context.items():
        if asset == target:
            continue
        close = frame["close"]
        source = pd.DataFrame({"return_1": close.pct_change(), "volatility_24": np.log(close / close.shift(1)).rolling(24).std()}, index=close.index)
        aligned = align_asof(source, target_index, max_staleness=pd.Timedelta(days=2))
        prefix = asset.replace("/", "_").lower()
        for column in aligned.columns:
            name = f"context_{prefix}_{column}"
            output[name] = aligned[column].to_numpy()
            provenance[name] = {"source_asset": asset, "source_timeframe": "input", "calculation": column, "lookback": 24 if column == "volatility_24" else 1, "availability_rule": "backward_closed_candle_max_2d"}
    return output, provenance


def build_labels(target: pd.DataFrame, horizon: str, observation_timeframe: str | None = None) -> pd.DataFrame:
    if horizon not in HORIZON_DELTAS:
        raise ValueError(f"unsupported prediction horizon: {horizon}")
    if observation_timeframe is None:
        observed_delta = target.index.to_series().diff().dropna().median()
    else:
        observed_delta = TIMEFRAME_DELTAS[observation_timeframe]
    steps = max(1, int(HORIZON_DELTAS[horizon] / observed_delta))
    future_return = target["close"].shift(-steps) / target["close"] - 1
    return pd.DataFrame({"future_return": future_return, "direction": (future_return > 0).astype("float64")}, index=target.index)


def build_dataset(
    data_root: str | Path,
    target: str,
    observation_timeframe: str,
    prediction_horizon: str,
    *,
    context_assets: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT"),
    feature_set_id: str = "crypto_multimarket_v1",
    data_universe_id: str = "trading_bot_default_universe",
    dataset_version: str = "1.0.0",
    macro_path: str | Path | None = None,
) -> DatasetResult:
    raw_target = load_local_ohlcv(data_root, target, observation_timeframe)
    target_frame = closed_candles(raw_target, observation_timeframe)
    local, provenance = build_local_features(target_frame, f"{target.replace('/', '_').lower()}_")
    context_frames: dict[str, pd.DataFrame] = {}
    unavailable: list[str] = []
    for asset in context_assets:
        try:
            context_frames[asset] = closed_candles(load_local_ohlcv(data_root, asset, observation_timeframe), observation_timeframe)
        except FileNotFoundError:
            unavailable.append(asset)
    context, context_provenance = build_context_features(context_frames, target_frame.index, target)
    macro_statuses: tuple[Any, ...] = ()
    macro_features = pd.DataFrame(index=target_frame.index)
    macro_provenance: dict[str, dict[str, Any]] = {}
    if macro_path is not None:
        macro_result = align_macro_proxies(macro_path, target_frame.index)
        macro_features = macro_result.features
        macro_statuses = macro_result.statuses
        macro_provenance = {
            feature: {
                "source_asset": group["source_asset"].iloc[0],
                "source_file": group["source_file"].iloc[0],
                "availability_rule": "explicit_conservative_lag_and_asof",
                "proxy_or_real": "PROXY",
            }
            for feature, group in macro_result.provenance.groupby("feature_name")
        }
    labels = build_labels(target_frame, prediction_horizon, observation_timeframe)
    combined = local.join(context).join(macro_features).join(labels)
    before = len(combined)
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()
    dropped = before - len(combined)
    metadata_payload = {"target": target, "context_assets": context_assets, "observation_timeframe": observation_timeframe, "prediction_horizon": prediction_horizon, "feature_set_id": feature_set_id, "data_universe_id": data_universe_id, "dataset_version": dataset_version, "unavailable_context": unavailable, "macro_path": str(macro_path) if macro_path is not None else None}
    context_tag = "macro_proxy" if macro_path is not None else "no_macro"
    dataset_id = f"{target.replace('/', '_')}_{observation_timeframe}_{prediction_horizon}_{feature_set_id}_{context_tag}_{dataset_version}"
    feature_columns = tuple(local.columns) + tuple(context.columns) + tuple(macro_features.columns)
    metadata = DatasetMetadata(dataset_id, dataset_version, target, context_assets, observation_timeframe, prediction_horizon, feature_set_id, data_universe_id, len(combined), combined.index.min().isoformat() if len(combined) else None, combined.index.max().isoformat() if len(combined) else None, feature_columns, tuple(labels.columns), dropped, {column: int(combined[column].isna().sum()) for column in combined.columns}, "INSUFFICIENT_DATA" if not len(combined) else "AVAILABLE", content_hash(metadata_payload))
    return DatasetResult(combined, metadata, {**provenance, **context_provenance, **macro_provenance}, macro_statuses)