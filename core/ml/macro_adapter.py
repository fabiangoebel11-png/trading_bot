"""Audited adapter for local daily macro proxy CSVs.

The cache contains daily period values but no publication timestamp. The
adapter therefore requires an explicit conservative availability lag and keeps
proxy identity visible in every feature name and provenance record.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.architecture import align_asof

MACRO_FILES = {
    "data/macro_ESF_VIX_URTH_3600d.csv": {
        "equity": "ES_F_PROXY", "vix": "VIX_PROXY", "world": "URTH_PROXY"
    },
    "data/macro_ESF_VIX_URTH_2600d.csv": {
        "equity": "ES_F_PROXY", "vix": "VIX_PROXY", "world": "URTH_PROXY"
    },
    "data/macro_ESF_VIX_URTH_730d.csv": {
        "equity": "ES_F_PROXY", "vix": "VIX_PROXY", "world": "URTH_PROXY"
    },
    "data/macro_ESF_VIX_730d.csv": {
        "equity": "ES_F_PROXY", "vix": "VIX_PROXY"
    },
    "data/ml_macro_ESF_NQF_VIX_URTH_2600d.csv": {
        "ES=F": "ES_F_PROXY", "NQ=F": "NQ_F_PROXY", "^VIX": "VIX_PROXY", "URTH": "URTH_PROXY"
    },
    "data/ml_macro_ESF_NQF_VIX_URTH_1095d.csv": {
        "ES=F": "ES_F_PROXY", "NQ=F": "NQ_F_PROXY", "^VIX": "VIX_PROXY", "URTH": "URTH_PROXY"
    },
}


@dataclass(frozen=True)
class MacroAssetStatus:
    requested_asset: str
    status: str
    proxy_asset: str | None
    source_file: str | None
    reason: str


@dataclass(frozen=True)
class MacroAdapterResult:
    features: pd.DataFrame
    provenance: pd.DataFrame
    statuses: tuple[MacroAssetStatus, ...]
    quality: dict[str, Any]


def inspect_macro_csv(path: str | Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError(f"macro CSV requires Date column: {path}")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["Date"], utc=True))
    deltas = timestamps.to_series().sort_values().diff().dropna()
    return {
        "source_file": str(path),
        "rows": len(frame),
        "columns": tuple(str(column) for column in frame.columns if column != "Date"),
        "start": timestamps.min().isoformat() if len(timestamps) else None,
        "end": timestamps.max().isoformat() if len(timestamps) else None,
        "timezone": str(timestamps.tz),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "sorted": timestamps.is_monotonic_increasing,
        "nulls": {str(column): int(value) for column, value in frame.isna().sum().items()},
        "weekend_rows": int((timestamps.dayofweek >= 5).sum()),
        "gap_count_over_2d": int((deltas > pd.Timedelta(days=2)).sum()),
        "median_interval": deltas.median().isoformat() if len(deltas) else None,
    }


def load_macro_csv(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    if "Date" not in frame.columns:
        raise ValueError(f"macro CSV requires Date column: {path}")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame.pop("Date"), utc=True))
    frame.index = timestamps
    frame = frame.apply(pd.to_numeric, errors="coerce").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame, inspect_macro_csv(path)


def _proxy_feature_frame(raw: pd.DataFrame, mapping: dict[str, str]) -> tuple[pd.DataFrame, dict[str, str]]:
    features: dict[str, pd.Series] = {}
    source_columns: dict[str, str] = {}
    for source_column, proxy_asset in mapping.items():
        if source_column not in raw:
            continue
        safe = proxy_asset.lower()
        price = raw[source_column]
        features[f"macro_{safe}_value"] = price
        features[f"macro_{safe}_return_1"] = price.pct_change()
        source_columns[f"macro_{safe}_value"] = source_column
        source_columns[f"macro_{safe}_return_1"] = source_column
    return pd.DataFrame(features, index=raw.index), source_columns


def align_macro_proxies(
    path: str | Path,
    target_index: pd.DatetimeIndex,
    *,
    availability_lag: pd.Timedelta = pd.Timedelta(days=2),
    max_staleness: pd.Timedelta = pd.Timedelta(days=5),
) -> MacroAdapterResult:
    """Load and causally align local proxies without forward-filling ad hoc."""
    raw, quality = load_macro_csv(path)
    normalized_path = str(path).replace("\\", "/")
    mapping = MACRO_FILES.get(normalized_path, {})
    if not mapping:
        mapping = next(
            (value for key, value in MACRO_FILES.items() if str(key).replace("\\", "/") == normalized_path),
            {},
        )
    if not mapping:
        raise ValueError(f"unregistered macro CSV proxy: {path}")
    source_features, source_columns = _proxy_feature_frame(raw, mapping)
    available_at = source_features.copy()
    available_at.index = available_at.index + availability_lag
    source_ns = pd.DataFrame(
        {f"__source_{column}": source_features.index.view("int64") for column in source_features.columns},
        index=available_at.index,
    )
    aligned = align_asof(
        pd.concat([available_at, source_ns], axis=1),
        target_index,
        max_staleness=max_staleness,
    )
    target = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True))
    provenance_rows: list[dict[str, Any]] = []
    for feature in source_features.columns:
        source_column = source_columns[feature]
        source_values = aligned[f"__source_{feature}"]
        for alignment_timestamp, source_timestamp_ns in source_values.items():
            if pd.isna(source_timestamp_ns):
                source_timestamp = pd.NaT
                availability_timestamp = pd.NaT
                staleness = np.nan
            else:
                source_timestamp = pd.Timestamp(int(source_timestamp_ns), tz="UTC")
                availability_timestamp = source_timestamp + availability_lag
                staleness = (pd.Timestamp(alignment_timestamp) - availability_timestamp).total_seconds()
            provenance_rows.append({
                "feature_name": feature,
                "source_asset": mapping.get(source_column),
                "source_file": str(path),
                "source_timestamp": source_timestamp,
                "availability_timestamp": availability_timestamp,
                "alignment_timestamp": alignment_timestamp,
                "staleness_seconds": staleness,
                "proxy_or_real": "PROXY",
            })
    features = aligned[source_features.columns].copy()
    features.index = target
    statuses = tuple(
        MacroAssetStatus(proxy, "AVAILABLE_PROXY", proxy, str(path), "local daily close proxy; publication timestamp unavailable")
        for proxy in dict.fromkeys(mapping.values())
    )
    return MacroAdapterResult(features, pd.DataFrame(provenance_rows), statuses, quality)


def requested_asset_statuses() -> tuple[MacroAssetStatus, ...]:
    return (
        MacroAssetStatus("GOLD", "INSUFFICIENT_DATA", None, None, "no verified local Gold source"),
        MacroAssetStatus("SPY", "AVAILABLE_PROXY", "ES_F_PROXY", "data/macro_ESF_VIX_URTH_3600d.csv", "ES futures proxy, not SPY"),
        MacroAssetStatus("QQQ", "AVAILABLE_PROXY", "NQ_F_PROXY", "data/ml_macro_ESF_NQF_VIX_URTH_2600d.csv", "NQ futures proxy, not QQQ"),
        MacroAssetStatus("DAX", "INSUFFICIENT_DATA", None, None, "no verified local DAX source"),
        MacroAssetStatus("VIX", "AVAILABLE_PROXY", "VIX_PROXY", "data/macro_ESF_VIX_URTH_3600d.csv", "VIX proxy column"),
        MacroAssetStatus("US10Y", "INSUFFICIENT_DATA", None, None, "no verified local US10Y source"),
        MacroAssetStatus("EURUSD", "INSUFFICIENT_DATA", None, None, "no verified local EURUSD source"),
        MacroAssetStatus("WTI", "INSUFFICIENT_DATA", None, None, "no verified local WTI source"),
        MacroAssetStatus("SOL/USDT", "INSUFFICIENT_DATA", None, None, "no verified local SOL source"),
    )