"""Leakage-safe alignment of independently supplied timeframe features."""
from __future__ import annotations

import pandas as pd

from core.config import TrendMLConfig
from core.ml.features import build_technical_features

_TIMEFRAME_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}


def align_closed_features(source: pd.DataFrame, target_index: pd.DatetimeIndex | pd.DataFrame) -> pd.DataFrame:
    """Align source rows to the latest source close at or before each target bar."""
    if source.empty:
        return pd.DataFrame(index=target_index)
    if isinstance(target_index, pd.DataFrame):
        target_index = target_index.index
    original_target = pd.DatetimeIndex(target_index)
    source = source.copy()
    source.index = pd.DatetimeIndex(pd.to_datetime(source.index, utc=True))
    target_index = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True))
    source = source.sort_index()
    target = pd.DataFrame(index=target_index.sort_values())
    aligned = pd.merge_asof(
        target.reset_index(names="timestamp"),
        source.reset_index(names="timestamp"),
        on="timestamp",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("timestamp")
    aligned = aligned.reindex(target_index)
    aligned.index = original_target
    return aligned


def build_multitimeframe_feature_matrix(
    ohlc_by_timeframe: dict[str, pd.DataFrame],
    target_timeframe: str,
    config: TrendMLConfig,
) -> pd.DataFrame:
    """Build features for all configured frames and align them to the target.

    The input candles must be indexed by their close timestamp. A higher
    timeframe row is visible only from its close onward; no forward fill from a
    not-yet-closed candle is possible.
    """
    if target_timeframe not in ohlc_by_timeframe:
        raise ValueError(f"Missing target timeframe: {target_timeframe}")
    if target_timeframe not in _TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {target_timeframe}")
    target_index = ohlc_by_timeframe[target_timeframe].index
    output = build_technical_features(ohlc_by_timeframe[target_timeframe], config)
    for timeframe, ohlc in ohlc_by_timeframe.items():
        if timeframe == target_timeframe:
            continue
        if timeframe not in _TIMEFRAME_MINUTES:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        if _TIMEFRAME_MINUTES[timeframe] < _TIMEFRAME_MINUTES[target_timeframe]:
            continue
        features = build_technical_features(ohlc, config).add_prefix(f"{timeframe}_")
        output = output.join(align_closed_features(features, target_index))
    return output
