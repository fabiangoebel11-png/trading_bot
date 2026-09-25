"""Contracts for the frozen Phase 3.3 ML research design.

The helpers in this module are deliberately small and pure. They validate
research metadata and split boundaries; they do not train models or select
strategies.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

TARGET_ASSETS = ("BTC/USDT", "ETH/USDT")
CONTEXT_ASSETS = ("ES_F_PROXY", "NQ_F_PROXY", "VIX", "URTH")
PROXY_ASSETS = CONTEXT_ASSETS
PRIMARY_FEATURE_SET = "crypto_core_v1"
OPTIONAL_PROXY_FEATURE_SET = "crypto_context_proxy_v1"
HOLDOUT_SPLIT = "final_holdout"


def validate_target_context_isolation(target: str, context_assets: Sequence[str]) -> None:
    """Reject a dataset contract that makes a target its own context."""
    if target not in TARGET_ASSETS:
        raise ValueError(f"target asset is not allowed in the frozen universe: {target}")
    overlap = set(context_assets).intersection({target})
    if overlap:
        raise ValueError(f"target_asset must differ from context_asset: {sorted(overlap)}")
    invalid = set(context_assets).difference(CONTEXT_ASSETS)
    if invalid:
        raise ValueError(f"context assets are not frozen research assets: {sorted(invalid)}")


def assert_selection_split(split_name: str) -> None:
    """Make accidental final-holdout selection fail closed."""
    if split_name == HOLDOUT_SPLIT:
        raise ValueError("final holdout is evaluation-only and cannot enter selection")
    if split_name not in {"train", "validation", "walk_forward_validation"}:
        raise ValueError(f"unknown split for selection: {split_name}")


def validate_feature_availability(
    feature_contract: Mapping[str, object],
    prediction_timestamp: pd.Timestamp,
) -> None:
    """Validate the causal timestamp and staleness rules for one feature."""
    required = {
        "feature_name", "source_asset", "source_timestamp", "availability_timestamp",
        "alignment_timestamp", "timeframe", "transformation", "staleness_limit",
        "proxy_or_real", "causal_status",
    }
    missing = required.difference(feature_contract)
    if missing:
        raise ValueError(f"feature contract missing fields: {sorted(missing)}")
    prediction = pd.Timestamp(prediction_timestamp)
    availability = pd.Timestamp(feature_contract["availability_timestamp"])
    alignment = pd.Timestamp(feature_contract["alignment_timestamp"])
    if prediction.tzinfo is None:
        prediction = prediction.tz_localize("UTC")
    else:
        prediction = prediction.tz_convert("UTC")
    if availability.tzinfo is None:
        availability = availability.tz_localize("UTC")
    else:
        availability = availability.tz_convert("UTC")
    if alignment.tzinfo is None:
        alignment = alignment.tz_localize("UTC")
    else:
        alignment = alignment.tz_convert("UTC")
    if availability > prediction:
        raise ValueError("feature availability_timestamp is after prediction timestamp")
    if alignment != prediction:
        raise ValueError("alignment_timestamp must equal prediction timestamp")
    staleness = prediction - availability
    limit = pd.Timedelta(feature_contract["staleness_limit"])
    if staleness > limit:
        raise ValueError("feature exceeds its configured staleness_limit")
    if feature_contract["proxy_or_real"] not in {"REAL", "PROXY"}:
        raise ValueError("proxy_or_real must be REAL or PROXY")
    if feature_contract["causal_status"] != "CAUSAL":
        raise ValueError("feature causal_status must be CAUSAL")


def label_end_times(sample_times: pd.DatetimeIndex, horizon: pd.Timedelta) -> pd.DatetimeIndex:
    """Return the latest timestamp that can influence each future label."""
    if horizon <= pd.Timedelta(0):
        raise ValueError("label horizon must be positive")
    return pd.DatetimeIndex(pd.to_datetime(sample_times, utc=True)) + horizon


def purged_train_mask(
    sample_times: pd.DatetimeIndex,
    label_ends: pd.DatetimeIndex,
    test_start: pd.Timestamp,
    embargo: pd.Timedelta,
) -> np.ndarray:
    """Keep only train rows strictly before the purged and embargoed boundary."""
    samples = pd.DatetimeIndex(pd.to_datetime(sample_times, utc=True))
    ends = pd.DatetimeIndex(pd.to_datetime(label_ends, utc=True))
    start = pd.Timestamp(test_start)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if len(samples) != len(ends):
        raise ValueError("sample_times and label_ends must have equal length")
    if embargo < pd.Timedelta(0):
        raise ValueError("embargo cannot be negative")
    boundary = start - embargo
    return (samples < boundary) & (ends < start)