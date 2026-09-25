"""Directional, setup-level trade-quality contracts.

The feature row is observable at the rule entry close. Labels are resolved only
from later bars and are never returned as model inputs. The module is research
and inference infrastructure only; it does not train or enable production.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from core.config import TradeQualityConfig
from core.ml.features import compute_rsi
from core.strategy import compute_atr, compute_donchian_channels

TRADE_QUALITY_FEATURES = (
    "direction",
    "atr_pct",
    "ema_fast_dist_atr",
    "ema_slow_dist_atr",
    "trend_strength",
    "momentum_6",
    "momentum_24",
    "rsi_14",
    "volatility_24",
    "volume_zscore",
    "breakout_strength_atr",
    "regime_trend_up",
    "regime_trend_down",
    "regime_high_volatility",
    "stop_distance_pct",
    "target_distance_pct",
    "reward_risk",
    "cost_round_trip_pct",
)


def _entry_mask(position: pd.Series) -> pd.Series:
    position = position.astype(float)
    return position.ne(0) & position.ne(position.shift(1).fillna(0))


def build_entry_features(signals: pd.DataFrame, config: TradeQualityConfig) -> pd.DataFrame:
    """Build only entry-time observable setup geometry and market context."""
    required = {"close", "high", "low", "position"}
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"signals missing required columns: {sorted(missing)}")
    frame = signals.copy().sort_index()
    close, high, low = frame["close"].astype(float), frame["high"].astype(float), frame["low"].astype(float)
    atr = frame["atr"].astype(float) if "atr" in frame else compute_atr(high, low, close, 14)
    atr_safe = atr.replace(0, np.nan)
    ema_fast = frame["ema_fast"].astype(float) if "ema_fast" in frame else close.ewm(span=20, adjust=False).mean()
    ema_slow = frame["ema_slow"].astype(float) if "ema_slow" in frame else close.ewm(span=100, adjust=False).mean()
    upper, lower, _, _ = compute_donchian_channels(high, low, 55, 20)
    log_return = np.log(close / close.shift(1))
    volatility = log_return.rolling(24).std()
    volume = frame.get("volume", pd.Series(np.nan, index=frame.index)).astype(float)
    volume_z = (volume - volume.rolling(288).mean()) / volume.rolling(288).std().replace(0, np.nan)
    trend_strength = (ema_fast - ema_slow) / atr_safe
    breakout = np.where(frame["position"].astype(float) > 0, (close - upper) / atr_safe, (lower - close) / atr_safe)
    atr_pct = atr / close
    direction = np.sign(frame["position"].astype(float))
    stop_pct = config.stop_atr_multiple * atr_pct
    target_pct = config.take_profit_atr_multiple * atr_pct
    out = pd.DataFrame(index=frame.index)
    out["direction"] = direction
    out["atr_pct"] = atr_pct
    out["ema_fast_dist_atr"] = (close - ema_fast) / atr_safe
    out["ema_slow_dist_atr"] = (close - ema_slow) / atr_safe
    out["trend_strength"] = trend_strength
    out["momentum_6"] = close.pct_change(6)
    out["momentum_24"] = close.pct_change(24)
    out["rsi_14"] = compute_rsi(close, 14)
    out["volatility_24"] = volatility
    out["volume_zscore"] = volume_z
    out["breakout_strength_atr"] = breakout
    out["regime_trend_up"] = (trend_strength >= 0.0).astype(float)
    out["regime_trend_down"] = (trend_strength < 0.0).astype(float)
    out["regime_high_volatility"] = (atr_pct >= volatility.rolling(288).median() * 1.5).astype(float)
    out["stop_distance_pct"] = stop_pct
    out["target_distance_pct"] = target_pct
    out["reward_risk"] = target_pct / np.maximum(stop_pct, 1e-12)
    out["cost_round_trip_pct"] = 2.0 * (config.fee_rate + config.slippage_rate)
    out = out.loc[_entry_mask(frame["position"])]
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=list(TRADE_QUALITY_FEATURES))


def build_directional_labels(
    ohlc: pd.DataFrame,
    entry_features: pd.DataFrame,
    config: TradeQualityConfig,
) -> pd.DataFrame:
    """Resolve Long/Short TP/SL/time barriers after each rule entry.

    The label is 1 only when the direction-adjusted TP is reached before SL and
    costs; 0 covers SL first or time expiry. ``realized_r`` is net of the fixed
    round-trip cost and is suitable for diagnostics, not as an input feature.
    """
    close, high, low = (ohlc[column].astype(float) for column in ("close", "high", "low"))
    atr = compute_atr(high, low, close, 14)
    rows: list[dict[str, Any]] = []
    for timestamp, feature in entry_features.iterrows():
        if timestamp not in ohlc.index:
            continue
        start = ohlc.index.get_loc(timestamp)
        end = min(len(ohlc), start + config.horizon_bars + 1)
        if end < start + config.horizon_bars + 1:
            continue
        future = ohlc.iloc[start + 1 : end]
        if future.empty or not np.isfinite(atr.loc[timestamp]):
            continue
        direction = float(feature["direction"])
        entry = float(close.loc[timestamp])
        stop_distance = config.stop_atr_multiple * float(atr.loc[timestamp])
        target_distance = config.take_profit_atr_multiple * float(atr.loc[timestamp])
        stop = entry - direction * stop_distance
        target = entry + direction * target_distance
        outcome = "TIME"
        exit_price = float(close.iloc[end - 1])
        exit_timestamp = future.index[-1]
        for future_timestamp, bar in future.iterrows():
            stop_hit = float(bar["low"]) <= stop if direction > 0 else float(bar["high"]) >= stop
            target_hit = float(bar["high"]) >= target if direction > 0 else float(bar["low"]) <= target
            # Conservative same-bar rule: if both barriers hit, assume the stop.
            if stop_hit:
                outcome, exit_price, exit_timestamp = "SL", stop, future_timestamp
                break
            if target_hit:
                outcome, exit_price, exit_timestamp = "TP", target, future_timestamp
                break
        gross = direction * (exit_price / entry - 1.0)
        net = gross - config.fee_rate * 2.0 - config.slippage_rate * 2.0
        rows.append({
            "timestamp": timestamp,
            "label": int(outcome == "TP" and net > 0),
            "outcome": outcome,
            "direction": int(direction),
            "entry_price": entry,
            "exit_price": exit_price,
            "exit_timestamp": exit_timestamp,
            "realized_return": net,
            "realized_r": net / max(config.stop_atr_multiple * float(atr.loc[timestamp]) / entry, 1e-12),
            "mfe": float((future["high"].max() / entry - 1.0) * direction),
            "mae": float((future["low"].min() / entry - 1.0) * direction),
            "t1": exit_timestamp,
        })
    if not rows:
        return pd.DataFrame(columns=["label", "outcome", "direction", "realized_return", "realized_r", "mfe", "mae", "t1"])
    return pd.DataFrame(rows).set_index("timestamp")


def build_trade_quality_dataset(ohlc: pd.DataFrame, signals: pd.DataFrame, config: TradeQualityConfig) -> pd.DataFrame:
    features = build_entry_features(signals, config)
    labels = build_directional_labels(ohlc, features, config)
    return features.join(labels, how="inner")


def purged_entry_splits(dataset: pd.DataFrame, n_splits: int = 3, embargo_bars: int = 24) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding chronological splits purged by each entry's label end ``t1``."""
    if len(dataset) < (n_splits + 1) * 10:
        raise ValueError("too few setup rows for purged trade-quality splits")
    timestamps = pd.DatetimeIndex(dataset.index)
    t1 = pd.DatetimeIndex(pd.to_datetime(dataset["t1"], utc=True))
    fold_size = len(dataset) // (n_splits + 1)
    folds = []
    for fold in range(1, n_splits + 1):
        test_start = fold * fold_size
        test_end = len(dataset) if fold == n_splits else (fold + 1) * fold_size
        train_candidates = np.arange(0, test_start)
        test_start_time = timestamps[test_start]
        purge = t1[train_candidates] < test_start_time
        embargo = train_candidates < max(0, test_start - embargo_bars)
        folds.append((train_candidates[purge & embargo], np.arange(test_start, test_end)))
    return folds


def fit_trade_quality_model(dataset: pd.DataFrame, config: TradeQualityConfig) -> dict[str, Any]:
    """Fit/evaluate HGB only when explicitly called by a manual training command."""
    X = dataset.loc[:, list(TRADE_QUALITY_FEATURES)]
    y = dataset["label"].astype(int)
    results = []
    for fold, (train_idx, test_idx) in enumerate(purged_entry_splits(dataset), start=1):
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, random_state=42)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probability = model.predict_proba(X.iloc[test_idx])[:, 1]
        test_y = y.iloc[test_idx]
        record: dict[str, Any] = {"fold": fold, "rows": len(test_idx), "positive_rate": float(test_y.mean())}
        if test_y.nunique() > 1:
            record.update({"roc_auc": float(roc_auc_score(test_y, probability)), "average_precision": float(average_precision_score(test_y, probability))})
        record["balanced_accuracy"] = float(balanced_accuracy_score(test_y, probability >= 0.5))
        results.append(record)
    return {"model_id": config.model_id, "model_version": config.model_version, "feature_version": config.feature_version, "label_version": config.label_version, "features": list(TRADE_QUALITY_FEATURES), "folds": results, "status": "RESEARCH_ONLY"}


def load_trade_quality_artifact(config: TradeQualityConfig) -> dict[str, Any] | None:
    path = Path(config.model_dir) / f"{config.model_id}.json"
    if not config.enabled or not config.production_enabled or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def trade_quality_score(probability: float | None, config: TradeQualityConfig) -> float | None:
    if probability is None or not np.isfinite(probability):
        return None
    return float(np.clip(probability * 100.0, 0.0, 100.0))
