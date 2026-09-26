"""Phase-2 strategy-family research on local crypto OHLCV data.

This module is intentionally independent from the production backtester. The
research question is broader than the current Donchian implementation, so each
candidate gets the same small, causal OHLC execution simulator:

* indicators and signals use data available at the signal-bar close;
* orders fill at the next bar open;
* stops can trigger inside the execution bar;
* fees and slippage are charged explicitly;
* funding is never silently assumed to be zero when no local series exists;
* strategy selection happens at 1x before the leverage study.

Run with::

    python -m research.phase2_strategy_research
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.metrics import summarize_performance
from core.monte_carlo import _batched_metrics_numpy


LEVERAGE_GRID = (1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
FAMILY_NAMES = (
    "donchian_trend",
    "ema_momentum",
    "breakout_atr",
    "volatility_breakout",
    "mean_reversion",
    "momentum_trend_filter",
    "trend_momentum_combo",
)


@dataclass
class Phase2Config:
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframe: str = "1h"
    history_days: int = 2500
    output_dir: str = "data/research/phase2"
    stress_runs: int = 200
    stress_window_days: int = 365
    train_fraction: float = 0.50
    validation_fraction: float = 0.20
    walk_forward_folds: int = 4
    fee_rate: float = 0.00035
    slippage_rate: float = 0.00020
    position_risk_fraction: float = 0.005
    annual_vol_target: float = 0.35


@dataclass(frozen=True)
class Phase2Candidate:
    candidate_id: str
    family: str
    description: str
    parameters: dict[str, Any]
    stop_atr_multiple: float = 2.5


def _local_path(data_dir: str | Path, symbol: str, timeframe: str, history_days: int) -> Path:
    return Path(data_dir) / f"ohlcv_{symbol.replace('/', '-')}_{timeframe}_{history_days}d.csv"


def load_local_ohlcv(
    symbols: list[str], data_dir: str | Path, timeframe: str, history_days: int
) -> dict[str, pd.DataFrame]:
    """Load only existing cached CSVs. This function never downloads data."""
    result: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in symbols:
        path = _local_path(data_dir, symbol, timeframe, history_days)
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.sort_index()[["open", "high", "low", "close"]]
        result[symbol] = frame[~frame.index.duplicated(keep="last")]
    if missing:
        raise FileNotFoundError("Missing local OHLCV files: " + ", ".join(missing))
    return result


def _candidate(candidate_id: str, family: str, description: str, **parameters: Any) -> Phase2Candidate:
    stop = float(parameters.pop("stop_atr_multiple", 2.5))
    return Phase2Candidate(candidate_id, family, description, parameters, stop)


def build_candidate_grid() -> list[Phase2Candidate]:
    """Build a deliberately small, interpretable grid across seven families."""
    candidates: list[Phase2Candidate] = []
    for entry, exit_window in itertools.product((40, 55, 80), (15, 20)):
        candidates.append(
            _candidate(
                f"donchian_{entry}_{exit_window}",
                "donchian_trend",
                "Prior-channel breakout with EMA direction filter and ATR gate.",
                entry_window=entry,
                exit_window=exit_window,
                fast_window=20,
                slow_window=100,
                atr_window=14,
                min_atr_pct=0.0015,
                stop_atr_multiple=3.0,
            )
        )
    for fast, slow, momentum_window, threshold in itertools.product(
        (12, 20), (50, 100), (6, 12), (0.002, 0.004)
    ):
        candidates.append(
            _candidate(
                f"ema_{fast}_{slow}_{momentum_window}_{threshold:.3f}",
                "ema_momentum",
                "EMA direction plus positive/negative trailing momentum impulse.",
                fast_window=fast,
                slow_window=slow,
                momentum_window=momentum_window,
                momentum_threshold=threshold,
                atr_window=14,
                min_atr_pct=0.001,
                stop_atr_multiple=2.5,
            )
        )
    for lookback, multiple in itertools.product((20, 40, 60), (0.5, 1.0)):
        candidates.append(
            _candidate(
                f"breakout_atr_{lookback}_{multiple:.1f}",
                "breakout_atr",
                "Breakout must clear the prior range by an ATR-scaled distance.",
                lookback=lookback,
                breakout_atr_multiple=multiple,
                atr_window=14,
                stop_atr_multiple=2.5,
            )
        )
    for lookback, vol_multiple in itertools.product((20, 40, 60), (1.0, 1.5)):
        candidates.append(
            _candidate(
                f"vol_breakout_{lookback}_{vol_multiple:.1f}",
                "volatility_breakout",
                "Range breakout only when ATR% is above its causal rolling baseline.",
                lookback=lookback,
                volatility_multiple=vol_multiple,
                atr_window=14,
                stop_atr_multiple=2.5,
            )
        )
    for window, entry_z, max_atr_pct in itertools.product((24, 48), (1.5, 2.0), (0.02, 0.04)):
        candidates.append(
            _candidate(
                f"mean_reversion_{window}_{entry_z:.1f}_{max_atr_pct:.2f}",
                "mean_reversion",
                "Z-score excursion toward a causal rolling mean, blocked in high ATR%.",
                z_window=window,
                entry_z=entry_z,
                exit_z=0.25,
                max_atr_pct=max_atr_pct,
                stop_atr_multiple=2.0,
            )
        )
    for fast, slow, momentum_window in itertools.product((6, 12), (24, 48), (6, 12)):
        candidates.append(
            _candidate(
                f"momentum_filter_{fast}_{slow}_{momentum_window}",
                "momentum_trend_filter",
                "Momentum impulse allowed only in the direction of a slower EMA trend.",
                fast_window=fast,
                slow_window=slow,
                momentum_window=momentum_window,
                momentum_threshold=0.001,
                atr_window=14,
                min_atr_pct=0.001,
                stop_atr_multiple=2.5,
            )
        )
    for entry, momentum_window, min_atr_pct in itertools.product((40, 55), (6, 12), (0.001, 0.002)):
        candidates.append(
            _candidate(
                f"combo_{entry}_{momentum_window}_{min_atr_pct:.3f}",
                "trend_momentum_combo",
                "Donchian breakout confirmed by EMA direction, momentum and ATR regime.",
                entry_window=entry,
                exit_window=20,
                fast_window=20,
                slow_window=100,
                momentum_window=momentum_window,
                momentum_threshold=0.002,
                atr_window=14,
                min_atr_pct=min_atr_pct,
                stop_atr_multiple=3.0,
            )
        )
    return candidates


def _true_range(frame: pd.DataFrame) -> pd.Series:
    previous_close = frame["close"].shift(1)
    return pd.concat(
        [frame["high"] - frame["low"], (frame["high"] - previous_close).abs(), (frame["low"] - previous_close).abs()],
        axis=1,
    ).max(axis=1)


def _features(frame: pd.DataFrame, candidate: Phase2Candidate) -> pd.DataFrame:
    params = candidate.parameters
    atr_window = int(params.get("atr_window", 14))
    close = frame["close"]
    features = frame.copy()
    features["atr"] = _true_range(frame).rolling(atr_window, min_periods=atr_window).mean()
    features["atr_pct"] = features["atr"] / close
    features["ret_1"] = close.pct_change()
    features["realized_vol"] = features["ret_1"].rolling(24, min_periods=24).std() * np.sqrt(24 * 365)
    features["ema_fast"] = close.ewm(span=int(params.get("fast_window", 20)), adjust=False).mean()
    features["ema_slow"] = close.ewm(span=int(params.get("slow_window", 100)), adjust=False).mean()
    return features


def _trend_position(features: pd.DataFrame, fast: pd.Series, slow: pd.Series, long_entry: pd.Series, short_entry: pd.Series, long_exit: pd.Series | None = None, short_exit: pd.Series | None = None) -> pd.Series:
    position = np.zeros(len(features), dtype=float)
    current = 0.0
    close = features["close"].to_numpy()
    for i in range(len(features)):
        if not np.isfinite(close[i]):
            current = 0.0
        elif current > 0 and (bool(long_exit.iloc[i]) if long_exit is not None else bool(short_entry.iloc[i])):
            current = 0.0
        elif current < 0 and (bool(short_exit.iloc[i]) if short_exit is not None else bool(long_entry.iloc[i])):
            current = 0.0
        if current == 0.0:
            if bool(long_entry.iloc[i]) and fast.iloc[i] > slow.iloc[i]:
                current = 1.0
            elif bool(short_entry.iloc[i]) and fast.iloc[i] < slow.iloc[i]:
                current = -1.0
        position[i] = current
    return pd.Series(position, index=features.index)


def generate_signal(frame: pd.DataFrame, candidate: Phase2Candidate) -> tuple[pd.DataFrame, str]:
    """Generate a causal close-time position series for one asset/candidate."""
    features = _features(frame, candidate)
    p = candidate.parameters
    close = features["close"]
    atr_pct = features["atr_pct"]
    fast = features["ema_fast"]
    slow = features["ema_slow"]
    family = candidate.family
    false = pd.Series(False, index=features.index)

    if family == "donchian_trend":
        upper = features["high"].rolling(int(p["entry_window"])).max().shift(1)
        lower = features["low"].rolling(int(p["entry_window"])).min().shift(1)
        exit_upper = features["high"].rolling(int(p["exit_window"])).max().shift(1)
        exit_lower = features["low"].rolling(int(p["exit_window"])).min().shift(1)
        gate = atr_pct >= float(p["min_atr_pct"])
        long_entry, short_entry = (close > upper) & gate, (close < lower) & gate
        long_exit, short_exit = (close < exit_lower) | (fast < slow), (close > exit_upper) | (fast > slow)
    elif family in {"ema_momentum", "momentum_trend_filter"}:
        momentum = close.pct_change(int(p["momentum_window"]))
        threshold = float(p["momentum_threshold"])
        gate = atr_pct >= float(p.get("min_atr_pct", 0.0))
        long_entry, short_entry = (momentum > threshold) & gate, (momentum < -threshold) & gate
        long_exit, short_exit = (momentum <= 0) | (fast < slow), (momentum >= 0) | (fast > slow)
    elif family == "breakout_atr":
        lookback = int(p["lookback"])
        upper = features["high"].rolling(lookback).max().shift(1)
        lower = features["low"].rolling(lookback).min().shift(1)
        distance = float(p["breakout_atr_multiple"]) * features["atr"]
        long_entry, short_entry = close > upper + distance, close < lower - distance
        long_exit, short_exit = (close < fast), (close > fast)
    elif family == "volatility_breakout":
        lookback = int(p["lookback"])
        upper = features["high"].rolling(lookback).max().shift(1)
        lower = features["low"].rolling(lookback).min().shift(1)
        baseline = atr_pct.rolling(lookback, min_periods=lookback).median().shift(1)
        gate = atr_pct > baseline * float(p["volatility_multiple"])
        long_entry, short_entry = (close > upper) & gate, (close < lower) & gate
        long_exit, short_exit = close < fast, close > fast
    elif family == "mean_reversion":
        window = int(p["z_window"])
        mean = close.rolling(window, min_periods=window).mean().shift(1)
        std = close.rolling(window, min_periods=window).std().shift(1).replace(0, np.nan)
        zscore = (close - mean) / std
        gate = atr_pct <= float(p["max_atr_pct"])
        long_entry, short_entry = (zscore < -float(p["entry_z"])) & gate, (zscore > float(p["entry_z"])) & gate
        long_exit, short_exit = zscore >= -float(p["exit_z"]), zscore <= float(p["exit_z"])
    elif family == "trend_momentum_combo":
        entry = int(p["entry_window"])
        upper = features["high"].rolling(entry).max().shift(1)
        lower = features["low"].rolling(entry).min().shift(1)
        momentum = close.pct_change(int(p["momentum_window"]))
        gate = (atr_pct >= float(p["min_atr_pct"])) & (momentum.abs() >= float(p["momentum_threshold"]))
        long_entry = (close > upper) & (momentum > 0) & gate
        short_entry = (close < lower) & (momentum < 0) & gate
        long_exit, short_exit = (close < fast) | (momentum <= 0), (close > fast) | (momentum >= 0)
    else:  # pragma: no cover - candidate grid is closed above
        raise ValueError(f"Unknown strategy family: {family}")

    signals = _trend_position(features, fast, slow, long_entry.fillna(False), short_entry.fillna(False), long_exit.fillna(false), short_exit.fillna(false))
    features["signal_position"] = signals
    features["regime_trend_strength"] = (fast / slow - 1.0).abs()
    return features, _rule_text(candidate)


def _rule_text(candidate: Phase2Candidate) -> str:
    p = candidate.parameters
    if candidate.family == "donchian_trend":
        entry = f"long close > prior {p['entry_window']}-bar high; short close < prior low"
        exit_rule = f"exit on prior {p['exit_window']}-bar channel break or EMA{p['fast_window']}/EMA{p['slow_window']} reversal"
    elif candidate.family == "ema_momentum":
        entry = f"EMA{p['fast_window']} > EMA{p['slow_window']} and {p['momentum_window']}-bar return > {p['momentum_threshold']:.3%}; inverse for short"
        exit_rule = "exit when momentum reaches zero or EMA direction reverses"
    elif candidate.family == "momentum_trend_filter":
        entry = f"{p['momentum_window']}-bar return exceeds {p['momentum_threshold']:.3%} and EMA trend agrees"
        exit_rule = "exit on zero momentum or EMA direction reversal"
    elif candidate.family == "breakout_atr":
        entry = f"close clears prior {p['lookback']}-bar range by {p['breakout_atr_multiple']} ATR"
        exit_rule = "exit on close across fast EMA"
    elif candidate.family == "volatility_breakout":
        entry = f"prior {p['lookback']}-bar range break with ATR% > {p['volatility_multiple']}x causal ATR% median"
        exit_rule = "exit on close across fast EMA"
    elif candidate.family == "mean_reversion":
        entry = f"z-score < -{p['entry_z']} long or > {p['entry_z']} short, only below ATR% cap"
        exit_rule = f"exit when z-score returns within +/-{p['exit_z']}"
    else:
        entry = f"Donchian range break confirmed by {p['momentum_window']}-bar momentum and ATR gate"
        exit_rule = "exit on fast EMA break or momentum reversal"
    return f"Entry: {entry}. Stop: {candidate.stop_atr_multiple}x ATR from entry. Exit: {exit_rule}. Flat during NaN/warmup; after an ATR stop, re-entry waits for a flat signal."


def _execute_candidate(features: pd.DataFrame, candidate: Phase2Candidate, config: Phase2Config, stop_multiple: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute next-open signals and return bar returns plus trade-level records."""
    index = features.index
    close = features["close"].to_numpy(float)
    open_price = features["open"].to_numpy(float)
    high = features["high"].to_numpy(float)
    low = features["low"].to_numpy(float)
    atr = features["atr"].to_numpy(float)
    signal = features["signal_position"].to_numpy(float)
    n = len(features)
    bar_return = np.zeros(n, dtype=float)
    gross_return = np.zeros(n, dtype=float)
    turnover = np.zeros(n, dtype=float)
    active_position = np.zeros(n, dtype=float)
    trades: list[dict[str, Any]] = []
    current = 0.0
    entry_i = -1
    entry_price = np.nan
    entry_atr = np.nan
    stop_price = np.nan
    previous_close = np.nan
    stop_reentry_blocked = False

    def finish(exit_i: int, exit_price: float, reason: str) -> None:
        nonlocal current, entry_i, entry_price, entry_atr, stop_price
        if entry_i < 0 or not np.isfinite(exit_price):
            return
        excursion_end = exit_i if reason in {"atr_stop", "end_of_data"} else max(entry_i, exit_i - 1)
        window = slice(entry_i, excursion_end + 1)
        side = current
        adverse_prices = low[window] if side > 0 else high[window]
        favorable_prices = high[window] if side > 0 else low[window]
        adverse = np.minimum((adverse_prices / entry_price - 1.0) * side, 0.0)
        favorable = np.maximum((favorable_prices / entry_price - 1.0) * side, 0.0)
        trades.append(
            {
                "entry_timestamp": index[entry_i],
                "exit_timestamp": index[exit_i],
                "side": int(side),
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "gross_return": float(side * (exit_price / entry_price - 1.0)),
                "mae": float(np.nanmin(adverse)),
                "mfe": float(np.nanmax(favorable)),
                "duration_bars": int(exit_i - entry_i + 1),
                "entry_atr_pct": float(entry_atr / entry_price) if entry_price else None,
                "stop_atr_multiple": stop_multiple,
                "exit_reason": reason,
            }
        )
        current = 0.0
        entry_i = -1
        entry_price = np.nan
        entry_atr = np.nan
        stop_price = np.nan

    for i in range(1, n):
        if not np.isfinite(open_price[i]) or not np.isfinite(close[i]) or not np.isfinite(previous_close):
            previous_close = close[i] if np.isfinite(close[i]) else previous_close
            continue
        desired = signal[i - 1] if np.isfinite(signal[i - 1]) else 0.0
        if desired == 0.0:
            stop_reentry_blocked = False
        current_bar_return = 0.0
        if current != 0.0:
            stop_hit = False
            if np.isfinite(stop_price):
                stop_hit = (current > 0 and low[i] <= stop_price) or (current < 0 and high[i] >= stop_price)
            if stop_hit:
                exit_price = stop_price
                current_bar_return = current * (exit_price / previous_close - 1.0)
                finish(i, exit_price, "atr_stop")
                turnover[i] += 1.0
                stop_reentry_blocked = True
            elif desired != current:
                exit_price = open_price[i]
                current_bar_return = current * (exit_price / previous_close - 1.0)
                finish(i, exit_price, "signal_exit")
                turnover[i] += 1.0
            else:
                current_bar_return = current * (close[i] / previous_close - 1.0)
        if current == 0.0 and desired != 0.0 and not stop_reentry_blocked:
            current = desired
            entry_i = i
            entry_price = open_price[i]
            entry_atr = atr[i - 1]
            stop_price = (
                entry_price - current * float(stop_multiple) * entry_atr
                if stop_multiple is not None and np.isfinite(entry_atr)
                else np.nan
            )
            turnover[i] += 1.0
            if current_bar_return == 0.0:
                current_bar_return = current * (close[i] / entry_price - 1.0)
        gross_return[i] = current_bar_return
        active_position[i] = current
        cost = turnover[i] * (config.fee_rate + config.slippage_rate)
        bar_return[i] = current_bar_return - cost
        previous_close = close[i]
    if current != 0.0 and n > 1:
        finish(n - 1, close[-1], "end_of_data")
    result = pd.DataFrame(
        {
            "gross_return": gross_return,
            "turnover": turnover,
            "cost_return": bar_return - gross_return,
            "strategy_return_1x": bar_return,
            "position": active_position,
            "atr_pct": features["atr_pct"],
            "realized_vol": features["realized_vol"],
        },
        index=index,
    )
    return result, pd.DataFrame(trades)


def _regime_flags(features: pd.DataFrame) -> pd.DataFrame:
    returns = features["close"].pct_change()
    vol = returns.rolling(24, min_periods=24).std() * np.sqrt(24 * 365)
    vol_baseline = vol.rolling(720, min_periods=120).median().shift(1)
    trend = (features["ema_fast"] / features["ema_slow"] - 1.0).abs() >= 0.01
    drawdown = features["close"] / features["close"].rolling(720, min_periods=24).max() - 1.0
    crash = returns < -3.0 * returns.rolling(24, min_periods=24).std().shift(1)
    recovery = (drawdown < -0.10) & (features["close"] > features["close"].shift(24))
    return pd.DataFrame(
        {
            "trend": trend.fillna(False),
            "range": (~trend).fillna(False),
            "high_volatility": (vol > vol_baseline * 1.5).fillna(False),
            "low_volatility": (vol < vol_baseline * 0.75).fillna(False),
            "crash": crash.fillna(False),
            "recovery": recovery.fillna(False),
        },
        index=features.index,
    )


def _periods_per_year(timeframe: str) -> int:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([a-zA-Z]+)", timeframe.strip())
    if match is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    value, unit = match.groups()
    unit = {"m": "min", "min": "min", "h": "h", "d": "D"}.get(unit.lower(), unit)
    minutes = pd.to_timedelta(float(value), unit=unit).total_seconds() / 60.0
    return int((365 * 24 * 60) / minutes)


def _metric_dict(returns: pd.Series, turnover: pd.Series, periods: int) -> dict[str, Any]:
    metrics = summarize_performance(returns, periods, turnover * returns)
    metrics["trades"] = int((turnover > 0).sum() / 2)
    return metrics


def _regime_metrics(result: pd.DataFrame, regimes: pd.DataFrame, periods: int) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for regime in regimes.columns:
        mask = regimes[regime]
        output[regime] = {
            "bars": int(mask.sum()),
            **(_metric_dict(result.loc[mask, "strategy_return_1x"], result.loc[mask, "turnover"], periods) if mask.any() else {"total_return_pct": 0.0, "sharpe": 0.0, "max_drawdown_pct": 0.0, "trades": 0}),
        }
    return output


def _split_ranges(index: pd.Index, config: Phase2Config) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    train_end = int(len(index) * config.train_fraction)
    validation_end = int(len(index) * (config.train_fraction + config.validation_fraction))
    if train_end < 100 or validation_end <= train_end or validation_end >= len(index):
        raise ValueError("Research splits are too small or invalid")
    return {
        "train": (index[0], index[train_end - 1]),
        "validation": (index[train_end], index[validation_end - 1]),
        "oos": (index[validation_end], index[-1]),
    }


def _walk_forward_ranges(index: pd.Index, folds: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    test_size = len(index) // (folds + 1)
    return [(index[(i + 1) * test_size], index[(i + 2) * test_size - 1]) for i in range(folds)]


def _mae_mfe_summary(raw_trades: pd.DataFrame, train_end: pd.Timestamp) -> dict[str, Any]:
    if raw_trades.empty:
        return {"trades": 0, "mae_q50": None, "mae_q75": None, "mae_q90": None, "mfe_q50": None, "mfe_q75": None, "mfe_q90": None, "stop_candidates": [], "exit_candidates": []}
    train = raw_trades[raw_trades["entry_timestamp"] <= train_end]
    if train.empty:
        train = raw_trades
    adverse = train["mae"].astype(float)
    favorable = train["mfe"].astype(float)
    return {
        "trades": int(len(train)),
        "mae_q50": float(adverse.quantile(0.50)),
        "mae_q75": float(adverse.quantile(0.75)),
        "mae_q90": float(adverse.quantile(0.90)),
        "mfe_q50": float(favorable.quantile(0.50)),
        "mfe_q75": float(favorable.quantile(0.75)),
        "mfe_q90": float(favorable.quantile(0.90)),
        "mae_abs_q75": float((-adverse).quantile(0.75)),
        "mae_abs_q90": float((-adverse).quantile(0.90)),
        "stop_candidates": [float((-adverse).quantile(q)) for q in (0.75, 0.90)],
        "exit_candidates": [float(favorable.quantile(q)) for q in (0.50, 0.75)],
    }


def _parameter_stability(rows: list[dict[str, Any]], row: dict[str, Any]) -> float:
    peers = [candidate for candidate in rows if candidate["family"] == row["family"] and candidate["candidate_id"] != row["candidate_id"]]
    if not peers:
        return 0.0
    distances = []
    for peer in peers:
        keys = set(row["parameters"]) | set(peer["parameters"])
        distance = sum(row["parameters"].get(key) != peer["parameters"].get(key) for key in keys)
        distances.append((distance, peer))
    nearest = [peer for _, peer in sorted(distances, key=lambda item: item[0])[: max(2, min(5, len(distances)))]]
    return float(np.mean([peer["validation"]["total_return_pct"] > 0 and peer["validation"]["sharpe"] > 0 for peer in nearest]))


def _robust_gate(row: dict[str, Any]) -> bool:
    validation = row["validation"]
    oos = row["oos"]
    wf = row["walk_forward"]
    stress = row["stress"]
    return bool(
        validation["total_return_pct"] > 0
        and validation["sharpe"] > 0
        and oos["total_return_pct"] > 0
        and oos["sharpe"] >= 0.5
        and wf["median_sharpe"] > 0
        and stress["median_sharpe"] > 0
        and stress["pct_profitable"] >= 50.0
        and row["parameter_stability"] >= 0.5
        and oos["max_drawdown_pct"] > -50.0
    )


def _leverage_metrics(
    base_result: pd.DataFrame, config: Phase2Config, periods: int, stop_multiple: float
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    gross = base_result["gross_return"].to_numpy(float)
    costs = base_result["cost_return"].to_numpy(float)
    for leverage in LEVERAGE_GRID:
        returns = pd.Series(np.clip(gross * leverage + costs * leverage, -0.99, None), index=base_result.index)
        output[str(leverage)] = _metric_dict(returns, base_result["turnover"], periods)

    stop_pct = base_result["atr_pct"] * stop_multiple
    realized_vol = base_result["realized_vol"].replace(0, np.nan)
    risk_cap = config.position_risk_fraction / stop_pct.replace(0, np.nan)
    vol_cap = config.annual_vol_target / realized_vol
    desired = pd.concat([risk_cap, vol_cap], axis=1).min(axis=1).clip(upper=max(LEVERAGE_GRID))
    dynamic = np.zeros(len(desired), dtype=float)
    for i, value in enumerate(desired.to_numpy(float)):
        if not np.isfinite(value) or value < 1.0 or base_result["position"].iloc[i] == 0:
            dynamic[i] = 0.0
        else:
            allowed = [tier for tier in LEVERAGE_GRID if tier <= value]
            dynamic[i] = max(allowed) if allowed else 0.0
    dynamic_returns = pd.Series(np.clip((gross + costs) * dynamic, -0.99, None), index=base_result.index)
    output["dynamic_risk_vol_target"] = {
        **_metric_dict(dynamic_returns, base_result["turnover"], periods),
        "tier_distribution": {str(tier): float(np.mean(dynamic == tier) * 100.0) for tier in LEVERAGE_GRID},
        "no_trade_or_below_1x_pct": float(np.mean(dynamic == 0.0) * 100.0),
        "rule": "desired=min(position_risk/stop_pct, annual_vol_target/realized_vol); choose largest tier <= desired; no trade below 1x",
    }
    return output


def _stress(result: pd.DataFrame, config: Phase2Config, periods: int) -> dict[str, Any]:
    returns = result["strategy_return_1x"].to_numpy(float)
    window = int(config.stress_window_days * periods / 365)
    if window >= len(returns):
        return {"status": "INSUFFICIENT_DATA", "median_sharpe": 0.0, "pct_profitable": 0.0}
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(returns) - window, size=config.stress_runs)
    windows = returns[starts[:, None] + np.arange(window)[None, :]]
    values = _batched_metrics_numpy(windows, periods, 500.0)
    return {
        "status": "TESTED",
        "median_sharpe": float(np.median(values["sharpe"])),
        "mean_sharpe": float(np.mean(values["sharpe"])),
        "pct_profitable": float(np.mean(values["total_return_pct"] > 0) * 100.0),
        "median_total_return_pct": float(np.median(values["total_return_pct"])),
        "worst_case_max_drawdown_usdt": float(np.min(values["max_drawdown_usdt"])),
    }


def _candidate_row(
    symbol: str,
    candidate: Phase2Candidate,
    features: pd.DataFrame,
    config: Phase2Config,
    splits: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    result, stopped_trades = _execute_candidate(features, candidate, config, candidate.stop_atr_multiple)
    _, raw_trades = _execute_candidate(features, candidate, config, None)
    regimes = _regime_flags(features)
    periods = _periods_per_year(config.timeframe)
    metrics = {name: _metric_dict(result.loc[start:end, "strategy_return_1x"], result.loc[start:end, "turnover"], periods) for name, (start, end) in splits.items()}
    wf_values = []
    for start, end in _walk_forward_ranges(features.index, config.walk_forward_folds):
        wf_values.append(_metric_dict(result.loc[start:end, "strategy_return_1x"], result.loc[start:end, "turnover"], periods))
    row: dict[str, Any] = {
        "symbol": symbol,
        "candidate_id": candidate.candidate_id,
        "family": candidate.family,
        "description": candidate.description,
        "parameters": candidate.parameters,
        "rules": _rule_text(candidate),
        "position_risk_fraction": config.position_risk_fraction,
        "stop_atr_multiple": candidate.stop_atr_multiple,
        "fees": {"fee_rate_per_turn": config.fee_rate, "slippage_rate_per_turn": config.slippage_rate, "funding": "NOT TESTED: no local funding series"},
        **metrics,
        "walk_forward": {"folds": wf_values, "median_sharpe": float(np.median([value["sharpe"] for value in wf_values])) if wf_values else 0.0},
        "regimes": _regime_metrics(result, regimes, periods),
        "mae_mfe": _mae_mfe_summary(raw_trades, splits["train"][1]),
        "stopped_trade_count": int((stopped_trades["exit_reason"] == "atr_stop").sum()) if not stopped_trades.empty else 0,
        "mae_mfe_stopped": _mae_mfe_summary(
            stopped_trades[stopped_trades["exit_reason"] == "atr_stop"] if not stopped_trades.empty else stopped_trades,
            splits["train"][1],
        ),
        "stress": _stress(result, config, periods),
        "leverage": {},
    }
    return row


def _derive_profiles(robust_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not robust_rows:
        return {"status": "NO ROBUST STRATEGY FOUND", "profiles": {}}
    unique = {row["symbol"] + ":" + row["candidate_id"]: row for row in robust_rows}
    if len(unique) < 3:
        return {"status": "INSUFFICIENT_ROBUST_CANDIDATES", "profiles": {}}
    remaining = list(unique.values())
    conservative = max(remaining, key=lambda row: row["oos"]["max_drawdown_pct"])
    remaining.remove(conservative)
    balanced = max(remaining, key=lambda row: row["oos"]["sharpe"])
    remaining.remove(balanced)
    aggressive = max(remaining, key=lambda row: row["oos"]["total_return_pct"])
    return {
        "status": "DERIVED_FROM_ROBUST_CANDIDATES",
        "profiles": {"CONSERVATIVE": conservative, "BALANCED": balanced, "AGGRESSIVE": aggressive},
    }


def run_phase2(config: Phase2Config | None = None) -> dict[str, Any]:
    config = config or Phase2Config()
    base_data = Path("data")
    data = load_local_ohlcv(config.symbols, base_data, config.timeframe, config.history_days)
    candidates = build_candidate_grid()
    rows: list[dict[str, Any]] = []
    for symbol, frame in data.items():
        splits = _split_ranges(frame.index, config)
        family_rows: list[dict[str, Any]] = []
        for candidate in candidates:
            features, _ = generate_signal(frame, candidate)
            row = _candidate_row(symbol, candidate, features, config, splits)
            family_rows.append(row)
        for row in family_rows:
            row["parameter_stability"] = _parameter_stability(family_rows, row)
            row["robust"] = _robust_gate(row)
            if row["robust"]:
                representative = next(candidate for candidate in candidates if candidate.candidate_id == row["candidate_id"])
                features, _ = generate_signal(frame, representative)
                periods = _periods_per_year(config.timeframe)
                row["leverage"] = _leverage_metrics(
                    _execute_candidate(features, representative, config, representative.stop_atr_multiple)[0],
                    config,
                    periods,
                    representative.stop_atr_multiple,
                )
        rows.extend(family_rows)
    robust_rows = [row for row in rows if row["robust"]]
    return {
        "decision": "ROBUST STRATEGY FOUND" if robust_rows else "NO ROBUST STRATEGY FOUND",
        "research": asdict(config),
        "data": {
            "symbols": config.symbols,
            "timeframe": config.timeframe,
            "families": list(FAMILY_NAMES),
            "candidate_count_per_asset": len(candidates),
            "history": {symbol: {"start": frame.index.min().isoformat(), "end": frame.index.max().isoformat(), "bars": len(frame)} for symbol, frame in data.items()},
            "SPY": "INSUFFICIENT_DATA",
            "QQQ": "INSUFFICIENT_DATA",
        },
        "cost_model": {"fees": config.fee_rate, "slippage": config.slippage_rate, "funding": "NOT TESTED: no local funding cache"},
        "candidates": rows,
        "robust_candidates": robust_rows,
        "profiles": _derive_profiles(robust_rows),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):  # JSON cannot encode pandas timestamps
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_outputs(report: dict[str, Any], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "phase2_strategy_research.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    rows = []
    for row in report["candidates"]:
        rows.append(
            {
                "symbol": row["symbol"],
                "candidate_id": row["candidate_id"],
                "family": row["family"],
                "stop_atr_multiple": row["stop_atr_multiple"],
                "parameter_stability": row["parameter_stability"],
                "robust": row["robust"],
                "validation_sharpe": row["validation"]["sharpe"],
                "validation_return_pct": row["validation"]["total_return_pct"],
                "walk_forward_median_sharpe": row["walk_forward"]["median_sharpe"],
                "oos_sharpe": row["oos"]["sharpe"],
                "oos_return_pct": row["oos"]["total_return_pct"],
                "oos_max_drawdown_pct": row["oos"]["max_drawdown_pct"],
                "stress_median_sharpe": row["stress"]["median_sharpe"],
                "stress_profitable_pct": row["stress"]["pct_profitable"],
                "trades": row["oos"]["trades"],
                "dynamic_leverage_sharpe": row["leverage"].get("dynamic_risk_vol_target", {}).get("sharpe"),
                "dynamic_leverage_return_pct": row["leverage"].get("dynamic_risk_vol_target", {}).get("total_return_pct"),
                "dynamic_leverage_max_drawdown_pct": row["leverage"].get("dynamic_risk_vol_target", {}).get("max_drawdown_pct"),
                "dynamic_leverage_no_trade_or_below_1x_pct": row["leverage"].get("dynamic_risk_vol_target", {}).get("no_trade_or_below_1x_pct"),
            }
        )
    pd.DataFrame(rows).to_csv(path / "phase2_strategy_research.csv", index=False)
    robust = report["robust_candidates"]
    lines = [
        "# Phase 2 Strategy Research",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"Families: {', '.join(report['data']['families'])}",
        f"Candidates per asset: {report['data']['candidate_count_per_asset']}",
        "SPY/QQQ: `INSUFFICIENT_DATA`; no local OHLCV series was used.",
        "Funding: `NOT TESTED`; no local funding cache was available and this runner does not download data.",
        "",
        "## Robust Candidates",
    ]
    if not robust:
        lines.append("NO ROBUST STRATEGY FOUND. No Conservative/Balanced/Aggressive profile is promoted.")
    for row in robust:
        lines.extend(
            [
                "",
                f"### {row['symbol']} / {row['candidate_id']}",
                f"Family: {row['family']}",
                row["rules"],
                f"Position risk: {row['position_risk_fraction']:.3%}; parameter stability: {row['parameter_stability']:.2f}",
                f"Validation: Sharpe {row['validation']['sharpe']:.3f}, return {row['validation']['total_return_pct']:.2f}%, max DD {row['validation']['max_drawdown_pct']:.2f}%",
                f"Walk-forward median Sharpe: {row['walk_forward']['median_sharpe']:.3f}",
                f"OOS: Sharpe {row['oos']['sharpe']:.3f}, return {row['oos']['total_return_pct']:.2f}%, max DD {row['oos']['max_drawdown_pct']:.2f}%, trades {row['oos']['trades']}",
                f"MAE/MFE train quantiles: MAE q50={row['mae_mfe']['mae_q50']}, q75={row['mae_mfe']['mae_q75']}, q90={row['mae_mfe']['mae_q90']}; MFE q50={row['mae_mfe']['mfe_q50']}, q75={row['mae_mfe']['mfe_q75']}, q90={row['mae_mfe']['mfe_q90']}",
                f"Stress: median Sharpe {row['stress']['median_sharpe']:.3f}, profitable windows {row['stress']['pct_profitable']:.1f}%",
                "Leverage: evaluated only after this strategy passed the underlying gate.",
            ]
        )
    (path / "phase2_strategy_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stress-runs", type=int, default=200)
    parser.add_argument("--output-dir", default="data/research/phase2")
    args = parser.parse_args()
    config = Phase2Config(stress_runs=args.stress_runs, output_dir=args.output_dir)
    report = run_phase2(config)
    write_outputs(report, config.output_dir)
    print(report["decision"])
    print(f"Candidates per asset: {report['data']['candidate_count_per_asset']}")
    print(f"Robust candidates: {len(report['robust_candidates'])}")


if __name__ == "__main__":
    main()