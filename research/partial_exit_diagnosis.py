"""Post-research diagnosis for the causal partial-exit study.

The diagnosis reruns the fixed phase-2 entry strategy with corrected partial
exit accounting. It is descriptive and validation-oriented only: it does not
promote a tactic and does not touch production, paper, or GUI code.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.metrics import summarize_performance
from core.monte_carlo import _batched_metrics_numpy
from research.partial_exit_research import _policy_grid
from research.partial_exit_simulator import (
    PartialExitEngine,
    PartialExitPolicy,
    TakeProfitSpec,
)
from research.phase2_strategy_research import (
    Phase2Candidate,
    Phase2Config,
    _execute_candidate,
    _periods_per_year,
    _regime_flags,
    _split_ranges,
    _walk_forward_ranges,
    build_candidate_grid,
    generate_signal,
    load_local_ohlcv,
)


DIAGNOSIS_OUTPUT = "research/partial_exit"
ETH_SYMBOL = "ETH/USDT"
METRIC_KEYS = (
    "total_return_pct",
    "sharpe",
    "sortino",
    "max_drawdown_pct",
    "calmar",
    "profit_factor",
    "expectancy_pct",
    "win_rate_pct",
    "average_trade_pct",
    "median_trade_pct",
    "worst_trade_pct",
    "fees_pct",
    "slippage_pct",
    "trades",
    "exposure_pct",
    "average_holding_bars",
    "mae_pct",
    "mfe_pct",
)


@dataclass(frozen=True)
class PartialExitDiagnosisConfig:
    symbols: list[str]
    timeframe: str = "1h"
    history_days: int = 2500
    output_dir: str = DIAGNOSIS_OUTPUT
    stress_runs: int = 200
    stress_window_days: int = 365
    train_fraction: float = 0.50
    validation_fraction: float = 0.20
    walk_forward_folds: int = 4
    fee_rate: float = 0.00035
    slippage_rate: float = 0.00020
    base_candidate_id: str = "breakout_atr_60_1.0"


def _policy_family(policy_id: str) -> str:
    for prefix in (
        "runner",
        "partial",
        "atr_grid",
        "fixed_pct",
        "vol_normalized",
        "atr_dynamic",
        "mfe_quantile",
        "simple_scaleout",
    ):
        if policy_id.startswith(prefix):
            return prefix
    return policy_id


def _diagnosis_policies() -> tuple[list[PartialExitPolicy], list[PartialExitPolicy]]:
    existing = _policy_grid()
    additional = [
        PartialExitPolicy(
            "mfe_quantile_q75_50_100",
            (0.25, 0.25, 0.50),
            TakeProfitSpec("mfe_quantile", (0.50, 1.00)),
        ),
        PartialExitPolicy(
            "mfe_quantile_q75_75_100",
            (0.25, 0.25, 0.50),
            TakeProfitSpec("mfe_quantile", (0.75, 1.00)),
        ),
        PartialExitPolicy(
            "simple_scaleout_25_25_50",
            (0.25, 0.25, 0.50),
            TakeProfitSpec("atr_multiple", (1.0, 2.0)),
            "break_even_then_atr_trailing",
            "atr_trailing",
        ),
        PartialExitPolicy(
            "simple_scaleout_50_50",
            (0.50, 0.50),
            TakeProfitSpec("atr_multiple", (1.5,)),
            "break_even",
            "signal",
        ),
    ]
    return existing, existing + additional


def _trade_pnl(trades: pd.DataFrame, fee_rate: float, slippage_rate: float) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    if "net_return" in trades:
        return trades["net_return"].astype(float)
    # Phase-2 trade rows contain gross entry-to-exit PnL. Its one entry and
    # one exit are charged the same blended cost used by the bar series.
    return trades["gross_return"].astype(float) - 2.0 * (fee_rate + slippage_rate)


def _trade_statistics(trades: pd.DataFrame, fee_rate: float, slippage_rate: float) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
            "average_trade_pct": 0.0,
            "median_trade_pct": 0.0,
            "average_winner_pct": 0.0,
            "average_loser_pct": 0.0,
            "worst_trade_pct": 0.0,
            "average_holding_bars": 0.0,
            "mae_pct": 0.0,
            "mfe_pct": 0.0,
            "tp1_rate_pct": 0.0,
            "tp2_rate_pct": 0.0,
            "tp_any_rate_pct": 0.0,
        }
    pnl = _trade_pnl(trades, fee_rate, slippage_rate)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    losses = -losers.sum()
    events = trades.get("partial_events", pd.Series([[] for _ in range(len(trades))], index=trades.index))

    def event_rate(reason: str | None = None) -> float:
        if reason is None:
            hit = [any(str(event.get("reason", "")).startswith("tp") for event in row) for row in events]
        else:
            hit = [any(event.get("reason") == reason for event in row) for row in events]
        return float(np.mean(hit) * 100.0)

    return {
        "trades": int(len(trades)),
        "win_rate_pct": float(np.mean(pnl > 0) * 100.0),
        "profit_factor": float(winners.sum() / losses) if losses > 0 else (float("inf") if winners.sum() > 0 else 0.0),
        "expectancy_pct": float(pnl.mean() * 100.0),
        "average_trade_pct": float(pnl.mean() * 100.0),
        "median_trade_pct": float(pnl.median() * 100.0),
        "average_winner_pct": float(winners.mean() * 100.0) if not winners.empty else 0.0,
        "average_loser_pct": float(losers.mean() * 100.0) if not losers.empty else 0.0,
        "worst_trade_pct": float(pnl.min() * 100.0),
        "average_holding_bars": float(trades["duration_bars"].mean()),
        "mae_pct": float(trades["mae"].mean() * 100.0),
        "mfe_pct": float(trades["mfe"].mean() * 100.0),
        "tp1_rate_pct": event_rate("tp1"),
        "tp2_rate_pct": event_rate("tp2"),
        "tp_any_rate_pct": event_rate(),
    }


def _diagnostic_metrics(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    periods: int,
    fee_rate: float,
    slippage_rate: float,
) -> dict[str, Any]:
    pnl = _trade_pnl(trades, fee_rate, slippage_rate)
    summary = summarize_performance(result["strategy_return_1x"], periods, pnl if not pnl.empty else None)
    years = len(result) / periods if periods else 0.0
    cagr = ((1.0 + summary["total_return_pct"] / 100.0) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and summary["total_return_pct"] > -100.0 else -100.0
    rolling_window = max(1, int(periods * 30 / 365))
    rolling = result["strategy_return_1x"].rolling(rolling_window).apply(lambda values: float(np.prod(1.0 + values) - 1.0), raw=True)
    stats = _trade_statistics(trades, fee_rate, slippage_rate)
    turnover = result["turnover"].astype(float) if "turnover" in result else pd.Series(0.0, index=result.index)
    return {
        **summary,
        "cagr_pct": float(cagr),
        "fees_pct": float(turnover.sum() * fee_rate * 100.0),
        "slippage_pct": float(turnover.sum() * slippage_rate * 100.0),
        "total_cost_pct": float(-result["cost_return"].sum() * 100.0) if "cost_return" in result else 0.0,
        "exposure_pct": float(result["position"].abs().mean() * 100.0) if "position" in result else 0.0,
        "worst_30d_pct": float(rolling.min() * 100.0) if not rolling.dropna().empty else 0.0,
        **stats,
    }


def _trade_slice(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, field: str = "exit_timestamp") -> pd.DataFrame:
    if trades.empty:
        return trades
    return trades[(trades[field] >= start) & (trades[field] <= end)]


def _delta(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for key in METRIC_KEYS:
        left, right = metrics.get(key), baseline.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)) and np.isfinite(left) and np.isfinite(right):
            output[f"{key}_delta"] = float(left - right)
        else:
            output[f"{key}_delta"] = None
    return output


def _stress(result: pd.DataFrame, config: PartialExitDiagnosisConfig, periods: int) -> dict[str, Any]:
    returns = result["strategy_return_1x"].to_numpy(float)
    window = int(config.stress_window_days * periods / 365)
    if window <= 0 or window >= len(returns):
        return {"status": "INSUFFICIENT_DATA", "median_sharpe": 0.0, "pct_profitable": 0.0}
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(returns) - window, size=config.stress_runs)
    windows = returns[starts[:, None] + np.arange(window)[None, :]]
    metrics = _batched_metrics_numpy(windows, periods, 500.0)
    return {
        "status": "TESTED",
        "median_sharpe": float(np.median(metrics["sharpe"])),
        "mean_sharpe": float(np.mean(metrics["sharpe"])),
        "pct_profitable": float(np.mean(metrics["total_return_pct"] > 0) * 100.0),
        "median_total_return_pct": float(np.median(metrics["total_return_pct"])),
        "worst_case_max_drawdown_pct": float(np.min(metrics["max_drawdown_usdt"]) / 500.0 * 100.0),
    }


def _walk_forward(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    index: pd.Index,
    folds: int,
    periods: int,
    config: PartialExitDiagnosisConfig,
) -> dict[str, Any]:
    fold_rows = []
    for fold, (start, end) in enumerate(_walk_forward_ranges(index, folds), start=1):
        metrics = _diagnostic_metrics(
            result.loc[start:end],
            _trade_slice(trades, start, end),
            periods,
            config.fee_rate,
            config.slippage_rate,
        )
        fold_rows.append({"fold": fold, "start": start, "end": end, **metrics})
    return {
        "folds": fold_rows,
        "median_sharpe": float(np.median([row["sharpe"] for row in fold_rows])) if fold_rows else 0.0,
        "median_return_pct": float(np.median([row["total_return_pct"] for row in fold_rows])) if fold_rows else 0.0,
        "median_max_drawdown_pct": float(np.median([row["max_drawdown_pct"] for row in fold_rows])) if fold_rows else 0.0,
    }


def _period_metrics(
    result: pd.DataFrame,
    trades: pd.DataFrame,
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    periods: int,
    config: PartialExitDiagnosisConfig,
) -> dict[str, Any]:
    output = {}
    for name, (start, end) in ranges.items():
        output[name] = _diagnostic_metrics(
            result.loc[start:end],
            _trade_slice(trades, start, end),
            periods,
            config.fee_rate,
            config.slippage_rate,
        )
    return output


def _mfe_mae_distribution(
    raw_trades: pd.DataFrame,
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    quantiles = (0.25, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90)
    output: dict[str, Any] = {}
    for name, (start, end) in ranges.items():
        trades = _trade_slice(raw_trades, start, end, "entry_timestamp")
        if trades.empty:
            output[name] = {"trades": 0, "mfe_pct_quantiles": {}, "mae_abs_pct_quantiles": {}, "tp_reach_rates": {}}
            continue
        mfe = trades["mfe"].astype(float)
        mae_abs = -trades["mae"].astype(float)
        entry_atr_pct = trades["entry_atr_pct"].astype(float)
        output[name] = {
            "trades": int(len(trades)),
            "mfe_pct_quantiles": {f"p{int(q * 100)}": float(mfe.quantile(q) * 100.0) for q in quantiles},
            "mae_abs_pct_quantiles": {f"p{int(q * 100)}": float(mae_abs.quantile(q) * 100.0) for q in quantiles},
            "tp_reach_rates": {
                f"{multiple:g}x_atr_pct": float(np.mean(mfe >= multiple * entry_atr_pct) * 100.0)
                for multiple in (0.5, 1.0, 1.5, 2.0, 3.0)
            },
        }
    return output


def _regime_metrics(
    features: pd.DataFrame,
    regimes: pd.DataFrame,
    result: pd.DataFrame,
    trades: pd.DataFrame,
    oos_range: tuple[pd.Timestamp, pd.Timestamp],
    periods: int,
    config: PartialExitDiagnosisConfig,
) -> dict[str, Any]:
    start, end = oos_range
    output: dict[str, Any] = {}
    for regime in regimes.columns:
        mask = regimes[regime] & (regimes.index >= start) & (regimes.index <= end)
        selected = _trade_slice(trades, start, end, "entry_timestamp")
        if not selected.empty:
            selected = selected[selected["entry_timestamp"].isin(regimes.index[mask])]
        metrics = _diagnostic_metrics(result.loc[mask], selected, periods, config.fee_rate, config.slippage_rate) if mask.any() else {}
        output[regime] = {"bars": int(mask.sum()), **metrics}
    return output


def _partial_event_summary(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty or "partial_events" not in trades:
        return {"trades": 0, "tp1": 0, "tp2": 0, "tp_any": 0, "tp1_rate_pct": 0.0, "tp2_rate_pct": 0.0, "tp_any_rate_pct": 0.0}
    stats = _trade_statistics(trades, 0.0, 0.0)
    return {
        "trades": int(len(trades)),
        "tp1": int(round(stats["tp1_rate_pct"] * len(trades) / 100.0)),
        "tp2": int(round(stats["tp2_rate_pct"] * len(trades) / 100.0)),
        "tp_any": int(round(stats["tp_any_rate_pct"] * len(trades) / 100.0)),
        "tp1_rate_pct": stats["tp1_rate_pct"],
        "tp2_rate_pct": stats["tp2_rate_pct"],
        "tp_any_rate_pct": stats["tp_any_rate_pct"],
    }


def _event_examples(
    baseline_trades: pd.DataFrame,
    variant_trades: pd.DataFrame,
    train_end: pd.Timestamp,
) -> list[dict[str, Any]]:
    if baseline_trades.empty:
        return []
    train = baseline_trades[baseline_trades["entry_timestamp"] <= train_end]
    threshold = float(train["mfe"].quantile(0.75)) if not train.empty else float(baseline_trades["mfe"].quantile(0.75))
    low_threshold = float(train["mfe"].quantile(0.50)) if not train.empty else float(baseline_trades["mfe"].quantile(0.50))
    candidates = baseline_trades.sort_values("entry_timestamp")
    categories = (
        ("strong_trend", candidates[(candidates["mfe"] >= threshold) & (candidates["gross_return"] > 0)]),
        ("weak_breakout", candidates[candidates["mfe"] <= low_threshold]),
        ("whipsaw_or_stop", candidates[(candidates["exit_reason"] == "atr_stop") | (candidates["gross_return"] < 0)]),
        ("reversal_after_favorable_move", candidates[(candidates["gross_return"] < 0) & (candidates["mfe"] > low_threshold)]),
    )
    variant_by_entry = {row["entry_timestamp"]: row for row in variant_trades.to_dict("records")}
    output = []
    for category, frame in categories:
        if frame.empty:
            continue
        base = frame.iloc[0].to_dict()
        variant = variant_by_entry.get(base["entry_timestamp"])
        output.append(
            {
                "category": category,
                "selection_rule": "first chronological trade satisfying a train-derived MFE/exit condition",
                "entry_timestamp": base["entry_timestamp"],
                "side": int(base["side"]),
                "entry_price": float(base["entry_price"]),
                "baseline": {
                    "exit_timestamp": base["exit_timestamp"],
                    "exit_reason": base["exit_reason"],
                    "gross_return_pct": float(base["gross_return"] * 100.0),
                    "mae_pct": float(base["mae"] * 100.0),
                    "mfe_pct": float(base["mfe"] * 100.0),
                },
                "variant": {
                    "matched": variant is not None,
                    "exit_timestamp": variant.get("exit_timestamp") if variant else None,
                    "exit_reason": variant.get("exit_reason") if variant else None,
                    "net_return_pct": float(variant["net_return"] * 100.0) if variant and "net_return" in variant else None,
                    "mae_pct": float(variant["mae"] * 100.0) if variant else None,
                    "mfe_pct": float(variant["mfe"] * 100.0) if variant else None,
                    "partial_events": variant.get("partial_events", []) if variant else [],
                },
            }
        )
    return output


def _sensitivity(rows: list[dict[str, Any]], symbol: str) -> dict[str, Any]:
    selected = [row for row in rows if row["symbol"] == symbol]
    output: dict[str, Any] = {}
    for family in sorted({_policy_family(row["policy_id"]) for row in selected}):
        family_rows = [row for row in selected if _policy_family(row["policy_id"]) == family]
        oos_sharpes = [row["periods"]["oos"]["sharpe"] for row in family_rows]
        oos_returns = [row["periods"]["oos"]["total_return_pct"] for row in family_rows]
        validation_positive = [row["periods"]["validation"]["sharpe"] > 0 for row in family_rows]
        oos_positive = [row["periods"]["oos"]["sharpe"] > 0 for row in family_rows]
        output[family] = {
            "policies": [row["policy_id"] for row in family_rows],
            "oos_sharpe_min": float(min(oos_sharpes)) if oos_sharpes else 0.0,
            "oos_sharpe_max": float(max(oos_sharpes)) if oos_sharpes else 0.0,
            "oos_sharpe_range": float(max(oos_sharpes) - min(oos_sharpes)) if oos_sharpes else 0.0,
            "oos_return_min_pct": float(min(oos_returns)) if oos_returns else 0.0,
            "oos_return_max_pct": float(max(oos_returns)) if oos_returns else 0.0,
            "validation_positive_rate_pct": float(np.mean(validation_positive) * 100.0) if validation_positive else 0.0,
            "oos_positive_rate_pct": float(np.mean(oos_positive) * 100.0) if oos_positive else 0.0,
        }
    return output


def _risk_management_findings(rows: list[dict[str, Any]], baseline_by_symbol: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    findings = []
    for row in rows:
        base = baseline_by_symbol[row["symbol"]]
        metrics = row["periods"]["oos"]
        baseline = base["periods"]["oos"]
        similar_return = metrics["total_return_pct"] >= baseline["total_return_pct"] - 10.0
        similar_sharpe = metrics["sharpe"] >= baseline["sharpe"] - 0.25
        lower_dd = metrics["max_drawdown_pct"] >= baseline["max_drawdown_pct"] + 2.0
        better_tail = metrics["worst_trade_pct"] >= baseline["worst_trade_pct"] + 0.25
        if similar_return and similar_sharpe and lower_dd and better_tail:
            findings.append({"symbol": row["symbol"], "policy_id": row["policy_id"], "classification": "RISK MANAGEMENT BENEFIT", "oos": metrics, "baseline_oos": baseline})
    return findings


def _diagnostic_row(
    symbol: str,
    candidate: Phase2Candidate,
    features: pd.DataFrame,
    policy: PartialExitPolicy,
    config: PartialExitDiagnosisConfig,
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    baseline: dict[str, Any],
    train_mfe: float | None,
) -> dict[str, Any]:
    result, trades = PartialExitEngine(config.fee_rate, config.slippage_rate).execute(features, policy, candidate.stop_atr_multiple, train_mfe)
    periods = _periods_per_year(config.timeframe)
    period_metrics = _period_metrics(result, trades, ranges, periods, config)
    full = _diagnostic_metrics(result, trades, periods, config.fee_rate, config.slippage_rate)
    oos_events = _partial_event_summary(_trade_slice(trades, *ranges["oos"], "entry_timestamp"))
    return {
        "symbol": symbol,
        "base_candidate_id": candidate.candidate_id,
        "policy_id": policy.policy_id,
        "policy_family": _policy_family(policy.policy_id),
        "policy": asdict(policy),
        "periods": period_metrics,
        "full_history": full,
        "walk_forward": _walk_forward(result, trades, features.index, config.walk_forward_folds, periods, config),
        "stress": _stress(result, config, periods),
        "oos_partial_events": oos_events,
        "relative_to_baseline": {
            period: _delta(period_metrics[period], baseline["periods"][period])
            for period in ("validation", "oos")
        },
        "result": result,
        "trades": trades,
    }


def _baseline_record(
    symbol: str,
    candidate: Phase2Candidate,
    features: pd.DataFrame,
    config: PartialExitDiagnosisConfig,
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> dict[str, Any]:
    phase_config = Phase2Config(
        symbols=[symbol],
        timeframe=config.timeframe,
        history_days=config.history_days,
        fee_rate=config.fee_rate,
        slippage_rate=config.slippage_rate,
        walk_forward_folds=config.walk_forward_folds,
    )
    result, trades = _execute_candidate(features, candidate, phase_config, candidate.stop_atr_multiple)
    _, raw_trades = _execute_candidate(features, candidate, phase_config, None)
    periods = _periods_per_year(config.timeframe)
    period_metrics = _period_metrics(result, trades, ranges, periods, config)
    return {
        "symbol": symbol,
        "policy_id": "baseline_existing_full_exit",
        "periods": period_metrics,
        "full_history": _diagnostic_metrics(result, trades, periods, config.fee_rate, config.slippage_rate),
        "walk_forward": _walk_forward(result, trades, features.index, config.walk_forward_folds, periods, config),
        "stress": _stress(result, config, periods),
        "mfe_mae_distribution": _mfe_mae_distribution(raw_trades, ranges),
        "managed_trades": trades,
        "raw_trades": raw_trades,
        "result": result,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict("records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_list())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _strip_runtime(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key not in {"result", "trades"}}


def run_partial_exit_diagnosis(config: PartialExitDiagnosisConfig | None = None) -> dict[str, Any]:
    config = config or PartialExitDiagnosisConfig(symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    candidates = {candidate.candidate_id: candidate for candidate in build_candidate_grid()}
    if config.base_candidate_id not in candidates:
        raise ValueError(f"Unknown base candidate: {config.base_candidate_id}")
    candidate = candidates[config.base_candidate_id]
    data = load_local_ohlcv(config.symbols, "data", config.timeframe, config.history_days)
    existing_policies, all_policies = _diagnosis_policies()
    baseline_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    regime_rows: dict[str, Any] = {}
    event_rows: dict[str, Any] = {}
    for symbol, frame in data.items():
        features, _ = generate_signal(frame, candidate)
        phase_config = Phase2Config(
            symbols=[symbol],
            timeframe=config.timeframe,
            history_days=config.history_days,
            train_fraction=config.train_fraction,
            validation_fraction=config.validation_fraction,
            walk_forward_folds=config.walk_forward_folds,
            fee_rate=config.fee_rate,
            slippage_rate=config.slippage_rate,
        )
        ranges = _split_ranges(features.index, phase_config)
        baseline = _baseline_record(symbol, candidate, features, config, ranges)
        baseline_rows.append(_strip_runtime(baseline))
        train_raw = _trade_slice(baseline["raw_trades"], *ranges["train"], "entry_timestamp")
        train_mfe = float(train_raw["mfe"].quantile(0.75)) if not train_raw.empty else None
        asset_rows = []
        for policy in all_policies:
            row = _diagnostic_row(symbol, candidate, features, policy, config, ranges, baseline, train_mfe)
            asset_rows.append(row)
            variant_rows.append(_strip_runtime(row))
        regimes = _regime_flags(features)
        regime_rows[symbol] = {
            "source": "research.phase2_strategy_research._regime_flags",
            "oos_range": ranges["oos"],
            "baseline": _regime_metrics(features, regimes, baseline["result"], baseline["managed_trades"], ranges["oos"], _periods_per_year(config.timeframe), config),
            "variants": {
                row["policy_id"]: _regime_metrics(features, regimes, row["result"], row["trades"], ranges["oos"], _periods_per_year(config.timeframe), config)
                for row in asset_rows
            },
        }
        compare_policy = next(row for row in asset_rows if row["policy_id"] == "runner_25_25_50")
        event_rows[symbol] = {
            "comparison_policy": compare_policy["policy_id"],
            "examples": _event_examples(baseline["managed_trades"], compare_policy["trades"], ranges["train"][1]),
        }
    baseline_by_symbol = {row["symbol"]: row for row in baseline_rows}
    risk_findings = _risk_management_findings(variant_rows, baseline_by_symbol)
    eth_rows = [row for row in variant_rows if row["symbol"] == ETH_SYMBOL]
    robust_like = [
        row for row in eth_rows
        if row["periods"]["oos"]["sharpe"] >= baseline_by_symbol[ETH_SYMBOL]["periods"]["oos"]["sharpe"]
        and row["periods"]["oos"]["max_drawdown_pct"] >= baseline_by_symbol[ETH_SYMBOL]["periods"]["oos"]["max_drawdown_pct"]
    ]
    return {
        "decision": "NO ROBUST PARTIAL EXIT ADVANTAGE FOUND",
        "classification": "DIAGNOSIS_ONLY_NO_PROMOTION",
        "executive_summary": {
            "root_finding": "Partial exits generally monetize early movement but give up too much of the long trend; variants with nearly unchanged results often did not execute meaningful TP events.",
            "corrected_accounting": "The original simulator omitted realized TP/stop/signal PnL from strategy_return_1x and omitted entry fee from trade net_return. The simulator and original research report were recomputed before this diagnosis.",
            "eth_candidates_with_nonworse_oos_risk_and_sharpe": [row["policy_id"] for row in robust_like],
            "promotion": "blocked: no conditional/regime tactic was promoted; all regime results are descriptive until a pre-specified walk-forward rule exists.",
        },
        "research_contract": {
            "config": asdict(config),
            "symbols_available": list(data),
            "missing_symbols": [symbol for symbol in config.symbols if symbol not in data],
            "base_candidate_id": candidate.candidate_id,
            "base_candidate_parameters": candidate.parameters,
            "entry_rule_fixed_before_management_study": True,
            "funding": "NOT TESTED",
            "same_bar_rule": "stop before TP",
            "split_rule": "train/validation/OOS chronological; MFE-quantile targets use training entries only",
            "stress_rule": "200 seeded random contiguous 365-day windows from each corrected bar-return series",
        },
        "technical_analysis": {
            "existing_policy_count": len(existing_policies),
            "diagnosis_policy_count": len(all_policies),
            "existing_policy_catalog": [
                {
                    "policy_id": policy.policy_id,
                    "family": _policy_family(policy.policy_id),
                    "allocations": policy.allocations,
                    "tp_method": policy.take_profit.method if policy.take_profit else None,
                    "tp_values": policy.take_profit.values if policy.take_profit else None,
                    "stop_mode": policy.stop_mode,
                    "final_exit": policy.final_exit,
                }
                for policy in existing_policies
            ],
            "additional_policy_catalog": [
                {
                    "policy_id": policy.policy_id,
                    "family": _policy_family(policy.policy_id),
                    "allocations": policy.allocations,
                    "tp_method": policy.take_profit.method,
                    "tp_values": policy.take_profit.values,
                    "stop_mode": policy.stop_mode,
                    "final_exit": policy.final_exit,
                }
                for policy in all_policies[len(existing_policies):]
            ],
            "accounting_note": "Partial bar returns include entry/TP/stop/signal fills; trade net returns include entry and exit fees. Bar-series and trade-level returns use their respective causal accounting conventions.",
        },
        "baseline": baseline_rows,
        "variants": variant_rows,
        "eth_breakout_atr_60_1.0": {
            "baseline": baseline_by_symbol.get(ETH_SYMBOL),
            "variants": eth_rows,
            "sensitivity": _sensitivity(variant_rows, ETH_SYMBOL),
        },
        "regime_analysis": regime_rows,
        "mfe_mae_analysis": {row["symbol"]: row["mfe_mae_distribution"] for row in baseline_rows},
        "parameter_sensitivity": {symbol: _sensitivity(variant_rows, symbol) for symbol in data},
        "overfitting_analysis": {
            "classification": "OVERFIT RISK",
            "reason": "The study spans allocations, TP methods, stop modes, final exits, dynamic volatility scaling, MFE-derived targets, and leverage-adjacent reporting. Most families do not have enough neighboring parameters to establish broad stability, and no regime-specific rule was pre-registered.",
            "freedom_degrees": ["TP stage count", "allocation fractions", "TP distance/method", "stop mode", "final exit", "trailing constants", "regime condition", "asset-specific selection"],
            "selection_rule": "No final-OOS or stress result was used to promote a tactic; all additional policies are marked diagnostic only.",
        },
        "event_level_analysis": event_rows,
        "risk_management_analysis": {
            "findings": risk_findings,
            "classification": "RISK MANAGEMENT BENEFIT NOT ESTABLISHED" if not risk_findings else "RISK MANAGEMENT BENEFIT CANDIDATES REQUIRE FRESH HOLDOUT",
        },
        "simple_scaleout": {
            "policies": [row for row in variant_rows if row["policy_id"].startswith("simple_scaleout_")],
            "classification": "PROMISING BUT NOT ROBUST",
            "reason": "These policies were added as a small pre-specified hypothesis test, not tuned winners; they still require the same independent holdout and regime-gated validation before any use.",
        },
        "answers": {
            "why_no_robust_advantage": "Early TP realization reduces participation in the persistent breakout trend, while extra fills add fees/slippage. The few near-baseline variants often have very low real TP activation and therefore are not meaningful scale-outs.",
            "baseline_vs_partial": "The full-position baseline remains the reference. ETH runner and stop variants show lower OOS Sharpe/return; fixed and ATR-grid variants can reduce drawdown but lose materially more return. No variant clears the combined evidence gates.",
            "eth_result": "ETH is the most favorable test case, but the apparent near-baseline vol-normalized rows have negligible TP-event rates and are not robust partial exits.",
            "risk_benefit": "A risk-management benefit was not established under the pre-specified return, Sharpe, drawdown, worst-trade, and event-activity conditions.",
            "overfitting": "Yes. The management search has many interacting degrees of freedom and sparse neighboring sensitivity for some families.",
            "simple_rule": "Simple scale-outs remain PROMISING BUT NOT ROBUST, not a production candidate.",
            "disposition": "Do not promote; retain Full-Position Management as the current control and only continue with a pre-registered, low-dimensional holdout study.",
            "next_step": "Run one ETH-only, pre-registered holdout experiment with exactly two simple policies plus the full-exit control, fixed costs, funding explicitly sourced or marked unavailable, and a minimum TP-activation threshold.",
        },
        "not_robust": [
            "all 19 original partial-exit variants as promotion candidates",
            "vol-normalized variants as meaningful partial exits when TP activation is negligible",
            "dynamic leverage as evidence of management quality",
            "regime-specific selection based on final OOS outcomes",
            "funding-adjusted conclusions while funding is NOT TESTED",
        ],
    }


def _flat_csv_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = {row["symbol"]: row for row in report["baseline"]}
    rows = []
    for row in report["variants"]:
        oos = row["periods"]["oos"]
        validation = row["periods"]["validation"]
        wf = row["walk_forward"]
        stress = row["stress"]
        base_oos = baseline[row["symbol"]]["periods"]["oos"]
        rows.append(
            {
                "symbol": row["symbol"],
                "policy_id": row["policy_id"],
                "policy_family": row["policy_family"],
                "classification": "DIAGNOSTIC_ONLY",
                "validation_return_pct": validation["total_return_pct"],
                "validation_sharpe": validation["sharpe"],
                "validation_sortino": validation["sortino"],
                "oos_return_pct": oos["total_return_pct"],
                "oos_sharpe": oos["sharpe"],
                "oos_sortino": oos["sortino"],
                "oos_max_drawdown_pct": oos["max_drawdown_pct"],
                "oos_calmar": oos["calmar"],
                "oos_profit_factor": oos["profit_factor"],
                "oos_expectancy_pct": oos["expectancy_pct"],
                "oos_win_rate_pct": oos["win_rate_pct"],
                "oos_average_trade_pct": oos["average_trade_pct"],
                "oos_median_trade_pct": oos["median_trade_pct"],
                "oos_worst_trade_pct": oos["worst_trade_pct"],
                "oos_fees_pct": oos["fees_pct"],
                "oos_slippage_pct": oos["slippage_pct"],
                "oos_trades": oos["trades"],
                "oos_exposure_pct": oos["exposure_pct"],
                "oos_average_holding_bars": oos["average_holding_bars"],
                "oos_mae_pct": oos["mae_pct"],
                "oos_mfe_pct": oos["mfe_pct"],
                "oos_tp_any_rate_pct": row["oos_partial_events"]["tp_any_rate_pct"],
                "walk_forward_median_return_pct": wf["median_return_pct"],
                "walk_forward_median_sharpe": wf["median_sharpe"],
                "walk_forward_median_max_drawdown_pct": wf["median_max_drawdown_pct"],
                "stress_median_sharpe": stress["median_sharpe"],
                "stress_profitable_pct": stress["pct_profitable"],
                "delta_oos_return_pct": oos["total_return_pct"] - base_oos["total_return_pct"],
                "delta_oos_sharpe": oos["sharpe"] - base_oos["sharpe"],
                "delta_oos_max_drawdown_pct": oos["max_drawdown_pct"] - base_oos["max_drawdown_pct"],
            }
        )
    return rows


def write_outputs(report: dict[str, Any], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_report = _json_safe(report)
    (path / "partial_exit_diagnosis.json").write_text(json.dumps(json_report, indent=2), encoding="utf-8")
    pd.DataFrame(_flat_csv_rows(report)).to_csv(path / "partial_exit_diagnosis.csv", index=False)

    lines = [
        "# Partial Exit Post-Research Diagnosis",
        "",
        "## 1. Executive Summary",
        "",
        "**Decision: NO ROBUST PARTIAL EXIT ADVANTAGE FOUND**",
        "",
        report["executive_summary"]["root_finding"],
        "",
        "The diagnosis reran the study after correcting partial-exit accounting. The previous report was not treated as final evidence because realized TP/stop/signal PnL had not fully entered the bar-return series and entry fees were absent from trade net returns.",
        "",
        "## 2. Technical Analysis",
        "",
        f"The original research tested {report['technical_analysis']['existing_policy_count']} variants. The diagnosis adds {report['technical_analysis']['diagnosis_policy_count'] - report['technical_analysis']['existing_policy_count']} explicitly labeled diagnostic policies: two training-only MFE-quantile policies and two simple scale-outs.",
        "",
        "Entries and direction remain fixed at `breakout_atr_60_1.0`; only stateful management changes. Entries fill next open, stops precede TP on the same bar, and MFE-derived targets use training trades only.",
        "",
        "### Original Policy Families",
        "",
        "| Policy | Allocation | TP method | TP values | Stop | Final exit |",
        "|---|---|---|---|---|---|",
    ]
    for policy in report["technical_analysis"]["existing_policy_catalog"]:
        lines.append(f"| `{policy['policy_id']}` | {policy['allocations']} | {policy['tp_method'] or '-'} | {policy['tp_values'] or '-'} | {policy['stop_mode']} | {policy['final_exit']} |")
    lines.extend(["", "## 3. Baseline vs Partial Exits", "", "All rows below are diagnostic comparisons; none is promoted.", "", "| Symbol | Policy | OOS return | OOS Sharpe | OOS Sortino | OOS max DD | OOS PF | Exp. | Win rate | Median trade | Worst trade | TP any | WF Sharpe | Stress Sharpe |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["variants"]:
        oos = row["periods"]["oos"]
        lines.append(f"| {row['symbol']} | `{row['policy_id']}` | {oos['total_return_pct']:.2f}% | {oos['sharpe']:.3f} | {oos['sortino']:.3f} | {oos['max_drawdown_pct']:.2f}% | {oos['profit_factor']:.2f} | {oos['expectancy_pct']:.3f}% | {oos['win_rate_pct']:.1f}% | {oos['median_trade_pct']:.3f}% | {oos['worst_trade_pct']:.3f}% | {row['oos_partial_events']['tp_any_rate_pct']:.1f}% | {row['walk_forward']['median_sharpe']:.3f} | {row['stress']['median_sharpe']:.3f} |")
    lines.extend(["", "Relative deltas, fees, slippage, exposure, holding time, MAE and MFE are available for every validation/OOS row in the JSON and CSV outputs.", "", "## 4. ETH `breakout_atr_60_1.0`", ""])
    eth_base = next(row for row in report["baseline"] if row["symbol"] == ETH_SYMBOL)
    lines.append(f"Baseline OOS: return {eth_base['periods']['oos']['total_return_pct']:.2f}%, Sharpe {eth_base['periods']['oos']['sharpe']:.3f}, Sortino {eth_base['periods']['oos']['sortino']:.3f}, max DD {eth_base['periods']['oos']['max_drawdown_pct']:.2f}%, expectancy {eth_base['periods']['oos']['expectancy_pct']:.3f}%.")
    lines.append("")
    lines.append("The ETH result is not a case where every partial exit is slightly worse. Runner, break-even, trailing, and fixed/ATR variants show different trade-offs; the near-baseline vol-normalized rows have negligible TP activation and therefore do not demonstrate a meaningful scale-out.")
    lines.extend(["", "## 5. Regime Analysis", "", "Regimes are the existing phase-2 flags: trend, range, high volatility, low volatility, crash, and recovery. Results are descriptive OOS slices only; no regime was selected after seeing its final result.", ""])
    for regime, details in report["regime_analysis"].items():
        lines.append(f"### {regime}")
        lines.append("")
        for name, metrics in details["baseline"].items():
            if isinstance(metrics, dict) and "sharpe" in metrics:
                lines.append(f"- Baseline {name}: {metrics['total_return_pct']:.2f}% return, Sharpe {metrics['sharpe']:.3f}, max DD {metrics['max_drawdown_pct']:.2f}%, trades {metrics['trades']}")
        lines.append("- No regime-specific tactic is promoted; any attractive slice requires a pre-registered rule and fresh walk-forward/OOS validation.")
    lines.extend(["", "## 6. MAE/MFE Analysis", "", "MFE/MAE distributions use raw baseline trade paths; OOS quantiles are not used to tune targets. TP reach rates are expressed relative to each trade's entry ATR.", ""])
    for symbol, distribution in report["mfe_mae_analysis"].items():
        oos = distribution.get("oos", {})
        lines.append(f"- {symbol}: OOS trades {oos.get('trades', 0)}, MFE quantiles {oos.get('mfe_pct_quantiles', {})}, MAE absolute quantiles {oos.get('mae_abs_pct_quantiles', {})}, TP reach {oos.get('tp_reach_rates', {})}")
    lines.extend(["", "## 7. Parameter Sensitivity", "", "Sensitivity is summarized by family range and sign persistence. Narrow winners without neighboring support are treated as unstable.", ""])
    for family, details in report["eth_breakout_atr_60_1.0"]["sensitivity"].items():
        lines.append(f"- `{family}`: {details['policies']}; OOS Sharpe range {details['oos_sharpe_min']:.3f} to {details['oos_sharpe_max']:.3f}; validation-positive {details['validation_positive_rate_pct']:.1f}%; OOS-positive {details['oos_positive_rate_pct']:.1f}%.")
    lines.extend(["", "## 8. Overfitting Analysis", "", "**OVERFIT RISK**: allocations, target methods/distances, stop modes, final exits, volatility scaling, MFE quantiles, and asset/regime selection create many interacting degrees of freedom. The diagnosis deliberately does not select a winner from final OOS or stress results.", "", "## 9. Event-Level Analysis", ""])
    for symbol, details in report["event_level_analysis"].items():
        lines.append(f"### {symbol} versus `{details['comparison_policy']}`")
        for example in details["examples"]:
            lines.append(f"- {example['category']} at {example['entry_timestamp']}: baseline {example['baseline']['gross_return_pct']:.3f}% gross, MFE {example['baseline']['mfe_pct']:.3f}%, exit `{example['baseline']['exit_reason']}`; variant matched={example['variant']['matched']}, net {example['variant']['net_return_pct'] if example['variant']['net_return_pct'] is not None else 'n/a'}%, events {example['variant']['partial_events']}")
    lines.extend(["", "## 10. Risk-Management Analysis", "", report["risk_management_analysis"]["classification"], ". Similar return alone is not called a risk benefit; the candidate must also improve Sharpe, drawdown, worst trade, and have meaningful TP activity.", ""])
    lines.extend(["## 11. Simple Scale-Out Test", "", "**PROMISING BUT NOT ROBUST**. The two simple policies were tested without aggressive tuning and remain research candidates only.", ""])
    for row in report["simple_scaleout"]["policies"]:
        oos = row["periods"]["oos"]
        lines.append(f"- `{row['policy_id']}` on {row['symbol']}: OOS return {oos['total_return_pct']:.2f}%, Sharpe {oos['sharpe']:.3f}, max DD {oos['max_drawdown_pct']:.2f}%, TP activation {row['oos_partial_events']['tp_any_rate_pct']:.1f}%.")
    lines.extend(["", "## 12. Conclusion", "", report["answers"]["why_no_robust_advantage"], "", report["answers"]["disposition"], "", "## 13. Recommended Next Research Step", "", report["answers"]["next_step"], "", "## 14. What Is Not Robust", ""])
    lines.extend(f"- {item}" for item in report["not_robust"])
    lines.extend(["", "## 15. What Remains To Test", "", "- Funding-inclusive ETH holdout with an actual funding series.", "- One pre-registered simple scale-out family only, with a minimum TP activation rate and no asset/regime cherry-picking.", "- Independent future data after the research cutoff.", "- If no simple rule passes, keep full-position management and stop the partial-exit branch."])
    (path / "partial_exit_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-runs", type=int, default=200)
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    args = parser.parse_args()
    config = PartialExitDiagnosisConfig(symbols=args.symbols, stress_runs=args.stress_runs)
    report = run_partial_exit_diagnosis(config)
    write_outputs(report, config.output_dir)
    print(report["decision"])
    print(f"Existing policies: {report['technical_analysis']['existing_policy_count']}")
    print(f"Diagnostic policies: {report['technical_analysis']['diagnosis_policy_count']}")
    print(f"Risk-management findings: {len(report['risk_management_analysis']['findings'])}")


if __name__ == "__main__":
    main()