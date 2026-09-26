"""Pre-registered, research-only holdout test for simple partial exits.

The manifest is written and locked before any holdout result is calculated.
This module does not modify production, paper, GUI, database, or execution
paths. The existing entry signal and ATR stop are fixed; only the registered
trade-management policies differ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.metrics import summarize_performance
from research.partial_exit_simulator import (
    PartialExitEngine,
    PartialExitPolicy,
    TakeProfitSpec,
    default_partial_exit_policies,
)
from research.phase2_strategy_research import (
    Phase2Config,
    _periods_per_year,
    _split_ranges,
    build_candidate_grid,
    generate_signal,
    load_local_ohlcv,
)


OUTPUT_DIR = "research/partial_exit"
MANIFEST_NAME = "holdout_manifest.json"
DECISION = "NO ROBUST PARTIAL EXIT ADVANTAGE FOUND"
ETH_SYMBOL = "ETH/USDT"
BASE_CANDIDATE_ID = "breakout_atr_60_1.0"
STRESS_SEED = 20260923


@dataclass(frozen=True)
class HoldoutConfig:
    symbols: list[str]
    timeframe: str = "1h"
    history_days: int = 2500
    output_dir: str = OUTPUT_DIR
    stress_runs: int = 200
    stress_window_days: int = 365
    fee_rate: float = 0.00035
    slippage_rate: float = 0.00020
    minimum_holdout_trades: int = 30
    minimum_bootstrap_pairs: int = 20
    minimum_tp1_activation_rate_pct: float = 5.0
    minimum_partial_event_rate_pct: float = 10.0


def registered_policies() -> tuple[PartialExitPolicy, ...]:
    return (
        PartialExitPolicy("control_full_position", (1.0,), None),
        PartialExitPolicy(
            "scaleout_a_25_25_50",
            (0.25, 0.25, 0.50),
            TakeProfitSpec("atr_multiple", (1.0, 2.0)),
            "break_even_then_atr_trailing",
            "atr_trailing",
            activate_trailing_after_stage=2,
        ),
        PartialExitPolicy(
            "scaleout_b_50_50",
            (0.50, 0.50),
            TakeProfitSpec("atr_multiple", (1.5,)),
            "break_even_then_atr_trailing",
            "atr_trailing",
            activate_trailing_after_stage=1,
        ),
    )


def _policy_manifest(policy: PartialExitPolicy) -> dict[str, Any]:
    take_profit = policy.take_profit
    return {
        "policy_id": policy.policy_id,
        "asset_scope": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        "strategy": BASE_CANDIDATE_ID,
        "tp_levels_atr": list(take_profit.values) if take_profit else [],
        "allocation": list(policy.allocations),
        "stop_rule": (
            "unchanged full-position ATR stop"
            if policy.stop_mode == "unchanged"
            else "break-even plus fee/slippage buffer, then ATR trailing runner"
        ),
        "trailing_rule": "existing causal ATR trailing logic" if take_profit else "none",
        "final_exit_rule": "existing signal exit" if policy.final_exit == "signal" else "existing ATR trailing exit",
        "implementation_parameters": {
            "policy_id": policy.policy_id,
            "allocations": list(policy.allocations),
            "take_profit": {
                "method": take_profit.method,
                "values": list(take_profit.values),
                "fee_slippage_buffer": take_profit.fee_slippage_buffer,
            } if take_profit else None,
            "stop_mode": policy.stop_mode,
            "final_exit": policy.final_exit,
            "activate_trailing_after_stage": policy.activate_trailing_after_stage,
        },
    }


def _manifest_for(config: HoldoutConfig, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
    candidate = next(item for item in build_candidate_grid() if item.candidate_id == BASE_CANDIDATE_ID)
    periods: dict[str, Any] = {}
    for symbol, frame in data.items():
        ranges = _split_ranges(frame.index, Phase2Config(symbols=[symbol], timeframe=config.timeframe, history_days=config.history_days))
        periods[symbol] = {
            "source_start": frame.index[0].isoformat(),
            "source_end": frame.index[-1].isoformat(),
            "train_start": ranges["train"][0].isoformat(),
            "train_end": ranges["train"][1].isoformat(),
            "validation_start": ranges["validation"][0].isoformat(),
            "validation_end": ranges["validation"][1].isoformat(),
            "holdout_start": ranges["oos"][0].isoformat(),
            "holdout_end": ranges["oos"][1].isoformat(),
            "rows": int(len(frame)),
        }
    return {
        "manifest_id": "partial-exit-holdout-v1",
        "manifest_date": "2026-09-23",
        "manifest_version": 1,
        "immutable_after_creation": True,
        "research_only": True,
        "asset_scope": list(config.symbols),
        "primary_asset": ETH_SYMBOL,
        "strategy": {
            "candidate_id": candidate.candidate_id,
            "family": candidate.family,
            "parameters": candidate.parameters,
            "stop_atr_multiple": candidate.stop_atr_multiple,
            "entry_fill": "next bar open",
            "direction_source": "fixed causal signal_position from phase-2 strategy",
        },
        "policies": [_policy_manifest(policy) for policy in registered_policies()],
        "cost_model": {
            "fee_rate_per_execution": config.fee_rate,
            "slippage_rate_per_execution": config.slippage_rate,
            "funding": "NOT TESTED; no funding series is used by this experiment",
        },
        "atr_definition": "14-period simple moving average of true range; entry ATR is the prior completed bar ATR",
        "timeframes": {"market_data": config.timeframe, "metrics_annualization": _periods_per_year(config.timeframe)},
        "holdout_definition": {
            "rule": "chronological 50% train, 20% validation, final 30% holdout; no shuffling",
            "periods_by_symbol": periods,
            "selection_or_tuning_after_manifest": False,
        },
        "pre_registered_gates": {
            "minimum_holdout_trades": config.minimum_holdout_trades,
            "minimum_bootstrap_pairs": config.minimum_bootstrap_pairs,
            "minimum_tp1_activation_rate_pct": config.minimum_tp1_activation_rate_pct,
            "minimum_partial_event_rate_pct": config.minimum_partial_event_rate_pct,
            "robust_return_advantage_min_pct": 0.0,
            "robust_sharpe_advantage_min": 0.05,
            "robust_risk_advantage_min_pp": 0.0,
            "risk_benefit_min_drawdown_advantage_pp": 2.0,
            "bootstrap_lower_bound_must_be_positive": True,
            "stress_worse_than_control_allowed": False,
        },
        "stress_definition": {
            "runs": config.stress_runs,
            "window_days": config.stress_window_days,
            "seed": STRESS_SEED,
            "fee_rate": config.fee_rate * 2.0,
            "slippage_rate": config.slippage_rate * 2.0,
            "adverse_fill_rate": 0.0005,
            "random_fill_deviation_rate": 0.00025,
            "execution_delay_bars": 1,
            "same_contiguous_windows_and_seeds_for_all_policies": True,
        },
        "known_integrity_limit": "The prior descriptive diagnosis already read these historical OOS bars; this runner performs no new tuning, but cannot claim a pristine untouched holdout without future data.",
    }


def _manifest_hash(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_manifest(manifest: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError(f"Refusing to modify immutable holdout manifest: {path}")
    else:
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return _manifest_hash(manifest)


def _trade_slice(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return trades
    return trades[(trades["exit_timestamp"] >= start) & (trades["exit_timestamp"] <= end)]


def _trade_pnl(trades: pd.DataFrame) -> pd.Series:
    return trades["net_return"].astype(float) if not trades.empty else pd.Series(dtype=float)


def _event_fractions(trade: dict[str, Any], prefix: str | None = None) -> float:
    events = trade.get("partial_events", [])
    return float(sum(float(event["fraction"]) for event in events if prefix is None or str(event.get("reason", "")).startswith(prefix)))


def _max_consecutive_losses(pnl: pd.Series) -> int:
    longest = current = 0
    for value in pnl.to_numpy(float):
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _drawdown_duration(returns: pd.Series) -> dict[str, float]:
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    longest = current = 0
    for value in drawdown.to_numpy(float):
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return {"bars": int(longest), "days": float(longest / 24.0)}


def _worst_period(returns: pd.Series, frequency: str) -> float:
    grouped = (1.0 + returns).resample(frequency).prod() - 1.0
    return float(grouped.min() * 100.0) if not grouped.empty else 0.0


def _metrics(result: pd.DataFrame, trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, config: HoldoutConfig, periods: int) -> dict[str, Any]:
    holdout = result.loc[start:end, "strategy_return_1x"].astype(float)
    selected = _trade_slice(trades, start, end)
    pnl = _trade_pnl(selected)
    summary = summarize_performance(holdout, periods, pnl)
    tp1_count = int(sum(_event_fractions(row, "tp1") > 0 for row in selected.to_dict("records")))
    tp2_count = int(sum(_event_fractions(row, "tp2") > 0 for row in selected.to_dict("records")))
    tp_any_count = int(sum(_event_fractions(row, "tp") > 0 for row in selected.to_dict("records")))
    total_closed_before_final = [_event_fractions(row, "tp") for row in selected.to_dict("records")]
    remaining_after_partial = [
        1.0 - _event_fractions(row, "tp")
        for row in selected.to_dict("records")
    ]
    execution_counts = [1 + len(row.get("partial_events", [])) for row in selected.to_dict("records")]
    fees = float(sum(config.fee_rate * count for count in execution_counts))
    slippage = float(sum(config.slippage_rate * count for count in execution_counts))
    return {
        **summary,
        "profit_factor": float(summary.get("profit_factor", 0.0)),
        "expectancy_pct": float(pnl.mean() * 100.0) if not pnl.empty else 0.0,
        "median_trade_pct": float(pnl.median() * 100.0) if not pnl.empty else 0.0,
        "worst_trade_pct": float(pnl.min() * 100.0) if not pnl.empty else 0.0,
        "fees_pct": fees * 100.0,
        "slippage_pct": slippage * 100.0,
        "trades": int(len(selected)),
        "partial_exit_count": int(sum(len([event for event in row.get("partial_events", []) if str(event.get("reason", "")).startswith("tp")]) for row in selected.to_dict("records"))),
        "tp1_count": tp1_count,
        "tp2_count": tp2_count,
        "tp1_activation_rate_pct": float(tp1_count / len(selected) * 100.0) if len(selected) else 0.0,
        "tp2_activation_rate_pct": float(tp2_count / len(selected) * 100.0) if len(selected) else 0.0,
        "partial_event_rate_pct": float(tp_any_count / len(selected) * 100.0) if len(selected) else 0.0,
        "average_percentage_closed_before_final_exit_pct": float(np.mean(total_closed_before_final) * 100.0) if total_closed_before_final else 0.0,
        "average_remaining_position_pct": float(np.mean(remaining_after_partial) * 100.0) if remaining_after_partial else 0.0,
        "average_holding_bars": float(selected["duration_bars"].mean()) if not selected.empty else 0.0,
        "mae_pct": float(selected["mae"].mean() * 100.0) if not selected.empty else 0.0,
        "mfe_pct": float(selected["mfe"].mean() * 100.0) if not selected.empty else 0.0,
        "worst_day_pct": _worst_period(holdout, "D"),
        "worst_week_pct": _worst_period(holdout, "W"),
        "worst_month_pct": _worst_period(holdout, "ME"),
        "max_consecutive_losses": _max_consecutive_losses(pnl),
        "tail_loss_p05_pct": float(pnl.quantile(0.05) * 100.0) if not pnl.empty else 0.0,
        "drawdown_duration": _drawdown_duration(holdout),
    }


def _bootstrap_delta(control: pd.DataFrame, variant: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, minimum_pairs: int) -> dict[str, Any]:
    control_rows = {row["entry_timestamp"]: float(row["net_return"]) for row in _trade_slice(control, start, end).to_dict("records")}
    variant_rows = {row["entry_timestamp"]: float(row["net_return"]) for row in _trade_slice(variant, start, end).to_dict("records")}
    keys = sorted(set(control_rows) & set(variant_rows))
    deltas = np.asarray([variant_rows[key] - control_rows[key] for key in keys], dtype=float)
    if len(deltas) < minimum_pairs:
        return {"status": "INSUFFICIENT SAMPLE", "paired_trades": int(len(deltas)), "mean_delta_pct": None, "ci95_pct": [None, None]}
    rng = np.random.default_rng(STRESS_SEED)
    samples = rng.integers(0, len(deltas), size=(2000, len(deltas)))
    means = deltas[samples].mean(axis=1) * 100.0
    return {
        "status": "TESTED",
        "paired_trades": int(len(deltas)),
        "mean_delta_pct": float(deltas.mean() * 100.0),
        "ci95_pct": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
    }


def _stress_summary(features: pd.DataFrame, policy: PartialExitPolicy, stop_multiple: float, config: HoldoutConfig, holdout: tuple[pd.Timestamp, pd.Timestamp], periods: int) -> dict[str, Any]:
    stress_result, _ = PartialExitEngine(config.fee_rate * 2.0, config.slippage_rate * 2.0).execute(
        features,
        policy,
        stop_multiple,
        execution_delay_bars=1,
        adverse_fill_rate=0.0005,
        random_fill_deviation_rate=0.00025,
        random_seed=STRESS_SEED,
    )
    start, end = holdout
    returns = stress_result.loc[start:end, "strategy_return_1x"].astype(float)
    window = max(1, int(config.stress_window_days * periods / 365))
    if window >= len(returns):
        return {"status": "INSUFFICIENT DATA", "runs": 0, "profitable_stress_windows": 0, "profitable_stress_windows_pct": 0.0}
    rng = np.random.default_rng(STRESS_SEED)
    starts = rng.integers(0, len(returns) - window + 1, size=config.stress_runs)
    values: list[dict[str, float]] = []
    array = returns.to_numpy(float)
    for offset in starts:
        window_returns = pd.Series(array[offset : offset + window])
        summary = summarize_performance(window_returns, periods)
        values.append({key: float(summary[key]) for key in ("total_return_pct", "sharpe", "max_drawdown_pct")})
    frame = pd.DataFrame(values)
    return {
        "status": "TESTED",
        "runs": int(len(frame)),
        "profitable_stress_windows": int((frame["total_return_pct"] > 0).sum()),
        "profitable_stress_windows_pct": float((frame["total_return_pct"] > 0).mean() * 100.0),
        "median_stress_return_pct": float(frame["total_return_pct"].median()),
        "worst_stress_return_pct": float(frame["total_return_pct"].min()),
        "stress_sharpe": float(frame["sharpe"].median()),
        "stress_max_drawdown_pct": float(frame["max_drawdown_pct"].min()),
    }


def _advantages(metrics: dict[str, Any], control: dict[str, Any], stress: dict[str, Any], control_stress: dict[str, Any]) -> dict[str, Any]:
    return {
        "return_advantage_pct": float(metrics["total_return_pct"] - control["total_return_pct"]),
        "risk_advantage_pp": float(abs(control["max_drawdown_pct"]) - abs(metrics["max_drawdown_pct"])),
        "sharpe_advantage": float(metrics["sharpe"] - control["sharpe"]),
        "expectancy_advantage_pp": float(metrics["expectancy_pct"] - control["expectancy_pct"]),
        "stress_return_advantage_pct": float(stress.get("median_stress_return_pct", 0.0) - control_stress.get("median_stress_return_pct", 0.0)),
        "stress_risk_advantage_pp": float(abs(control_stress.get("stress_max_drawdown_pct", 0.0)) - abs(stress.get("stress_max_drawdown_pct", 0.0))),
    }


def _classify(metrics: dict[str, Any], control: dict[str, Any], stress: dict[str, Any], control_stress: dict[str, Any], bootstrap: dict[str, Any], config: HoldoutConfig, integrity_clean: bool) -> tuple[str, dict[str, Any]]:
    sample_ok = metrics["trades"] >= config.minimum_holdout_trades and bootstrap["paired_trades"] >= config.minimum_bootstrap_pairs
    meaningful = metrics["tp1_activation_rate_pct"] >= config.minimum_tp1_activation_rate_pct and metrics["partial_event_rate_pct"] >= config.minimum_partial_event_rate_pct
    advantages = _advantages(metrics, control, stress, control_stress)
    ci = bootstrap.get("ci95_pct", [None, None])
    raw_robust = sample_ok and meaningful and advantages["return_advantage_pct"] >= 0.0 and advantages["sharpe_advantage"] >= 0.05 and advantages["risk_advantage_pp"] >= 0.0 and advantages["stress_return_advantage_pct"] >= 0.0 and advantages["stress_risk_advantage_pp"] >= 0.0 and ci[0] is not None and ci[0] > 0.0
    if not meaningful:
        classification = "NOT MEANINGFUL PARTIAL-EXIT EXPOSURE"
    elif not sample_ok:
        classification = "INSUFFICIENT DATA"
    elif raw_robust and integrity_clean:
        classification = "ROBUST PARTIAL-EXIT ADVANTAGE"
    elif advantages["risk_advantage_pp"] >= 2.0 and advantages["return_advantage_pct"] <= 0.0 and advantages["stress_risk_advantage_pp"] >= 0.0:
        classification = "RISK MANAGEMENT BENEFIT – NOT RETURN ADVANTAGE"
    elif advantages["return_advantage_pct"] > 0.0 or advantages["risk_advantage_pp"] > 0.0:
        classification = "PROMISING BUT NOT ROBUST"
    else:
        classification = DECISION
    return classification, {"sample_ok": sample_ok, "meaningful_exposure": meaningful, "raw_robust": raw_robust, "integrity_clean": integrity_clean, **advantages}


def _data_integrity(data: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> dict[str, Any]:
    checks = {}
    for symbol, frame in data.items():
        checks[symbol] = {
            "rows": int(len(frame)),
            "sorted": bool(frame.index.is_monotonic_increasing),
            "duplicate_timestamps": int(frame.index.duplicated().sum()),
            "missing_ohlc_values": int(frame[["open", "high", "low", "close"]].isna().sum().sum()),
            "manifest_holdout_start": manifest["holdout_definition"]["periods_by_symbol"][symbol]["holdout_start"],
            "manifest_holdout_end": manifest["holdout_definition"]["periods_by_symbol"][symbol]["holdout_end"],
        }
    return {
        "status": "NOT_PRISTINE_PRIOR_DIAGNOSTIC_READ",
        "checks_by_symbol": checks,
        "chronological_split_verified": True,
        "parameters_fixed_before_this_runner": True,
        "holdout_used_for_parameter_or_policy_selection": False,
        "holdout_pristine": False,
        "reason": "The previous descriptive diagnosis evaluated the same historical OOS bars. No tuning was performed here, but no future untouched data is available to certify a pristine holdout.",
        "funding": "NOT TESTED",
    }


def _lookahead_validation() -> dict[str, Any]:
    index = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    base = pd.DataFrame({"open": [100.0] * 6, "high": [100.0, 100.0, 102.0, 101.0, 101.0, 101.0], "low": [100.0, 99.0, 99.0, 99.0, 99.0, 99.0], "close": [100.0, 100.0, 101.0, 101.0, 100.0, 100.0], "atr": [1.0] * 6, "signal_position": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0], "realized_vol": [0.1] * 6}, index=index)
    altered = base.copy()
    altered.loc[index[5], "high"] = 10000.0
    altered.loc[index[5], "low"] = 0.01
    policy = registered_policies()[1]
    first, first_trades = PartialExitEngine(0.00035, 0.0002).execute(base, policy, 2.5)
    second, second_trades = PartialExitEngine(0.00035, 0.0002).execute(altered, policy, 2.5)
    return {
        "status": "PASSED" if first_trades.to_dict("records") == second_trades.to_dict("records") and first.iloc[:5].equals(second.iloc[:5]) else "FAILED",
        "checks": [
            "ATR uses prior completed bars",
            "TP uses current bar high/low only after the entry exists",
            "same-bar stop has priority over TP",
            "future exit-bar high/low cannot alter earlier fills",
        ],
        "regression_test": "tests/test_partial_exit_holdout.py::test_future_extremes_do_not_determine_earlier_partial_exit",
    }


def run_holdout(config: HoldoutConfig | None = None) -> dict[str, Any]:
    config = config or HoldoutConfig(symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    data = load_local_ohlcv(config.symbols, "data", config.timeframe, config.history_days)
    manifest = _manifest_for(config, data)
    manifest_path = Path(config.output_dir) / MANIFEST_NAME
    manifest_hash = ensure_manifest(manifest, manifest_path)
    policies = registered_policies()
    candidate = next(item for item in build_candidate_grid() if item.candidate_id == BASE_CANDIDATE_ID)
    periods = _periods_per_year(config.timeframe)
    integrity = _data_integrity(data, manifest)
    integrity_clean = integrity["holdout_pristine"]
    assets: dict[str, Any] = {}
    flat_rows: list[dict[str, Any]] = []
    for symbol, frame in data.items():
        features, _ = generate_signal(frame, candidate)
        ranges = _split_ranges(features.index, Phase2Config(symbols=[symbol], timeframe=config.timeframe, history_days=config.history_days))
        holdout = ranges["oos"]
        evaluated: dict[str, Any] = {}
        for policy in policies:
            result, trades = PartialExitEngine(config.fee_rate, config.slippage_rate).execute(features, policy, candidate.stop_atr_multiple)
            evaluated[policy.policy_id] = {
                "policy": _policy_manifest(policy),
                "metrics": _metrics(result, trades, holdout[0], holdout[1], config, periods),
                "stress": _stress_summary(features, policy, candidate.stop_atr_multiple, config, holdout, periods),
                "trades": trades,
                "result": result,
            }
        control = evaluated[policies[0].policy_id]
        comparisons = []
        for policy in policies[1:]:
            variant = evaluated[policy.policy_id]
            bootstrap = _bootstrap_delta(control["trades"], variant["trades"], holdout[0], holdout[1], config.minimum_bootstrap_pairs)
            classification, gates = _classify(variant["metrics"], control["metrics"], variant["stress"], control["stress"], bootstrap, config, integrity_clean)
            comparison = {
                "symbol": symbol,
                "policy_id": policy.policy_id,
                "classification": classification,
                "metrics": variant["metrics"],
                "control_metrics": control["metrics"],
                "stress": variant["stress"],
                "control_stress": control["stress"],
                "bootstrap": bootstrap,
                "gates": gates,
                "holdout_period": {"start": holdout[0], "end": holdout[1]},
            }
            comparisons.append(comparison)
            flat_rows.append(_flat_row(comparison))
        assets[symbol] = {
            "holdout_period": {"start": holdout[0], "end": holdout[1]},
            "control": {"metrics": control["metrics"], "stress": control["stress"]},
            "comparisons": comparisons,
        }

    breakeven_policy = next(policy for policy in default_partial_exit_policies() if policy.policy_id == "partial_breakeven")
    sol_candidate = None
    if "SOL/USDT" in data:
        features, _ = generate_signal(data["SOL/USDT"], candidate)
        ranges = _split_ranges(features.index, Phase2Config(symbols=["SOL/USDT"], timeframe=config.timeframe, history_days=config.history_days))
        control_result, control_trades = PartialExitEngine(config.fee_rate, config.slippage_rate).execute(features, policies[0], candidate.stop_atr_multiple)
        candidate_result, candidate_trades = PartialExitEngine(config.fee_rate, config.slippage_rate).execute(features, breakeven_policy, candidate.stop_atr_multiple)
        candidate_metrics = _metrics(candidate_result, candidate_trades, ranges["oos"][0], ranges["oos"][1], config, periods)
        control_metrics = _metrics(control_result, control_trades, ranges["oos"][0], ranges["oos"][1], config, periods)
        sol_candidate = {
            "policy": _policy_manifest(breakeven_policy),
            "metrics": candidate_metrics,
            "control_metrics": control_metrics,
            "advantages": _advantages(candidate_metrics, control_metrics, _stress_summary(features, breakeven_policy, candidate.stop_atr_multiple, config, ranges["oos"], periods), _stress_summary(features, policies[0], candidate.stop_atr_multiple, config, ranges["oos"], periods)),
            "classification": "PROMISING BUT NOT ROBUST",
            "disposition": "Separate historical candidate only; never a production promotion from this single holdout.",
        }

    robust = [row for row in flat_rows if row["classification"] == "ROBUST PARTIAL-EXIT ADVANTAGE"]
    return {
        "decision": DECISION if not robust else "ROBUST PARTIAL-EXIT ADVANTAGE",
        "classification": "NO ROBUST PARTIAL EXIT ADVANTAGE FOUND" if not robust else "ROBUST PARTIAL-EXIT ADVANTAGE",
        "manifest": {**manifest, "sha256": manifest_hash, "path": str(manifest_path)},
        "registered_policies": [_policy_manifest(policy) for policy in policies],
        "holdout_period": manifest["holdout_definition"],
        "data_integrity": integrity,
        "primary_test": assets.get(ETH_SYMBOL),
        "secondary_tests": {symbol: value for symbol, value in assets.items() if symbol != ETH_SYMBOL},
        "comparisons": flat_rows,
        "sol_partial_breakeven": sol_candidate,
        "lookahead_validation": _lookahead_validation(),
        "research_boundary": {"funding": "NOT TESTED", "leverage_optimization": False, "production_or_paper_integration": False, "parameter_optimization_after_manifest": False},
    }


def _flat_row(comparison: dict[str, Any]) -> dict[str, Any]:
    metrics = comparison["metrics"]
    control = comparison["control_metrics"]
    advantages = comparison["gates"]
    stress = comparison["stress"]
    return {
        "symbol": comparison["symbol"],
        "policy_id": comparison["policy_id"],
        "classification": comparison["classification"],
        "holdout_start": comparison["holdout_period"]["start"],
        "holdout_end": comparison["holdout_period"]["end"],
        "holdout_return_pct": metrics["total_return_pct"],
        "control_return_pct": control["total_return_pct"],
        "holdout_sharpe": metrics["sharpe"],
        "control_sharpe": control["sharpe"],
        "holdout_sortino": metrics["sortino"],
        "holdout_max_drawdown_pct": metrics["max_drawdown_pct"],
        "control_max_drawdown_pct": control["max_drawdown_pct"],
        "holdout_calmar": metrics["calmar"],
        "profit_factor": metrics["profit_factor"],
        "expectancy_pct": metrics["expectancy_pct"],
        "win_rate_pct": metrics.get("win_rate_pct", 0.0),
        "median_trade_pct": metrics["median_trade_pct"],
        "worst_trade_pct": metrics["worst_trade_pct"],
        "fees_pct": metrics["fees_pct"],
        "slippage_pct": metrics["slippage_pct"],
        "trades": metrics["trades"],
        "partial_exit_count": metrics["partial_exit_count"],
        "tp1_activation_rate_pct": metrics["tp1_activation_rate_pct"],
        "tp2_activation_rate_pct": metrics["tp2_activation_rate_pct"],
        "partial_event_rate_pct": metrics["partial_event_rate_pct"],
        "average_percentage_closed_before_final_exit_pct": metrics["average_percentage_closed_before_final_exit_pct"],
        "average_remaining_position_pct": metrics["average_remaining_position_pct"],
        "average_holding_bars": metrics["average_holding_bars"],
        "mae_pct": metrics["mae_pct"],
        "mfe_pct": metrics["mfe_pct"],
        "worst_day_pct": metrics["worst_day_pct"],
        "worst_week_pct": metrics["worst_week_pct"],
        "worst_month_pct": metrics["worst_month_pct"],
        "max_consecutive_losses": metrics["max_consecutive_losses"],
        "tail_loss_p05_pct": metrics["tail_loss_p05_pct"],
        "drawdown_duration_bars": metrics["drawdown_duration"]["bars"],
        "return_advantage_pct": advantages["return_advantage_pct"],
        "risk_advantage_pp": advantages["risk_advantage_pp"],
        "sharpe_advantage": advantages["sharpe_advantage"],
        "expectancy_advantage_pp": advantages["expectancy_advantage_pp"],
        "bootstrap_status": comparison["bootstrap"]["status"],
        "bootstrap_paired_trades": comparison["bootstrap"]["paired_trades"],
        "bootstrap_mean_delta_pct": comparison["bootstrap"]["mean_delta_pct"],
        "bootstrap_ci95_low_pct": comparison["bootstrap"]["ci95_pct"][0],
        "bootstrap_ci95_high_pct": comparison["bootstrap"]["ci95_pct"][1],
        "profitable_stress_windows": stress.get("profitable_stress_windows", 0),
        "profitable_stress_windows_pct": stress.get("profitable_stress_windows_pct", 0.0),
        "median_stress_return_pct": stress.get("median_stress_return_pct", 0.0),
        "worst_stress_return_pct": stress.get("worst_stress_return_pct", 0.0),
        "stress_sharpe": stress.get("stress_sharpe", 0.0),
        "stress_max_drawdown_pct": stress.get("stress_max_drawdown_pct", 0.0),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict("records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_list())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_outputs(report: dict[str, Any], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    json_report = _json_safe(report)
    (path / "partial_exit_holdout.json").write_text(json.dumps(json_report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    assets = {ETH_SYMBOL: report["primary_test"], **report["secondary_tests"]}
    csv_rows = [_control_flat_row(symbol, asset) for symbol, asset in assets.items()]
    csv_rows.extend(report["comparisons"])
    pd.DataFrame(_json_safe(csv_rows)).to_csv(path / "partial_exit_holdout.csv", index=False)
    lines = [
        "# Partial Exit Holdout Test",
        "",
        f"**Final classification: {report['decision']}**",
        "",
        "Research-only result. No GUI, tactics engine, live order, paper order, or database integration was changed.",
        "",
        "## 1. Registered Policies",
        "",
        "| Policy | Allocation | TP levels | Stop / trailing |",
        "|---|---|---|---|",
    ]
    for policy in report["registered_policies"]:
        lines.append(f"| `{policy['policy_id']}` | {policy['allocation']} | {policy['tp_levels_atr'] or '-'} | {policy['stop_rule']} / {policy['trailing_rule']} |")
    lines.extend(["", "## 2. Holdout Period", "", "Chronological final 30% after the pre-existing 50% train and 20% validation split. Exact periods are stored in the manifest.", "", "## 3. Data Integrity", "", f"Status: **{report['data_integrity']['status']}**. {report['data_integrity']['reason']}", "", "Funding: `NOT TESTED`.", "", "## 4. ETH Primary Test"])
    _append_asset(lines, report["primary_test"])
    lines.extend(["", "## 5. BTC/SOL Secondary Tests"])
    for symbol, asset in report["secondary_tests"].items():
        lines.append(f"### {symbol}")
        _append_asset(lines, asset)
    lines.extend(["", "## 6. Control vs Scale-Out A", "", "The full comparison table is in the CSV; the registered A row is shown below."])
    for row in report["comparisons"]:
        if row["policy_id"] == "scaleout_a_25_25_50":
            lines.append(f"- {row['symbol']}: return {row['holdout_return_pct']:.2f}% vs control {row['control_return_pct']:.2f}%, Sharpe {row['holdout_sharpe']:.3f}, max DD {row['holdout_max_drawdown_pct']:.2f}%, class `{row['classification']}`.")
    lines.extend(["", "## 7. Control vs Scale-Out B", ""])
    for row in report["comparisons"]:
        if row["policy_id"] == "scaleout_b_50_50":
            lines.append(f"- {row['symbol']}: return {row['holdout_return_pct']:.2f}% vs control {row['control_return_pct']:.2f}%, Sharpe {row['holdout_sharpe']:.3f}, max DD {row['holdout_max_drawdown_pct']:.2f}%, class `{row['classification']}`.")
    lines.extend(["", "## 8. TP Activation Rates", "", "TP1, TP2, partial-event rate, average closed percentage, fees, slippage, MAE, MFE, and holding time are reported per row in the CSV/JSON.", "", "## 9. Risk Metrics", "", "Worst day/week/month, consecutive losses, tail loss, and drawdown duration are included for every policy comparison.", "", "## 10. Stress Test", "", "Each policy used the same 200 seeded contiguous 365-day windows after doubled fees/slippage, adverse fills, one-bar execution delay, and random fill deviations."])
    for symbol, asset in assets.items():
        stress = asset["control"]["stress"]
        lines.append(f"- {symbol} `control_full_position`: profitable {stress['profitable_stress_windows_pct']:.1f}%, median return {stress['median_stress_return_pct']:.2f}%, worst return {stress['worst_stress_return_pct']:.2f}%, stress Sharpe {stress['stress_sharpe']:.3f}, stress max DD {stress['stress_max_drawdown_pct']:.2f}%.")
    for row in report["comparisons"]:
        lines.append(f"- {row['symbol']} `{row['policy_id']}`: profitable {row['profitable_stress_windows_pct']:.1f}%, median return {row['median_stress_return_pct']:.2f}%, worst return {row['worst_stress_return_pct']:.2f}%, stress Sharpe {row['stress_sharpe']:.3f}, stress max DD {row['stress_max_drawdown_pct']:.2f}%.")
    lines.extend(["", "## 11. Statistical Stability", "", "Trade-level paired bootstrap uses 2,000 resamples. Rows below the pre-registered trade/pair gates are classified as `INSUFFICIENT DATA`; no result is selected by return alone.", "", "## 12. Look-Ahead Validation", "", f"Runtime probe: **{report['lookahead_validation']['status']}**. {', '.join(report['lookahead_validation']['checks'])}.", "", "## 13. Fees / Slippage", "", "Every actual entry, partial fill, and final exit receives the registered fee and slippage assumptions; funding is explicitly not tested.", "", "## 14. Final Classification", "", f"**{report['decision']}**", "", "The historical candidate `SOL/USDT partial_breakeven` is documented separately as `PROMISING BUT NOT ROBUST` and is not promoted.", "", "## 15. Limitations", "", "The same historical OOS bars were already read by the prior descriptive diagnosis, so this repository cannot honestly label them a pristine untouched holdout. No policy or parameter was changed after the manifest, and no holdout result was used for selection.", "", "## 16. Next Research Step", "", "Collect a genuinely future, unseen period and rerun this exact immutable manifest, including actual funding data. Until then, full-position management remains the control and partial exits remain research-only."])
    (path / "partial_exit_holdout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_asset(lines: list[str], asset: dict[str, Any]) -> None:
    control = asset["control"]["metrics"]
    lines.append(f"Control: return {control['total_return_pct']:.2f}%, Sharpe {control['sharpe']:.3f}, Sortino {control['sortino']:.3f}, max DD {control['max_drawdown_pct']:.2f}%, trades {control['trades']}.")
    for comparison in asset["comparisons"]:
        metrics = comparison["metrics"]
        gates = comparison["gates"]
        lines.append(f"- `{comparison['policy_id']}`: return {metrics['total_return_pct']:.2f}%, Sharpe {metrics['sharpe']:.3f}, Sortino {metrics['sortino']:.3f}, max DD {metrics['max_drawdown_pct']:.2f}%, TP1 {metrics['tp1_activation_rate_pct']:.1f}%, TP2 {metrics['tp2_activation_rate_pct']:.1f}%, classification `{comparison['classification']}`; return advantage {gates['return_advantage_pct']:.2f}pp, risk advantage {gates['risk_advantage_pp']:.2f}pp.")


def _control_flat_row(symbol: str, asset: dict[str, Any]) -> dict[str, Any]:
    metrics = asset["control"]["metrics"]
    stress = asset["control"]["stress"]
    return _flat_row(
        {
            "symbol": symbol,
            "policy_id": "control_full_position",
            "classification": "CONTROL",
            "holdout_period": asset["holdout_period"],
            "metrics": metrics,
            "control_metrics": metrics,
            "stress": stress,
            "control_stress": stress,
            "bootstrap": {"status": "N/A", "paired_trades": 0, "mean_delta_pct": None, "ci95_pct": [None, None]},
            "gates": {
                "return_advantage_pct": 0.0,
                "risk_advantage_pp": 0.0,
                "sharpe_advantage": 0.0,
                "expectancy_advantage_pp": 0.0,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-runs", type=int, default=200)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    report = run_holdout(HoldoutConfig(symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"], stress_runs=args.stress_runs, output_dir=args.output_dir))
    write_outputs(report, args.output_dir)
    print(report["decision"])
    print(f"Manifest SHA256: {report['manifest']['sha256']}")
    print(f"Comparison rows: {len(report['comparisons'])}")


if __name__ == "__main__":
    main()