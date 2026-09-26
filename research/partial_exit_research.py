"""Research partial exits against the validated phase-2 strategy layer.

This runner treats entry and direction as fixed. Only stateful trade
management changes, and every policy is evaluated through the same causal
train/validation/walk-forward/OOS/stress workflow.
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
from research.phase2_strategy_research import (
    LEVERAGE_GRID,
    Phase2Candidate,
    Phase2Config,
    _periods_per_year,
    _split_ranges,
    _walk_forward_ranges,
    _execute_candidate,
    build_candidate_grid,
    generate_signal,
    load_local_ohlcv,
)
from research.partial_exit_simulator import (
    PartialExitEngine,
    PartialExitPolicy,
    TakeProfitSpec,
    default_partial_exit_policies,
)


BASE_CANDIDATE_ID = "breakout_atr_60_1.0"


@dataclass(frozen=True)
class PartialExitConfig:
    symbols: list[str]
    timeframe: str = "1h"
    history_days: int = 2500
    output_dir: str = "data/research/partial_exit"
    stress_runs: int = 200
    stress_window_days: int = 365
    train_fraction: float = 0.50
    validation_fraction: float = 0.20
    walk_forward_folds: int = 4
    fee_rate: float = 0.00035
    slippage_rate: float = 0.00020
    base_candidate_id: str = BASE_CANDIDATE_ID
    position_risk_fraction: float = 0.005
    max_crypto_leverage: float = 10.0


def _policy_grid() -> list[PartialExitPolicy]:
    policies = default_partial_exit_policies()[1:]
    allocations = (0.25, 0.25, 0.50)
    for values in ((0.5, 1.0), (1.0, 2.0), (1.5, 3.0), (2.0, 4.0)):
        policies.append(PartialExitPolicy(f"atr_grid_{values[0]:.1f}_{values[1]:.1f}", allocations, TakeProfitSpec("atr_multiple", values)))
    for values in ((0.005, 0.01), (0.01, 0.02)):
        policies.append(PartialExitPolicy(f"fixed_pct_{values[0]:.3f}_{values[1]:.3f}", allocations, TakeProfitSpec("fixed_pct", values)))
    for values in ((0.5, 0.75), (0.75, 1.0)):
        policies.append(PartialExitPolicy(f"vol_normalized_{values[0]:.2f}_{values[1]:.2f}", allocations, TakeProfitSpec("vol_normalized", values)))
    for values in ((0.5, 1.0), (0.75, 1.0)):
        policies.append(PartialExitPolicy(f"atr_dynamic_{values[0]:.2f}_{values[1]:.2f}", allocations, TakeProfitSpec("atr_dynamic", values)))
    return policies


def _trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
            "average_trade_pct": 0.0,
            "average_winner_pct": 0.0,
            "average_loser_pct": 0.0,
            "fees_pct": 0.0,
            "average_holding_bars": 0.0,
            "mae_pct": 0.0,
            "mfe_pct": 0.0,
            "worst_trade_pct": 0.0,
            "exit_type_counts": {},
        }
    pnl = trades["net_return"].astype(float) if "net_return" in trades else trades["gross_return"].astype(float)
    winners = pnl[pnl > 0]
    losers = pnl[pnl < 0]
    losses = -losers.sum()
    return {
        "trades": int(len(trades)),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "profit_factor": float(winners.sum() / losses) if losses > 0 else (float("inf") if winners.sum() > 0 else 0.0),
        "expectancy_pct": float(pnl.mean() * 100.0),
        "average_trade_pct": float(pnl.mean() * 100.0),
        "average_winner_pct": float(winners.mean() * 100.0) if not winners.empty else 0.0,
        "average_loser_pct": float(losers.mean() * 100.0) if not losers.empty else 0.0,
        "fees_pct": float(trades["cost_return"].sum() * 100.0) if "cost_return" in trades else 0.0,
        "average_holding_bars": float(trades["duration_bars"].mean()),
        "mae_pct": float(trades["mae"].mean() * 100.0),
        "mfe_pct": float(trades["mfe"].mean() * 100.0),
        "worst_trade_pct": float(pnl.min() * 100.0),
        "exit_type_counts": {str(key): int(value) for key, value in trades["exit_reason"].value_counts().items()},
    }


def _metrics(result: pd.DataFrame, trades: pd.DataFrame, periods: int) -> dict[str, Any]:
    trade_pnl = None
    if not trades.empty:
        trade_pnl = trades["net_return"] if "net_return" in trades else trades["gross_return"]
    summary = summarize_performance(result["strategy_return_1x"], periods, trade_pnl)
    years = len(result) / periods if periods else 0.0
    cagr = ((1.0 + summary["total_return_pct"] / 100.0) ** (1.0 / years) - 1.0) * 100.0 if years > 0 and summary["total_return_pct"] > -100.0 else -100.0
    rolling = result["strategy_return_1x"].rolling(max(1, int(periods * 30 / 365))).apply(lambda values: float(np.prod(1.0 + values) - 1.0), raw=True)
    return {
        **summary,
        "cagr_pct": float(cagr),
        "return_drawdown": float(summary["total_return_pct"] / abs(summary["max_drawdown_pct"])) if summary["max_drawdown_pct"] < 0 else 0.0,
        "exposure_pct": float(result["position"].abs().mean() * 100.0),
        "worst_period_pct": float(rolling.min() * 100.0) if not rolling.dropna().empty else 0.0,
        **_trade_metrics(trades),
    }


def _stress(result: pd.DataFrame, config: PartialExitConfig, periods: int) -> dict[str, Any]:
    returns = result["strategy_return_1x"].to_numpy(float)
    window = int(config.stress_window_days * periods / 365)
    if window >= len(returns) or window <= 0:
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
        "worst_case_max_drawdown_pct": float(np.min(values["max_drawdown_usdt"])) / 500.0 * 100.0,
    }


def _slice_trades(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return trades
    return trades[(trades["exit_timestamp"] >= start) & (trades["exit_timestamp"] <= end)]


def _evaluate_series(result: pd.DataFrame, trades: pd.DataFrame, ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]], periods: int) -> dict[str, Any]:
    return {
        name: _metrics(result.loc[start:end], _slice_trades(trades, start, end), periods)
        for name, (start, end) in ranges.items()
    }


def _walk_forward(result: pd.DataFrame, trades: pd.DataFrame, index: pd.Index, folds: int, periods: int) -> dict[str, Any]:
    fold_metrics = []
    for start, end in _walk_forward_ranges(index, folds):
        fold_metrics.append(_metrics(result.loc[start:end], _slice_trades(trades, start, end), periods))
    return {
        "folds": fold_metrics,
        "median_sharpe": float(np.median([row["sharpe"] for row in fold_metrics])) if fold_metrics else 0.0,
        "median_return_pct": float(np.median([row["total_return_pct"] for row in fold_metrics])) if fold_metrics else 0.0,
    }


def _parameter_stability(rows: list[dict[str, Any]], row: dict[str, Any]) -> float:
    peers = [peer for peer in rows if peer["policy_family"] == row["policy_family"] and peer["policy_id"] != row["policy_id"]]
    if not peers:
        return 0.0
    successful = [peer for peer in peers if peer["validation"]["sharpe"] > 0 and peer["oos"]["sharpe"] > 0]
    return float(len(successful) / len(peers))


def _robust_advantage(row: dict[str, Any], baseline: dict[str, Any]) -> bool:
    validation = row["validation"]
    oos = row["oos"]
    stress = row["stress"]
    return bool(
        row["partial_event_rate_pct"] >= 25.0
        and validation["sharpe"] >= baseline["validation"]["sharpe"] + 0.05
        and validation["max_drawdown_pct"] >= baseline["validation"]["max_drawdown_pct"] - 5.0
        and row["walk_forward"]["median_sharpe"] >= baseline["walk_forward"]["median_sharpe"]
        and oos["sharpe"] >= baseline["oos"]["sharpe"] + 0.05
        and oos["total_return_pct"] >= baseline["oos"]["total_return_pct"] - 2.0
        and oos["max_drawdown_pct"] >= baseline["oos"]["max_drawdown_pct"] - 5.0
        and stress["median_sharpe"] >= baseline["stress"]["median_sharpe"]
        and stress["pct_profitable"] >= baseline["stress"]["pct_profitable"]
        and row["parameter_stability"] >= 0.5
    )


def _classification(row: dict[str, Any], baseline: dict[str, Any]) -> str:
    if row["robust_advantage"]:
        return "ROBUST"
    validation = row["validation"]["sharpe"] > baseline["validation"]["sharpe"]
    oos = row["oos"]["sharpe"] > baseline["oos"]["sharpe"]
    stress = row["stress"]["median_sharpe"] >= baseline["stress"]["median_sharpe"]
    if validation and oos and not stress:
        return "OVERFIT RISK"
    if validation and stress:
        return "POTENTIALLY ROBUST"
    return "NOT ROBUST"


def _leverage_report(result: pd.DataFrame, config: PartialExitConfig, periods: int, stop_multiple: float) -> dict[str, Any]:
    gross = result["gross_return"].to_numpy(float)
    costs = result["cost_return"].to_numpy(float)
    stop_pct = (result["atr_pct"] * stop_multiple).replace(0, np.nan)
    risk_cap = config.position_risk_fraction / stop_pct
    output: dict[str, Any] = {
        "tested_grid": list(LEVERAGE_GRID),
        "exchange_ceiling": config.max_crypto_leverage,
        "risk_rule": "leverage is capped by position_risk_fraction / (ATR% * stop_multiple); no leverage is selected by return alone",
        "tiers": {},
    }
    for leverage in LEVERAGE_GRID:
        returns = pd.Series(np.clip((gross + costs) * leverage, -0.99, None), index=result.index)
        metrics = summarize_performance(returns, periods)
        metrics["risk_eligible_bar_pct"] = float(np.mean(risk_cap >= leverage) * 100.0)
        output["tiers"][str(leverage)] = metrics
    output["median_risk_cap"] = float(risk_cap.replace([np.inf, -np.inf], np.nan).median())
    return output


def _policy_row(
    symbol: str,
    candidate: Phase2Candidate,
    features: pd.DataFrame,
    policy: PartialExitPolicy,
    baseline: dict[str, Any],
    config: PartialExitConfig,
    ranges: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    train_mfe: float | None,
) -> dict[str, Any]:
    engine = PartialExitEngine(config.fee_rate, config.slippage_rate)
    result, trades = engine.execute(features, policy, candidate.stop_atr_multiple, train_mfe)
    periods = _periods_per_year(config.timeframe)
    row = {
        "symbol": symbol,
        "base_candidate_id": candidate.candidate_id,
        "policy_id": policy.policy_id,
        "policy_family": f"{policy.take_profit.method if policy.take_profit else 'baseline'}:{policy.stop_mode}:{policy.final_exit}:{policy.allocations}",
        "policy": asdict(policy),
        "validation": _evaluate_series(result, trades, {"validation": ranges["validation"]}, periods)["validation"],
        "oos": _evaluate_series(result, trades, {"oos": ranges["oos"]}, periods)["oos"],
        "walk_forward": _walk_forward(result, trades, features.index, config.walk_forward_folds, periods),
        "stress": _stress(result, config, periods),
        "partial_event_rate_pct": float(
            np.mean([
                any(str(event.get("reason", "")).startswith("tp") for event in trade.get("partial_events", []))
                for trade in trades.to_dict("records")
            ]) * 100.0
        ) if not trades.empty else 0.0,
        "parameter_stability": 0.0,
        "robust_advantage": False,
        "classification": "NOT ROBUST",
        "mae_mfe": {
            "mean_mae_pct": float(trades["mae"].mean() * 100.0) if not trades.empty else 0.0,
            "mean_mfe_pct": float(trades["mfe"].mean() * 100.0) if not trades.empty else 0.0,
            "by_exit_type": {
                str(reason): {
                    "trades": int(len(group)),
                    "mean_mae_pct": float(group["mae"].mean() * 100.0),
                    "mean_mfe_pct": float(group["mfe"].mean() * 100.0),
                }
                for reason, group in trades.groupby("exit_reason")
            },
        },
        "leverage": {},
    }
    return row


def run_partial_exit_research(config: PartialExitConfig | None = None) -> dict[str, Any]:
    config = config or PartialExitConfig(symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    candidates = {candidate.candidate_id: candidate for candidate in build_candidate_grid()}
    if config.base_candidate_id not in candidates:
        raise ValueError(f"Unknown base candidate: {config.base_candidate_id}")
    candidate = candidates[config.base_candidate_id]
    data = load_local_ohlcv(config.symbols, "data", config.timeframe, config.history_days)
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    policies = _policy_grid()
    periods = _periods_per_year(config.timeframe)
    for symbol, frame in data.items():
        features, _ = generate_signal(frame, candidate)
        ranges = _split_ranges(features.index, config)
        baseline_result, baseline_trades = _execute_candidate(features, candidate, Phase2Config(
            symbols=[symbol], timeframe=config.timeframe, history_days=config.history_days,
            fee_rate=config.fee_rate, slippage_rate=config.slippage_rate,
        ), candidate.stop_atr_multiple)
        baseline = {
            "symbol": symbol,
            "policy_id": "baseline_existing_full_exit",
            "validation": _metrics(baseline_result.loc[ranges["validation"][0]:ranges["validation"][1]], _slice_trades(baseline_trades, *ranges["validation"]), periods),
            "oos": _metrics(baseline_result.loc[ranges["oos"][0]:ranges["oos"][1]], _slice_trades(baseline_trades, *ranges["oos"]), periods),
            "walk_forward": _walk_forward(baseline_result, baseline_trades, features.index, config.walk_forward_folds, periods),
            "stress": _stress(baseline_result, config, periods),
            "full_history": _metrics(baseline_result, baseline_trades, periods),
        }
        baseline_rows.append(baseline)
        train_mask = baseline_trades["entry_timestamp"] <= ranges["train"][1] if not baseline_trades.empty else pd.Series(dtype=bool)
        train_mfe = float(baseline_trades.loc[train_mask, "mfe"].quantile(0.75)) if train_mask.any() else None
        asset_rows = [_policy_row(symbol, candidate, features, policy, baseline, config, ranges, train_mfe) for policy in policies]
        for row in asset_rows:
            row["parameter_stability"] = _parameter_stability(asset_rows, row)
            row["robust_advantage"] = _robust_advantage(row, baseline)
            row["classification"] = _classification(row, baseline)
            if row["robust_advantage"]:
                policy = next(item for item in policies if item.policy_id == row["policy_id"])
                result, _ = PartialExitEngine(config.fee_rate, config.slippage_rate).execute(features, policy, candidate.stop_atr_multiple, train_mfe)
                row["leverage"] = _leverage_report(result, config, periods, candidate.stop_atr_multiple)
        rows.extend(asset_rows)
    robust = [row for row in rows if row["robust_advantage"]]
    return {
        "decision": "ROBUST PARTIAL EXIT ADVANTAGE FOUND" if robust else "NO ROBUST PARTIAL EXIT ADVANTAGE FOUND",
        "classification": "ROBUST" if robust else "NOT ROBUST",
        "research": asdict(config),
        "strategy_layer": {
            "base_candidate_id": candidate.candidate_id,
            "base_candidate_parameters": candidate.parameters,
            "stop_atr_multiple": candidate.stop_atr_multiple,
            "selection_note": "Base entry strategy was fixed before this tactic study; OOS is not used for parameter tuning.",
        },
        "data": {"symbols": list(data), "missing_symbols": [symbol for symbol in config.symbols if symbol not in data], "timeframe": config.timeframe},
        "cost_model": {"fee_rate": config.fee_rate, "slippage_rate": config.slippage_rate, "funding": "NOT TESTED: no local funding cache"},
        "baseline": baseline_rows,
        "variants": rows,
        "robust_variants": robust,
        "tested_policy_count": len(policies),
        "not_tested": ["swing/structure targets: no separate causal swing contract was available", "live execution: advisory only, no orders"],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
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
    (path / "partial_exit_research.json").write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")
    rows = []
    for row in report["variants"]:
        rows.append({
            "symbol": row["symbol"],
            "base_candidate_id": row["base_candidate_id"],
            "policy_id": row["policy_id"],
            "classification": row["classification"],
            "parameter_stability": row["parameter_stability"],
            "validation_sharpe": row["validation"]["sharpe"],
            "validation_return_pct": row["validation"]["total_return_pct"],
            "oos_sharpe": row["oos"]["sharpe"],
            "oos_return_pct": row["oos"]["total_return_pct"],
            "oos_max_drawdown_pct": row["oos"]["max_drawdown_pct"],
            "walk_forward_median_sharpe": row["walk_forward"]["median_sharpe"],
            "stress_median_sharpe": row["stress"]["median_sharpe"],
            "stress_profitable_pct": row["stress"]["pct_profitable"],
            "partial_event_rate_pct": row["partial_event_rate_pct"],
            "robust_advantage": row["robust_advantage"],
            "trades": row["oos"]["trades"],
        })
    pd.DataFrame(rows).to_csv(path / "partial_exit_research.csv", index=False)
    lines = [
        "# Partial Exit Research",
        "",
        f"Decision: **{report['decision']}**",
        f"Base strategy: `{report['strategy_layer']['base_candidate_id']}`",
        f"Policies tested: {report['tested_policy_count']}",
        f"Funding: `{report['cost_model']['funding']}`",
        f"Missing data: `{', '.join(report['data']['missing_symbols']) or 'none'}`",
        "",
        "## Baseline",
    ]
    for baseline in report["baseline"]:
        lines.append(f"- {baseline['symbol']}: validation Sharpe {baseline['validation']['sharpe']:.3f}, OOS Sharpe {baseline['oos']['sharpe']:.3f}, OOS return {baseline['oos']['total_return_pct']:.2f}%, OOS max DD {baseline['oos']['max_drawdown_pct']:.2f}%")
    lines.extend(["", "## Variant Results", "", "| Symbol | Policy | Class | Validation Sharpe | OOS Sharpe | OOS Return | OOS Max DD | Stress Sharpe | Stress profitable |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for row in report["variants"]:
        lines.append(f"| {row['symbol']} | `{row['policy_id']}` | {row['classification']} | {row['validation']['sharpe']:.3f} | {row['oos']['sharpe']:.3f} | {row['oos']['total_return_pct']:.2f}% | {row['oos']['max_drawdown_pct']:.2f}% | {row['stress']['median_sharpe']:.3f} | {row['stress']['pct_profitable']:.1f}% |")
    lines.extend(["", "## Guardrails", "", "- Strategy selection is separate from risk/product leverage.", "- OOS was not used for tactic parameter tuning.", "- Same-bar stop precedence is conservative and deterministic.", "- No live or paper order path was changed by this research runner.", "- Structure/swing targets were not tested because no separate causal structure contract was available."])
    (path / "partial_exit_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-runs", type=int, default=200)
    parser.add_argument("--candidate-id", default=BASE_CANDIDATE_ID)
    args = parser.parse_args()
    config = PartialExitConfig(
        symbols=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        stress_runs=args.stress_runs,
        base_candidate_id=args.candidate_id,
    )
    report = run_partial_exit_research(config)
    write_outputs(report, config.output_dir)
    print(report["decision"])
    print(f"Policies tested: {report['tested_policy_count']}")
    print(f"Robust variants: {len(report['robust_variants'])}")


if __name__ == "__main__":
    main()