"""Research runner for the rule-based crypto strategy candidates.

The runner deliberately keeps candidate definitions explicit and evaluates them
on the same local OHLCV history. It does not optimize parameters on OOS data,
does not fetch missing data, and does not test leverage as a source of edge.
Leverage-grid research belongs after the underlying signal has survived this
comparison.

Run with::

    python -m research.run_strategy_research
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.backtester import _periods_per_year, run_backtest
from core.config import TradingBotConfig
from core.metrics import summarize_performance
from core.monte_carlo import run_monte_carlo_stress_test


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    description: str
    fast_ma_window: int
    slow_ma_window: int
    donchian_entry_window: int
    donchian_exit_window: int
    atr_window: int
    min_atr_pct: float
    atr_stop_multiplier: float
    cooldown_candles: int
    max_portfolio_risk_pct: float
    min_leverage: float
    max_leverage: float


@dataclass
class ResearchConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframe: str = "1h"
    history_days: int = 2500
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    output_dir: str = "data/research"
    stress_runs: int = 200
    stress_window_days: int = 365


def candidate_definitions() -> list[StrategyCandidate]:
    """Return the three pre-registered candidates; no data-derived tuning occurs here."""
    return [
        StrategyCandidate(
            name="conservative",
            description="Longer breakout confirmation, wider stop and half portfolio risk budget.",
            fast_ma_window=30,
            slow_ma_window=150,
            donchian_entry_window=80,
            donchian_exit_window=30,
            atr_window=20,
            min_atr_pct=0.0025,
            atr_stop_multiplier=3.5,
            cooldown_candles=12,
            max_portfolio_risk_pct=0.005,
            min_leverage=1.0,
            max_leverage=2.0,
        ),
        StrategyCandidate(
            name="balanced",
            description="Reference EMA/Donchian trend-following specification.",
            fast_ma_window=20,
            slow_ma_window=100,
            donchian_entry_window=55,
            donchian_exit_window=20,
            atr_window=14,
            min_atr_pct=0.0015,
            atr_stop_multiplier=3.0,
            cooldown_candles=6,
            max_portfolio_risk_pct=0.01,
            min_leverage=2.0,
            max_leverage=4.0,
        ),
        StrategyCandidate(
            name="aggressive",
            description="Faster breakout response and shorter cooldown within the same execution model.",
            fast_ma_window=10,
            slow_ma_window=50,
            donchian_entry_window=30,
            donchian_exit_window=12,
            atr_window=10,
            min_atr_pct=0.0010,
            atr_stop_multiplier=2.5,
            cooldown_candles=3,
            max_portfolio_risk_pct=0.01,
            min_leverage=2.0,
            max_leverage=4.0,
        ),
    ]


def _local_path(data_dir: Path, symbol: str, timeframe: str, history_days: int) -> Path:
    return data_dir / f"ohlcv_{symbol.replace('/', '-')}_{timeframe}_{history_days}d.csv"


def load_local_ohlcv(
    symbols: list[str], data_dir: str | Path, timeframe: str, history_days: int
) -> dict[str, pd.DataFrame]:
    """Load cached OHLCV only; missing files are a hard, visible research error."""
    data_path = Path(data_dir)
    raw: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in symbols:
        path = _local_path(data_path, symbol, timeframe, history_days)
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        frame.index = pd.to_datetime(frame.index, utc=True)
        frame = frame.sort_index()[["open", "high", "low", "close"]]
        raw[symbol] = frame[~frame.index.duplicated(keep="last")]
    if missing:
        raise FileNotFoundError(
            "Missing local OHLCV files (no network fetch performed): " + ", ".join(missing)
        )
    union_index = pd.Index([], dtype="datetime64[ns, UTC]")
    for frame in raw.values():
        union_index = union_index.union(frame.index)
    union_index = union_index.sort_values()
    return {symbol: frame.reindex(union_index) for symbol, frame in raw.items()}


def _candidate_config(base: TradingBotConfig, candidate: StrategyCandidate) -> TradingBotConfig:
    config = copy.deepcopy(base)
    config.trend.fast_ma_window = candidate.fast_ma_window
    config.trend.slow_ma_window = candidate.slow_ma_window
    config.trend.donchian_entry_window = candidate.donchian_entry_window
    config.trend.donchian_exit_window = candidate.donchian_exit_window
    config.trend.atr_window = candidate.atr_window
    config.trend.min_atr_pct = candidate.min_atr_pct
    config.risk.atr_stop_multiplier = candidate.atr_stop_multiplier
    config.risk.min_leverage = candidate.min_leverage
    config.risk.max_leverage = candidate.max_leverage
    config.capital.cooldown_candles = candidate.cooldown_candles
    config.capital.max_portfolio_risk_pct = candidate.max_portfolio_risk_pct
    return config


def _split_bounds(index: pd.Index, research: ResearchConfig) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    if not 0 < research.train_fraction < 1 or not 0 < research.validation_fraction < 1:
        raise ValueError("train_fraction and validation_fraction must be between 0 and 1")
    if research.train_fraction + research.validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")
    n = len(index)
    train_end = int(n * research.train_fraction)
    validation_end = int(n * (research.train_fraction + research.validation_fraction))
    if min(train_end, validation_end - train_end, n - validation_end) < 100:
        raise ValueError("Each research split needs at least 100 bars")
    return {
        "train": (index[0], index[train_end - 1]),
        "validation": (index[train_end], index[validation_end - 1]),
        "oos": (index[validation_end], index[-1]),
    }


def _metrics(frame: pd.DataFrame, periods_per_year: int) -> dict[str, float | int]:
    trade_pnl = frame["trade"] * frame["strategy_return"]
    return summarize_performance(frame["strategy_return"], periods_per_year, trade_pnl)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _robust(
    validation: dict[str, float | int],
    oos: dict[str, float | int],
    stress: dict[str, float | int],
) -> bool:
    """Require a positive validation and OOS result plus stress confirmation."""
    return (
        float(validation["total_return_pct"]) > 0
        and float(validation["sharpe"]) > 0
        and float(oos["total_return_pct"]) > 0
        and float(oos["sharpe"]) >= 0.5
        and float(stress["median_sharpe"]) > 0
        and float(stress["pct_profitable"]) >= 50.0
    )


def run_research(
    research: ResearchConfig,
    base_config: TradingBotConfig | None = None,
    run_stress: bool = True,
) -> dict[str, object]:
    base = copy.deepcopy(base_config or TradingBotConfig())
    base.data.symbols = list(research.symbols)
    base.data.base_timeframe = research.timeframe
    base.data.history_days = research.history_days
    base.macro.enabled = False
    base.funding.enabled = False
    base.ml.enabled = False
    multi_ohlc = load_local_ohlcv(research.symbols, base.data.cache_dir, research.timeframe, research.history_days)
    index = next(iter(multi_ohlc.values())).index
    splits = _split_bounds(index, research)
    periods_per_year = _periods_per_year(research.timeframe)
    candidates: dict[str, object] = {}

    for candidate in candidate_definitions():
        config = _candidate_config(base, candidate)
        result = run_backtest(multi_ohlc, config)
        split_metrics: dict[str, object] = {}
        for split_name, (start, end) in splits.items():
            segment = result.loc[start:end]
            split_metrics[split_name] = _metrics(segment, periods_per_year)

        stress = {
            "status": "NOT_TESTED",
            "median_sharpe": 0.0,
            "pct_profitable": 0.0,
        }
        if run_stress:
            stress_config = copy.deepcopy(config)
            stress_config.monte_carlo.n_runs = research.stress_runs
            stress_config.monte_carlo.window_days = research.stress_window_days
            stress_config.monte_carlo.use_gpu = False
            summary = run_monte_carlo_stress_test(
                multi_ohlc, stress_config, precomputed_backtest=result
            )
            stress = {
                "status": "TESTED",
                "median_sharpe": summary.median_sharpe,
                "mean_sharpe": summary.mean_sharpe,
                "pct_profitable": summary.pct_profitable,
                "mean_total_return_pct": summary.mean_total_return_pct,
                "median_total_return_pct": float(summary.runs["total_return_pct"].median()),
                "worst_case_max_drawdown_usdt": summary.worst_case_max_drawdown_usdt,
            }
        candidates[candidate.name] = {
            "rules": asdict(candidate),
            "splits": split_metrics,
            "stress": stress,
            "robust": _robust(split_metrics["validation"], split_metrics["oos"], stress),
        }

    robust_names = [name for name, value in candidates.items() if value["robust"]]
    return {
        "research": asdict(research),
        "data": {
            "symbols": research.symbols,
            "timeframe": research.timeframe,
            "history_start": index.min().isoformat(),
            "history_end": index.max().isoformat(),
            "bars": len(index),
            "equity_assets": {"SPY": "INSUFFICIENT_DATA", "QQQ": "INSUFFICIENT_DATA"},
        },
        "candidates": candidates,
        "decision": "ROBUST STRATEGY FOUND" if robust_names else "NO ROBUST STRATEGY FOUND",
        "robust_candidates": robust_names,
    }


def _write_outputs(report: dict[str, object], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    safe_report = _json_safe(report)
    (path / "strategy_research.json").write_text(json.dumps(safe_report, indent=2), encoding="utf-8")

    rows: list[dict[str, object]] = []
    for name, candidate in report["candidates"].items():
        for split, metrics in candidate["splits"].items():
            rows.append({"candidate": name, "sample": split, **metrics})
        rows.append({"candidate": name, "sample": "stress", **candidate["stress"]})
    pd.DataFrame(rows).to_csv(path / "strategy_research.csv", index=False)

    lines = [
        "# Strategy Research",
        "",
        f"Decision: **{report['decision']}**",
        "",
        f"Data: {report['data']['history_start']} to {report['data']['history_end']} "
        f"({report['data']['bars']} union bars), crypto only.",
        "SPY/QQQ: `INSUFFICIENT_DATA` because no local OHLCV files were available.",
        "",
        "## Candidates",
    ]
    for name, candidate in report["candidates"].items():
        rules = candidate["rules"]
        lines.extend(
            [
                "",
                f"### {name}",
                candidate["rules"]["description"],
                "",
                "Rules: " + ", ".join(
                    f"{key}={value}" for key, value in rules.items() if key != "description"
                ),
                "",
                "| Sample | Sharpe | Return % | Max DD % | Win rate % |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for sample, metrics in candidate["splits"].items():
            lines.append(
                f"| {sample} | {metrics['sharpe']:.3f} | {metrics['total_return_pct']:.2f} | "
                f"{metrics['max_drawdown_pct']:.2f} | {metrics.get('win_rate_pct', 0.0):.2f} |"
            )
        stress = candidate["stress"]
        lines.append(
            f"| stress median | {stress['median_sharpe']:.3f} | {stress.get('median_total_return_pct', 0.0):.2f} | "
            f"n/a | {stress['pct_profitable']:.2f} profitable windows |"
        )
        lines.append(f"\nRobust gate: **{'PASS' if candidate['robust'] else 'FAIL'}**")
    (path / "strategy_research.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-stress", action="store_true", help="skip the random-window stress test")
    parser.add_argument("--stress-runs", type=int, default=200)
    parser.add_argument("--output-dir", default="data/research")
    args = parser.parse_args()
    research = ResearchConfig(stress_runs=args.stress_runs, output_dir=args.output_dir)
    report = run_research(research, run_stress=not args.no_stress)
    _write_outputs(report, research.output_dir)
    print(report["decision"])
    for name, candidate in report["candidates"].items():
        oos = candidate["splits"]["oos"]
        stress = candidate["stress"]
        print(
            f"{name}: OOS Sharpe={oos['sharpe']:.3f}, OOS return={oos['total_return_pct']:.2f}%, "
            f"stress median Sharpe={stress['median_sharpe']:.3f}, "
            f"profitable windows={stress['pct_profitable']:.1f}%"
        )


if __name__ == "__main__":
    main()