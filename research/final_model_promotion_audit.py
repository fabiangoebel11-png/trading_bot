"""Final, bounded promotion audit for the four permitted model variants.

This runner never retrains or promotes a model.  It evaluates only causal,
closed-candle signals and rejects a historical TCN path unless its persisted OOS
series has the same frequency as the strategy task.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.backtester import _periods_per_year, run_backtest_from_signals
from core.config import TradingBotConfig
from core.metrics import max_drawdown, profit_factor, sharpe_ratio, sortino_ratio
from core.ml.inference import load_oos_confidence
from core.strategy import generate_trend_signals
from decision_pipeline import load_cached_ohlcv
from model_integration import _load_chronos2_pipeline
from train import load_config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "research" / "final_model_promotion"
VARIANTS = ("RULE_ONLY", "RULE_TCN", "RULE_CHRONOS", "RULE_TCN_CHRONOS")
SPLITS = (
    ("WF1", "2022-01-01", "2022-12-31"),
    ("WF2", "2023-01-01", "2023-12-31"),
    ("WF3", "2024-01-01", "2024-12-31"),
    ("FINAL_HOLDOUT", "2025-01-01", "2026-09-30"),
)
TASKS = (
    ("BTC Intraday", "BTC/USDT", "1h", 4),
    ("ETH Intraday", "ETH/USDT", "1h", 4),
    ("BTC Swing", "BTC/USDT", "4h", 42),
    ("ETH Swing", "ETH/USDT", "4h", 42),
)
CONTEXT_BARS = 512


@dataclass(frozen=True)
class Task:
    name: str
    asset: str
    timeframe: str
    horizon: int


def _td(timeframe: str) -> pd.Timedelta:
    units = {"m": "minutes", "h": "hours", "d": "days"}
    value, unit = timeframe[:-1], timeframe[-1:]
    if not value.isdigit() or unit not in units:
        raise ValueError(f"Unsupported audit timeframe: {timeframe}")
    return pd.Timedelta(timedelta(**{units[unit]: int(value)}))


def _tcn_status(asset: str, timeframe: str) -> tuple[str, str, pd.Series | None]:
    """Only accept an OOS series whose timestamp frequency matches the task."""
    cfg = load_config("configs/training_crypto_intraday.yaml")
    confidence = load_oos_confidence(asset, cfg.ml)
    if confidence is None or confidence.dropna().empty:
        return "MODEL_UNAVAILABLE", "No persisted leak-free TCN OOS confidence series.", None
    index = pd.DatetimeIndex(confidence.index)
    observed = index.to_series().diff().dropna().median()
    if observed != _td(timeframe):
        return (
            "MODEL_UNAVAILABLE",
            f"TCN OOS cadence is {observed}, not the {timeframe} task cadence; final checkpoint inference would be leaky.",
            None,
        )
    return "AVAILABLE", "Persisted leak-free TCN OOS confidence matches task cadence.", confidence


def _rule_config(asset: str, timeframe: str) -> TradingBotConfig:
    config = TradingBotConfig()
    config.data.symbols = [asset]
    config.data.base_timeframe = timeframe
    config.data.resample_to = None
    # The audit has no local historical funding series. It uses the same zero
    # funding assumption for every variant and records that limitation.
    config.macro.enabled = False
    config.funding.enabled = False
    config.ml.enabled = False
    return config


def _entry_times(signals: pd.DataFrame) -> pd.DatetimeIndex:
    position = signals["position"].astype(float)
    entries = position.ne(position.shift()).fillna(False) & position.ne(0)
    return pd.DatetimeIndex(signals.index[entries])


def chronos_entry_forecasts(frame: pd.DataFrame, entries: pd.DatetimeIndex, horizon: int, batch_size: int = 32) -> tuple[pd.DataFrame, dict[str, float]]:
    """Forecast only entry candles, each using at most its prior closed history."""
    import torch

    eligible = [timestamp for timestamp in entries if frame.index.get_loc(timestamp) + 1 >= CONTEXT_BARS]
    columns = ["direction", "model_score", "forecast_low", "forecast_median", "forecast_high", "uncertainty", "inference_seconds"]
    if not eligible:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz=frame.index.tz)), {"load_seconds": 0.0, "mean_inference_seconds": 0.0}
    started = time.perf_counter()
    pipeline = _load_chronos2_pipeline("amazon/chronos-2", "cpu")
    load_seconds = time.perf_counter() - started
    rows: list[dict[str, Any]] = []
    for offset in range(0, len(eligible), batch_size):
        batch = eligible[offset : offset + batch_size]
        contexts = np.stack([
            frame["close"].iloc[frame.index.get_loc(timestamp) + 1 - CONTEXT_BARS : frame.index.get_loc(timestamp) + 1].to_numpy(dtype=np.float32)
            for timestamp in batch
        ])
        started = time.perf_counter()
        outputs = pipeline.predict(torch.as_tensor(contexts).unsqueeze(1), prediction_length=horizon)
        per_item_seconds = (time.perf_counter() - started) / len(batch)
        for timestamp, output in zip(batch, outputs, strict=True):
            values = output.detach().cpu().numpy() if hasattr(output, "detach") else np.asarray(output)
            if values.ndim != 3 or values.shape[1] < 20:
                continue
            low, median, high = (float(values[0, quantile, -1]) for quantile in (3, 11, 19))
            close = float(frame.at[timestamp, "close"])
            if not np.isfinite([low, median, high]).all() or close <= 0:
                continue
            low_return, median_return, high_return = low / close - 1.0, median / close - 1.0, high / close - 1.0
            spread = abs(high_return - low_return)
            rows.append({
                "timestamp": timestamp,
                "direction": "LONG" if median_return > 0 else "SHORT" if median_return < 0 else "UNCERTAIN",
                "model_score": float(np.clip(50.0 + 50.0 * abs(median_return) / max(spread, 1e-12), 0.0, 100.0)),
                "forecast_low": low,
                "forecast_median": median,
                "forecast_high": high,
                "uncertainty": "LOW" if spread < 0.01 else "MEDIUM" if spread < 0.03 else "HIGH",
                "inference_seconds": per_item_seconds,
            })
    forecast = pd.DataFrame(rows).set_index("timestamp").sort_index() if rows else pd.DataFrame(columns=columns)
    return forecast, {"load_seconds": load_seconds, "mean_inference_seconds": float(forecast["inference_seconds"].mean()) if not forecast.empty else 0.0}


def gate_positions(rule: pd.DataFrame, forecast: pd.DataFrame) -> pd.DataFrame:
    """Apply a model only when a new causal rule entry has matching direction."""
    out = rule.copy()
    gated = np.zeros(len(out), dtype=float)
    raw = out["position"].astype(float).to_numpy()
    prior_raw = 0.0
    active = 0.0
    for index, (timestamp, position) in enumerate(zip(out.index, raw, strict=True)):
        if position == 0.0:
            active = 0.0
        elif position != prior_raw:
            direction = forecast.at[timestamp, "direction"] if timestamp in forecast.index else "UNAVAILABLE"
            active = position if (position > 0 and direction == "LONG") or (position < 0 and direction == "SHORT") else 0.0
        gated[index] = active
        prior_raw = position
    out["position"] = gated
    return out


def _trade_returns(result: pd.DataFrame) -> np.ndarray:
    position = result["execution_position"].to_numpy(float)
    returns = result["strategy_return"].to_numpy(float)
    completed: list[float] = []
    active: list[float] = []
    for current, value in zip(position, returns, strict=True):
        if current != 0:
            active.append(value)
        elif active:
            completed.append(float(np.prod(1 + np.asarray(active)) - 1))
            active = []
    if active:
        completed.append(float(np.prod(1 + np.asarray(active)) - 1))
    return np.asarray(completed, dtype=float)


def _performance(result: pd.DataFrame, timeframe: str) -> dict[str, float]:
    returns = result["strategy_return"].astype(float)
    trades = _trade_returns(result)
    cumulative = (1 + returns).cumprod()
    positive, negative = trades[trades > 0], trades[trades < 0]
    return {
        "return_pct": float((cumulative.iloc[-1] - 1) * 100) if len(cumulative) else 0.0,
        "sharpe": sharpe_ratio(returns, _periods_per_year(timeframe)),
        "sortino": sortino_ratio(returns, _periods_per_year(timeframe)),
        "max_drawdown_pct": max_drawdown(cumulative) * 100 if len(cumulative) else 0.0,
        "profit_factor": profit_factor(pd.Series(trades)),
        "expectancy_pct": float(trades.mean() * 100) if len(trades) else 0.0,
        "win_rate_pct": float((trades > 0).mean() * 100) if len(trades) else 0.0,
        "average_win_pct": float(positive.mean() * 100) if len(positive) else 0.0,
        "average_loss_pct": float(negative.mean() * 100) if len(negative) else 0.0,
        "trades": float(len(trades)),
        "coverage_pct": float(result["execution_position"].ne(0).mean() * 100),
        "turnover": float(result["trade"].sum()),
        "fees_pct": float(result["fee_cost"].sum() * 100),
        "funding_pct": float(result.get("funding_cost", pd.Series(0.0, index=result.index)).sum() * 100),
    }


def _forecast_metrics(frame: pd.DataFrame, forecasts: pd.DataFrame, horizon: int) -> dict[str, float | None]:
    if forecasts.empty:
        return {"directional_accuracy_pct": None, "mae_pct": None, "range_coverage_pct": None}
    realized = frame["close"].shift(-horizon).reindex(forecasts.index)
    valid = forecasts.assign(realized=realized).dropna(subset=["realized"])
    if valid.empty:
        return {"directional_accuracy_pct": None, "mae_pct": None, "range_coverage_pct": None}
    direction = np.sign(valid["realized"].to_numpy() - frame.loc[valid.index, "close"].to_numpy())
    predicted = np.where(valid["direction"].eq("LONG"), 1, -1)
    return {
        "directional_accuracy_pct": float((direction == predicted).mean() * 100),
        "mae_pct": float((valid["forecast_median"] / valid["realized"] - 1.0).abs().mean() * 100),
        "range_coverage_pct": float(((valid["realized"] >= valid["forecast_low"]) & (valid["realized"] <= valid["forecast_high"])).mean() * 100),
    }


def _mc(returns: pd.Series, periods_per_year: int, runs: int, seed: int) -> dict[str, float]:
    """Trade-sequence and fixed-block bootstrap diagnostics, both batched."""
    values = returns.to_numpy(float)
    window = min(len(values), periods_per_year)
    if window < 24:
        return {"mc_status": "INSUFFICIENT_DATA"}
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    block = max(1, periods_per_year // 365)  # one day of bars
    blocks = int(np.ceil(window / block))
    for _ in range(0, runs, 250):
        count = min(250, runs - len(samples) * 250)
        starts = rng.integers(0, max(1, len(values) - block), size=(count, blocks))
        indexes = (starts[..., None] + np.arange(block)).reshape(count, -1)[:, :window]
        samples.append(values[indexes])
    stacked = np.vstack(samples)
    cumulative = np.cumprod(1 + stacked, axis=1)
    total = (cumulative[:, -1] - 1) * 100
    drawdown = cumulative / np.maximum.accumulate(cumulative, axis=1) - 1
    drawdown_abs = -drawdown.min(axis=1) * 100
    return {
        "mc_status": "TESTED",
        "mc_runs": float(runs),
        "mc_median_return_pct": float(np.median(total)),
        "mc_return_p05_pct": float(np.quantile(total, 0.05)),
        "mc_return_p95_pct": float(np.quantile(total, 0.95)),
        "mc_median_max_drawdown_pct": float(np.median(drawdown_abs)),
        "mc_max_drawdown_p95_pct": float(np.quantile(drawdown_abs, 0.95)),
        "mc_probability_negative_pct": float((total < 0).mean() * 100),
        "mc_worst_drawdown_pct": float(drawdown_abs.max()),
    }


def _split_rows(task: Task, variant: str, result: pd.DataFrame, model_status: str, model_reason: str, forecast: dict[str, float | None], mc: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for split, start, end in SPLITS:
        part = result.loc[(result.index >= pd.Timestamp(start, tz="UTC")) & (result.index <= pd.Timestamp(end, tz="UTC"))]
        metrics = _performance(part, task.timeframe) if len(part) >= 24 else {"return_pct": np.nan, "sharpe": np.nan, "max_drawdown_pct": np.nan, "trades": 0.0, "coverage_pct": 0.0}
        rows.append({"task": task.name, "asset": task.asset, "category": "INTRADAY" if task.timeframe == "1h" else "SWING", "timeframe": task.timeframe, "horizon_bars": task.horizon, "variant": variant, "split": split, "status": model_status, "reason": model_reason, **metrics, **forecast, **(mc if split == "FINAL_HOLDOUT" else {})})
    return rows


def audit_task(task: Task, runs: int) -> list[dict[str, Any]]:
    frame = load_cached_ohlcv(task.asset, task.timeframe, ROOT / "data")
    if frame is None or len(frame) < CONTEXT_BARS + task.horizon:
        return [{"task": task.name, "variant": variant, "split": "ALL", "status": "INSUFFICIENT_DATA", "reason": "No sufficient local closed-candle cache."} for variant in VARIANTS]
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    config = _rule_config(task.asset, task.timeframe)
    rule = generate_trend_signals(frame, config.trend)
    entries = _entry_times(rule)
    forecasts, timing = chronos_entry_forecasts(frame, entries, task.horizon)
    forecast_metrics = {**_forecast_metrics(frame, forecasts, task.horizon), "chronos_load_seconds": timing["load_seconds"], "chronos_mean_inference_seconds": timing["mean_inference_seconds"], "chronos_forecasts": float(len(forecasts))}
    tcn_status, tcn_reason, _ = _tcn_status(task.asset, task.timeframe)

    rule_result = run_backtest_from_signals({task.asset: rule}, config).attrs["per_symbol"][task.asset]
    chronos_result = run_backtest_from_signals({task.asset: gate_positions(rule, forecasts)}, config).attrs["per_symbol"][task.asset]
    rows = _split_rows(task, "RULE_ONLY", rule_result, "EVALUATED", "Rule-only baseline; zero funding assumption applied equally.", {key: None for key in forecast_metrics}, _mc(rule_result["strategy_return"], _periods_per_year(task.timeframe), runs, 42))
    rows += _split_rows(task, "RULE_CHRONOS", chronos_result, "EVALUATED", "Causal Chronos-2 entry-direction gate; identical downstream execution/risk/cost path.", forecast_metrics, _mc(chronos_result["strategy_return"], _periods_per_year(task.timeframe), runs, 43))
    # A final TCN checkpoint may only score truly future bars. The 5m OOS file
    # cannot be silently reinterpreted as an hourly/four-hour model signal.
    for variant in ("RULE_TCN", "RULE_TCN_CHRONOS"):
        rows += _split_rows(task, variant, rule_result, tcn_status, tcn_reason, {key: None for key in forecast_metrics}, {"mc_status": "NOT_RUN_MODEL_UNAVAILABLE"})
    return rows


def _write_reports(results: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / "final_model_promotion_results.csv", index=False)
    results.to_json(OUTPUT_DIR / "final_model_promotion_results.json", orient="records", indent=2, date_format="iso")
    holdout = results[results["split"].eq("FINAL_HOLDOUT")].copy()
    columns = [column for column in ("task", "variant", "status", "return_pct", "sharpe", "max_drawdown_pct", "trades", "coverage_pct", "fees_pct", "mc_median_return_pct", "mc_probability_negative_pct") if column in holdout]
    if holdout.empty:
        table = "No evaluated holdout rows."
    else:
        try:
            table = holdout[columns].to_markdown(index=False)
        except ImportError:
            table = "```text\n" + holdout[columns].to_string(index=False) + "\n```"
    status = "NOT COMPLETE"
    model_audit = f"""# Final Model Combination Audit\n\nStatus: **{status}**\n\nThe audit evaluated causal Rule-only and Rule+Chronos-2 variants. `RULE_TCN` and `RULE_TCN_CHRONOS` are `MODEL_UNAVAILABLE` for the 1h/4h tasks because their only persisted leak-free TCN OOS series is 5m. Using the final checkpoint on earlier history would be leakage.\n\nNo variant is promoted: current model training cutoffs are at the end of the available local cache, so no untouched post-cutoff forward holdout exists. `RULE_ONLY` remains active.\n\n## Final-Holdout Diagnostics\n\n{table}\n\nChronos is evaluated only at causal rule-entry candles and gates a trade only when its direction agrees. The TCN/Chronos combination is not averaged or promoted.\n\n## Agreement and Selection\n\n`MODEL_AGREEMENT`, model conflicts, confidence-weighted ensemble scores, and ensemble value are `NOT_AVAILABLE` for every audited task: a task-compatible historical TCN OOS stream is absent. The runner deliberately refuses to resample the 5m TCN scores onto 1h/4h candles.\n\nQQQ/SPY intraday and swing are `INSUFFICIENT_DATA` for this final promotion run. They are proxy-labelled local series, begin in 2023, and do not cover the declared WF1/WF2/WF3 protocol.\n"""
    mc_audit = f"""# Final Monte Carlo Audit\n\nStatus: **{status}**\n\nEach evaluated Rule-only and Rule+Chronos row uses 5,000 fixed-block bootstrap simulations over up to one year of its return path. Metrics are diagnostics for sensitivity to ordering/dependence, not proof of profitability. TCN-containing variants are not simulated because the task-compatible historical TCN signal is unavailable.\n\n## Final-Holdout Monte Carlo\n\n{table}\n\nCost stress, parameter perturbation, and model-noise promotion tests remain blocked until a task-matched, leakage-safe TCN OOS artifact and a true post-training forward holdout exist. The identical fee/slippage assumptions are already applied to all evaluated variants; historical funding was unavailable locally and set to zero for every variant.\n"""
    registry = "# Final Model Registry\n\nStatus: **NOT COMPLETE**\n\n| Asset | Category | Timeframe | Horizon | Strategy | Model | Model version | Ensemble version | Status | Reason |\n|---|---|---|---|---|---|---|---|---|---|\n"
    for task in sorted(results["task"].dropna().unique()):
        row = results[results["task"].eq(task)].iloc[0]
        registry += f"| {row['asset']} | {row['category']} | {row['timeframe']} | {int(row['horizon_bars'])} bars | Donchian/EMA/ATR | RULE_ONLY | v1.0 | v1.0-no-promotion | ACTIVE | TCN OOS cadence mismatch; Chronos remains observe-only; no untouched forward holdout. |\n"
    registry += "| QQQ (proxy) | INTRADAY | 1h | 4 bars | Rule-only | none | n/a | v1.0-no-promotion | INSUFFICIENT_DATA | Proxy history begins in 2023 and does not cover WF1/WF2/WF3. |\n"
    registry += "| QQQ (proxy) | SWING | 4h | 42 bars | Rule-only | none | n/a | v1.0-no-promotion | INSUFFICIENT_DATA | Proxy history begins in 2023 and does not cover WF1/WF2/WF3. |\n"
    registry += "| SPY (proxy) | INTRADAY | 1h | 4 bars | Rule-only | none | n/a | v1.0-no-promotion | INSUFFICIENT_DATA | Proxy history begins in 2023 and does not cover WF1/WF2/WF3. |\n"
    registry += "| SPY (proxy) | SWING | 4h | 42 bars | Rule-only | none | n/a | v1.0-no-promotion | INSUFFICIENT_DATA | Proxy history begins in 2023 and does not cover WF1/WF2/WF3. |\n"
    reports_dir = ROOT / "docs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "FINAL_MODEL_COMBINATION_AUDIT.md").write_text(model_audit, encoding="utf-8")
    (reports_dir / "FINAL_MONTE_CARLO_AUDIT.md").write_text(mc_audit, encoding="utf-8")
    (reports_dir / "FINAL_MODEL_REGISTRY.md").write_text(registry, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--task", action="append", choices=[task[0] for task in TASKS])
    args = parser.parse_args()
    selected = [Task(*task) for task in TASKS if not args.task or task[0] in args.task]
    rows = [row for task in selected for row in audit_task(task, args.runs)]
    _write_reports(pd.DataFrame(rows))
    print(f"Wrote {len(rows)} audit rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()