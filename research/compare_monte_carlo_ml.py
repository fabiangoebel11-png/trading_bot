"""Paired Monte Carlo comparison: rule strategy versus OOS-ML confirmation.

Both runs use identical data, funding, Monte Carlo seed and sampled start
indices. The resulting per-window deltas therefore isolate the historical
effect of the leak-free stitched OOS ML confidence series.

Run with: python -m uv run research/compare_monte_carlo_ml.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from core.config import TradingBotConfig
from core.data_loader import load_multi_asset_data
from core.funding import load_portfolio_funding
from core.monte_carlo import run_monte_carlo_stress_test
from core.ml.regime import classify_regimes


def main() -> None:
    config = TradingBotConfig()
    config.data.history_days = 2500
    multi_ohlc = load_multi_asset_data(config.data)
    funding_df = load_portfolio_funding(config.data, config.funding) if config.funding.enabled else None

    rule_config = copy.deepcopy(config)
    rule_config.ml.enabled = False
    ml_config = copy.deepcopy(config)
    ml_config.ml.enabled = True
    gated_config = copy.deepcopy(config)
    gated_config.ml.enabled = True
    gated_config.ml.regime_gate_enabled = True

    print("\n=== Paired Monte Carlo: Regelstrategie ===")
    rule_summary = run_monte_carlo_stress_test(multi_ohlc, rule_config, funding_df=funding_df)
    print("=== Paired Monte Carlo: Regelstrategie + OOS-ML ===")
    ml_summary = run_monte_carlo_stress_test(multi_ohlc, ml_config, funding_df=funding_df)
    print("=== Paired Monte Carlo: Regelstrategie + OOS-ML + Regime-Gate ===")
    gated_summary = run_monte_carlo_stress_test(multi_ohlc, gated_config, funding_df=funding_df)

    if not (
        rule_summary.runs["start_idx"].equals(ml_summary.runs["start_idx"])
        and rule_summary.runs["start_idx"].equals(gated_summary.runs["start_idx"])
    ):
        raise RuntimeError("Monte-Carlo start windows differ; paired ML comparison is invalid.")

    metrics = ["sharpe", "total_return_pct", "max_drawdown_pct", "final_capital_usdt"]
    comparison = rule_summary.runs[["start_idx", "start_ts", *metrics]].copy()
    for metric in metrics:
        comparison[f"rule_{metric}"] = comparison.pop(metric)
        comparison[f"ml_{metric}"] = ml_summary.runs[metric].to_numpy()
        comparison[f"delta_{metric}"] = comparison[f"ml_{metric}"] - comparison[f"rule_{metric}"]
        comparison[f"gated_{metric}"] = gated_summary.runs[metric].to_numpy()
        comparison[f"delta_gated_{metric}"] = comparison[f"gated_{metric}"] - comparison[f"rule_{metric}"]

    delta_sharpe = comparison["delta_sharpe"]
    delta_return = comparison["delta_total_return_pct"]
    delta_drawdown = comparison["delta_max_drawdown_pct"]
    print("\n=== ML minus Regelstrategie, gleiche Fenster ===")
    print(f"  Sharpe: mean {delta_sharpe.mean():+.3f}, median {delta_sharpe.median():+.3f}, p10 {delta_sharpe.quantile(0.10):+.3f}")
    print(f"  Return: mean {delta_return.mean():+.2f}%, median {delta_return.median():+.2f}%")
    print(f"  Max DD: mean {delta_drawdown.mean():+.2f}pp, median {delta_drawdown.median():+.2f}pp (positiv = besser)")
    print(f"  Fenster mit höherem Sharpe: {(delta_sharpe > 0).mean():.1%}")
    print(f"  Fenster mit höherem Return: {(delta_return > 0).mean():.1%}")

    gated_sharpe = comparison["delta_gated_sharpe"]
    gated_return = comparison["delta_gated_total_return_pct"]
    gated_drawdown = comparison["delta_gated_max_drawdown_pct"]
    print("\n=== Regime-Gate-ML minus Regelstrategie, gleiche Fenster ===")
    print(f"  Sharpe: mean {gated_sharpe.mean():+.3f}, median {gated_sharpe.median():+.3f}, p10 {gated_sharpe.quantile(0.10):+.3f}")
    print(f"  Return: mean {gated_return.mean():+.2f}%, median {gated_return.median():+.2f}%")
    print(f"  Max DD: mean {gated_drawdown.mean():+.2f}pp, median {gated_drawdown.median():+.2f}pp (positiv = besser)")
    print(f"  Fenster mit höherem Sharpe: {(gated_sharpe > 0).mean():.1%}")
    print(f"  Fenster mit höherem Return: {(gated_return > 0).mean():.1%}")

    print("\n=== ML-Gewichts-Sweep minus Regelstrategie, gleiche Fenster ===")
    weight_rows = []
    for weight in (0.10, 0.25, 0.50, 1.00):
        if weight == 1.00:
            weighted_summary = ml_summary
        else:
            weighted_config = copy.deepcopy(config)
            weighted_config.ml.enabled = True
            weighted_config.ml.confirmation_weight = weight
            weighted_summary = run_monte_carlo_stress_test(multi_ohlc, weighted_config, funding_df=funding_df)
        if not rule_summary.runs["start_idx"].equals(weighted_summary.runs["start_idx"]):
            raise RuntimeError(f"Monte-Carlo start windows differ for ML weight={weight}.")
        sharpe_delta = weighted_summary.runs["sharpe"] - rule_summary.runs["sharpe"]
        return_delta = weighted_summary.runs["total_return_pct"] - rule_summary.runs["total_return_pct"]
        drawdown_delta = weighted_summary.runs["max_drawdown_pct"] - rule_summary.runs["max_drawdown_pct"]
        row = {
            "confirmation_weight": weight,
            "mean_delta_sharpe": sharpe_delta.mean(),
            "median_delta_sharpe": sharpe_delta.median(),
            "p10_delta_sharpe": sharpe_delta.quantile(0.10),
            "pct_higher_sharpe": (sharpe_delta > 0).mean(),
            "median_delta_return_pct": return_delta.median(),
            "median_delta_max_drawdown_pct": drawdown_delta.median(),
        }
        weight_rows.append(row)
        print(
            f"  weight={weight:.2f}: Sharpe mean={row['mean_delta_sharpe']:+.3f}, "
            f"median={row['median_delta_sharpe']:+.3f}, "
            f"besser={row['pct_higher_sharpe']:.1%}, "
            f"Return median={row['median_delta_return_pct']:+.2f}%, "
            f"DD median={row['median_delta_max_drawdown_pct']:+.2f}pp"
        )

    btc_close = multi_ohlc["BTC/USDT"]["close"]
    regimes = classify_regimes(btc_close, config.ml).reindex(rule_summary.runs["start_ts"]).to_numpy()
    comparison["start_regime"] = regimes
    print("\n=== ML minus Regelstrategie nach Start-Regime ===")
    for regime, subset in comparison.groupby("start_regime", dropna=False):
        print(
            f"  {regime}: n={len(subset)}, "
            f"Sharpe median={subset['delta_sharpe'].median():+.3f}, "
            f"Return median={subset['delta_total_return_pct'].median():+.2f}%, "
            f"DD median={subset['delta_max_drawdown_pct'].median():+.2f}pp, "
            f"Sharpe besser={(subset['delta_sharpe'] > 0).mean():.1%}"
        )

    output_path = Path(config.monte_carlo.output_dir) / "ml_vs_rule_monte_carlo.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)
    print(f"\nPaarweise Ergebnisse: {output_path}")
    weight_path = Path(config.monte_carlo.output_dir) / "ml_weight_sweep.csv"
    pd.DataFrame(weight_rows).to_csv(weight_path, index=False)
    print(f"Gewichts-Sweep: {weight_path}")


if __name__ == "__main__":
    main()