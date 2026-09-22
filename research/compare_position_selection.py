"""Paired Monte Carlo comparison: today's all-active-signals baseline versus
the conviction-ranked top-N position-selection variant (PositionSelectionConfig).

Both runs use identical data, funding, Monte Carlo seed and sampled start
indices, so the resulting per-window deltas isolate the historical effect of
selecting only the strongest-scoring symbol(s) instead of trading every
active signal. This is the validation step required before
``PositionSelectionConfig.enabled`` may ever be set to True (see its
docstring in core/config.py) -- this script NEVER flips that flag itself.

Run with: python -m uv run research/compare_position_selection.py
"""
from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from core.config import TradingBotConfig
from core.data_loader import load_multi_asset_data
from core.funding import load_portfolio_funding
from core.monte_carlo import run_monte_carlo_stress_test


def main() -> None:
    config = TradingBotConfig()
    config.data.history_days = 2500
    multi_ohlc = load_multi_asset_data(config.data)
    funding_df = load_portfolio_funding(config.data, config.funding) if config.funding.enabled else None

    baseline_config = copy.deepcopy(config)
    baseline_config.selection.enabled = False
    print("\n=== Paired Monte Carlo: Baseline (alle aktiven Signale, inverse-Leverage-Gewichtung) ===")
    baseline_summary = run_monte_carlo_stress_test(multi_ohlc, baseline_config, funding_df=funding_df)

    metrics = ["sharpe", "total_return_pct", "max_drawdown_pct", "final_capital_usdt"]
    comparison = baseline_summary.runs[["start_idx", "start_ts", *metrics]].copy()
    for metric in metrics:
        comparison[f"baseline_{metric}"] = comparison.pop(metric)

    summary_rows = []
    for max_active_positions in (1, 2):
        variant_config = copy.deepcopy(config)
        variant_config.selection.enabled = True
        variant_config.selection.max_active_positions = max_active_positions
        print(f"\n=== Paired Monte Carlo: Top-{max_active_positions}-Auswahl (score-proportionale Gewichte) ===")
        variant_summary = run_monte_carlo_stress_test(multi_ohlc, variant_config, funding_df=funding_df)

        if not baseline_summary.runs["start_idx"].equals(variant_summary.runs["start_idx"]):
            raise RuntimeError(f"Monte-Carlo-Startfenster weichen ab fuer max_active_positions={max_active_positions}.")

        prefix = f"top{max_active_positions}"
        for metric in metrics:
            comparison[f"{prefix}_{metric}"] = variant_summary.runs[metric].to_numpy()
            comparison[f"delta_{prefix}_{metric}"] = comparison[f"{prefix}_{metric}"] - comparison[f"baseline_{metric}"]

        delta_sharpe = comparison[f"delta_{prefix}_sharpe"]
        delta_return = comparison[f"delta_{prefix}_total_return_pct"]
        delta_drawdown = comparison[f"delta_{prefix}_max_drawdown_pct"]
        row = {
            "max_active_positions": max_active_positions,
            "mean_delta_sharpe": delta_sharpe.mean(),
            "median_delta_sharpe": delta_sharpe.median(),
            "pct_windows_better_sharpe": (delta_sharpe > 0).mean(),
            "median_delta_return_pct": delta_return.median(),
            "median_delta_max_drawdown_pct": delta_drawdown.median(),
        }
        summary_rows.append(row)
        print(
            f"  Sharpe: mean {delta_sharpe.mean():+.3f}, median {delta_sharpe.median():+.3f}, "
            f"Fenster besser={row['pct_windows_better_sharpe']:.1%}"
        )
        print(
            f"  Return median {delta_return.median():+.2f}%, "
            f"Max-DD median {delta_drawdown.median():+.2f}pp (positiv = besser)"
        )

    output_dir = Path(config.monte_carlo.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "position_selection_vs_baseline_monte_carlo.csv"
    comparison.to_csv(comparison_path, index=False)
    summary_path = output_dir / "position_selection_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nPaarweise Ergebnisse: {comparison_path}")
    print(f"Zusammenfassung: {summary_path}")
    print(
        "\nHinweis: PositionSelectionConfig.enabled bleibt False, bis ein Mensch dieses "
        "Ergebnis manuell geprueft und die Config-Datei explizit geaendert hat -- dieses "
        "Skript setzt die Flag nie selbst."
    )


if __name__ == "__main__":
    main()
