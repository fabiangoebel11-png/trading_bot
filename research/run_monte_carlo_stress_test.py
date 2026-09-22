"""Entry point: full-history backtest + Monte Carlo random-start stress test.

Run with:  uv run research/run_monte_carlo_stress_test.py
"""
from __future__ import annotations

from core.config import TradingBotConfig
from core.data_loader import load_multi_asset_data
from core.funding import load_portfolio_funding
from core.monte_carlo import plot_monte_carlo_distribution, run_monte_carlo_stress_test

def main() -> None:
    config = TradingBotConfig()
    
    # 1. Daten auf ca. 5,5 Jahre erweitern, um mehr Marktphasen zu testen
    config.data.history_days = 2500
    
    # ML remains disabled for the production candidate: the paired OOS
    # comparison did not establish a robust Sharpe improvement. Experimental
    # ML variants are evaluated only through compare_monte_carlo_ml.py.

    multi_ohlc = load_multi_asset_data(config.data)
    funding_df = load_portfolio_funding(config.data, config.funding) if config.funding.enabled else None

    print("\n=== Monte-Carlo-Stresstest: zufällige Start-Fenster über die volle Historie ===")
    summary = run_monte_carlo_stress_test(multi_ohlc, config, funding_df=funding_df)

    print(f"  Anzahl Läufe:              {len(summary.runs)}")
    print(f"  Fenstergröße:              {config.monte_carlo.window_days} Tage")
    print(f"  Mittlere Sharpe-Ratio:     {summary.mean_sharpe:+.3f} (std={summary.std_sharpe:.3f})")
    print(f"  Median Sharpe-Ratio:       {summary.median_sharpe:+.3f}  <-- Erfolgskriterium: > 0")
    print(f"  Mittlere Gesamtrendite:    {summary.mean_total_return_pct:+.2f}%")
    print(f"  Anteil profitabler Fenster:{summary.pct_profitable:.1f}%")
    print(f"  t-Stat (Sharpe > 0):       {summary.t_stat_sharpe:+.2f} (p={summary.p_value_sharpe:.4f})")
    print(
        "  Hinweis: Fenster überlappen sich stark (Block-Bootstrap, keine unabhängigen "
        "Stichproben) -- p-Wert ist ein Anhaltspunkt, kein strenger Signifikanztest."
    )
    print(f"\n  --- Money Management (Startkapital {summary.initial_capital_usdt:,.0f} USDT) ---")
    print(f"  Mittleres Endkapital:      {summary.mean_final_capital_usdt:,.0f} USDT")
    print(f"  Median Endkapital:         {summary.median_final_capital_usdt:,.0f} USDT")
    print(f"  Mittlerer Max Drawdown:    {summary.mean_max_drawdown_usdt:,.0f} USDT")
    print(f"  Worst-Case Max Drawdown:   {summary.worst_case_max_drawdown_usdt:,.0f} USDT")

    path = plot_monte_carlo_distribution(summary, config)
    print(f"\n  Verteilungs-Plot gespeichert: {path}")


if __name__ == "__main__":
    main()