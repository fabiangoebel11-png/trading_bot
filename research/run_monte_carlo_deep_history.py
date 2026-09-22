"""Entry point: Deep-History Monte Carlo stress test on SPOT data.

Binance's USDT-M perpetual *futures* (``core.config.DataConfig.market_type ==
"swap"``, the production/live-trading market) only exist since ~2019-09/11 --
that's a hard exchange-side data ceiling, not a config choice, and it's why
the regular ``run_monte_carlo_stress_test.py`` can never see the 2017 bull run
or the 2018 crypto winter. Binance *spot* BTC/USDT and ETH/USDT, however, have
traded since mid-2017, so this script isolates a one-off deep-history stress
test on spot candles to probe the strategy/risk-management across those extra
bear/bull regimes -- for research only.

Deliberately isolated from the production config:
- Only *this script's own* ``TradingBotConfig`` instance is mutated (market
  type, history length, macro lookback); ``core/config.py`` defaults --
  and therefore ``execution/live_trader.py`` and the regular backtest/Monte
  Carlo scripts -- are completely untouched.
- Spot markets have no funding rate (that's a perpetual-futures-only
  mechanism), so funding is force-disabled here and never fetched -- calling
  ``fetch_funding_rate_history`` against a spot symbol would either error or
  return nothing meaningful.
- Note: ``core/data_loader.py``'s on-disk OHLCV cache filename only encodes
  symbol/timeframe/``history_days``, not ``market_type`` -- there's no cache
  collision today since this script's 3500-day request differs from the
  production 2500-day one, but if those two numbers ever match, the cache
  would need to be namespaced by market type too.
- The strategy itself is fully price-based (Donchian breakout + EMA regime
  filter operates on OHLC closes/highs/lows) and simulates long/short
  directionally regardless of whether the underlying candles came from a
  spot or a perpetual market, so no strategy/risk code changes are needed --
  only the data source and funding handling differ from the production path.

Run with:  uv run research/run_monte_carlo_deep_history.py
"""
from __future__ import annotations

from core.config import TradingBotConfig
from core.data_loader import load_multi_asset_data
from core.monte_carlo import plot_monte_carlo_distribution, run_monte_carlo_stress_test


def main() -> None:
    config = TradingBotConfig()

    # --- Deep-history overrides (isolated to this script's config instance) ---
    config.data.market_type = "spot"  # spot BTC/ETH/SOL candles reach back to ~2017, unlike swap futures
    config.data.history_days = 3500  # ~9.6 years back from today, into the 2017 bull run
    config.macro.lookback_days = 3600  # macro (ES=F/VIX/URTH) must cover at least as much history

    # Spot markets have no funding rate mechanism -- disable it outright so
    # nothing downstream (core.funding) is ever fetched or applied for this run.
    config.funding.enabled = False

    multi_ohlc = load_multi_asset_data(config.data)

    print("\n=== Deep-History Monte-Carlo-Stresstest (SPOT-Daten, 2017er Bull + 2018er Krypto-Winter) ===")
    summary = run_monte_carlo_stress_test(multi_ohlc, config, funding_df=None)

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

    # Distinct filename so this research-only run never overwrites the
    # production run_monte_carlo_stress_test.py's plot.
    path = plot_monte_carlo_distribution(summary, config, filename="monte_carlo_deep_history.png")
    print(f"\n  Verteilungs-Plot gespeichert: {path}")


if __name__ == "__main__":
    main()
