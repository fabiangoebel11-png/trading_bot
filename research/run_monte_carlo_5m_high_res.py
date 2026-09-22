"""Entry point: high-resolution Monte Carlo stress test on the exact live-execution
timeframe (5m), to validate the strategy's microdynamics rather than its 1h-smoothed
behavior.

Deliberately isolated from the production config: only *this script's own*
``TradingBotConfig`` instance is mutated. ``core/config.py`` defaults --
and therefore ``execution/live_trader.py`` and the regular 1h backtest/Monte
Carlo scripts -- are completely untouched.

Important: bar-based windows are time-scaled, not reused unchanged. ``TrendConfig``/
``RiskConfig`` windows (EMA/Donchian/ATR lookbacks) are *bar counts*, not fixed
time durations -- reusing the production 1h bar counts directly on 5m bars
would shrink every lookback to 1/12th of its intended wall-clock duration
(e.g. the 55-bar Donchian entry channel would span ~4.6h instead of ~55h),
turning the strategy into extreme overtrading on pure microstructure noise
rather than validating its actual trend logic at finer resolution. All
bar-based windows are therefore multiplied by 12 below (1h = 12 x 5m bars) so
every window covers the *same wall-clock duration* as the production 1h
config; only the sampling resolution changes, not the strategy's time horizon.

Matching the window *duration* alone is not enough for ATR, though: ATR
averages the true range *per bar*, and per-bar intra-bar volatility scales
with sqrt(time) (Brownian-motion scaling), not linearly. A 5m bar's true
range is therefore only ~1/sqrt(12) of an otherwise-comparable 1h bar's true
range even when both ATRs are computed over a matched 14h/24h window -- so a
1h-calibrated stop multiplier (``RiskConfig.atr_stop_multiplier``) applied to
a 5m ATR produces a stop distance ~sqrt(12) too tight, causing exactly the
immediate whipsaw stop-outs the -85% run showed. The fix multiplies the stop
multiplier by ``sqrt(12)`` (restores the 1h absolute stop distance) and
divides the vol-filter threshold ``TrendConfig.min_atr_pct`` by the same
factor (a 5m ATR% is likewise ~sqrt(12) smaller in calm-vs-choppy regimes, so
the unscaled 1h threshold would reject almost every 5m bar as "too quiet").

Run with:  uv run research/run_monte_carlo_5m_high_res.py
"""
from __future__ import annotations

import math

from core.config import TradingBotConfig
from core.data_loader import load_multi_asset_data
from core.funding import load_portfolio_funding
from core.monte_carlo import plot_monte_carlo_distribution, run_monte_carlo_stress_test

# 1h = 12 x 5m bars: scales bar-count windows so they keep the same wall-clock
# duration as the production 1h config instead of silently shrinking 12x.
_TIMEFRAME_SCALE = 12
# Square-root-of-time rule: intra-bar volatility (and thus ATR's absolute
# magnitude) scales with sqrt(bar duration), not bar duration itself.
_VOL_SCALE = math.sqrt(_TIMEFRAME_SCALE)


def main() -> None:
    config = TradingBotConfig()

    # --- High-resolution overrides (isolated to this script's config instance) ---
    config.data.base_timeframe = "5m"  # exact live-trader execution resolution
    config.data.history_days = 1095  # capped (~3 years) so 5m candle volume stays RAM-friendly
    config.monte_carlo.n_runs = 500  # more iterations now that the stop-loss/vol-filter scaling is fixed
    config.monte_carlo.output_dir = "data/monte_carlo_5m"  # own folder, never overwrites the 1h plots

    # We're back in the current market era (last ~3 years) with the live
    # trading product -- swap (perpetual futures) + funding-rate accounting
    # both stay exactly as in production, unlike the deep-history spot script.
    config.data.market_type = "swap"
    config.funding.enabled = True

    # --- Time-scale every bar-based strategy/risk window by 12x (see module
    # docstring): keeps the same *time duration* as the 1h production config,
    # only the sampling resolution changes.
    config.trend.fast_ma_window *= _TIMEFRAME_SCALE  # 20 -> 240
    config.trend.slow_ma_window *= _TIMEFRAME_SCALE  # 100 -> 1200
    config.trend.donchian_entry_window *= _TIMEFRAME_SCALE  # 55 -> 660
    config.trend.donchian_exit_window *= _TIMEFRAME_SCALE  # 20 -> 240
    config.trend.atr_window *= _TIMEFRAME_SCALE  # 14 -> 168, vol filter lookback
    config.risk.atr_window *= _TIMEFRAME_SCALE  # 24 -> 288, trailing-stop ATR lookback

    # --- sqrt(time) volatility scaling (see module docstring): a 5m ATR is
    # ~sqrt(12)x smaller in absolute terms than a duration-matched 1h ATR, so
    # widen the stop multiplier and loosen the vol-filter threshold by the
    # same factor to keep the same *effective* stop distance/selectivity.
    config.risk.atr_stop_multiplier *= _VOL_SCALE  # 3.0 -> ~10.39
    config.trend.min_atr_pct /= _VOL_SCALE  # 0.0015 -> ~0.000433

    multi_ohlc = load_multi_asset_data(config.data)
    funding_df = load_portfolio_funding(config.data, config.funding)

    print("\n=== High-Res Monte-Carlo-Stresstest (5m-Live-Timeframe) ===")
    print(
        f"  Zeit-skalierte Fenster (Faktor {_TIMEFRAME_SCALE}x): "
        f"EMA {config.trend.fast_ma_window}/{config.trend.slow_ma_window}, "
        f"Donchian {config.trend.donchian_entry_window}/{config.trend.donchian_exit_window}, "
        f"ATR (Trend) {config.trend.atr_window}, ATR (Risk) {config.risk.atr_window}"
    )
    print(
        f"  Vol-skaliert (sqrt({_TIMEFRAME_SCALE})={_VOL_SCALE:.3f}x), um das 1h-Risikoprofil auf 5m "
        f"abzubilden: atr_stop_multiplier={config.risk.atr_stop_multiplier:.3f}, "
        f"min_atr_pct={config.trend.min_atr_pct:.6f}"
    )
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

    path = plot_monte_carlo_distribution(summary, config, filename="monte_carlo_5m.png")
    print(f"\n  Verteilungs-Plot gespeichert: {path}")


if __name__ == "__main__":
    main()
