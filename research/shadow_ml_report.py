"""Shadow-mode comparison report: rule strategy vs. shadow-ML on identical
time windows.

Two sections are produced:

1. **Historical illustration**, via ``core.backtester.run_rule_vs_ml_backtest``:
   rule vs. ML shadow runs go through the exact same macro gate, ATR stop,
   cooldown, execution latency, funding cost and shared-wallet leverage logic
   -- the only difference between them is the ML confirmation block itself
   (using the leak-free, purged-walk-forward OOS confidence already produced
   by ``research/train_ml_trend_model.py``). Results are aggregated into
   non-overlapping chronological windows for comparison. Still illustrative
   only, not a substitute for a genuine forward holdout: it reuses historical
   data the model's final fit has already seen in some fold.

2. **Production holdout-gate evaluation**, PORTFOLIO-level (real money trades
   as ONE shared wallet, not N independent per-symbol books): uses the live
   ``ml.portfolio_shadow_log_path`` (see ``core/ml/shadow.py:
   PortfolioShadowSimulator``), where rule and ML each maintain their OWN
   hypothetical shared-wallet capital/cold-start/absolute-cap trajectory
   (never the real account's fetched equity) and funding cost is included via
   a periodic funding-rate fetch. This is the ONLY section whose result is
   valid evidence for ``core/ml/promotion.py: evaluate_promotion``. Only bars
   after every traded symbol's model was frozen (``max`` of all
   ``training_cutoff`` values) are used. If insufficient live shadow data has
   accumulated yet, this section reports how much more is needed and does not
   print a promotion recommendation.

This script NEVER sets ``ml.production_enabled`` -- promotion is manual-only
(see ``core/config.py: TrendMLConfig.production_enabled``).

Run with: python -m uv run research/shadow_ml_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.backtester import _periods_per_year, run_rule_vs_ml_backtest
from core.config import TradingBotConfig
from core.data_loader import load_multi_asset_data
from core.funding import load_portfolio_funding
from core.ml.holdout import paired_window_metrics
from core.ml.promotion import PromotionCriteria, evaluate_promotion
from core.ml.shadow import build_shadow_log_from_backtest


def _historical_shadow_log(config: TradingBotConfig) -> tuple[pd.DataFrame, int]:
    multi_ohlc = load_multi_asset_data(config.data)
    funding_df = load_portfolio_funding(config.data, config.funding) if config.funding.enabled else None
    periods_per_year = _periods_per_year(config.data.resample_to or config.data.base_timeframe)

    portfolio_rule, portfolio_ml = run_rule_vs_ml_backtest(multi_ohlc, config, funding_df=funding_df)
    per_symbol_rule = portfolio_rule.attrs["per_symbol"]
    per_symbol_ml = portfolio_ml.attrs["per_symbol"]

    logs = []
    for symbol in config.data.symbols:
        if symbol not in per_symbol_rule or symbol not in per_symbol_ml:
            continue
        logs.append(build_shadow_log_from_backtest(symbol, per_symbol_rule[symbol], per_symbol_ml[symbol]))
    if not logs:
        return pd.DataFrame(), periods_per_year
    return pd.concat(logs), periods_per_year


def _report_historical(config: TradingBotConfig, window_days: int) -> None:
    print("\n=== 1) Historische Illustration: Regelstrategie vs. Shadow-ML (leak-freie OOS-Konfidenz) ===")
    combined, periods_per_year = _historical_shadow_log(config)
    if combined.empty:
        print("  Keine Symbole mit trainiertem Modell + OOS-Konfidenz gefunden -- Report uebersprungen.")
        return

    for symbol, log in combined.groupby("symbol"):
        rule_returns = log["rule_pnl"]
        ml_returns = log["ml_pnl"]
        holdout_start = log.index[0]  # entire available history, purely illustrative
        windows = paired_window_metrics(rule_returns, ml_returns, holdout_start, periods_per_year, window_days)
        if windows.empty:
            print(f"  [{symbol}] zu wenig Historie fuer ein vollstaendiges {window_days}-Tage-Fenster.")
            continue
        print(
            f"  [{symbol}] {len(windows)} Fenster: "
            f"Sharpe-Delta median={windows['delta_sharpe'].median():+.3f}, "
            f"Return-Delta median={windows['delta_return_pct'].median():+.2f}pp, "
            f"DD-Delta median={windows['delta_max_drawdown_pct'].median():+.2f}pp, "
            f"Fenster verbessert={((windows['delta_sharpe'] > 0).mean()):.1%}"
        )

    output_dir = Path(config.monte_carlo.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "shadow_ml_historical_comparison.csv"
    combined.to_csv(output_path)
    print(f"  Bar-Level-Vergleich gespeichert: {output_path}")


def _training_cutoffs(config: TradingBotConfig) -> dict[str, pd.Timestamp]:
    cutoffs = {}
    model_dir = Path(config.ml.model_dir)
    for symbol in config.data.symbols:
        safe_symbol = symbol.replace("/", "-")
        meta_path = model_dir / f"{safe_symbol}_tcn_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        cutoff = meta.get("training_cutoff")
        if cutoff:
            cutoffs[symbol] = pd.Timestamp(cutoff)
    return cutoffs


def _report_production_holdout(config: TradingBotConfig) -> None:
    print("\n=== 2) Portfolio-weites Produktions-Holdout-Gate (Shared-Wallet-Shadow-Simulation) ===")
    log_path = Path(config.ml.portfolio_shadow_log_path)
    if not log_path.exists():
        print(
            f"  Kein Portfolio-Shadow-Log unter {log_path} gefunden -- Live-Trader muss erst mit "
            "ml.enabled=True (production_enabled=False) im Shadow-Modus laufen, bevor eine "
            "Promotion-Entscheidung moeglich ist."
        )
        return

    portfolio_log = pd.read_csv(log_path, parse_dates=["timestamp"]).set_index("timestamp")
    cutoffs = _training_cutoffs(config)
    if not cutoffs:
        print("  Keine trainierten Modelle mit training_cutoff gefunden -- Promotion nicht moeglich.")
        return
    # Conservative: a portfolio bar is only genuinely "never seen" evidence
    # for EVERY symbol in the shared wallet once ALL of their models have
    # been frozen, so use the latest (max) training_cutoff across symbols.
    holdout_start = max(cutoffs.values())
    periods_per_year = _periods_per_year(config.data.resample_to or config.data.base_timeframe)
    criteria = PromotionCriteria()

    result = evaluate_promotion(
        portfolio_log["rule_return"], portfolio_log["ml_return"], holdout_start, periods_per_year, criteria
    )
    print(f"  Holdout-Start = max(training_cutoff ueber alle Symbole) = {holdout_start.isoformat()}")
    for line in result.summary().splitlines():
        print(f"    {line}")
    if not result.passed:
        print("    -> ml.production_enabled bleibt False. Keine automatische Aenderung vorgenommen.")


def main() -> None:
    config = TradingBotConfig()
    config.ml.enabled = True  # only to enable feature/OOS-confidence loading for this read-only report
    print(
        "Hinweis: dieser Report setzt niemals ml.production_enabled -- Promotion bleibt "
        "ausschliesslich eine manuelle Entscheidung nach bestandenem Holdout-Gate."
    )
    _report_historical(config, window_days=30)
    _report_production_holdout(config)


if __name__ == "__main__":
    main()
