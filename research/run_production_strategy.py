"""Entry point: loads data, runs a full-sample backtest and a walk-forward
validation for the BTC/ETH stat-arb pair, using the shared core/ library.

Run with:  uv run research/run_production_strategy.py
"""
from __future__ import annotations

from core.backtester import run_backtest
from core.config import TradingBotConfig
from core.data_loader import load_pair_data
from core.metrics import summarize_performance
from core.validation import run_walk_forward, summarize_folds


def main() -> None:
    config = TradingBotConfig()

    df = load_pair_data(config.data)

    print("\n=== Full-sample backtest (in-sample reference, expect optimistic bias) ===")
    full_result = run_backtest(df, config)
    periods_per_year = int((365 * 24 * 60) / (60 * 10))  # 10m candles
    full_summary = summarize_performance(
        full_result["strategy_return"], periods_per_year, full_result["trade"] * full_result["strategy_return"]
    )
    for key, value in full_summary.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n=== Walk-forward validation (out-of-sample, overfitting check) ===")
    try:
        folds = run_walk_forward(df, config)
    except ValueError as exc:
        print(f"  Skipped: {exc}")
        return

    fold_table = summarize_folds(folds)
    print(fold_table.to_string(index=False))
    print("\n  Mean OOS Sharpe:", fold_table["sharpe"].mean())
    print("  Mean OOS return %:", fold_table["total_return_pct"].mean())
    print("  Worst OOS max drawdown %:", fold_table["max_drawdown_pct"].min())


if __name__ == "__main__":
    main()
