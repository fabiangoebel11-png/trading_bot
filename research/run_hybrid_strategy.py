"""Entry point for the hybrid stat-arb + ML strategy: walk-forward validation with
per-fold model retraining, run alongside the pure rule-based baseline for comparison.

Run with:  uv run research/run_hybrid_strategy.py
"""
from __future__ import annotations

from core.config import TradingBotConfig
from core.data_loader import load_pair_data
from core.validation import run_walk_forward, run_walk_forward_hybrid, summarize_folds


def main() -> None:
    config = TradingBotConfig()
    # Switch to config.ml.model_type = "lstm" to train the GPU LSTM variant instead
    # (requires `uv sync --extra ml` for torch/CUDA on the RTX 3070).

    df = load_pair_data(config.data)

    print("\n=== Baseline: rule-based stat-arb, walk-forward (no ML gate) ===")
    baseline_folds = run_walk_forward(df, config)
    baseline_table = summarize_folds(baseline_folds)
    print(baseline_table.to_string(index=False))
    print("  Mean OOS Sharpe:", baseline_table["sharpe"].mean())

    print("\n=== Hybrid: stat-arb + ML reversion gate, walk-forward (model retrained per fold) ===")
    hybrid_folds = run_walk_forward_hybrid(df, config)
    hybrid_table = summarize_folds(hybrid_folds)
    print(hybrid_table.to_string(index=False))
    print("  Mean OOS Sharpe:", hybrid_table["sharpe"].mean())
    print("  Mean OOS return %:", hybrid_table["total_return_pct"].mean())
    print("  Worst OOS max drawdown %:", hybrid_table["max_drawdown_pct"].min())

    print("\n=== Comparison ===")
    print(f"  Baseline mean Sharpe:  {baseline_table['sharpe'].mean():+.3f}")
    print(f"  Hybrid mean Sharpe:    {hybrid_table['sharpe'].mean():+.3f}")
    print(f"  Baseline mean trades:  {baseline_table['n_periods'].mean():.0f} periods/fold")


if __name__ == "__main__":
    main()
