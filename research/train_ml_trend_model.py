"""Entry point: train the GPU trend-continuation confirmation model.

Trains one TCN per symbol on raw (unsmoothed) intraday OHLCV plus cross-asset
macro context (S&P 500 / Nasdaq 100 / MSCI World / VIX), validated with
purged/embargoed walk-forward cross-validation, and persists the final model
under ``config.ml.model_dir`` for use by ``core/ml/inference.py``.

Run with:  uv run research/train_ml_trend_model.py
"""
from __future__ import annotations

from core.config import TradingBotConfig
from core.ml.train import run_training_pipeline


def main() -> None:
    config = TradingBotConfig()
    config.ml.enabled = True

    print("=== Trend-continuation ML pipeline ===")
    print(f"Timeframe: {config.ml.base_timeframe}, history: {config.ml.history_days}d")
    print(f"Macro symbols: {config.ml.macro_symbols}")
    print(f"Label horizon: {config.ml.label_horizon} bars, barrier: {config.ml.barrier_atr_multiple} * ATR")
    print(f"Purged walk-forward folds: {config.ml.n_splits}, embargo: {config.ml.embargo_fraction:.2%}")

    results = run_training_pipeline(config)

    print("\n=== Zusammenfassung: purged walk-forward OOS-Metriken ===")
    for symbol, result in results.items():
        print(f"\n{symbol}:")
        fold_metrics = result["fold_metrics"]
        if fold_metrics.empty:
            print("  keine validen Folds.")
            continue
        print(fold_metrics.to_string(index=False))
        print(f"  Mean OOS accuracy: {fold_metrics['accuracy'].mean():.4f}")
        print(f"  Mean OOS directional hit rate: {fold_metrics['directional_hit_rate'].mean():.4f}")
        print(f"  Mean OOS directional edge: {fold_metrics['directional_edge'].mean():.4f}")


if __name__ == "__main__":
    main()
