"""Manual Trade-Quality research command; never runs implicitly.

Examples:
    python -m research.train_trade_quality --config configs/training_trade_quality.yaml --evaluate
    python -m research.train_trade_quality --config configs/training_trade_quality.yaml --train
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from core.config import TradeQualityConfig
from core.ml.trade_quality import (
    TRADE_QUALITY_FEATURES,
    build_trade_quality_dataset,
    fit_trade_quality_model,
)
from core.strategy import generate_trend_signals
from decision_pipeline import load_cached_ohlcv
from train import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only setup Trade-Quality HGB pipeline")
    parser.add_argument("--config", default="configs/training_trade_quality.yaml")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evaluate", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config).trade_quality
    frame = load_cached_ohlcv("BTC/USDT", config.timeframe, "data")
    if frame is None:
        raise FileNotFoundError(f"missing local cache for BTC/USDT {config.timeframe}")
    base = load_config(args.config)
    signals = generate_trend_signals(frame, base.trend)
    dataset = build_trade_quality_dataset(frame, signals, config)
    if dataset.empty:
        raise RuntimeError("trade-quality dataset is empty")
    report = fit_trade_quality_model(dataset, config)
    output_dir = Path(config.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{config.model_id}_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(dataset), "report": report, "mode": "evaluate" if args.evaluate else "train"}, indent=2))
    if args.train:
        from sklearn.ensemble import HistGradientBoostingClassifier

        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, random_state=42)
        model.fit(dataset.loc[:, list(TRADE_QUALITY_FEATURES)], dataset["label"].astype(int))
        with (output_dir / f"{config.model_id}.pkl").open("wb") as handle:
            pickle.dump(model, handle)
        (output_dir / f"{config.model_id}.json").write_text(json.dumps({**report, "trained_rows": len(dataset)}, indent=2), encoding="utf-8")
        print(f"Wrote research artifact to {output_dir}")


if __name__ == "__main__":
    main()
