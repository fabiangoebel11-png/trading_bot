"""Train Model 2: QQQ/SPY daily-primary leveraged swing model."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from core.ml.swing_data import build_swing_dataset, discover_swing_history
from core.ml.swing_model import SwingModel
from train import load_config


def _print_coverage(asset: str, report: dict[str, dict[str, object]]) -> None:
    print(f"{asset}:")
    for timeframe in ("1d", "1h", "5m"):
        item = report[timeframe]
        print(f"  {timeframe}: earliest={item['earliest']} latest={item['latest']} rows={item['rows']}")


def train_asset(asset: str, config) -> dict[str, object]:
    swing = config.swing
    report = discover_swing_history(swing.market_cache_dir, asset)
    _print_coverage(asset, report)
    dataset = build_swing_dataset(swing.market_cache_dir, asset, swing.daily_lookback, swing.target_horizons, swing.macro_symbols)
    n = len(dataset.timestamps)
    test_start = int(n * (1 - swing.test_fraction))
    validation_start = int(n * (1 - swing.test_fraction - swing.validation_fraction))
    train_end = max(0, validation_start - swing.purge_days)
    validation_end = max(validation_start, test_start - swing.purge_days)
    if train_end <= 0 or validation_end <= validation_start:
        raise ValueError(f"Not enough chronological samples for purged split: {n}")
    print(f"{asset}: daily-only samples={dataset.metadata['daily_only_samples']}; full multi-timeframe samples={dataset.metadata['full_multitimeframe_samples']}; final={n}")
    print(f"{asset}: train={train_end} validation={validation_end - validation_start} test={n - test_start} purge={swing.purge_days} trading days")
    model = SwingModel(swing, dataset.X_daily.shape[1:], dataset.X_entry.shape[1])
    model.fit(dataset.X_daily[:train_end], dataset.X_entry[:train_end], dataset.branch_mask[:train_end], {key: value[:train_end] for key, value in dataset.targets.items()})
    if model.training_history:
        print(f"{asset}: Model 2 loss {model.training_history[0]['loss']:.6f} -> {model.training_history[-1]['loss']:.6f}")
    checkpoint = Path(swing.model_dir) / f"{asset.lower()}_swing.pt"
    model.save(checkpoint, {**dataset.metadata, "asset": asset, "asset_data_start": dataset.timestamps.min().isoformat(), "asset_data_end": dataset.timestamps.max().isoformat(), "feature_version": swing.feature_version, "target_version": swing.target_version, "lookback": swing.daily_lookback, "target_horizons": swing.target_horizons, "feature_count": len(dataset.feature_names), "train_samples": train_end, "validation_samples": validation_end - validation_start, "test_samples": n - test_start, "config": vars(swing), "max_model2_leverage": min(float(swing.max_model2_leverage), 10.0)})
    return {"asset": asset, "checkpoint": str(checkpoint), "samples": n, "feature_count": len(dataset.feature_names), "metadata": dataset.metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the independent QQQ/SPY swing model")
    parser.add_argument("--config", default="configs/training_swing.yaml")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if not args.train:
        parser.print_help()
        return
    config = load_config(args.config)
    random.seed(42)
    np.random.seed(42)
    print("=== MODEL 2 SWING PREPARATION ===")
    results = [train_asset(asset, config) for asset in config.swing.assets]
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()