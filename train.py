"""Config-driven ML research training entry point.

Run with ``python train.py --config configs/training.yaml``. This command only
loads historical data and writes research artifacts; it has no execution path.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import fields
from pathlib import Path

import numpy as np
try:
    import yaml
except ImportError:  # pragma: no cover - exercised on minimal environments
    yaml = None

from core.config import TradingBotConfig
from core.ml.train import load_ml_ohlc, live_safe_feature_columns, run_training_pipeline
from core.ml.providers import TwelveDataHistoricalProvider, asset_specs_from_config, prepare_asset_specs


def _apply_section(target: object, values: dict) -> None:
    allowed = {field.name for field in fields(target)}
    for name, value in values.items():
        if name not in allowed:
            raise ValueError(f"Unknown configuration field: {name}")
        setattr(target, name, value)


def load_config(path: str | Path) -> TradingBotConfig:
    text = Path(path).read_text(encoding="utf-8")
    payload = (yaml.safe_load(text) if yaml is not None else json.loads(text)) or {}
    config = TradingBotConfig()
    for section, values in payload.items():
        if section == "seed":
            config.ml.random_state = int(values)
        elif hasattr(config, section):
            target = getattr(config, section)
            if isinstance(target, (str, int, float, bool)) or not hasattr(target, "__dataclass_fields__"):
                setattr(config, section, values)
            elif isinstance(values, dict):
                _apply_section(target, values)
            else:
                raise ValueError(f"Configuration section {section!r} expects a mapping of dataclass fields, got {type(values).__name__}")
        else:
            raise ValueError(f"Unknown configuration section: {section}")
    return config


def print_training_summary(config: TradingBotConfig) -> None:
    ml = config.ml
    print("=== Training configuration ===")
    print(f"Assets: {ml.assets}")
    print(f"Timeframes: {ml.timeframes}; target: {ml.base_timeframe}")
    print(f"Sequence length: {ml.sequence_length}; prediction horizon: {ml.label_horizon}")
    from core.ml.date_validation import horizon_description

    print(f"Effective horizon: {horizon_description(ml.base_timeframe, ml.label_horizon)}")
    print(f"Model: TCN channels={ml.hidden_channels}, layers={ml.num_layers}, dropout={ml.dropout}")
    print(f"Batch size: {ml.batch_size}; epochs: {ml.max_epochs}; learning rate: {ml.learning_rate}")
    print(f"Train: asset-specific earliest available -> {ml.train_end}")
    print(f"Validation: {ml.validation_start} -> {ml.validation_end}")
    print(f"Test: {ml.test_start} -> {ml.test_end}")
    print(f"Calibration: {ml.calibration_method}; baselines: {ml.baseline_logistic_enabled}/{ml.baseline_gradient_boosting_enabled}")
    print(f"Score thresholds: {ml.score_thresholds}")
    print(f"Feature contract: live_safe_local_v1 ({len(live_safe_feature_columns(ml))} columns; local OHLCV/HTF only; <=5s live budget)")
    print("Training samples and class distribution are printed after each configured dataset is prepared.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the leakage-safe TCN research model")
    parser.add_argument("--config", default="configs/training.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-data", action="store_true", help="explicitly fetch/cache configured historical data")
    mode.add_argument("--train", action="store_true", help="explicitly start model training")
    args = parser.parse_args()
    if not args.prepare_data and not args.train:
        parser.print_help()
        return
    config = load_config(args.config)
    seed = int(config.ml.random_state)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    if args.prepare_data:
        specs = asset_specs_from_config(config)
        estimated_requests = sum(
            TwelveDataHistoricalProvider.estimate_requests(spec.history_days, spec.timeframe)
            for spec in specs
            if spec.provider == "twelve_data"
        )
        print(f"Estimated Twelve Data API requests: {estimated_requests}")
        print(f"Configured daily limit: {config.data.twelve_data.get('max_requests_per_day', 800)}")
        _, reports = prepare_asset_specs(
            specs,
            cache_root=config.ml.market_cache_dir,
            cache_dir=config.data.cache_dir,
            exchange_id=config.data.exchange_id,
            default_market_type=config.data.market_type,
            twelve_data_settings=config.data.twelve_data,
        )
        if reports:
            import pandas as pd

            print(pd.DataFrame(reports).to_string(index=False))
        complete = sum(report.get("status") in {"success", "complete"} for report in reports)
        partial = sum(report.get("status") == "partial" for report in reports)
        unavailable = sum(report.get("status") == "unavailable" for report in reports)
        optional = sum(bool(report.get("optional")) for report in reports)
        print(f"Prepared datasets: {len(reports)}")
        print(f"Complete: {complete}; Partial: {partial}; Unavailable: {unavailable}; Optional: {optional}")

        required_primary = {"NASDAQ100_PROXY": "QQQ", "SP500_PROXY": "SPY", "BTC/USDT": "BTC", "ETH/USDT": "ETH"}
        primary_ready = {}
        for asset, label in required_primary.items():
            match = next((report for report in reports if report.get("asset") == asset and report.get("timeframe") == "5m"), None)
            primary_ready[label] = bool(match and int(match.get("candles", 0) or 0) > 0 and match.get("status") != "unavailable")
        derived_ready = all(
            any(report.get("asset") == asset and report.get("timeframe") == timeframe and report.get("provider") == "local_derived" and int(report.get("candles", 0) or 0) > 0 for report in reports)
            for asset in ("NASDAQ100_PROXY", "SP500_PROXY", "BTC/USDT", "ETH/USDT")
            for timeframe in ("15m", "1h", "4h")
            if any(report.get("asset") == asset and report.get("timeframe") == timeframe for report in reports)
        )
        print("PRIMARY 5m:")
        for label, ready in primary_ready.items():
            print(f"{label:<8} {'READY' if ready else 'UNAVAILABLE'}")
        print("DERIVED:")
        for timeframe in ("15m", "1h", "4h"):
            ready = all(
                any(report.get("asset") == asset and report.get("timeframe") == timeframe and report.get("provider") == "local_derived" and int(report.get("candles", 0) or 0) > 0 for report in reports)
                for asset in required_primary
            )
            print(f"{timeframe:<8} {'READY' if ready else 'UNAVAILABLE'}")
        context_ready = all(
            any(report.get("asset") == asset and int(report.get("candles", 0) or 0) > 0 for report in reports)
            for asset in ("VIX", "US10Y", "EURUSD", "GOLD", "WTI")
        )
        print(f"TRAINING_READY = {'YES' if all(primary_ready.values()) and derived_ready and context_ready else 'NO'}")
        return

    print_training_summary(config)
    print(f"Training historical ML pipeline from {args.config}")
    results = run_training_pipeline(config)
    for symbol, result in results.items():
        print(f"{symbol}: {result['artifact_path']}")


if __name__ == "__main__":
    main()