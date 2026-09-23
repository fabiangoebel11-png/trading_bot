"""Train Model 1B on QQQ/SPY only."""
from __future__ import annotations

from core.ml.train import run_training_pipeline
from train import load_config


if __name__ == "__main__":
    config = load_config("configs/training_equity_intraday.yaml")
    config.ml.enabled = True
    if config.ml.entry_quality_mode != "payoff_volatility_compression":
        raise ValueError("Model 1B requires payoff_volatility_compression entry targets")
    config.ml.direction_loss_weight = 0.0
    print("Model 1B: direction is auxiliary; training continuous entry quality only")
    run_training_pipeline(config)
