"""Train Model 1A on BTC/ETH only."""
from __future__ import annotations

from core.ml.train import run_training_pipeline
from train import load_config


if __name__ == "__main__":
    config = load_config("configs/training_crypto_intraday.yaml")
    config.ml.enabled = True
    run_training_pipeline(config)
