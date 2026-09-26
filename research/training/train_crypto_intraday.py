"""Train Model 1A on BTC/ETH only."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ml.train import run_training_pipeline
from train import load_config


if __name__ == "__main__":
    config = load_config(PROJECT_ROOT / "configs" / "training_crypto_intraday.yaml")
    config.ml.enabled = True
    run_training_pipeline(config)
