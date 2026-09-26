"""Train Model 1B on QQQ/SPY only."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.ml.train import run_training_pipeline
from train import load_config


if __name__ == "__main__":
    config = load_config(PROJECT_ROOT / "configs" / "training_equity_intraday.yaml")
    config.ml.enabled = True
    if config.ml.entry_quality_mode != "payoff_volatility_compression":
        raise ValueError("Model 1B requires payoff_volatility_compression entry targets")
    config.ml.direction_loss_weight = 0.0
    print("Model 1B: direction is auxiliary; training continuous entry quality only")
    run_training_pipeline(config)
