from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import SwingMLConfig
from core.ml.swing_data import build_swing_dataset
from core.ml.swing_model import combine_scores, horizon_opportunity_score
from core.risk import cap_model2_leverage


def test_model2_score_is_monotonic_and_bounded() -> None:
    assert combine_scores(94, 52) >= 80
    assert combine_scores(80, 40) <= combine_scores(90, 40)
    assert combine_scores(80, 40) <= combine_scores(80, 50)
    assert 0 <= combine_scores(-10, 200) <= 100


def test_model2_leverage_has_independent_absolute_cap() -> None:
    assert cap_model2_leverage(25, 100) == 10.0
    assert cap_model2_leverage(4, 5) == 4.0
    assert cap_model2_leverage(-1) == 0.0
    assert cap_model2_leverage(float("nan")) == 0.0


def test_model2_targets_include_all_horizons_and_excursions(tmp_path) -> None:
    index = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 130, len(index)), index=index)
    frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0})
    import core.ml.swing_data as swing_data

    original = swing_data._load
    swing_data._load = lambda cache_dir, asset, timeframe: frame if timeframe == "1d" else None
    try:
        dataset = build_swing_dataset(str(tmp_path), "QQQ", 20, [1, 3, 5, 10, 20], macro_symbols=[])
    finally:
        swing_data._load = original
    for horizon in (1, 3, 5, 10, 20):
        assert f"return_{horizon}d" in dataset.targets
        assert f"mfe_{horizon}d" in dataset.targets
        assert f"mae_{horizon}d" in dataset.targets
        assert f"duration_{horizon}d" in dataset.targets
    assert dataset.metadata["daily_only_samples"] == len(dataset.timestamps)
    assert np.all(dataset.branch_mask == 0)


def test_model2_config_defaults() -> None:
    config = SwingMLConfig()
    assert config.assets == ["QQQ", "SPY"]
    assert config.target_horizons == [1, 3, 5, 10, 20]
    assert config.max_model2_leverage == 10.0


def test_swing_horizon_score_changes_with_horizon_forecast() -> None:
    config = SwingMLConfig()
    short_horizon = horizon_opportunity_score(0.001, 0.01, -0.006, 0.3, config)
    longer_horizon = horizon_opportunity_score(0.012, 0.05, -0.02, 0.6, config)
    assert short_horizon != longer_horizon
    assert 0.0 <= short_horizon <= 100.0
    assert 0.0 <= longer_horizon <= 100.0