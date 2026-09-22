"""Tests for the ML shadow-mode audit log (core/ml/shadow.py)."""
from __future__ import annotations

import pandas as pd
import pytest

from core.config import ExecutionConfig
from core.ml.shadow import SHADOW_LOG_COLUMNS, ShadowLogger, build_shadow_log


def _signals(index: pd.DatetimeIndex, position: list[float], close: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"position": position, "close": close, "atr": 1.0}, index=index)


def test_build_shadow_log_matches_columns_and_is_causal() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="1h")
    rule = _signals(index, [0, 1, 1, 0, -1], [100, 101, 102, 101, 99])
    ml = _signals(index, [0, 0.5, 1, 0, -0.5], [100, 101, 102, 101, 99])
    ml["ml_confidence"] = [0.0, 0.2, 0.9, 0.0, -0.4]
    ml["ml_regime"] = ["sideways"] * 5

    log = build_shadow_log("BTC/USDT", rule, ml, ExecutionConfig())

    expected_columns = set(SHADOW_LOG_COLUMNS) - {"timestamp"}
    assert expected_columns <= set(log.columns) | {"symbol"}
    assert "symbol" in log.columns
    # First bar has no prior position -> zero pnl/cost (causal, no look-ahead).
    assert log["rule_pnl"].iloc[0] == 0.0
    assert log["ml_pnl"].iloc[0] == 0.0
    # A position change incurs cost on both sides independently.
    assert log["rule_cost"].iloc[1] > 0
    assert log["ml_cost"].iloc[1] > 0


def test_shadow_logger_appends_persistent_rows(tmp_path) -> None:
    path = tmp_path / "shadow.csv"
    logger = ShadowLogger(str(path), ExecutionConfig())

    logger.log_bar("BTC/USDT", pd.Timestamp("2024-01-01T00:00"), 0.0, 0.0, 0.0, "sideways", 1.0, 100.0)
    logger.log_bar("BTC/USDT", pd.Timestamp("2024-01-01T01:00"), 1.0, 0.5, 0.3, "strong_trend", 1.1, 102.0)

    assert path.exists()
    logged = logger.read_all()
    assert len(logged) == 2
    assert list(logged.columns) == SHADOW_LOG_COLUMNS
    # Second row: prior position was flat (0), so no realized pnl/cost yet
    # from the *position held* into this bar, but taking the new position
    # itself is recorded as a cost.
    second = logged.iloc[1]
    assert second["rule_cost"] == pytest.approx(1.0 * ShadowLogger(str(path), ExecutionConfig())._fee)
    assert second["ml_cost"] == pytest.approx(0.5 * ShadowLogger(str(path), ExecutionConfig())._fee)


def test_shadow_logger_never_touches_production_flag(tmp_path) -> None:
    """Shadow logging is purely observational -- writing rows must have no
    side effect on any config/production state."""
    path = tmp_path / "shadow.csv"
    execution_config = ExecutionConfig()
    logger = ShadowLogger(str(path), execution_config)
    logger.log_bar("ETH/USDT", pd.Timestamp("2024-01-01"), 1.0, 1.0, 0.5, "strong_trend", 2.0, 200.0)
    assert not hasattr(execution_config, "production_enabled")
