from datetime import timedelta

import pandas as pd
import pytest

from model_schedule import GUI_REFRESH_SECONDS, get_task_schedule


def test_crypto_schedule_separates_closed_candle_inference_from_gui_refresh() -> None:
    schedule = get_task_schedule("BTC/USDT", "1h")
    assert schedule.gui_refresh_seconds == GUI_REFRESH_SECONDS == 15
    assert "Once per new closed candle" in schedule.model_inference
    assert schedule.is_stale(pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T01:16:00Z"))
    assert not schedule.is_stale(pd.Timestamp("2026-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T01:15:00Z"))


def test_equity_schedule_declares_market_hours_and_rejects_unknown_timeframe() -> None:
    schedule = get_task_schedule("QQQ", "15m")
    assert "regular session" in schedule.market_hours
    assert schedule.expected_latency == timedelta(minutes=5)
    with pytest.raises(ValueError, match="No closed-candle cadence"):
        get_task_schedule("QQQ", "2m")