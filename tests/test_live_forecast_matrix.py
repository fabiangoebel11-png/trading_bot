from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

import live_daemon
import state_db
from core.ml.assistant_forecasts import PUBLIC_HORIZONS, direct_swing_forecasts, direct_tcn_forecasts


def _tcn_output() -> dict:
    return {
        "probabilities": (0.2, 0.1, 0.7), "expected_return": 0.01, "expected_mfe": 0.02,
        "expected_mae": -0.01, "expected_duration": 4.0, "opportunity_score": 65.0,
        "direction": "LONG",
    }


def test_complete_four_asset_matrix_persists_all_public_horizons(tmp_path) -> None:
    database = tmp_path / "state.db"
    state_db.init_db(database)
    crypto = direct_tcn_forecasts(asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1", timestamp="2026-01-01T00:00:00+00:00", outputs={1: _tcn_output(), 4: _tcn_output(), 8: _tcn_output(), 12: _tcn_output(), 24: _tcn_output()})
    equity = direct_swing_forecasts(asset="SPY", model="swing", model_version="v1", timestamp="2026-01-01T00:00:00+00:00", outputs={day: {"expected_return": 0.01, "expected_mfe": 0.02, "expected_mae": -0.01, "expected_duration": float(day), "opportunity_score": 60.0} for day in (1, 3, 5, 10, 20)})
    with state_db.connect(database) as connection:
        for asset, forecasts in (("BTC/USDT", crypto), ("ETH/USDT", [item.__class__(**{**item.as_dict(), "asset": "ETH/USDT"}) for item in crypto]), ("SPY", equity), ("QQQ", [item.__class__(**{**item.as_dict(), "asset": "QQQ"}) for item in equity])):
            state_db.replace_assistant_forecasts(connection, asset, [item.as_dict() for item in forecasts])
        rows = connection.execute("SELECT asset, horizon FROM assistant_forecasts").fetchall()

    assert len(rows) == 36
    for asset in ("BTC/USDT", "ETH/USDT", "SPY", "QQQ"):
        assert {row["horizon"] for row in rows if row["asset"] == asset} == set(PUBLIC_HORIZONS)


def test_swing_forecast_is_returned_below_alert_threshold(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "spy_swing.pt"
    checkpoint.touch()
    timestamps = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    dataset = SimpleNamespace(
        timestamps=timestamps,
        X_daily=np.zeros((2, 3), dtype=np.float32), X_entry=np.zeros((2, 3), dtype=np.float32),
        branch_mask=np.zeros(2, dtype=np.float32), metadata={"daily_data_end": timestamps[-1].isoformat()},
    )

    class FakeModel:
        def predict(self, *_args):
            return {
                "returns": np.full((1, 5), 0.001), "mfe": np.array([0.002]), "mae": np.array([-0.002]),
                "entry": np.array([-1.0]), "duration": np.full((1, 5), 1.0),
            }

    daemon = live_daemon.LiveDaemon.__new__(live_daemon.LiveDaemon)
    daemon.swing_cfg = SimpleNamespace(swing=SimpleNamespace(model_dir=str(tmp_path), market_cache_dir="unused", daily_lookback=90, target_horizons=[1, 3, 5, 10, 20], macro_symbols=[], swing_score_weight=0.7, entry_score_weight=0.3))
    daemon._swing_models = {}
    daemon._notified_swing = set()
    daemon._pending_notifications = []
    daemon._write_snapshot = lambda *_args: None
    daemon._persist_strategy_profiles = lambda *_args: None
    monkeypatch.setattr(live_daemon, "build_swing_dataset", lambda *_args: dataset)
    monkeypatch.setattr(live_daemon, "load_swing_model", lambda *_args: FakeModel())

    result = daemon._refresh_swing(SimpleNamespace(), "SPY", pd.Timestamp("2026-01-02", tz="UTC").to_pydatetime())

    assert result is not None
    _, forecasts = result
    assert set(forecasts) == {1, 3, 5, 10, 20}