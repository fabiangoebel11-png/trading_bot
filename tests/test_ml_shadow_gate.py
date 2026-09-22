"""Tests that ML confirmation runs only in shadow (audit) mode by default and
never affects real position sizing unless ``ml.production_enabled`` is
manually set to True (core/config.py, execution/live_trader.py)."""
from __future__ import annotations

import pandas as pd

from core.config import TradingBotConfig
from execution.live_trader import LiveTrader


def _config(tmp_path) -> TradingBotConfig:
    config = TradingBotConfig()
    config.execution.state_path = str(tmp_path / "live_state.json")
    config.execution.audit_log_path = str(tmp_path / "live_audit.jsonl")
    config.ml.shadow_log_path = str(tmp_path / "shadow.csv")
    config.ml.portfolio_shadow_log_path = str(tmp_path / "portfolio_shadow.csv")
    config.funding.enabled = False
    return config


def _fake_ml_signals(raw_signals: pd.DataFrame) -> pd.DataFrame:
    out = raw_signals.copy()
    # Flip every non-zero rule position to a clearly different fractional
    # size, so a test can tell which side actually drove `_symbol_candidate`.
    out["position"] = raw_signals["position"] * 0.25
    out["ml_confidence"] = 0.9
    out["ml_regime"] = "strong_trend"
    return out


def test_ml_enabled_without_production_flag_does_not_change_target_position(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    config = _config(tmp_path)
    config.ml.enabled = True
    config.ml.production_enabled = False  # default -- must stay this way unless set manually
    trader = LiveTrader(config)

    assert trader._shadow_logger is not None

    calls = []

    def fake_apply_ml_confirmation(symbol, signals):
        calls.append(symbol)
        return _fake_ml_signals(signals)

    monkeypatch.setattr(trader, "_apply_ml_confirmation", fake_apply_ml_confirmation)

    index = pd.date_range("2024-01-01", periods=60, freq="1h")
    ohlc = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": [100.0 + i * 0.5 for i in range(60)],
            "volume": 1.0,
        },
        index=index,
    )
    monkeypatch.setattr(trader, "_load_recent_ohlc", lambda symbol: ohlc)

    candidate = trader._symbol_candidate("BTC/USDT")

    assert calls == ["BTC/USDT"]  # shadow ML was computed for audit purposes
    assert candidate is not None
    # Staged but not yet persisted until the poll's shared-wallet leverage
    # pass flushes it (mirrors how poll_once calls this after building every
    # symbol's candidate).
    assert "BTC/USDT" in trader._pending_shadow_rows
    trader._flush_shadow_log({"BTC/USDT": candidate}, cold_start_scale=1.0)
    logged = trader._shadow_logger.read_all()
    assert len(logged) == 1
    assert logged.iloc[0]["ml_confidence"] == 0.9


def test_ml_production_enabled_uses_ml_confirmed_signal(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    config = _config(tmp_path)
    config.ml.enabled = True
    config.ml.production_enabled = True  # explicit manual opt-in
    trader = LiveTrader(config)

    def fake_apply_ml_confirmation(symbol, signals):
        out = signals.copy()
        out["position"] = 0.0  # ML forces flat -- should now win over the raw rule signal
        out["ml_confidence"] = -0.9
        out["ml_regime"] = "crash"
        return out

    monkeypatch.setattr(trader, "_apply_ml_confirmation", fake_apply_ml_confirmation)

    index = pd.date_range("2024-01-01", periods=60, freq="1h")
    ohlc = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": [100.0 + i * 0.5 for i in range(60)],
            "volume": 1.0,
        },
        index=index,
    )
    monkeypatch.setattr(trader, "_load_recent_ohlc", lambda symbol: ohlc)

    candidate = trader._symbol_candidate("BTC/USDT")

    assert candidate is not None
    assert candidate["target_position"] == 0  # ML's flat call was actually applied


def test_ml_disabled_never_instantiates_shadow_logger(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    config = _config(tmp_path)
    config.ml.enabled = False
    trader = LiveTrader(config)
    assert trader._shadow_logger is None
