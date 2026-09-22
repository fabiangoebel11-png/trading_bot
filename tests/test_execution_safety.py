from __future__ import annotations

import json

from core.config import TradingBotConfig
from execution.exchange_client import OrderResult
from execution.live_trader import LiveTrader


def _config(tmp_path) -> TradingBotConfig:
    config = TradingBotConfig()
    config.execution.state_path = str(tmp_path / "live_state.json")
    config.execution.audit_log_path = str(tmp_path / "live_audit.jsonl")
    return config


def test_account_loss_halts_and_persists_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    trader = LiveTrader(_config(tmp_path))

    assert trader._evaluate_account_risk(500.0)
    assert not trader._evaluate_account_risk(480.0)
    assert trader.safety_state.data["entries_halted"] is True


def test_reconciliation_accepts_exchange_stop_and_blocks_unknown_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    trader = LiveTrader(_config(tmp_path))

    class FakeClient:
        dry_run = False

        def __init__(self) -> None:
            self.orders = [{"id": "stop-1", "type": "STOP_MARKET", "reduceOnly": True}]

        def fetch_signed_position_amount(self, symbol: str, strict: bool = False) -> float:
            return 0.4 if symbol == "BTC/USDT" else 0.0

        def fetch_open_orders(self, symbol: str, strict: bool = False) -> list[dict]:
            return self.orders if symbol == "BTC/USDT" else []

    trader.client = FakeClient()
    trader.current_position_amount["BTC/USDT"] = 1.0
    trader.current_positions["BTC/USDT"] = 1

    assert trader._reconcile_exchange_state("partial_fill")
    assert trader.current_position_amount["BTC/USDT"] == 0.4
    assert trader.active_stop_orders["BTC/USDT"] == "stop-1"
    assert not trader._entries_halted

    trader.client.orders.append({"id": "unexpected-1", "type": "limit", "reduceOnly": False})
    assert trader._reconcile_exchange_state("unexpected_order")
    assert trader._entries_halted


def test_unknown_order_blocks_new_entries_and_is_audited(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    config = _config(tmp_path)
    trader = LiveTrader(config)

    class FakeClient:
        dry_run = False

        def place_maker_order(self, *args, **kwargs) -> OrderResult:
            return OrderResult("unknown-1", "unknown", None, True)

    trader.client = FakeClient()
    assert not trader._rebalance("BTC/USDT", "buy", 0.1)
    assert trader._entries_halted
    assert "unknown-1" in trader.pending_order_ids["BTC/USDT"]
    records = [json.loads(line) for line in (tmp_path / "live_audit.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(record["event"] == "order_result" for record in records)