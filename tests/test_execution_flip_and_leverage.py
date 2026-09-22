"""Tests for delta-only order sizing with a two-step (close-then-open) flip
instead of a single execution-risky oversized order, and discrete
exchange-leverage caching (execution/live_trader.py)."""
from __future__ import annotations

from core.config import TradingBotConfig
from execution.exchange_client import OrderResult
from execution.live_trader import LiveTrader


def _config(tmp_path) -> TradingBotConfig:
    config = TradingBotConfig()
    config.execution.state_path = str(tmp_path / "live_state.json")
    config.execution.audit_log_path = str(tmp_path / "live_audit.jsonl")
    config.ml.shadow_log_path = str(tmp_path / "shadow.csv")
    config.ml.portfolio_shadow_log_path = str(tmp_path / "portfolio_shadow.csv")
    config.funding.enabled = False
    return config


class _RecordingClient:
    dry_run = False

    def __init__(self, position_after_close: float = 0.0) -> None:
        self.orders: list[dict] = []
        self._position_after_close = position_after_close
        self._closed = False

    def place_maker_order(self, symbol, side, amount, reduce_only=False) -> OrderResult:
        self.orders.append({"symbol": symbol, "side": side, "amount": amount, "reduce_only": reduce_only})
        if reduce_only:
            self._closed = True
        return OrderResult(order_id=f"order-{len(self.orders)}", status="filled", filled_price=100.0, is_maker=True)

    def fetch_signed_position_amount(self, symbol: str, strict: bool = False) -> float:
        return self._position_after_close if self._closed else 1.0

    def fetch_open_orders(self, symbol: str, strict: bool = False) -> list[dict]:
        return []


def test_non_flip_delta_is_a_single_order(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    trader = LiveTrader(_config(tmp_path))
    trader.client = _RecordingClient()
    trader.client.dry_run = True

    status = trader._execute_symbol_delta("BTC/USDT", signed_current=0.0, signed_target=0.5)
    assert status == "ok"
    assert len(trader.client.orders) == 1
    assert trader.client.orders[0]["reduce_only"] is False


def test_direction_flip_sends_close_then_open_and_confirms_flat(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    trader = LiveTrader(_config(tmp_path))
    trader.client = _RecordingClient(position_after_close=0.0)
    trader.current_position_amount["BTC/USDT"] = 1.0

    status = trader._execute_symbol_delta("BTC/USDT", signed_current=1.0, signed_target=-0.5)

    assert status == "ok"
    assert len(trader.client.orders) == 2
    close_order, open_order = trader.client.orders
    assert close_order["reduce_only"] is True
    assert close_order["side"] == "sell"
    assert close_order["amount"] == 1.0
    assert open_order["reduce_only"] is False
    assert open_order["side"] == "sell"
    assert open_order["amount"] == 0.5


def test_direction_flip_aborts_if_exchange_not_confirmed_flat_after_close(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    trader = LiveTrader(_config(tmp_path))
    # Exchange still reports a residual position after the "close" order --
    # must NOT proceed to open the opposite side on unconfirmed state.
    trader.client = _RecordingClient(position_after_close=0.2)
    trader.current_position_amount["BTC/USDT"] = 1.0

    status = trader._execute_symbol_delta("BTC/USDT", signed_current=1.0, signed_target=-0.5)

    assert status == "failed"
    assert len(trader.client.orders) == 1  # only the close order was sent, never the open


def test_resolve_exchange_leverage_only_calls_set_leverage_on_change(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    trader = LiveTrader(_config(tmp_path))
    trader.config.risk.max_leverage = 10.0  # keep the risk cap out of this test's way

    calls: list[float] = []

    class FakeClient:
        dry_run = True

        def fetch_leverage_brackets(self, symbol, fallback_steps=None) -> list[float]:
            return [1, 2, 3, 5, 10]

        def set_leverage(self, symbol, leverage) -> None:
            calls.append(leverage)

    trader.client = FakeClient()

    first = trader._resolve_exchange_leverage("BTC/USDT", required_leverage=2.3)
    second = trader._resolve_exchange_leverage("BTC/USDT", required_leverage=2.3)  # unchanged -> no extra call
    third = trader._resolve_exchange_leverage("BTC/USDT", required_leverage=4.5)  # changed -> new call

    assert first == 3
    assert second == 3
    assert third == 5
    assert calls == [3, 5]


def test_resolve_exchange_leverage_never_exceeds_risk_cap(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DRY_RUN", "true")
    trader = LiveTrader(_config(tmp_path))
    trader.config.risk.max_leverage = 4.0

    class FakeClient:
        dry_run = True

        def fetch_leverage_brackets(self, symbol, fallback_steps=None) -> list[float]:
            return [1, 2, 3, 5, 10, 20]

        def set_leverage(self, symbol, leverage) -> None:
            pass

    trader.client = FakeClient()
    leverage = trader._resolve_exchange_leverage("BTC/USDT", required_leverage=8.0)
    assert leverage <= 4.0
