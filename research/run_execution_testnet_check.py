"""Read-only Binance Futures testnet checks and an explicit order-lifecycle probe.

Default mode is read-only: it verifies credentials, account state, open orders,
and local audit/state artifacts. ``--order-lifecycle`` is deliberately gated
by *both* ``DRY_RUN=false`` and ``TESTNET_ORDER_TEST=true``; it may create a
testnet order and, if filled, creates/cancels a native stop and flattens the
testnet position using the production client methods.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from core.config import TradingBotConfig
from execution.exchange_client import BinanceFuturesClient


def _require_testnet(client: BinanceFuturesClient) -> None:
    if not client.testnet:
        raise RuntimeError("Refusing: BINANCE_TESTNET must be true.")


def _check_local_artifacts(config: TradingBotConfig) -> None:
    state_path = Path(config.execution.state_path)
    audit_path = Path(config.execution.audit_log_path)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(f"state: entries_halted={state.get('entries_halted', False)}, reason={state.get('halt_reason')}")
    else:
        print("state: no live-state file yet")
    if audit_path.exists():
        records = audit_path.read_text(encoding="utf-8").splitlines()
        for line in records:
            json.loads(line)
        print(f"audit: {len(records)} valid JSONL records")
    else:
        print("audit: no audit file yet")


def _read_only_check(client: BinanceFuturesClient, config: TradingBotConfig) -> None:
    client._ensure_markets()
    equity = client.fetch_equity(fallback_equity=-1.0)
    print(f"testnet equity: {equity}")
    for symbol in config.data.symbols:
        amount = client.fetch_signed_position_amount(symbol, strict=True)
        orders = client.fetch_open_orders(symbol, strict=True)
        print(f"{symbol}: position={amount}, open_orders={len(orders)}")
    _check_local_artifacts(config)


def _order_lifecycle(client: BinanceFuturesClient, symbol: str, amount: float) -> None:
    dry_run = os.getenv("DRY_RUN", "true").strip().lower()
    if dry_run != "false" or os.getenv("TESTNET_ORDER_TEST", "").strip().lower() != "true":
        raise RuntimeError("Refusing order test: require DRY_RUN=false and TESTNET_ORDER_TEST=true.")

    before = client.fetch_signed_position_amount(symbol, strict=True)
    if before != 0.0:
        raise RuntimeError(f"Refusing order test: {symbol} is not flat ({before}).")

    result = client.place_maker_order(symbol, "buy", amount)
    after = client.fetch_signed_position_amount(symbol, strict=True)
    print(f"entry: status={result.status}, order_id={result.order_id}, exchange_position={after}")
    if after == 0.0:
        print("No fill observed; maker cancel/reprice path was exercised. Re-run during more active testnet liquidity for fill/stop coverage.")
        return

    try:
        last_price = client.fetch_last_price(symbol)
        stop = client.place_stop_loss_order(symbol, "long" if after > 0 else "short", abs(after), last_price * 0.5)
        print(f"stop: status={stop.status}, order_id={stop.order_id}")
        restarted = BinanceFuturesClient(testnet=True, dry_run=False)
        print(f"restart reconciliation: position={restarted.fetch_signed_position_amount(symbol, strict=True)}, open_orders={len(restarted.fetch_open_orders(symbol, strict=True))}")
        if stop.order_id is not None:
            client.cancel_stop_loss_order(symbol, stop.order_id)
    finally:
        current = client.fetch_signed_position_amount(symbol, strict=True)
        if current:
            close = client.place_taker_fallback(symbol, "sell" if current > 0 else "buy", abs(current), reduce_only=True)
            print(f"flatten: status={close.status}, order_id={close.order_id}")
        final = client.fetch_signed_position_amount(symbol, strict=True)
        if final != 0.0:
            raise RuntimeError(f"Cleanup failed: {symbol} remains open at {final}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order-lifecycle", action="store_true", help="Explicitly run a testnet order/stop/flatten probe.")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--amount", type=float, default=0.001)
    args = parser.parse_args()

    config = TradingBotConfig()
    # Read-only verification must query the real testnet account even while
    # the normal bot remains configured with DRY_RUN=true.
    client = BinanceFuturesClient(testnet=True, dry_run=False)
    _require_testnet(client)
    _read_only_check(client, config)
    if args.order_lifecycle:
        _order_lifecycle(client, args.symbol, args.amount)


if __name__ == "__main__":
    main()