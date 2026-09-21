"""CCXT-compatible exchange client focused on maker (post-only) execution.

``CCXTExchangeClient`` is exchange-agnostic: it works with any CCXT unified
perpetual-futures market -- Binance USDT-M futures by default, but also
CCXT-compatible DEX/perp venues (e.g. Hyperliquid, Bybit, OKX) by passing a
different ``exchange_id``/``market_type``/credential env-var names.

Design goals:
- Never place naive market/taker orders by default; always try a post-only
  limit order first (maker fee), and only fall back to a taker order if the
  caller explicitly allows it.
- Centralize leverage and testnet/dry-run handling so the live trader never
  talks to ``ccxt`` directly.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import ccxt
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OrderResult:
    order_id: str | None
    status: str  # "filled" | "open" | "canceled" | "dry_run"
    filled_price: float | None
    is_maker: bool


class CCXTExchangeClient:
    """Generic maker-first execution client for any CCXT unified perpetual market."""

    def __init__(
        self,
        exchange_id: str = "binance",
        market_type: str = "future",
        api_key_env: str = "BINANCE_API_KEY",
        api_secret_env: str = "BINANCE_API_SECRET",
        testnet_env: str = "BINANCE_TESTNET",
        supports_sandbox: bool = True,
        testnet: bool | None = None,
        dry_run: bool | None = None,
    ) -> None:
        self.dry_run = _env_bool("DRY_RUN", True) if dry_run is None else dry_run
        self.testnet = _env_bool(testnet_env, True) if testnet is None else testnet
        self.supports_sandbox = supports_sandbox

        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class(
            {
                "apiKey": os.getenv(api_key_env, ""),
                "secret": os.getenv(api_secret_env, ""),
                "enableRateLimit": True,
                "options": {"defaultType": market_type},
            }
        )
        if self.testnet and self.supports_sandbox:
            self.exchange.set_sandbox_mode(True)

        # Binance requires the extra GTX time-in-force for a guaranteed-maker order;
        # most other CCXT-unified venues accept plain postOnly.
        self.maker_order_params = (
            {"postOnly": True, "timeInForce": "GTX"} if exchange_id == "binance" else {"postOnly": True}
        )

    def set_leverage(self, symbol: str, leverage: float) -> None:
        if self.dry_run:
            print(f"[DRY_RUN] set_leverage {symbol} -> {leverage}x")
            return
        self.exchange.set_leverage(int(leverage), symbol)

    def fetch_order_book_top(self, symbol: str) -> tuple[float, float]:
        book = self.exchange.fetch_order_book(symbol, limit=5)
        return book["bids"][0][0], book["asks"][0][0]

    def place_maker_order(
        self,
        symbol: str,
        side: str,  # "buy" | "sell"
        amount: float,
        price: float | None = None,
        reprice_timeout_s: float = 5.0,
        max_reprices: int = 3,
    ) -> OrderResult:
        """Place a post-only limit order priced at (or improving) the current best
        bid/ask, so the order is virtually guaranteed to execute as a maker fill.
        If unfilled after ``reprice_timeout_s``, cancels and re-quotes at the new
        top of book, up to ``max_reprices`` times."""
        best_bid, best_ask = self.fetch_order_book_top(symbol)
        quote_price = price or (best_bid if side == "buy" else best_ask)

        if self.dry_run:
            print(f"[DRY_RUN] maker {side} {amount} {symbol} @ {quote_price}")
            return OrderResult(order_id=None, status="dry_run", filled_price=quote_price, is_maker=True)

        for attempt in range(max_reprices + 1):
            order = self.exchange.create_order(
                symbol,
                type="limit",
                side=side,
                amount=amount,
                price=quote_price,
                params=self.maker_order_params,
            )
            order_id = order["id"]
            deadline = time.time() + reprice_timeout_s
            while time.time() < deadline:
                time.sleep(0.5)
                status = self.exchange.fetch_order(order_id, symbol)
                if status["status"] == "closed":
                    return OrderResult(order_id, "filled", status.get("average") or quote_price, True)
                if status["status"] == "canceled":
                    break

            self.exchange.cancel_order(order_id, symbol)
            best_bid, best_ask = self.fetch_order_book_top(symbol)
            quote_price = best_bid if side == "buy" else best_ask

        return OrderResult(order_id=None, status="canceled", filled_price=None, is_maker=True)

    def place_taker_fallback(self, symbol: str, side: str, amount: float) -> OrderResult:
        """Explicit taker (market) order fallback; only used when maker execution
        repeatedly fails and immediate execution is required (e.g. stop-loss)."""
        if self.dry_run:
            print(f"[DRY_RUN] taker {side} {amount} {symbol}")
            return OrderResult(order_id=None, status="dry_run", filled_price=None, is_maker=False)
        order = self.exchange.create_order(symbol, type="market", side=side, amount=amount)
        return OrderResult(order["id"], "filled", order.get("average"), False)


class BinanceFuturesClient(CCXTExchangeClient):
    """Preconfigured client for Binance USDT-M perpetual futures (backward-compat
    alias kept for existing call sites, e.g. ``execution/live_trader.py``)."""

    def __init__(self, testnet: bool | None = None, dry_run: bool | None = None) -> None:
        super().__init__(
            exchange_id="binance",
            market_type="future",
            api_key_env="BINANCE_API_KEY",
            api_secret_env="BINANCE_API_SECRET",
            testnet_env="BINANCE_TESTNET",
            supports_sandbox=True,
            testnet=testnet,
            dry_run=dry_run,
        )


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}
