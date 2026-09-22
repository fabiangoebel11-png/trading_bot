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

# Transient/recoverable CCXT errors worth a short retry-with-backoff instead of
# either crashing the poll loop or silently treating the call as failed.
_TRANSIENT_CCXT_ERRORS = (
    ccxt.NetworkError,  # includes RequestTimeout, ExchangeNotAvailable, DDoSProtection
    ccxt.RateLimitExceeded,
)


def _call_with_retry(fn, *args, max_retries: int = 3, base_delay_s: float = 1.0, **kwargs):
    """Retry a CCXT call a few times with exponential backoff on transient
    network/rate-limit errors. Re-raises immediately on any other exception
    (e.g. InsufficientFunds, InvalidOrder) since those are not retryable."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except _TRANSIENT_CCXT_ERRORS as exc:
            last_exc = exc
            delay = base_delay_s * (2**attempt)
            print(f"[CCXT] transient error ({exc.__class__.__name__}): {exc} -- retry in {delay:.1f}s")
            time.sleep(delay)
    raise last_exc  # noqa: RSE102 - last_exc is always set if we reach here


@dataclass
class OrderResult:
    order_id: str | None
    status: str  # "filled" | "open" | "canceled" | "unknown" | "dry_run"
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
                "timeout": 15_000,  # ms; avoid hanging indefinitely on a stalled API/crash-day congestion
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

    def fetch_equity(self, quote_currency: str = "USDT", fallback_equity: float = 0.0) -> float:
        """Total account equity in ``quote_currency`` (free + used margin), used
        by the live trader as the shared wallet's current size for position
        sizing. Falls back to ``fallback_equity`` (e.g. the configured starting
        capital) in dry-run mode or if the exchange call fails, rather than
        crashing the poll loop over a transient balance-fetch error."""
        if self.dry_run:
            return fallback_equity
        try:
            balance = _call_with_retry(self.exchange.fetch_balance)
            total = balance.get("total", {}).get(quote_currency)
            return float(total) if total is not None else fallback_equity
        except Exception as exc:  # noqa: BLE001 - never let a balance-fetch failure crash the live loop
            print(f"[CCXT] fetch_equity failed: {exc} -- falling back to {fallback_equity}")
            return fallback_equity

    def fetch_signed_position_amount(self, symbol: str) -> float:
        """Signed base-asset position size currently held on the exchange
        (positive = long, negative = short, 0.0 = flat), used to seed/reconcile
        ``LiveTrader``'s in-memory position tracking at startup so short/flip
        order sizing (buy vs. sell, and how much) is computed from the real
        exchange state rather than an assumed-flat 0 -- critical for the short
        side, since a wrong assumed sign here silently mis-sizes the very next
        rebalancing order. Falls back to 0.0 in dry-run mode or on any fetch
        error (matches ``fetch_equity``'s fail-safe rather than fail-crash
        behavior for a non-critical bookkeeping read)."""
        if self.dry_run:
            return 0.0
        try:
            positions = _call_with_retry(self.exchange.fetch_positions, [symbol])
        except Exception as exc:  # noqa: BLE001 - never let this crash the live loop
            print(f"[CCXT] fetch_signed_position_amount failed for {symbol}: {exc} -- assuming flat")
            return 0.0
        for pos in positions:
            contracts = pos.get("contracts") or 0.0
            if not contracts:
                continue
            side = (pos.get("side") or "").lower()
            sign = -1.0 if side == "short" else 1.0
            return sign * abs(float(contracts))
        return 0.0

    def fetch_order_book_top(self, symbol: str) -> tuple[float, float]:
        book = _call_with_retry(self.exchange.fetch_order_book, symbol, limit=5)
        if not book["bids"] or not book["asks"]:
            # Empty book (delisting, extreme illiquidity, exchange outage): no
            # safe price to quote against -- surface this explicitly instead
            # of crashing on an IndexError deep inside order placement.
            raise RuntimeError(f"Empty order book for {symbol}: cannot determine a maker quote price")
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
            order = _call_with_retry(
                self.exchange.create_order,
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
                try:
                    status = _call_with_retry(self.exchange.fetch_order, order_id, symbol, max_retries=2)
                except Exception as exc:  # noqa: BLE001 - can't confirm state, don't guess
                    # We genuinely don't know if this order filled (it may have,
                    # with the confirmation lost to the network error). Treating
                    # this as "canceled" would make the caller re-place a
                    # duplicate order into what could already be an open
                    # position/order. Surface it as "unknown" so the caller can
                    # refuse to update its position state and require a manual
                    # reconciliation instead of silently doubling exposure.
                    print(f"[CCXT] fetch_order failed for {order_id} ({symbol}): {exc} -- state unknown")
                    return OrderResult(order_id, "unknown", None, True)
                if status["status"] == "closed":
                    return OrderResult(order_id, "filled", status.get("average") or quote_price, True)
                if status["status"] == "canceled":
                    break

            try:
                _call_with_retry(self.exchange.cancel_order, order_id, symbol, max_retries=2)
            except Exception as exc:  # noqa: BLE001 - cancel failing doesn't mean the order is gone
                print(f"[CCXT] cancel_order failed for {order_id} ({symbol}): {exc} -- state unknown")
                return OrderResult(order_id, "unknown", None, True)
            best_bid, best_ask = self.fetch_order_book_top(symbol)
            quote_price = best_bid if side == "buy" else best_ask

        return OrderResult(order_id=None, status="canceled", filled_price=None, is_maker=True)

    def place_taker_fallback(self, symbol: str, side: str, amount: float) -> OrderResult:
        """Explicit taker (market) order fallback; only used when maker execution
        repeatedly fails and immediate execution is required (e.g. stop-loss)."""
        if self.dry_run:
            print(f"[DRY_RUN] taker {side} {amount} {symbol}")
            return OrderResult(order_id=None, status="dry_run", filled_price=None, is_maker=False)
        order = _call_with_retry(self.exchange.create_order, symbol, type="market", side=side, amount=amount)
        return OrderResult(order["id"], "filled", order.get("average"), False)

    def place_stop_loss_order(
        self, symbol: str, position_side: str, amount: float, stop_price: float
    ) -> OrderResult:
        """Place a real, server-side reduce-only stop order ("catastrophe
        stop"): if this process crashes, loses network, or the poll loop
        stalls, the exchange itself still force-closes the position at
        ``stop_price`` -- the in-memory ATR-trailing stop (``core.risk.
        apply_dynamic_stop``) only protects while the bot is actively polling.
        ``position_side`` is the side being protected ("long"/"short"); the
        stop order itself is the opposite side (sell stop under a long,
        buy stop above a short). Order type string is Binance-specific
        (STOP_MARKET); other CCXT venues may need a different unified type.
        """
        side = "sell" if position_side == "long" else "buy"
        if self.dry_run:
            print(f"[DRY_RUN] stop_market {side} {amount} {symbol} @ stop={stop_price}")
            return OrderResult(order_id=None, status="dry_run", filled_price=None, is_maker=False)
        order = _call_with_retry(
            self.exchange.create_order,
            symbol,
            type="STOP_MARKET",
            side=side,
            amount=amount,
            params={"stopPrice": stop_price, "reduceOnly": True},
        )
        return OrderResult(order["id"], "open", None, False)

    def cancel_stop_loss_order(self, symbol: str, order_id: str) -> None:
        """Best-effort cancel of a previously placed stop order (e.g. before
        re-quoting it at a new ATR-trailing level). Does not raise: if the
        order already triggered/was already canceled, there is nothing to
        clean up, and the caller (``LiveTrader``) always re-derives the
        correct in-memory state from its own tracking afterward."""
        if self.dry_run:
            print(f"[DRY_RUN] cancel stop order {order_id} ({symbol})")
            return
        try:
            _call_with_retry(self.exchange.cancel_order, order_id, symbol, max_retries=2)
        except Exception as exc:  # noqa: BLE001 - stale/already-gone order is not fatal
            print(f"[CCXT] cancel_stop_loss_order failed for {order_id} ({symbol}): {exc}")

    def fetch_open_stop_orders(self, symbol: str) -> list[dict]:
        """Reduce-only stop orders currently open on the exchange for
        ``symbol`` -- used at startup/after a restart (state recovery) to
        check whether an open position is already protected before assuming
        it isn't and placing a duplicate. Falls back to an empty list in
        dry-run mode or on any fetch error (fail-safe: the caller then places
        a fresh stop rather than leaving a possibly-unprotected position)."""
        if self.dry_run:
            return []
        try:
            orders = _call_with_retry(self.exchange.fetch_open_orders, symbol)
        except Exception as exc:  # noqa: BLE001 - never let this crash the live loop
            print(f"[CCXT] fetch_open_stop_orders failed for {symbol}: {exc} -- assuming none")
            return []
        return [
            o
            for o in orders
            if (o.get("type") or "").lower() in ("stop_market", "stop", "stop_loss")
            and (o.get("reduceOnly") or (o.get("info") or {}).get("reduceOnly"))
        ]


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
