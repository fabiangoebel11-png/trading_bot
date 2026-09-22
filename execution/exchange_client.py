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
    status: str  # "filled" | "open" | "canceled" | "unknown" | "dry_run" | "skipped"
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
        self._markets_loaded = False

    def _ensure_markets(self) -> None:
        """Load CCXT market metadata (step size / precision / min notional) once
        per process. This is an unauthenticated, public call -- done even in
        dry-run/testnet mode, since realistic order-size rounding is exactly
        what needs verifying *before* real capital is at risk."""
        if self._markets_loaded:
            return
        try:
            _call_with_retry(self.exchange.load_markets)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, don't block trading on this
            print(f"[CCXT] load_markets failed: {exc} -- lot-size/min-notional checks disabled this session")
        self._markets_loaded = True

    def _round_to_step(self, symbol: str, amount: float) -> float:
        """Round ``amount`` down to the exchange's allowed step size/precision
        only (no minimum-size enforcement) -- used for reduce-only stop orders
        where an amount below the minimum must still be sent (it protects an
        existing position) rather than dropped."""
        if amount <= 0:
            return 0.0
        self._ensure_markets()
        market = (self.exchange.markets or {}).get(symbol)
        if market is None:
            return amount
        try:
            return float(self.exchange.amount_to_precision(symbol, amount))
        except Exception as exc:  # noqa: BLE001 - precision helper failing shouldn't crash sizing
            print(f"[CCXT] amount_to_precision failed for {symbol}: {exc} -- using unrounded amount")
            return amount

    def _round_price_to_tick(self, symbol: str, price: float) -> float:
        """Round ``price`` to the exchange's allowed tick size for ``symbol``.
        Every order below (limit price AND stop price) must go through this --
        exchanges reject prices with more precision than their ``PRICE_FILTER``
        tick size just as readily as they reject an oversized ``amount``."""
        if price is None or price <= 0:
            return price
        self._ensure_markets()
        market = (self.exchange.markets or {}).get(symbol)
        if market is None:
            return price
        try:
            return float(self.exchange.price_to_precision(symbol, price))
        except Exception as exc:  # noqa: BLE001 - precision helper failing shouldn't crash sizing
            print(f"[CCXT] price_to_precision failed for {symbol}: {exc} -- using unrounded price")
            return price

    def _apply_market_limits(
        self, symbol: str, amount: float, price: float | None, reduce_only: bool = False
    ) -> float:
        """Round ``amount`` down to the exchange's allowed step size/precision
        and enforce the minimum order size / minimum notional (``minNotional``)
        for ``symbol``. A 500 USDT shared wallet split across up to 3 leveraged
        symbols can easily compute a raw target size that Binance would reject
        outright (``LOT_SIZE``/``MIN_NOTIONAL`` filter errors) -- rounding and
        clamping here up front means the live loop never crashes on this and
        never silently sends a size the exchange would refuse.

        ``reduce_only=True`` (dust-handling / position-closing orders):
        SKIPS the minimum-notional/minimum-amount floor entirely, only
        rounds to the step size. A 500 USDT wallet split across several
        leveraged symbols will regularly compute a closing remainder below
        the exchange minimum (e.g. closing down to exactly flat, or a
        partial reduction) -- silently refusing to close it (as a normal,
        non-reduce-only order correctly would, to protect the risk budget)
        would instead leave an un-closeable "dust" position open forever.
        Reduce-only orders only ever shrink existing exposure, never grow it,
        so bypassing the minimum here cannot violate the risk budget.

        Returns ``0.0`` if the amount can't clear the exchange minimum even
        after rounding (caller should skip the order rather than pad the size
        up to the minimum, which would risk exceeding the intended risk budget).
        """
        rounded = self._round_to_step(symbol, amount)
        if rounded <= 0:
            return 0.0
        if reduce_only:
            return rounded
        self._ensure_markets()
        market = (self.exchange.markets or {}).get(symbol)
        if market is None:
            return rounded

        limits = market.get("limits", {}) or {}
        min_amount = (limits.get("amount", {}) or {}).get("min") or 0.0
        min_cost = (limits.get("cost", {}) or {}).get("min") or 0.0
        if min_amount and rounded < min_amount:
            return 0.0
        if min_cost and price and rounded * price < min_cost:
            return 0.0
        return rounded

    def set_leverage(self, symbol: str, leverage: float) -> None:
        if self.dry_run:
            print(f"[DRY_RUN] set_leverage {symbol} -> {leverage}x")
            return
        self.exchange.set_leverage(int(leverage), symbol)

    def fetch_leverage_brackets(self, symbol: str, fallback_steps: list[float] | None = None) -> list[float]:
        """Discrete leverage values ``symbol`` is actually allowed to run at
        (Bybit/Binance both only accept specific tiers, never an arbitrary
        vol-targeted float like "2.37x") -- read from the exchange's own
        market metadata, never hardcoded, since tiers differ per symbol/
        margin bracket/exchange. Falls back to ``fallback_steps``
        (``ExecutionConfig.fallback_leverage_steps``) if the exchange doesn't
        expose tiers via ccxt or the call fails -- degrade gracefully rather
        than block trading on this being unavailable."""
        fallback_steps = fallback_steps or []
        if self.dry_run:
            return fallback_steps
        self._ensure_markets()
        try:
            if self.exchange.has.get("fetchLeverageTiers"):
                tiers = _call_with_retry(self.exchange.fetch_leverage_tiers, [symbol])
                symbol_tiers = tiers.get(symbol, [])
                steps = sorted({float(t["maxLeverage"]) for t in symbol_tiers if t.get("maxLeverage")})
                if steps:
                    return steps
            market = (self.exchange.markets or {}).get(symbol, {})
            max_leverage = ((market.get("limits") or {}).get("leverage") or {}).get("max")
            if max_leverage:
                # Only the ceiling is known (no per-tier breakdown) -- degrade
                # to every whole-number step up to it as a reasonable
                # approximation of what the exchange is likely to offer.
                return list(range(1, int(max_leverage) + 1))
        except Exception as exc:  # noqa: BLE001 - leverage discovery is best-effort, never fatal
            print(f"[ExchangeClient] {symbol}: fetch_leverage_brackets failed: {exc}")
        return fallback_steps

    def configure_account(
        self, symbols: list[str], leverage_ceiling: int, margin_mode: str = "cross"
    ) -> None:
        """Set the exchange-side margin mode + leverage CEILING once per symbol
        at process startup -- never per order. This only widens/narrows what
        the account is *allowed* to hold; actual exposure is controlled
        exclusively through order ``amount`` (the strategy's own vol-targeted/
        risk-capped leverage from ``core.risk`` stays far below this ceiling
        by construction). Idempotent and crash-safe: many exchanges (Binance
        included) reject a redundant "set it to what it already is" call with
        a benign error -- caught and logged per symbol rather than aborting
        startup, since a failure here must never block the bot from at least
        trading with whatever mode/leverage the account already has."""
        self._ensure_markets()
        for symbol in symbols:
            if self.dry_run:
                print(f"[DRY_RUN] configure_account {symbol}: margin_mode={margin_mode}, leverage_ceiling={leverage_ceiling}x")
                continue
            try:
                _call_with_retry(self.exchange.set_margin_mode, margin_mode, symbol)
            except Exception as exc:  # noqa: BLE001 - e.g. "already cross" / open position blocks the switch
                print(f"[CCXT] {symbol}: set_margin_mode({margin_mode}) failed/no-op: {exc}")
            try:
                _call_with_retry(self.exchange.set_leverage, int(leverage_ceiling), symbol)
            except Exception as exc:  # noqa: BLE001 - e.g. "leverage not modified"
                print(f"[CCXT] {symbol}: set_leverage({leverage_ceiling}x) failed/no-op: {exc}")

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

    def fetch_signed_position_amount(self, symbol: str, strict: bool = False) -> float:
        """Signed base-asset position size currently held on the exchange
        (positive = long, negative = short, 0.0 = flat), used to seed/reconcile
        ``LiveTrader``'s in-memory position tracking at startup so short/flip
        order sizing (buy vs. sell, and how much) is computed from the real
        exchange state rather than an assumed-flat 0 -- critical for the short
        side, since a wrong assumed sign here silently mis-sizes the very next
        rebalancing order. Falls back to 0.0 in dry-run mode or on any fetch
        error unless ``strict=True``. Reconciliation must use strict mode:
        assuming a failed request means "flat" can create duplicate exposure.
        """
        if self.dry_run:
            return 0.0
        try:
            positions = _call_with_retry(self.exchange.fetch_positions, [symbol])
        except Exception as exc:  # noqa: BLE001 - never let this crash the live loop
            if strict:
                raise RuntimeError(f"Unable to reconcile position for {symbol}") from exc
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

    def fetch_open_orders(self, symbol: str, strict: bool = False) -> list[dict]:
        """Return all open orders for a symbol for state reconciliation.

        In strict mode a read failure is surfaced to the safety layer instead
        of being silently converted into an empty order list.
        """
        if self.dry_run:
            return []
        try:
            return list(_call_with_retry(self.exchange.fetch_open_orders, symbol))
        except Exception as exc:  # noqa: BLE001 - caller chooses strict safety behavior
            if strict:
                raise RuntimeError(f"Unable to reconcile open orders for {symbol}") from exc
            print(f"[CCXT] fetch_open_orders failed for {symbol}: {exc}")
            return []

    def fetch_last_price(self, symbol: str) -> float:
        """Single real-time last-traded price via a lightweight ticker call --
        used by the live trader's high-frequency risk loop (every ~60s,
        between candle closes) to re-check the ATR-trailing stop against the
        current market price. Deliberately NOT an OHLCV/candle fetch: this
        must never be fed into signal generation (that would reintroduce the
        exact train-serve skew the ML confirmation model is trained to avoid,
        see ``TrendMLConfig`` docstring) -- it is a price scalar only used for
        stop-distance comparisons.
        """
        if self.dry_run:
            best_bid, best_ask = self.fetch_order_book_top(symbol)
            return (best_bid + best_ask) / 2.0
        ticker = _call_with_retry(self.exchange.fetch_ticker, symbol)
        last = ticker.get("last") or ticker.get("close")
        if last is None:
            best_bid, best_ask = self.fetch_order_book_top(symbol)
            return (best_bid + best_ask) / 2.0
        return float(last)

    def fetch_current_funding_rate(self, symbol: str) -> float:
        """Current perpetual-swap funding rate (fraction, e.g. 0.0001 = 0.01%)
        for live shadow-mode funding-cost simulation (``core.ml.shadow``) --
        NOT used for real order sizing (the exchange charges/credits funding
        automatically on whichever position is actually held). Dry-run/no-data
        fallback is 0.0 (no simulated funding cost rather than a crash)."""
        if self.dry_run:
            return 0.0
        try:
            data = _call_with_retry(self.exchange.fetch_funding_rate, symbol)
            rate = data.get("fundingRate")
            return float(rate) if rate is not None else 0.0
        except Exception as exc:  # noqa: BLE001 - funding rate is best-effort audit data, never fatal
            print(f"[ExchangeClient] {symbol}: fetch_current_funding_rate failed: {exc}")
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
        reduce_only: bool = False,
    ) -> OrderResult:
        """Place a post-only limit order priced at (or improving) the current best
        bid/ask, so the order is virtually guaranteed to execute as a maker fill.
        If unfilled after ``reprice_timeout_s``, cancels and re-quotes at the new
        top of book, up to ``max_reprices`` times.

        ``reduce_only=True`` (this order can only ever shrink an existing
        position, e.g. rebalancing down to a smaller target or fully closing
        it): bypasses the minimum-notional/minimum-amount floor (dust
        handling, see ``_apply_market_limits``) and tags the order
        ``reduceOnly`` so the exchange itself refuses to let it accidentally
        open/flip a position beyond what actually exists."""
        best_bid, best_ask = self.fetch_order_book_top(symbol)
        quote_price = price or (best_bid if side == "buy" else best_ask)
        quote_price = self._round_price_to_tick(symbol, quote_price)
        amount = self._apply_market_limits(symbol, amount, quote_price, reduce_only=reduce_only)
        if amount <= 0:
            print(f"[CCXT] {symbol}: order amount below exchange lot-size/min-notional after rounding -- skipping")
            return OrderResult(order_id=None, status="skipped", filled_price=None, is_maker=True)

        if self.dry_run:
            print(f"[DRY_RUN] maker {side} {amount} {symbol} @ {quote_price}" + (" (reduceOnly)" if reduce_only else ""))
            return OrderResult(order_id=None, status="dry_run", filled_price=quote_price, is_maker=True)

        order_params = dict(self.maker_order_params)
        if reduce_only:
            order_params["reduceOnly"] = True

        for attempt in range(max_reprices + 1):
            order = _call_with_retry(
                self.exchange.create_order,
                symbol,
                type="limit",
                side=side,
                amount=amount,
                price=quote_price,
                params=order_params,
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
            quote_price = self._round_price_to_tick(symbol, best_bid if side == "buy" else best_ask)

        return OrderResult(order_id=None, status="canceled", filled_price=None, is_maker=True)

    def place_taker_fallback(
        self, symbol: str, side: str, amount: float, reduce_only: bool = False
    ) -> OrderResult:
        """Explicit taker (market) order fallback; only used when maker execution
        repeatedly fails and immediate execution is required (e.g. stop-loss/
        emergency flatten). ``reduce_only=True`` (used by
        ``LiveTrader._emergency_flatten``): bypasses the minimum-notional/
        minimum-amount floor -- an emergency close must always be able to
        fully flatten the position down to exactly zero, including a dust
        remainder below the exchange minimum, and is tagged ``reduceOnly`` so
        the exchange rejects it outright rather than accidentally flipping
        into a new position if the local size estimate is stale/wrong."""
        best_bid, best_ask = self.fetch_order_book_top(symbol)
        reference_price = best_ask if side == "buy" else best_bid
        amount = self._apply_market_limits(symbol, amount, reference_price, reduce_only=reduce_only)
        if amount <= 0:
            print(f"[CCXT] {symbol}: taker order amount below exchange lot-size/min-notional after rounding -- skipping")
            return OrderResult(order_id=None, status="skipped", filled_price=None, is_maker=False)

        if self.dry_run:
            print(f"[DRY_RUN] taker {side} {amount} {symbol}" + (" (reduceOnly)" if reduce_only else ""))
            return OrderResult(order_id=None, status="dry_run", filled_price=None, is_maker=False)
        params = {"reduceOnly": True} if reduce_only else {}
        order = _call_with_retry(
            self.exchange.create_order, symbol, type="market", side=side, amount=amount, params=params
        )
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
        # Only round to the allowed step size here (not the min-notional floor):
        # this order reduces/closes an existing position of exactly ``amount``,
        # so silently skipping it below a notional minimum would leave that
        # position unprotected rather than just under-sized.
        amount = self._round_to_step(symbol, amount) or amount
        stop_price = self._round_price_to_tick(symbol, stop_price)
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

    def cancel_all_orders(self, symbol: str) -> None:
        """Best-effort cancel of EVERY open order for ``symbol`` (resting
        maker limit orders from a stalled rebalance, stray stop orders, ...).
        Used before an emergency market flatten (``LiveTrader.
        _emergency_flatten``) so a leftover limit/stop order from before the
        breach cannot later fill/trigger unexpectedly against a position that
        no longer exists (or exists with a different size/direction).
        Not fatal if unsupported/fails: falls back to fetching + canceling
        open orders individually, and never raises up into the caller (an
        emergency flatten must proceed even if this cleanup step degrades)."""
        if self.dry_run:
            print(f"[DRY_RUN] cancel_all_orders {symbol}")
            return
        try:
            _call_with_retry(self.exchange.cancel_all_orders, symbol)
            return
        except Exception as exc:  # noqa: BLE001 - some venues don't support the bulk endpoint
            print(f"[CCXT] cancel_all_orders failed for {symbol}: {exc} -- falling back to per-order cancel")
        try:
            open_orders = self.fetch_open_orders(symbol, strict=True)
        except Exception as exc:  # noqa: BLE001 - never let cleanup itself block the emergency flatten
            print(f"[CCXT] fetch_open_orders failed for {symbol}: {exc} -- cannot clean up resting orders")
            return
        for order in open_orders:
            try:
                _call_with_retry(self.exchange.cancel_order, order["id"], symbol, max_retries=2)
            except Exception as exc:  # noqa: BLE001 - best-effort, keep trying the rest
                print(f"[CCXT] cancel_order failed for {order.get('id')} ({symbol}): {exc}")

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
