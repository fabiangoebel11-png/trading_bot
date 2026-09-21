"""Live trading loop skeleton: polls candles, reuses the exact same trend-following
signal logic as the backtester, and routes orders through the maker-first
execution client.

Position sizing mirrors ``core.backtester`` exactly: every symbol's *candidate*
leverage is computed independently (vol-targeted, capped by the shared wallet's
whole risk budget), then every poll's currently active (non-flat) symbols are
combined into inverse-leverage weights renormalized to sum to 1 across only
those active symbols, plus the same defensive ``shared_wallet_risk_scale``
safety valve -- so live sizing can never silently drift from what was
backtested (Konzept 1 & 2 of the shared-wallet review).

Intended to run headless on a VPS. Defaults to DRY_RUN=true (see .env.example)
so it can be exercised safely before real capital is at risk.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from core.backtester import _periods_per_year
from core.config import TradingBotConfig
from core.data_loader import drop_incomplete_last_candle, fetch_ohlcv_history, resample_ohlcv
from core.macro import apply_macro_gate, fetch_macro_data
from core.risk import (
    apply_dynamic_stop,
    apply_trade_cooldown,
    cap_absolute_notional,
    cold_start_scale_live,
    dynamic_leverage,
    dynamic_stop_distance,
    risk_based_leverage_cap,
    shared_wallet_risk_scale,
)
from core.strategy import generate_trend_signals
from execution.exchange_client import BinanceFuturesClient


class LiveTrader:
    """Trades every symbol in ``config.data.symbols`` out of one shared wallet
    (one exchange client per process, multiple symbols share both the client
    and the capital/risk budget -- see module docstring)."""

    def __init__(self, config: TradingBotConfig, macro_refresh_every_polls: int = 24) -> None:
        self.config = config
        self.client = BinanceFuturesClient()
        self.current_positions: dict[str, int] = {symbol: 0 for symbol in config.data.symbols}
        # Signed base-asset quantity actually held per symbol (positive=long,
        # negative=short) -- reconciled against the real exchange state below
        # rather than assumed flat, so a direct long<->short flip is sized as
        # "close old + open new" instead of only the new leg's notional (a
        # naive size-only-the-new-target order would leave a stale opposite
        # remainder open, or under/overshoot, exactly the failure mode Konzept
        # 3's short-logic review is meant to catch).
        self.current_position_amount: dict[str, float] = {
            symbol: self.client.fetch_signed_position_amount(symbol) for symbol in config.data.symbols
        }
        for symbol, amount in self.current_position_amount.items():
            self.current_positions[symbol] = int(np.sign(amount))
        self.macro_refresh_every_polls = macro_refresh_every_polls
        self._polls_since_macro_refresh = 0
        self._macro_df = None
        # Cold-start / equity-cushion ratchet (Konzept 1): once True, never
        # reset back to False for the lifetime of this process.
        self._cold_start_buffer_reached = False

    def _load_recent_ohlc(self, symbol: str) -> pd.DataFrame:
        cfg = self.config.data
        lookback_days = max(3, cfg.history_days // 100)  # small rolling window, warm-up only
        raw = fetch_ohlcv_history(symbol, cfg.base_timeframe, lookback_days, cache_dir=cfg.cache_dir, use_cache=False)
        raw = drop_incomplete_last_candle(raw, cfg.base_timeframe)
        resampled = resample_ohlcv(raw, cfg.resample_to) if cfg.resample_to else raw
        return drop_incomplete_last_candle(resampled, cfg.resample_to) if cfg.resample_to else resampled

    def _symbol_candidate(self, symbol: str) -> dict | None:
        """Latest-bar equivalent of ``core.backtester._symbol_candidate_frame``:
        target position (post stop/cooldown) plus the leverage this symbol
        could run if it were the wallet's only open position."""
        ohlc = self._load_recent_ohlc(symbol)
        signals = generate_trend_signals(ohlc, self.config.trend)
        if signals.empty:
            return None
        signals = apply_macro_gate(signals, self.config, self._macro_df)

        close = signals["close"]
        stop_distance = dynamic_stop_distance(close, self.config.risk)
        position = apply_dynamic_stop(signals["position"], close, stop_distance)
        position = apply_trade_cooldown(position, self.config.capital.cooldown_candles)

        periods_per_year = _periods_per_year(self.config.data.resample_to or self.config.data.base_timeframe)
        price_returns = close.pct_change()
        vol_leverage = dynamic_leverage(price_returns, self.config.risk, periods_per_year)
        risk_cap = risk_based_leverage_cap(
            stop_distance, close, self.config.capital.max_portfolio_risk_pct, self.config.risk.max_leverage
        )
        leverage_candidate = min(vol_leverage.iloc[-1], risk_cap.iloc[-1])
        leverage_candidate = max(leverage_candidate, self.config.risk.min_leverage)

        stop_pct = stop_distance.iloc[-1] / close.iloc[-1] if close.iloc[-1] else 0.0
        return {
            "target_position": int(position.iloc[-1]),
            "leverage_candidate": float(leverage_candidate),
            "stop_pct": float(stop_pct) if stop_pct == stop_pct else 0.0,  # NaN-safe
            "close": float(close.iloc[-1]),
        }

    def poll_once(self) -> None:
        if self.config.macro.enabled:
            if self._macro_df is None or self._polls_since_macro_refresh >= self.macro_refresh_every_polls:
                self._macro_df = fetch_macro_data(self.config.macro, use_cache=False)
                self._polls_since_macro_refresh = 0
            self._polls_since_macro_refresh += 1

        candidates = {}
        for symbol in self.config.data.symbols:
            candidate = self._symbol_candidate(symbol)
            if candidate is not None:
                candidates[symbol] = candidate

        # --- Shared-wallet sizing across every symbol active THIS poll (Konzept 1 & 2) ---
        active = {s: c for s, c in candidates.items() if c["target_position"] != 0}
        weight_sum = sum(c["leverage_candidate"] for c in active.values())
        weights = {s: c["leverage_candidate"] / weight_sum for s, c in active.items()} if weight_sum > 0 else {}
        implied_risk = sum(
            weights[s] * active[s]["leverage_candidate"] * active[s]["stop_pct"] for s in active
        )
        scale = min(1.0, self.config.capital.max_portfolio_risk_pct / implied_risk) if implied_risk > 0 else 1.0

        equity = self.client.fetch_equity(fallback_equity=self.config.capital.initial_capital_usdt)

        # --- Cold-start / equity-cushion protection (Konzept 1) ---
        # Half leverage (default) until the account has ever closed +10% above
        # the initial capital, then permanently normal -- ratchet, never
        # re-triggers on a later drawdown once the cushion has been built.
        cold_start_scale, self._cold_start_buffer_reached = cold_start_scale_live(
            equity,
            self._cold_start_buffer_reached,
            self.config.capital.initial_capital_usdt,
            self.config.capital.cold_start_buffer_pct,
            self.config.capital.cold_start_risk_scale,
        )

        for symbol, candidate in candidates.items():
            target_position = candidate["target_position"]

            if target_position == 0:
                desired_amount = 0.0
            else:
                weight = weights.get(symbol, 0.0)
                leverage = candidate["leverage_candidate"] * scale * cold_start_scale
                notional = equity * weight * leverage
                # --- Absolute margin ceiling (Konzept 2): hard USDT cap per
                # symbol regardless of how large the wallet has grown ---
                notional = cap_absolute_notional(notional, self.config.capital.max_absolute_position_size_usdt)
                desired_amount = notional / candidate["close"] if candidate["close"] else 0.0

            # Signed target quantity (+long/-short/0 flat); the order size is
            # the *delta* against the currently held signed amount, not just
            # the new target's own notional -- this is what correctly sizes a
            # direct long<->short flip (close old + open new in one order)
            # instead of only ever placing the new leg's size.
            signed_target = target_position * desired_amount
            signed_current = self.current_position_amount[symbol]
            delta = signed_target - signed_current
            if abs(delta) < 1e-12:
                continue  # already at target, nothing to rebalance this poll

            side = "buy" if delta > 0 else "sell"
            order_amount = abs(delta)
            if self._rebalance(symbol, side, order_amount):
                self.current_position_amount[symbol] = signed_target
                self.current_positions[symbol] = target_position

    def _rebalance(self, symbol: str, side: str, order_amount: float) -> bool:
        print(f"{symbol}: {side} {order_amount:.6f} (maker-first, taker fallback on failure)...")
        result = self.client.place_maker_order(symbol, side, amount=order_amount)
        if result.status == "unknown":
            # We can't confirm whether the order actually filled (lost network
            # confirmation). Do NOT update in-memory position state on a guess
            # -- that risks placing a duplicate order into an already-open
            # position on the next poll. Leave state untouched and require a
            # manual reconciliation against the exchange before continuing.
            print(f"[LiveTrader] {symbol}: order state UNKNOWN after network error -- "
                  "skipping state update, please reconcile manually against the exchange.")
            return False
        return True

    def run_forever(self, poll_interval_s: float = 60.0) -> None:
        while True:
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive on transient errors
                print(f"[LiveTrader] error during poll: {exc}")
            time.sleep(poll_interval_s)


if __name__ == "__main__":
    trader = LiveTrader(TradingBotConfig())
    trader.run_forever()
