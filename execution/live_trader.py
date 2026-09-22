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

Polling is candle-close-aligned by default (see ``run_forever``): since
``core.config.DataConfig.base_timeframe`` is 1h system-wide (data, ML
confirmation model and execution all unified after the 2026-09-22 Monte Carlo
stress tests showed 5m execution was catastrophic even after correct time/
volatility scaling -- see ``TrendMLConfig`` docstring), a strategy whose
signal can only change once per hour gains nothing from polling every few
seconds in between and would only burn exchange API rate-limit budget for no
new information.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from core.backtester import _periods_per_year
from core.config import TradingBotConfig
from core.data_loader import drop_incomplete_last_candle, fetch_ohlcv_history, resample_ohlcv
from core.macro import apply_macro_gate, fetch_macro_data
from core.ml.shadow import PortfolioShadowLogger, ShadowLogger, is_funding_event_hour
from core.risk import (
    apply_dynamic_stop_with_level,
    apply_trade_cooldown,
    cap_absolute_notional,
    cap_total_notional,
    cold_start_scale_live,
    dynamic_leverage,
    dynamic_stop_distance,
    resolve_shared_wallet_weights,
    risk_based_leverage_cap,
    select_discrete_exchange_leverage,
    shared_wallet_risk_scale,
)
from core.strategy import generate_trend_signals
from execution.audit import LiveAudit, LiveSafetyState
from execution.exchange_client import BinanceFuturesClient


def _seconds_until_next_candle_close(timeframe: str, buffer_s: float) -> float:
    """Seconds to sleep until shortly after the next ``timeframe`` candle
    closes (e.g. the next top-of-hour for "1h"), plus a small buffer so the
    exchange has finished finalizing/publishing that candle before we poll --
    aligns the poll loop to when new information can actually arrive instead
    of an arbitrary fixed interval."""
    now = pd.Timestamp.utcnow().tz_localize(None)
    duration = pd.Timedelta(timeframe)
    epoch = pd.Timestamp("1970-01-01")
    elapsed_in_candle = (now - epoch) % duration
    seconds_to_close = (duration - elapsed_in_candle).total_seconds()
    return max(0.0, seconds_to_close) + buffer_s


class LiveTrader:
    """Trades every symbol in ``config.data.symbols`` out of one shared wallet
    (one exchange client per process, multiple symbols share both the client
    and the capital/risk budget -- see module docstring)."""

    def __init__(self, config: TradingBotConfig, macro_refresh_every_polls: int = 24) -> None:
        self.config = config
        self.client = BinanceFuturesClient()
        self.audit = LiveAudit(config.execution.audit_log_path)
        self.safety_state = LiveSafetyState(config.execution.state_path)
        self._entries_halted = bool(self.safety_state.data.get("entries_halted", False)) or not config.execution.trading_enabled
        self._last_reconciliation = 0.0
        # Exchange-side leverage CEILING + cross-margin mode, set ONCE per
        # symbol at startup (never per order -- see ExecutionConfig docstring).
        # Failure here (e.g. an already-open position blocking a margin-mode
        # switch) must never prevent the bot from starting; it degrades to
        # whatever mode/leverage the account already has.
        try:
            self.client.configure_account(
                config.data.symbols,
                config.execution.exchange_leverage_ceiling,
                config.execution.exchange_margin_mode,
            )
        except Exception as exc:  # noqa: BLE001 - startup account setup must not be fatal
            print(f"[LiveTrader] configure_account failed: {exc} -- continuing with account's current settings")
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
        # Server-side catastrophe-stop tracking (order id + last-quoted stop
        # price per symbol). ``None`` means "no order tracked in-memory yet"
        # -- deliberately ambiguous between "never placed" and "placed in a
        # prior process that crashed/restarted", so it always triggers the
        # exchange-side reconciliation in ``_sync_stop_loss`` before assuming
        # either.
        self.active_stop_orders: dict[str, str | None] = {symbol: None for symbol in config.data.symbols}
        self.active_stop_prices: dict[str, float | None] = {symbol: None for symbol in config.data.symbols}
        self.pending_order_ids: dict[str, set[str]] = {symbol: set() for symbol in config.data.symbols}
        # ATR stop *distance* (price units) per symbol, refreshed only once per
        # candle close in ``poll_once`` -- the high-frequency risk loop below
        # reuses this fixed distance against a fresh real-time price to ratchet
        # the stop intra-candle, it never recomputes ATR itself from partial data.
        self.active_stop_distance: dict[str, float | None] = {symbol: None for symbol in config.data.symbols}
        # ML confirmation model cache (Problem: previously this layer was only
        # ever exercised by the backtester's ``generate_portfolio_signals`` --
        # never wired into live trading at all, a real backtest/live gap).
        # Models are loaded once and reused for the life of this process
        # (unlike the backtester's per-fold fresh instantiation) since a
        # long-running live loop should not repeatedly pay GPU init cost.
        self._ml_models: dict[str, tuple] = {}
        self._ml_models_attempted: set[str] = set()
        self._ml_macro_df = None
        self._ml_breadth_return = None
        # Shadow-mode audit logger (Konzept: ML runs parallel to the rule
        # strategy, never affecting real orders unless a human has manually
        # set ``config.ml.production_enabled = True`` -- see
        # ``core/config.py: TrendMLConfig.production_enabled`` and
        # ``core/ml/promotion.py``). Only instantiated when the ML layer is
        # enabled at all, so a plain rule-only deployment writes nothing.
        self._shadow_logger = (
            ShadowLogger(config.ml.shadow_log_path, config.execution) if config.ml.enabled else None
        )
        self._pending_shadow_rows: dict[str, dict] = {}
        # Portfolio-level (shared-wallet) shadow simulation -- see
        # core/ml/shadow.py: PortfolioShadowSimulator. Independent of the
        # per-symbol ShadowLogger above; this is what a genuine promotion
        # decision must be based on (real money trades as ONE shared wallet).
        self._portfolio_shadow_logger = (
            PortfolioShadowLogger(
                config.ml.portfolio_shadow_log_path,
                config.capital,
                config.execution,
                config.selection.enabled,
                config.selection.max_active_positions,
            )
            if config.ml.enabled
            else None
        )
        self._funding_rates: dict[str, float] = {}
        # Discrete exchange leverage state (Bybit/Binance only accept
        # specific tiers, see core.risk.select_discrete_exchange_leverage):
        # tiers cached per symbol for the process lifetime (rarely change),
        # last actually-SET value tracked so set_leverage is only called again
        # when the discretized target changes, not every poll.
        self._leverage_steps: dict[str, list[float]] = {}
        self._current_exchange_leverage: dict[str, float] = {}
        self.audit.write("startup", entries_halted=self._entries_halted, symbols=config.data.symbols)

    def _halt_entries(self, reason: str, flatten: bool = False) -> None:
        """Persistently stop new exposure; optionally flatten every position."""
        if not self._entries_halted:
            print(f"[LiveTrader] ENTRY HALT: {reason}")
            self.audit.write("entry_halt", reason=reason, flatten=flatten)
        self._entries_halted = True
        self.safety_state.data["entries_halted"] = True
        self.safety_state.data["halt_reason"] = reason
        self.safety_state.save()
        if flatten:
            for symbol in self.config.data.symbols:
                self._emergency_flatten(symbol)

    def _evaluate_account_risk(self, equity: float) -> bool:
        """Update persistent equity baselines and enforce hard account limits."""
        if equity <= 0:
            self._halt_entries("invalid_account_equity", flatten=False)
            return False

        today = pd.Timestamp.utcnow().date().isoformat()
        state = self.safety_state.data
        if state.get("daily_baseline_date") != today:
            state["daily_baseline_date"] = today
            state["daily_baseline_equity"] = equity
        high_water = max(float(state.get("high_water_equity", equity)), equity)
        state["high_water_equity"] = high_water
        state["last_equity"] = equity
        self.safety_state.save()

        daily_baseline = float(state["daily_baseline_equity"])
        daily_loss = max(0.0, 1.0 - equity / daily_baseline) if daily_baseline > 0 else 0.0
        drawdown = max(0.0, 1.0 - equity / high_water) if high_water > 0 else 0.0
        self.audit.write("account_risk_check", equity=equity, daily_loss_pct=daily_loss, drawdown_pct=drawdown)

        if daily_loss >= self.config.execution.max_daily_loss_pct:
            self._halt_entries(f"max_daily_loss:{daily_loss:.4%}", flatten=True)
            return False
        if drawdown >= self.config.execution.max_account_drawdown_pct:
            self._halt_entries(f"max_account_drawdown:{drawdown:.4%}", flatten=True)
            return False
        return True

    def _reconcile_exchange_state(self, reason: str) -> bool:
        """Make the exchange authoritative for positions, open orders and stops."""
        try:
            for symbol in self.config.data.symbols:
                exchange_amount = self.client.fetch_signed_position_amount(symbol, strict=True)
                previous_amount = self.current_position_amount[symbol]
                if not np.isclose(exchange_amount, previous_amount, atol=1e-12):
                    self.audit.write(
                        "position_reconciled",
                        reason=reason,
                        symbol=symbol,
                        local_amount=previous_amount,
                        exchange_amount=exchange_amount,
                    )
                    self.current_position_amount[symbol] = exchange_amount
                    self.current_positions[symbol] = int(np.sign(exchange_amount))

                open_orders = self.client.fetch_open_orders(symbol, strict=True)
                open_ids = {str(order.get("id")) for order in open_orders if order.get("id") is not None}
                self.pending_order_ids[symbol].intersection_update(open_ids)
                tracked_stop = self.active_stop_orders.get(symbol)
                if tracked_stop is not None and tracked_stop not in open_ids:
                    self.audit.write("tracked_stop_missing", reason=reason, symbol=symbol, order_id=tracked_stop)
                    self.active_stop_orders[symbol] = None
                    self.active_stop_prices[symbol] = None
                exchange_stop_ids = {
                    str(order["id"])
                    for order in open_orders
                    if order.get("id") is not None
                    and (order.get("type") or "").lower() in ("stop_market", "stop", "stop_loss")
                    and (order.get("reduceOnly") or (order.get("info") or {}).get("reduceOnly"))
                }
                if self.active_stop_orders.get(symbol) is None and exchange_stop_ids:
                    self.active_stop_orders[symbol] = next(iter(exchange_stop_ids))
                known_ids = self.pending_order_ids[symbol] | exchange_stop_ids
                unexpected_ids = open_ids - known_ids
                if unexpected_ids:
                    self._halt_entries(f"unexpected_open_orders:{symbol}:{sorted(unexpected_ids)}", flatten=False)
            self._last_reconciliation = time.monotonic()
            self.audit.write("reconciliation_complete", reason=reason)
            return True
        except Exception as exc:  # noqa: BLE001 - unsafe state must block new entries
            self._halt_entries(f"reconciliation_failed:{reason}:{exc}", flatten=False)
            return False

    def _halted_target(self, signed_current: float, signed_target: float) -> float:
        """Allow only risk-reducing orders while the account is halted."""
        if signed_current == 0.0:
            return 0.0
        if signed_target == 0.0 or np.sign(signed_target) != np.sign(signed_current):
            return 0.0
        return signed_target if abs(signed_target) < abs(signed_current) else signed_current

    def _resolve_exchange_leverage(self, symbol: str, required_leverage: float) -> float:
        """Discretize the risk engine's continuous vol-targeted leverage down
        to the smallest exchange-allowed step that still covers it, capped at
        ``RiskConfig.max_leverage`` (``core.risk.
        select_discrete_exchange_leverage``) -- a real exchange never accepts
        a fantasy leverage like "2.37x". Only calls ``set_leverage`` when the
        discretized value actually changed since the last poll (matches the
        existing "set once, not per order" convention)."""
        if symbol not in self._leverage_steps:
            self._leverage_steps[symbol] = self.client.fetch_leverage_brackets(
                symbol, self.config.execution.fallback_leverage_steps
            )
        discrete_leverage = select_discrete_exchange_leverage(
            required_leverage, self._leverage_steps[symbol], self.config.risk.max_leverage
        )
        if not np.isclose(self._current_exchange_leverage.get(symbol, -1.0), discrete_leverage):
            try:
                self.client.set_leverage(symbol, discrete_leverage)
                self._current_exchange_leverage[symbol] = discrete_leverage
            except Exception as exc:  # noqa: BLE001 - a leverage adjustment failure must not crash the poll
                print(f"[LiveTrader] {symbol}: set_leverage({discrete_leverage}) failed: {exc}")
        return discrete_leverage

    def _send_order(self, symbol: str, delta: float, signed_target: float, reduce_only: bool) -> str:
        """Send a single order for ``delta`` (signed base-asset quantity).
        Returns ``"ok"`` (nothing to do, or state confirmed), ``"failed"``
        (order rejected/skipped -- non-fatal, this symbol's sequence stops
        but other symbols keep polling), or ``"abort_poll"`` (a live order
        went through but exchange-state reconciliation afterward failed --
        state is uncertain, the whole poll must stop, mirrors the prior
        inline behavior in ``poll_once``)."""
        if abs(delta) < 1e-12:
            return "ok"
        side = "buy" if delta > 0 else "sell"
        if not self._rebalance(symbol, side, abs(delta), reduce_only=reduce_only):
            return "failed"
        if self.client.dry_run:
            self.current_position_amount[symbol] = signed_target
            self.current_positions[symbol] = int(np.sign(signed_target))
            return "ok"
        return "ok" if self._reconcile_exchange_state("post_order") else "abort_poll"

    def _execute_symbol_delta(self, symbol: str, signed_current: float, signed_target: float) -> str:
        """Move ``symbol`` from ``signed_current`` to ``signed_target``. A
        direct long<->short flip is NEVER sent as one oversized order (that
        is execution-risky -- a partial fill/reject could leave an ambiguous
        mixed exposure): the existing side is fully closed (reduce-only)
        first, the exchange state is confirmed flat, and only then is the
        opposite side opened as a separate order."""
        if self._entries_halted:
            signed_target = self._halted_target(signed_current, signed_target)

        is_flip = signed_current != 0 and signed_target != 0 and np.sign(signed_current) != np.sign(signed_target)
        if not is_flip:
            # reduce_only ONLY for a strict same-direction shrink (or full
            # close to flat) -- covers the shrink/grow/open cases, all of
            # which are safe as a single order.
            reduce_only = signed_current != 0 and (
                signed_target == 0
                or (np.sign(signed_target) == np.sign(signed_current) and abs(signed_target) < abs(signed_current))
            )
            return self._send_order(symbol, signed_target - signed_current, signed_target, reduce_only)

        close_status = self._send_order(symbol, -signed_current, 0.0, reduce_only=True)
        if close_status != "ok":
            return close_status  # closing leg failed/uncertain -- never compound with an opening order

        if not self.client.dry_run:
            if not self._reconcile_exchange_state("pre_flip_confirm"):
                return "abort_poll"
            if not np.isclose(self.current_position_amount[symbol], 0.0, atol=1e-9):
                print(f"[LiveTrader] {symbol}: Flip abgebrochen -- Exchange zeigt nach Close keine flache Position.")
                return "failed"

        return self._send_order(symbol, signed_target, signed_target, reduce_only=False)

    def _load_recent_ohlc(self, symbol: str) -> pd.DataFrame:
        cfg = self.config.data
        lookback_days = max(3, cfg.history_days // 100)  # small rolling window, warm-up only
        if self.config.ml.enabled:
            # The ML confirmation model needs its own, longer warm-up: the
            # 1d higher-timeframe features (see core/ml/features.py:
            # build_higher_timeframe_features) reuse a 100-period slow EMA,
            # i.e. a ~100-*day* trend filter, plus ``sequence_length`` (128h)
            # valid rows after that warm-up before the model can predict at
            # all. A too-short live fetch would silently leave those features
            # NaN forever (never actually broken, just permanently neutral --
            # exactly the kind of gap that's easy to miss without this note).
            lookback_days = max(lookback_days, 150)
        raw = fetch_ohlcv_history(symbol, cfg.base_timeframe, lookback_days, cache_dir=cfg.cache_dir, use_cache=False)
        raw = drop_incomplete_last_candle(raw, cfg.base_timeframe)
        resampled = resample_ohlcv(raw, cfg.resample_to) if cfg.resample_to else raw
        return drop_incomplete_last_candle(resampled, cfg.resample_to) if cfg.resample_to else resampled

    def _apply_ml_confirmation(self, symbol: str, signals: pd.DataFrame) -> pd.DataFrame:
        """Live-trading counterpart of ``core.strategy.generate_portfolio_
        signals``'s ML block. Deliberately uses ``predict_trend_confidence``
        with the full-history-trained final model (``load_symbol_model``),
        NOT the out-of-sample confidence (``load_oos_confidence``) that
        ``generate_portfolio_signals`` uses for backtesting -- here that's
        correct rather than a leak: live inference only ever scores bars at
        or after the model's training cutoff (i.e. the present), so there is
        no future information for the final model to have leaked from. Same
        causal feature pipeline (``core.ml.features.build_feature_matrix``),
        just applied per-symbol per-poll
        instead of over a full historical DataFrame. Order matches the
        backtester exactly: ML confidence scaling happens BEFORE the macro
        gate (``core.backtester.run_backtest_from_signals`` calls
        ``generate_portfolio_signals`` -- which includes this step -- then
        ``apply_macro_gate`` afterward)."""
        if not self.config.ml.enabled:
            return signals
        from core.ml.inference import apply_regime_gated_ml_confirmation, load_symbol_model, predict_trend_confidence

        if symbol not in self._ml_models and symbol not in self._ml_models_attempted:
            self._ml_models_attempted.add(symbol)
            try:
                self._ml_models[symbol] = load_symbol_model(symbol, self.config.ml)
            except FileNotFoundError:
                print(f"[LiveTrader] {symbol}: kein ML-Modell gefunden -- nutze rohes Signal.")

        cached = self._ml_models.get(symbol)
        if cached is None:
            return signals
        model, meta = cached
        try:
            confidence = predict_trend_confidence(
                signals, self._ml_macro_df, model, meta, self.config.ml, self._ml_breadth_return
            )
            return apply_regime_gated_ml_confirmation(signals, confidence, self.config.ml)
        except Exception as exc:  # noqa: BLE001 - never let a feature/inference bug block the risk-safe raw signal
            print(f"[LiveTrader] {symbol}: ML-Konfidenz uebersprungen ({exc}) -- nutze rohes Signal.")
            return signals

    def _build_candidate(self, signals: pd.DataFrame) -> dict:
        """Shared macro/stop/cooldown/leverage pipeline, identical for the
        rule and ML variants (see ``_symbol_candidate`` below) -- guarantees
        both are scored through byte-identical logic, differing only in
        whichever ``signals["position"]`` they were handed."""
        signals = apply_macro_gate(signals, self.config, self._macro_df)

        close = signals["close"]
        stop_distance = dynamic_stop_distance(close, self.config.risk)
        position, stop_level = apply_dynamic_stop_with_level(signals["position"], close, stop_distance)
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
        last_stop_level = stop_level.iloc[-1]
        last_stop_distance = stop_distance.iloc[-1]
        # Ranking score for the opt-in position-selection variant
        # (PositionSelectionConfig, default disabled) -- same formula as
        # core.backtester._symbol_candidate_frame, purely derived from
        # already-computed causal columns (EMA regime divergence, ATR%).
        ema_fast, ema_slow, atr = signals["ema_fast"].iloc[-1], signals["ema_slow"].iloc[-1], signals["atr"].iloc[-1]
        trend_strength = (ema_fast / ema_slow - 1.0) if ema_slow else 0.0
        atr_pct = (atr / close.iloc[-1]) if close.iloc[-1] else 0.0
        score = abs(trend_strength) / max(atr_pct, self.config.selection.score_epsilon)
        return {
            "target_position": int(position.iloc[-1]),
            "leverage_candidate": float(leverage_candidate),
            "stop_pct": float(stop_pct) if stop_pct == stop_pct else 0.0,  # NaN-safe
            "stop_price": float(last_stop_level) if last_stop_level == last_stop_level else None,  # NaN-safe
            "stop_distance": float(last_stop_distance) if last_stop_distance == last_stop_distance else None,  # NaN-safe
            "close": float(close.iloc[-1]),
            "score": float(score) if score == score else 0.0,  # NaN-safe
        }

    def _symbol_candidate(self, symbol: str) -> dict | None:
        """Latest-bar equivalent of ``core.backtester._symbol_candidate_frame``:
        target position (post stop/cooldown) plus the leverage this symbol
        could run if it were the wallet's only open position.

        The ML confirmation layer is ALWAYS computed and run through
        ``_build_candidate`` whenever ``config.ml.enabled`` (for audit/
        comparison purposes, with the SAME macro/stop/cooldown/leverage logic
        as the rule variant -- see ``_build_candidate``), but it only drives
        the actually traded candidate returned here when
        ``config.ml.production_enabled`` is also True (manual-only switch,
        see ``core/config.py``) -- otherwise the rule strategy's raw signal
        is what gets traded, unchanged, exactly as if ML were absent."""
        ohlc = self._load_recent_ohlc(symbol)
        raw_signals = generate_trend_signals(ohlc, self.config.trend)
        if raw_signals.empty:
            return None

        rule_candidate = self._build_candidate(raw_signals)
        ml_candidate = rule_candidate
        ml_signals = None
        if self.config.ml.enabled:
            ml_signals = self._apply_ml_confirmation(symbol, raw_signals)
            ml_candidate = self._build_candidate(ml_signals)
            self._log_shadow(symbol, raw_signals, ml_signals, rule_candidate, ml_candidate)

        active_candidate = ml_candidate if (self.config.ml.enabled and self.config.ml.production_enabled) else rule_candidate
        result = dict(active_candidate)
        result["rule"] = rule_candidate
        result["ml"] = ml_candidate if self.config.ml.enabled else None
        return result

    def _log_shadow(
        self,
        symbol: str,
        raw_signals: pd.DataFrame,
        ml_signals: pd.DataFrame,
        rule_candidate: dict,
        ml_candidate: dict,
    ) -> None:
        """Persist the latest bar's rule-vs-shadow-ML comparison, using the
        SAME post-macro/stop/cooldown position and shared-wallet-scaled
        leverage that would have been traded had that variant been active
        (see ``poll_once``, which fills in ``rule_leverage``/``ml_leverage``
        after the shared-wallet weight pass below). Never raises into the
        trading path -- a logging failure must not block a poll."""
        if self._shadow_logger is None:
            return
        try:
            confidence = float(ml_signals["ml_confidence"].iloc[-1]) if "ml_confidence" in ml_signals.columns else 0.0
            regime = str(ml_signals["ml_regime"].iloc[-1]) if "ml_regime" in ml_signals.columns else None
            atr = float(raw_signals["atr"].iloc[-1]) if "atr" in raw_signals.columns else None
            self._pending_shadow_rows[symbol] = {
                "timestamp": raw_signals.index[-1],
                "rule_position": float(rule_candidate["target_position"]),
                "ml_position": float(ml_candidate["target_position"]),
                "ml_confidence": confidence,
                "regime": regime,
                "atr": atr,
                "close": float(raw_signals["close"].iloc[-1]),
            }
        except Exception as exc:  # noqa: BLE001 - shadow logging is diagnostic only
            print(f"[LiveTrader] {symbol}: Shadow-Log fehlgeschlagen ({exc})")

    def _shared_wallet_leverage(self, candidates: dict[str, dict], cold_start_scale: float) -> dict[str, float]:
        """Same shared-wallet weighting/risk-capped scaling as the real-order
        block in ``poll_once`` (Konzept 1 & 2, plus the opt-in position-
        selection variant -- see ``core.risk.resolve_shared_wallet_weights``),
        applied to an arbitrary candidate set -- used both for the actually
        traded variant and, for the other (shadow-only) variant, purely so
        the shadow log reflects the leverage that variant would really have
        run at instead of an unweighted leverage=1.0 placeholder."""
        active = {s: c for s, c in candidates.items() if c["target_position"] != 0}
        weights = resolve_shared_wallet_weights(
            candidates, self.config.selection.enabled, self.config.selection.max_active_positions
        )
        implied_risk = sum(weights.get(s, 0.0) * active[s]["leverage_candidate"] * active[s]["stop_pct"] for s in active)
        scale = min(1.0, self.config.capital.max_portfolio_risk_pct / implied_risk) if implied_risk > 0 else 1.0
        return {
            symbol: (candidates[symbol]["leverage_candidate"] * scale * cold_start_scale) if symbol in weights else 0.0
            for symbol in candidates
        }

    def _flush_shadow_log(self, candidates: dict[str, dict], cold_start_scale: float) -> None:
        """Resolve shared-wallet leverage for both variants and persist the
        rows staged by ``_log_shadow`` this poll -- deliberately runs AFTER
        the real order block above has computed its own weights/scale so a
        shadow-logging bug can never delay/affect a real trade."""
        if self._shadow_logger is None or not self._pending_shadow_rows:
            return
        rule_candidates = {s: c["rule"] for s, c in candidates.items()}
        ml_candidates = {s: c["ml"] for s, c in candidates.items() if c["ml"] is not None}
        rule_leverage = self._shared_wallet_leverage(rule_candidates, cold_start_scale)
        ml_leverage = self._shared_wallet_leverage(ml_candidates, cold_start_scale) if ml_candidates else {}
        for symbol, row in self._pending_shadow_rows.items():
            try:
                self._shadow_logger.log_bar(
                    symbol=symbol,
                    timestamp=row["timestamp"],
                    rule_position=row["rule_position"],
                    ml_position=row["ml_position"],
                    ml_confidence=row["ml_confidence"],
                    regime=row["regime"],
                    atr=row["atr"],
                    close=row["close"],
                    rule_leverage=rule_leverage.get(symbol, 1.0),
                    ml_leverage=ml_leverage.get(symbol, 1.0),
                )
            except Exception as exc:  # noqa: BLE001 - shadow logging is diagnostic only
                print(f"[LiveTrader] {symbol}: Shadow-Log-Flush fehlgeschlagen ({exc})")
        self._pending_shadow_rows.clear()

    def _step_portfolio_shadow(self, candidates: dict[str, dict]) -> None:
        """Advance the portfolio-level shadow simulators by one poll (see
        core.ml.shadow.PortfolioShadowSimulator) -- runs unconditionally once
        per poll, independent of ``production_enabled``, so both hypothetical
        shared-wallet capital trajectories stay continuously up to date for a
        genuine portfolio-level promotion decision. Never raises into the
        trading path."""
        if self._portfolio_shadow_logger is None:
            return
        try:
            rule_candidates = {s: c["rule"] for s, c in candidates.items()}
            ml_candidates = (
                {s: c["ml"] for s, c in candidates.items() if c["ml"] is not None}
                if self.config.ml.enabled
                else None
            )
            timestamp = pd.Timestamp.utcnow().tz_localize(None)
            self._portfolio_shadow_logger.step_and_log(
                timestamp,
                rule_candidates,
                ml_candidates,
                funding_rates=self._funding_rates,
                funding_event=is_funding_event_hour(timestamp),
            )
        except Exception as exc:  # noqa: BLE001 - shadow simulation is diagnostic only
            print(f"[LiveTrader] Portfolio-Shadow-Simulation fehlgeschlagen: {exc}")

    def poll_once(self) -> None:
        if time.monotonic() - self._last_reconciliation >= self.config.execution.reconciliation_interval_s:
            if not self._reconcile_exchange_state("pre_signal_poll"):
                return

        equity = self.client.fetch_equity(fallback_equity=self.config.capital.initial_capital_usdt)
        if not self._evaluate_account_risk(equity):
            return
        refresh_slow_data = self._macro_df is None or self._polls_since_macro_refresh >= self.macro_refresh_every_polls

        if self.config.macro.enabled and refresh_slow_data:
            self._macro_df = fetch_macro_data(self.config.macro, use_cache=False)

        if self.config.ml.enabled and (self._ml_macro_df is None or refresh_slow_data):
            # Independent config objects from ``core.macro``'s
            # (``TrendMLConfig.macro_symbols``/``breadth_symbols``), but same
            # slow-moving-daily-data refresh cadence -- reuses the same
            # counter/threshold, no need for a second one.
            from core.ml.features import fetch_breadth_basket, fetch_macro_matrix

            self._ml_macro_df = fetch_macro_matrix(self.config.ml, use_cache=False)
            self._ml_breadth_return = fetch_breadth_basket(self.config.data, self.config.ml, use_cache=False)

        if self.config.funding.enabled and (not self._funding_rates or refresh_slow_data):
            # Current funding rate only, for shadow-mode PnL simulation (see
            # core.ml.shadow.PortfolioShadowSimulator) -- NOT used for real
            # order sizing, the exchange charges/credits funding directly on
            # whichever position is actually held.
            self._funding_rates = {
                symbol: self.client.fetch_current_funding_rate(symbol) for symbol in self.config.data.symbols
            }

        if refresh_slow_data:
            self._polls_since_macro_refresh = 0
        self._polls_since_macro_refresh += 1

        candidates = {}
        for symbol in self.config.data.symbols:
            try:
                candidate = self._symbol_candidate(symbol)
            except Exception as exc:  # noqa: BLE001 - one symbol's data/network hiccup must not abort the whole poll
                print(f"[LiveTrader] {symbol}: failed to build candidate this poll: {exc} -- skipping symbol")
                continue
            if candidate is not None:
                candidates[symbol] = candidate

        # --- Shared-wallet sizing across every symbol active THIS poll (Konzept 1 & 2,
        # plus the opt-in position-selection variant -- core.risk.resolve_shared_wallet_weights) ---
        active = {s: c for s, c in candidates.items() if c["target_position"] != 0}
        weights = resolve_shared_wallet_weights(
            candidates, self.config.selection.enabled, self.config.selection.max_active_positions
        )
        implied_risk = sum(
            weights.get(s, 0.0) * active[s]["leverage_candidate"] * active[s]["stop_pct"] for s in active
        )
        scale = min(1.0, self.config.capital.max_portfolio_risk_pct / implied_risk) if implied_risk > 0 else 1.0

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

        self._flush_shadow_log(candidates, cold_start_scale)
        self._step_portfolio_shadow(candidates)

        # --- Phase 1: target notional per symbol (Konzept 2's per-symbol cap
        # applied here; the exchange leverage actually set is the discretized
        # step that covers the desired notional, never a raw fractional
        # "2.37x") ---
        signed_notionals: dict[str, float] = {}
        for symbol, candidate in candidates.items():
            target_position = candidate["target_position"]
            if target_position == 0:
                signed_notionals[symbol] = 0.0
                continue
            weight = weights.get(symbol, 0.0)
            required_leverage = candidate["leverage_candidate"] * scale * cold_start_scale
            leverage = self._resolve_exchange_leverage(symbol, required_leverage)
            notional = equity * weight * leverage
            # --- Absolute margin ceiling (Konzept 2): hard USDT cap per
            # symbol regardless of how large the wallet has grown ---
            notional = cap_absolute_notional(notional, self.config.capital.max_absolute_position_size_usdt)
            signed_notionals[symbol] = target_position * notional

        # --- Hard portfolio-wide notional ceiling: scale EVERY symbol down
        # uniformly (never selectively) if the combined notional would still
        # exceed the budget -- applied after every per-symbol cap above. ---
        signed_notionals = cap_total_notional(signed_notionals, self.config.capital.max_total_notional_usdt)

        # --- Phase 2: compare against the real exchange position and trade
        # only the delta; a direction flip is split into close-then-open
        # (see _execute_symbol_delta) instead of one execution-risky order. ---
        for symbol, candidate in candidates.items():
            signed_target = signed_notionals.get(symbol, 0.0) / candidate["close"] if candidate["close"] else 0.0
            signed_current = self.current_position_amount[symbol]
            status = self._execute_symbol_delta(symbol, signed_current, signed_target)
            if status == "abort_poll":
                return

        # --- Catastrophe stop (Konzept 1 & 2) ---
        # Keep the exchange-side protective stop in sync with the
        # just-updated position and the latest ATR-trailing stop level for
        # every symbol currently holding a position (not only the ones
        # rebalanced this poll), and recover it from the exchange if this
        # process just (re)started without an in-memory record of it.
        for symbol, candidate in candidates.items():
            self._sync_stop_loss(
                symbol,
                self.current_positions[symbol],
                abs(self.current_position_amount[symbol]),
                candidate.get("stop_price"),
            )
            # Refreshed once per candle close only -- the ATR itself is a
            # candle-derived quantity (Problem B: must never be recomputed
            # from a partial/incomplete intra-hour candle). The high-frequency
            # risk loop reuses this fixed distance against fresh real-time
            # prices between candle closes (Problem A).
            self.active_stop_distance[symbol] = candidate.get("stop_distance")

    def _sync_stop_loss(
        self, symbol: str, target_position: int, amount: float, stop_price: float | None
    ) -> None:
        """Server-side 'catastrophe stop': mirrors the internal ATR-trailing
        stop as a real reduce-only STOP_MARKET order on the exchange, so the
        position stays protected even if this process crashes, loses network,
        or the poll loop stalls -- the in-memory ``apply_dynamic_stop_with_
        level`` ratchet alone only protects while the bot is actively polling.
        Also doubles as startup state recovery (Konzept 2): whenever this
        symbol has no in-memory order id tracked, the exchange is queried
        first for an already-open reduce-only stop before concluding one
        needs to be placed.
        """
        if target_position == 0 or stop_price is None or amount <= 0:
            # Flat (or about to be): no protective stop needed -- cancel any
            # leftover order instead of leaving a stray reduce-only stop open.
            self._cancel_tracked_stop(symbol)
            return

        position_side = "long" if target_position > 0 else "short"

        if self.active_stop_orders.get(symbol) is None:
            existing = self.client.fetch_open_stop_orders(symbol)
            if existing:
                order = existing[0]
                self.active_stop_orders[symbol] = order["id"]
                self.active_stop_prices[symbol] = float(
                    order.get("stopPrice") or order.get("price") or stop_price
                )

        current_price = self.active_stop_prices.get(symbol)
        # Only re-quote when the trailing stop has meaningfully moved --
        # avoids pointless cancel/replace churn on floating-point noise.
        moved = current_price is None or abs(stop_price - current_price) / stop_price > 1e-4
        if self.active_stop_orders.get(symbol) is not None and not moved:
            return

        self._cancel_tracked_stop(symbol)
        result = self.client.place_stop_loss_order(symbol, position_side, amount, stop_price)
        if result.status != "dry_run" and result.order_id is None:
            print(f"[LiveTrader] {symbol}: failed to place catastrophe stop at {stop_price}")
            return
        self.active_stop_orders[symbol] = result.order_id
        self.active_stop_prices[symbol] = stop_price

    def _cancel_tracked_stop(self, symbol: str) -> None:
        order_id = self.active_stop_orders.get(symbol)
        if order_id is not None:
            self.client.cancel_stop_loss_order(symbol, order_id)
        self.active_stop_orders[symbol] = None
        self.active_stop_prices[symbol] = None

    def risk_check_tick(self) -> None:
        """High-frequency safety net (Problem A: 'Sleep of Death' fix), run
        every ``risk_poll_interval_s`` (default 60s) *between* candle closes
        -- while ``poll_once`` (the signal-generation/rebalancing path) only
        runs once per closed candle.

        Deliberately narrow in scope: it only re-prices the *already known*
        ATR-trailing stop level against a fresh real-time last price (via
        ``CCXTExchangeClient.fetch_last_price``, a ticker call, not an OHLCV
        fetch) and ratchets it or force-flattens if breached. It never calls
        ``generate_trend_signals``/the ML confirmation model and never
        recomputes ATR/indicators from a partial, still-forming candle --
        that would reintroduce exactly the train-serve skew the model is
        trained to avoid (Problem B). The stop *distance* itself
        (``active_stop_distance``) is refreshed only once per candle close in
        ``poll_once``, from a fully closed candle, same as the backtester.

        This is intentionally redundant with the exchange-side STOP_MARKET
        catastrophe stop placed by ``_sync_stop_loss``: that order protects
        the account even if this whole process is down; this tick is a
        faster, software-side layer that can react within
        ``risk_poll_interval_s`` instead of waiting for the exchange to
        trigger its own resting order, and is also the natural place to add
        portfolio-wide (multi-symbol) real-time risk checks.
        """
        for symbol in self.config.data.symbols:
            try:
                self._risk_check_symbol(symbol)
            except Exception as exc:  # noqa: BLE001 - one symbol's failure must never stall/crash the risk loop
                print(f"[LiveTrader] risk-tick: {symbol} failed: {exc}")

    def _risk_check_symbol(self, symbol: str) -> None:
        position = self.current_positions[symbol]
        stop_price = self.active_stop_prices.get(symbol)
        distance = self.active_stop_distance.get(symbol)
        if position == 0 or stop_price is None or distance is None or distance <= 0:
            return

        last_price = self.client.fetch_last_price(symbol)

        if position > 0:
            candidate_stop = last_price - distance
            new_stop = max(stop_price, candidate_stop)
            breached = last_price <= new_stop
        else:
            candidate_stop = last_price + distance
            new_stop = min(stop_price, candidate_stop)
            breached = last_price >= new_stop

        if breached:
            print(
                f"[LiveTrader] risk-tick: {symbol} breached intra-candle stop "
                f"({last_price} vs {new_stop}) -- emergency taker flatten"
            )
            self._emergency_flatten(symbol)
            return

        if new_stop != stop_price:
            self.active_stop_prices[symbol] = new_stop
            # Keep the exchange-side resting stop order in sync with the
            # tightened level too, so a crash between now and the next
            # candle close still protects the ratcheted (not stale) price.
            self._sync_stop_loss(symbol, position, abs(self.current_position_amount[symbol]), new_stop)

    def _emergency_flatten(self, symbol: str) -> None:
        """Immediate, unconditional taker (market) close of ``symbol``'s full
        position, used when the fast risk-tick detects the real-time price
        has already breached the ATR-trailing stop between candle closes --
        a maker-first limit order is not appropriate here, the whole point is
        guaranteed, immediate execution.

        Three hardening steps beyond a plain market order, in order:
          1. Cancel every resting order for this symbol first (the tracked
             server-side catastrophe stop AND any other open order, e.g. a
             maker rebalance limit order that hasn't filled yet) -- a
             leftover order left open could otherwise fill/trigger *after*
             this flatten and silently re-open exposure.
          2. Reconcile against the exchange's actual position size rather
             than trusting only local bookkeeping: a nonzero exchange read
             wins; only if that read itself fails/returns 0 do we fall back
             to the local ``current_position_amount`` (never skip closing a
             position we believe is open just because one reconciliation
             call failed).
          3. Send the close tagged ``reduce_only=True`` (dust-safe: bypasses
             the minimum-notional floor so even a small remainder can always
             be fully closed, and the exchange itself refuses to let this
             order accidentally flip into a new position).
        """
        self.client.cancel_all_orders(symbol)
        self._cancel_tracked_stop(symbol)

        try:
            exchange_amount = self.client.fetch_signed_position_amount(symbol)
        except Exception as exc:  # noqa: BLE001 - fall back to local state below
            print(f"[LiveTrader] {symbol}: fetch_signed_position_amount failed during emergency flatten: {exc}")
            exchange_amount = 0.0
        local_amount = self.current_position_amount[symbol]
        # A 0.0 exchange read is ambiguous (genuinely flat vs. a failed fetch
        # that fell back to 0.0) -- a nonzero local state always wins as the
        # more conservative assumption (never skip closing a position we
        # believe is open).
        signed_amount = exchange_amount if exchange_amount else local_amount
        amount = abs(signed_amount)
        if amount <= 0:
            self.current_position_amount[symbol] = 0.0
            self.current_positions[symbol] = 0
            self.active_stop_distance[symbol] = None
            return

        side = "sell" if signed_amount > 0 else "buy"
        result = self.client.place_taker_fallback(symbol, side, amount, reduce_only=True)
        if result.status in ("filled", "dry_run"):
            if self.client.dry_run:
                self.current_position_amount[symbol] = 0.0
                self.current_positions[symbol] = 0
            else:
                self._reconcile_exchange_state("post_emergency_flatten")
            self.active_stop_distance[symbol] = None
        elif result.status != "skipped":
            print(f"[LiveTrader] {symbol}: emergency flatten status={result.status} -- state left untouched, reconcile manually")

    def _rebalance(self, symbol: str, side: str, order_amount: float, reduce_only: bool = False) -> bool:
        suffix = " [reduce_only]" if reduce_only else ""
        print(f"{symbol}: {side} {order_amount:.6f} (maker-first, taker fallback on failure){suffix}...")
        result = self.client.place_maker_order(symbol, side, amount=order_amount, reduce_only=reduce_only)
        self.audit.write(
            "order_result",
            symbol=symbol,
            side=side,
            requested_amount=order_amount,
            reduce_only=reduce_only,
            order_id=result.order_id,
            status=result.status,
            filled_price=result.filled_price,
            is_maker=result.is_maker,
        )
        if result.status == "skipped":
            # Amount rounds to zero or below the exchange's lot-size/min-notional
            # floor (plausible on a 500 USDT wallet split across active symbols)
            # -- not an error, just hold the current position until the target
            # notional grows enough (e.g. equity growth, fewer symbols active,
            # or leverage/weight shift) to clear the exchange minimum.
            print(f"[LiveTrader] {symbol}: order skipped (below exchange lot-size/min-notional) -- holding current position")
            return False
        if result.status == "unknown":
            # We can't confirm whether the order actually filled (lost network
            # confirmation). Do NOT update in-memory position state on a guess
            # -- that risks placing a duplicate order into an already-open
            # position on the next poll. Leave state untouched and require a
            # manual reconciliation against the exchange before continuing.
            print(f"[LiveTrader] {symbol}: order state UNKNOWN after network error -- "
                  "skipping state update, please reconcile manually against the exchange.")
            if result.order_id is not None:
                self.pending_order_ids[symbol].add(result.order_id)
            self._halt_entries(f"unknown_order_status:{symbol}", flatten=False)
            return False
        return True

    def run_forever(
        self,
        poll_interval_s: float = 60.0,
        max_backoff_s: float = 300.0,
        candle_aligned: bool = True,
        poll_buffer_s: float = 10.0,
        risk_poll_interval_s: float = 60.0,
    ) -> None:
        """Hybrid polling loop (Problem A fix): signal generation/rebalancing
        (``poll_once``, which touches OHLC candles and the ML confirmation
        model) only ever runs once per closed candle, but the ATR-trailing
        stop / portfolio risk check (``risk_check_tick``, real-time ticker
        prices only) runs every ``risk_poll_interval_s`` seconds in between --
        so a flash crash mid-candle is caught within seconds instead of only
        at the next hourly close, without ever feeding a partial/incomplete
        candle into the strategy or ML model (Problem B stays satisfied:
        ``risk_check_tick`` never touches signal generation).

        Resuming from wherever local/exchange state currently is: an exchange
        outage or transient error during a poll never crashes the loop, just
        backs off exponentially (capped at ``max_backoff_s``) and retries --
        ``poll_once``/``risk_check_tick`` re-derive everything they need
        (candles, equity, positions, prices) fresh from the exchange/local
        state each call, and only ever mutate in-memory state after a
        confirmed exchange response, so a failed poll/tick can never leave
        that state corrupted or half-applied.

        ``candle_aligned=True`` (default): after each successful signal poll,
        the loop sleeps in ``risk_poll_interval_s``-sized steps (running
        ``risk_check_tick`` at each step) until shortly after the *next*
        ``config.data.base_timeframe`` candle closes (e.g. the following
        top-of-hour + ``poll_buffer_s``) instead of a fixed
        ``poll_interval_s`` for the *signal* cadence -- the strategy/ML signal
        cannot change until a new candle closes, so recomputing it more often
        than that only burns exchange API rate-limit budget for no new
        information; the risk checks in between are cheap ticker calls, not
        full candle fetches. ``poll_interval_s`` is still used as the
        retry-backoff base on error, and as the fixed signal-poll cadence if
        ``candle_aligned=False`` (e.g. for fast local testing/debugging).
        """
        backoff_s = poll_interval_s
        timeframe = self.config.data.resample_to or self.config.data.base_timeframe
        while True:
            try:
                self.poll_once()
                backoff_s = poll_interval_s  # reset after any clean poll
            except Exception as exc:  # noqa: BLE001 - keep the loop alive on transient errors
                print(f"[LiveTrader] error during poll: {exc} -- retrying in {backoff_s:.0f}s")
                time.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, max_backoff_s)
                continue

            total_sleep_s = (
                _seconds_until_next_candle_close(timeframe, poll_buffer_s) if candle_aligned else poll_interval_s
            )
            remaining_s = total_sleep_s
            while remaining_s > 0:
                step_s = min(risk_poll_interval_s, remaining_s)
                time.sleep(step_s)
                remaining_s -= step_s
                if remaining_s <= 0:
                    break  # about to hit the candle close anyway, let poll_once run next
                try:
                    self.risk_check_tick()
                except Exception as exc:  # noqa: BLE001 - a failed risk tick must not kill the loop either
                    print(f"[LiveTrader] error during risk-tick: {exc}")


if __name__ == "__main__":
    config = TradingBotConfig()
    # Default module entry point: collect the frozen model's live shadow
    # evidence while the rule strategy remains the sole order driver.
    config.ml.enabled = True
    config.ml.production_enabled = False
    trader = LiveTrader(config)
    trader.run_forever()
