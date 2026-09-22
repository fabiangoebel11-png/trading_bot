"""Shadow-mode audit logging: rule strategy vs. ML confirmation, bar-by-bar.

The rule strategy stays the sole driver of real orders. This module only
*observes* what the ML confirmation layer would have done on the identical
bar and persists both sides side-by-side -- it never influences the trade
itself. See ``core/config.py: TrendMLConfig.production_enabled`` for the
manual-only switch that would ever let ML affect real sizing, and
``core/ml/promotion.py`` for the (read-only) gate that recommends whether
that switch should be flipped.

Costs use the same blended maker/taker fee model as
``core/backtester.py: _finalize_symbol_backtest`` (fill probability blended
with a taker+slippage fallback).

Two ways to build the comparison, in increasing order of realism:
  - ``build_shadow_log_from_backtest`` (preferred for historical/illustrative
    use): both sides come from ``core.backtester.run_rule_vs_ml_backtest``, so
    macro gate, ATR stop, cooldown, execution latency, funding cost and
    shared-wallet leverage are IDENTICAL between the rule and ML runs.
  - ``PortfolioShadowSimulator``/``PortfolioShadowLogger`` (live, per-poll,
    PREFERRED for a genuine promotion decision): each variant maintains its
    OWN hypothetical shared-wallet capital trajectory (cold-start ratchet,
    absolute-notional cap, funding cost included via a periodic funding-rate
    fetch) -- never the real account's fetched equity, which only ever
    reflects whichever variant is actually being live-traded. Real money
    trades as ONE shared wallet, not N independent per-symbol books, so this
    is portfolio-level, not per-symbol.
  - ``build_shadow_log``/``ShadowLogger`` (live, per-symbol, diagnostic only):
    both variants go through the exact same stop/cooldown/leverage code path
    (see ``execution/live_trader.py: _build_candidate``), but leverage is
    still resolved per-symbol, independent of the other currently-open
    symbols' shared-wallet interaction -- useful for per-symbol debugging,
    NOT a substitute for the portfolio-level simulator above.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from core.config import CapitalConfig, ExecutionConfig
from core.risk import cap_absolute_notional, cold_start_scale_live, resolve_shared_wallet_weights

SHADOW_LOG_COLUMNS = [
    "timestamp",
    "symbol",
    "rule_position",
    "ml_position",
    "ml_confidence",
    "regime",
    "atr",
    "close",
    "rule_cost",
    "ml_cost",
    "rule_pnl",
    "ml_pnl",
]


def blended_fee(execution_config: ExecutionConfig) -> float:
    """Same formula as ``core.backtester._finalize_symbol_backtest``'s
    ``blended_fee`` -- kept as a standalone helper so the shadow log's cost
    assumption never silently drifts from the real backtester's."""
    fill_prob = execution_config.maker_fill_probability
    return float(
        fill_prob * execution_config.maker_fee
        + (1 - fill_prob) * (execution_config.taker_fee + execution_config.slippage_bps_on_taker_fallback / 10_000)
    )


def build_shadow_log_from_backtest(
    symbol: str,
    rule_result: pd.DataFrame,
    ml_result: pd.DataFrame,
) -> pd.DataFrame:
    """Parity-correct shadow comparison from two FULL per-symbol backtest
    result frames (``core.backtester.run_rule_vs_ml_backtest(...)``'s
    ``portfolio.attrs["per_symbol"][symbol]`` for each side). Both frames
    already went through identical macro gate, ATR stop, cooldown, latency,
    funding and shared-wallet leverage logic -- ``strategy_return`` is the
    true net PnL, ``fee_cost``/``funding_cost`` the true costs, no
    recomputation/approximation needed here."""
    index = rule_result.index.intersection(ml_result.index)
    confidence = (
        ml_result.loc[index, "ml_confidence"] if "ml_confidence" in ml_result.columns else pd.Series(0.0, index=index)
    )
    regime = ml_result.loc[index, "ml_regime"] if "ml_regime" in ml_result.columns else pd.Series(None, index=index)
    rule_funding = rule_result.loc[index, "funding_cost"] if "funding_cost" in rule_result.columns else 0.0
    ml_funding = ml_result.loc[index, "funding_cost"] if "funding_cost" in ml_result.columns else 0.0

    out = pd.DataFrame(
        {
            "symbol": symbol,
            "rule_position": rule_result.loc[index, "execution_position"].astype(float),
            "ml_position": ml_result.loc[index, "execution_position"].astype(float),
            "ml_confidence": confidence,
            "regime": regime,
            "atr": rule_result.loc[index, "atr"] if "atr" in rule_result.columns else np.nan,
            "close": rule_result.loc[index, "close"].astype(float),
            "rule_cost": rule_result.loc[index, "fee_cost"] + rule_funding,
            "ml_cost": ml_result.loc[index, "fee_cost"] + ml_funding,
            "rule_pnl": rule_result.loc[index, "strategy_return"],
            "ml_pnl": ml_result.loc[index, "strategy_return"],
        },
        index=index,
    )
    out.index.name = "timestamp"
    return out


def build_shadow_log(
    symbol: str,
    rule_signals: pd.DataFrame,
    ml_signals: pd.DataFrame,
    execution_config: ExecutionConfig,
) -> pd.DataFrame:
    """Simplified, leverage=1.0 bar-by-bar shadow comparison from two raw
    signal frames, WITHOUT macro gate/stop/cooldown/latency/funding/
    shared-wallet parity -- a lightweight fallback for contexts without a
    full backtest result (e.g. quick ad-hoc checks/unit tests). Prefer
    ``build_shadow_log_from_backtest`` for anything feeding a promotion
    decision."""
    index = rule_signals.index.intersection(ml_signals.index)
    rule_pos = rule_signals.loc[index, "position"].astype(float)
    ml_pos = ml_signals.loc[index, "position"].astype(float)
    close = rule_signals.loc[index, "close"].astype(float)
    atr = rule_signals.loc[index, "atr"] if "atr" in rule_signals.columns else pd.Series(np.nan, index=index)
    confidence = (
        ml_signals.loc[index, "ml_confidence"] if "ml_confidence" in ml_signals.columns else pd.Series(0.0, index=index)
    )
    regime = ml_signals.loc[index, "ml_regime"] if "ml_regime" in ml_signals.columns else pd.Series(None, index=index)

    price_return = close.pct_change().fillna(0.0)
    fee = blended_fee(execution_config)

    prev_rule = rule_pos.shift(1).fillna(0.0)
    prev_ml = ml_pos.shift(1).fillna(0.0)
    rule_cost = rule_pos.diff().abs().fillna(0.0) * fee
    ml_cost = ml_pos.diff().abs().fillna(0.0) * fee
    rule_pnl = prev_rule * price_return - rule_cost
    ml_pnl = prev_ml * price_return - ml_cost

    out = pd.DataFrame(
        {
            "symbol": symbol,
            "rule_position": rule_pos,
            "ml_position": ml_pos,
            "ml_confidence": confidence,
            "regime": regime,
            "atr": atr,
            "close": close,
            "rule_cost": rule_cost,
            "ml_cost": ml_cost,
            "rule_pnl": rule_pnl,
            "ml_pnl": ml_pnl,
        },
        index=index,
    )
    out.index.name = "timestamp"
    return out


class ShadowLogger:
    """Persistent, append-only per-bar shadow log for live trading.

    One CSV row per symbol per poll, same schema as ``build_shadow_log``.
    Keeps a tiny in-memory previous-bar cache per symbol so cost/PnL use the
    same causal "yesterday's position times today's return" formula as
    ``core.backtester`` (``execution_position`` there is itself the *lagged*
    signal, i.e. the same latency semantics), without needing full history in
    memory. Never touches order execution -- purely additive audit output.
    """

    def __init__(self, path: str, execution_config: ExecutionConfig) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fee = blended_fee(execution_config)
        self._prev: dict[str, dict] = {}

    def log_bar(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        rule_position: float,
        ml_position: float,
        ml_confidence: float,
        regime: str | None,
        atr: float | None,
        close: float,
        rule_leverage: float = 1.0,
        ml_leverage: float = 1.0,
    ) -> None:
        """``rule_leverage``/``ml_leverage`` should be the same
        shared-wallet-scaled final leverage used for real order sizing (see
        ``execution/live_trader.py: _build_candidate``) -- passing 1.0
        degrades to a leverage-normalized comparison only, which is still
        useful but not what real notional exposure would have looked like."""
        prev = self._prev.get(symbol)
        if prev is None or not prev["close"]:
            rule_cost = 0.0
            ml_cost = 0.0
            rule_pnl = 0.0
            ml_pnl = 0.0
        else:
            price_return = close / prev["close"] - 1.0
            # Mirrors core.backtester._finalize_symbol_backtest exactly:
            # cost = |position change| * blended_fee * current-bar leverage,
            # return = previous (already latency-lagged) position * this
            # bar's price return * current-bar leverage.
            rule_cost = abs(rule_position - prev["rule_position"]) * self._fee * rule_leverage
            ml_cost = abs(ml_position - prev["ml_position"]) * self._fee * ml_leverage
            rule_pnl = prev["rule_position"] * price_return * rule_leverage - rule_cost
            ml_pnl = prev["ml_position"] * price_return * ml_leverage - ml_cost
            # Same simulated forced-liquidation floor as the backtester.
            rule_pnl = max(rule_pnl, -0.99)
            ml_pnl = max(ml_pnl, -0.99)
        self._prev[symbol] = {"rule_position": rule_position, "ml_position": ml_position, "close": close}

        row = {
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            "symbol": symbol,
            "rule_position": rule_position,
            "ml_position": ml_position,
            "ml_confidence": ml_confidence,
            "regime": regime,
            "atr": atr,
            "close": close,
            "rule_cost": rule_cost,
            "ml_cost": ml_cost,
            "rule_pnl": rule_pnl,
            "ml_pnl": ml_pnl,
        }
        header_needed = not self.path.exists() or self.path.stat().st_size == 0
        pd.DataFrame([row], columns=SHADOW_LOG_COLUMNS).to_csv(self.path, mode="a", header=header_needed, index=False)

    def read_all(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=SHADOW_LOG_COLUMNS)
        return pd.read_csv(self.path, parse_dates=["timestamp"])


def is_funding_event_hour(timestamp: pd.Timestamp, funding_hours: tuple[int, ...] = (0, 8, 16)) -> bool:
    """Binance USDT-M perpetuals default to an 8h funding schedule anchored at
    00:00/08:00/16:00 UTC. Approximation used for live shadow funding-cost
    simulation only (``PortfolioShadowSimulator``) -- a handful of symbols may
    run a different schedule, but this matches the majors traded here."""
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.hour in funding_hours


PORTFOLIO_SHADOW_LOG_COLUMNS = [
    "timestamp",
    "rule_capital_usdt",
    "ml_capital_usdt",
    "rule_return",
    "ml_return",
    "rule_fee_cost_usdt",
    "ml_fee_cost_usdt",
    "rule_funding_cost_usdt",
    "ml_funding_cost_usdt",
]


class PortfolioShadowSimulator:
    """Maintains ONE variant's own hypothetical shared-wallet capital
    trajectory, so a promotion decision never reuses the real account's
    fetched equity/cold-start ratchet -- which only ever reflects whichever
    variant is actually being live-traded (see ``core/config.py:
    TrendMLConfig.production_enabled``). Mirrors the same shared-wallet
    weighting (inverse-leverage, risk-capped), cold-start scale-down and
    absolute-notional-cap logic as the real per-poll order-sizing block in
    ``execution/live_trader.py: poll_once`` -- just driven by this variant's
    own simulated capital instead of the real exchange equity.

    Uses dollar-based bookkeeping rather than the vectorized backtester's
    fractional-return matrix math: simpler to run incrementally per poll, and
    yields a single portfolio-level USDT capital number to compare rule vs.
    ML on directly (real money trades as ONE shared wallet, not N independent
    per-symbol books).
    """

    def __init__(
        self,
        initial_capital_usdt: float,
        cold_start_buffer_pct: float,
        cold_start_risk_scale: float,
        max_portfolio_risk_pct: float,
        max_absolute_position_size_usdt: float | None,
        fee: float,
        selection_enabled: bool = False,
        max_active_positions: int | None = None,
    ) -> None:
        self.capital = initial_capital_usdt
        self._initial_capital = initial_capital_usdt
        self._cold_start_buffer_pct = cold_start_buffer_pct
        self._cold_start_risk_scale = cold_start_risk_scale
        self._max_portfolio_risk_pct = max_portfolio_risk_pct
        self._max_absolute_position_size_usdt = max_absolute_position_size_usdt
        self._fee = fee
        self._selection_enabled = selection_enabled
        self._max_active_positions = max_active_positions
        self._cold_start_reached = False
        self._prev_signed_notional: dict[str, float] = {}
        self._prev_close: dict[str, float] = {}

    def step(
        self,
        candidates: dict[str, dict],
        funding_rates: dict[str, float] | None = None,
        funding_event: bool = False,
    ) -> dict:
        """``candidates``: symbol -> {"target_position", "leverage_candidate",
        "stop_pct", "close", optionally "score"} for THIS variant only (rule
        or ML), for every symbol in the traded universe this poll. Updates
        ``self.capital`` in place and returns this bar's realized
        return/cost breakdown."""
        funding_rates = funding_rates or {}

        active = {s: c for s, c in candidates.items() if c["target_position"] != 0}
        weights = resolve_shared_wallet_weights(candidates, self._selection_enabled, self._max_active_positions)
        implied_risk = sum(weights.get(s, 0.0) * active[s]["leverage_candidate"] * active[s]["stop_pct"] for s in active)
        scale = min(1.0, self._max_portfolio_risk_pct / implied_risk) if implied_risk > 0 else 1.0

        cold_start_scale, self._cold_start_reached = cold_start_scale_live(
            self.capital,
            self._cold_start_reached,
            self._initial_capital,
            self._cold_start_buffer_pct,
            self._cold_start_risk_scale,
        )

        pnl_usdt = 0.0
        fee_cost_usdt = 0.0
        funding_cost_usdt = 0.0
        next_signed_notional: dict[str, float] = {}
        for symbol, candidate in candidates.items():
            close = candidate["close"]
            target_position = candidate["target_position"]
            if target_position == 0:
                signed_notional = 0.0
            else:
                leverage = candidate["leverage_candidate"] * scale * cold_start_scale
                notional = self.capital * weights.get(symbol, 0.0) * leverage
                notional = cap_absolute_notional(notional, self._max_absolute_position_size_usdt)
                signed_notional = target_position * notional

            prev_notional = self._prev_signed_notional.get(symbol, 0.0)
            prev_close = self._prev_close.get(symbol)
            if prev_close:
                # Previous poll's notional (already latency-lagged, same
                # semantics as core.backtester's shifted execution_position)
                # times this bar's price move -- causal, no look-ahead.
                price_return = close / prev_close - 1.0
                pnl_usdt += prev_notional * price_return
                funding_rate = funding_rates.get(symbol, 0.0)
                if funding_event and funding_rate:
                    # Long (positive notional) pays a positive funding rate;
                    # short receives it -- same convention as core.funding.
                    funding_cost_usdt += prev_notional * funding_rate
            fee_cost_usdt += abs(signed_notional - prev_notional) * self._fee

            next_signed_notional[symbol] = signed_notional
            self._prev_close[symbol] = close

        self._prev_signed_notional = next_signed_notional

        net_pnl_usdt = pnl_usdt - fee_cost_usdt - funding_cost_usdt
        bar_return = net_pnl_usdt / self.capital if self.capital > 0 else 0.0
        bar_return = max(bar_return, -0.99)  # same simulated forced-liquidation floor as core.backtester
        self.capital = max(self.capital * (1 + bar_return), 0.0)

        return {
            "capital_usdt": self.capital,
            "return": bar_return,
            "fee_cost_usdt": fee_cost_usdt,
            "funding_cost_usdt": funding_cost_usdt,
        }


class PortfolioShadowLogger:
    """Persistent, append-only PORTFOLIO-level shadow log: one row per poll,
    each side (rule/ML) driven by its own ``PortfolioShadowSimulator``. This
    -- not the per-symbol ``ShadowLogger`` above -- is what
    ``core/ml/promotion.py: evaluate_promotion`` should be run against for a
    genuine portfolio-level promotion decision.
    """

    def __init__(
        self,
        path: str,
        capital_config: CapitalConfig,
        execution_config: ExecutionConfig,
        selection_enabled: bool = False,
        max_active_positions: int | None = None,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fee = blended_fee(execution_config)
        self.rule = PortfolioShadowSimulator(
            capital_config.initial_capital_usdt,
            capital_config.cold_start_buffer_pct,
            capital_config.cold_start_risk_scale,
            capital_config.max_portfolio_risk_pct,
            capital_config.max_absolute_position_size_usdt,
            fee,
            selection_enabled,
            max_active_positions,
        )
        self.ml = PortfolioShadowSimulator(
            capital_config.initial_capital_usdt,
            capital_config.cold_start_buffer_pct,
            capital_config.cold_start_risk_scale,
            capital_config.max_portfolio_risk_pct,
            capital_config.max_absolute_position_size_usdt,
            fee,
            selection_enabled,
            max_active_positions,
        )

    def step_and_log(
        self,
        timestamp: pd.Timestamp,
        rule_candidates: dict[str, dict],
        ml_candidates: dict[str, dict] | None,
        funding_rates: dict[str, float] | None = None,
        funding_event: bool = False,
    ) -> None:
        rule_result = self.rule.step(rule_candidates, funding_rates, funding_event)
        # ML disabled/unavailable this poll -- mirror the rule side so the ML
        # capital trajectory never silently drifts out of sync with a skipped
        # poll (should be rare: the caller only builds this logger when
        # config.ml.enabled).
        ml_result = self.ml.step(ml_candidates or rule_candidates, funding_rates, funding_event)

        row = {
            "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
            "rule_capital_usdt": rule_result["capital_usdt"],
            "ml_capital_usdt": ml_result["capital_usdt"],
            "rule_return": rule_result["return"],
            "ml_return": ml_result["return"],
            "rule_fee_cost_usdt": rule_result["fee_cost_usdt"],
            "ml_fee_cost_usdt": ml_result["fee_cost_usdt"],
            "rule_funding_cost_usdt": rule_result["funding_cost_usdt"],
            "ml_funding_cost_usdt": ml_result["funding_cost_usdt"],
        }
        header_needed = not self.path.exists() or self.path.stat().st_size == 0
        pd.DataFrame([row], columns=PORTFOLIO_SHADOW_LOG_COLUMNS).to_csv(
            self.path, mode="a", header=header_needed, index=False
        )

    def read_all(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=PORTFOLIO_SHADOW_LOG_COLUMNS)
        return pd.read_csv(self.path, parse_dates=["timestamp"])
