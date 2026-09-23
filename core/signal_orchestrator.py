"""Human-in-the-loop fusion of swing setup and intraday entry signals."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np

from core.market_state import DataHealth, MarketState, PredictionSnapshot
from core.ml.scoring import continuous_opportunity_score


@dataclass(frozen=True)
class AlertGateConfig:
    """Policy only; this module never sends Telegram messages or places orders."""

    swing_threshold: float = 80.0
    intraday_threshold: float = 60.0
    require_same_direction: bool = True
    require_fresh_market_data: bool = True


@dataclass(frozen=True)
class FusionDecision:
    allowed: bool
    reason: str
    asset: str
    direction: str
    swing_score: float
    intraday_score: float
    swing_threshold: float
    intraday_threshold: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def intraday_snapshot(
    asset: str,
    timestamp,
    outputs: dict[str, float],
) -> PredictionSnapshot:
    """Convert one Model 1B multitask output into the fusion contract."""
    entry_score = float(outputs.get("entry_score", 100.0 * outputs["opportunity"]))
    if not np.isfinite(entry_score) or not 0.0 <= entry_score <= 100.0:
        raise ValueError("Model 1B entry_score must be finite and within [0, 100]")
    return PredictionSnapshot(
        score=entry_score,
        direction="UNKNOWN",
        expected_return=0.0,
        expected_duration_bars=float(outputs.get("expected_duration", 0.0)),
        expected_mfe=float(outputs["expected_mfe"]),
        expected_mae=float(outputs["expected_mae"]),
        timestamp=timestamp,
        model_type="intraday",
        asset=asset,
    )


class AlertGate:
    """Gate QQQ/SPY alerts on setup quality and entry timing together."""

    def __init__(self, config: AlertGateConfig | None = None) -> None:
        self.config = config or AlertGateConfig()

    def evaluate(
        self,
        swing: PredictionSnapshot,
        intraday: PredictionSnapshot,
        market: MarketState,
        health: DataHealth,
    ) -> FusionDecision:
        config = self.config
        base = {
            "asset": swing.asset,
            "direction": swing.direction,
            "swing_score": float(swing.swing_opportunity_score if swing.swing_opportunity_score is not None else swing.score),
            "intraday_score": float(intraday.score),
            "swing_threshold": config.swing_threshold,
            "intraday_threshold": config.intraday_threshold,
        }
        if swing.asset not in {"QQQ", "SPY"} or intraday.asset != swing.asset:
            return FusionDecision(False, "fusion is restricted to matching QQQ/SPY assets", **base)
        values = (base["swing_score"], base["intraday_score"], swing.expected_return, intraday.expected_return)
        if not all(np.isfinite(value) for value in values):
            return FusionDecision(False, "non-finite swing or intraday prediction", **base)
        if base["swing_score"] <= config.swing_threshold:
            return FusionDecision(False, "swing opportunity is not strictly above threshold", **base)
        if base["intraday_score"] < config.intraday_threshold:
            return FusionDecision(False, "intraday entry momentum is below threshold", **base)
        if swing.direction not in {"LONG", "SHORT"} or intraday.direction not in {"LONG", "SHORT"}:
            return FusionDecision(False, "direction is uncertain", **base)
        if not market.market_open or market.session_type not in {"REGULAR", "24_7"}:
            return FusionDecision(False, f"market session not alertable: {market.session_type}", **base)
        if config.require_fresh_market_data and (not health.feed_healthy or not health.freshness_ok):
            return FusionDecision(False, "primary data stale or unhealthy", **base)
        if not health.context_available or not health.context_freshness_ok:
            return FusionDecision(False, "context unavailable or stale", **base)
        return FusionDecision(True, "swing setup and intraday entry both passed", **base)


def evaluate_alert_gate(
    swing: PredictionSnapshot,
    intraday: PredictionSnapshot,
    market: MarketState,
    health: DataHealth,
    config: AlertGateConfig | None = None,
) -> FusionDecision:
    return AlertGate(config).evaluate(swing, intraday, market, health)
