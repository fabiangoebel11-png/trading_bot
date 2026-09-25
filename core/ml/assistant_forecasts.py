"""Public forecast contract for the manual trading decision assistant.

The registry deliberately exposes only exact model horizons.  It never maps a
requested public horizon to a different prediction head: unsupported routes
remain explicit ``UNAVAILABLE`` records.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np


PUBLIC_HORIZONS = ("1h", "4h", "8h", "12h", "1d", "3d", "5d", "10d", "20d")


@dataclass(frozen=True)
class AssistantForecast:
    asset: str
    asset_class: str
    timeframe: str
    horizon: str
    forecast_source: str
    model: str | None
    model_version: str | None
    direction: str | None
    probability_short: float | None
    probability_neutral: float | None
    probability_long: float | None
    expected_return: float | None
    expected_mfe: float | None
    expected_mae: float | None
    expected_duration: float | None
    opportunity_score: float | None
    confidence: str
    forecast_timestamp: str | None
    target_timestamp: str | None
    data_quality: str
    forecast_status: str
    reason: str = ""
    quality_status: str = "UNVERIFIED"
    usable_for_decision: bool = False
    freshness: str = "UNKNOWN"
    forecast_age_minutes: float | None = None
    derived_from: str | None = None
    derivation_method: str | None = None
    model_sources: tuple[str, ...] = ()
    individual_forecasts: tuple[dict[str, Any], ...] = ()
    combination_method: str = "SINGLE_MODEL"
    combined_confidence: str = "UNKNOWN"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_forecast_age_minutes(timestamp: str | None) -> tuple[str, float | None]:
    if timestamp is None:
        return "UNKNOWN", None
    try:
        source = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if source.tzinfo is None:
            source = source.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        age = (now - source).total_seconds() / 60.0
        return "STALE" if age > 75.0 else "FRESH", age
    except ValueError:
        return "UNKNOWN", None


def _asset_horizon_oos_gate(asset: str, horizon: str) -> tuple[str, bool, str]:
    gate = {
        ("BTC/USDT", "1h"): ("DEGRADED", False, "OOS gate: 1h is degraded and must never trigger a decision."),
        ("BTC/USDT", "4h"): ("PARTIALLY_VALIDATED", True, "OOS gate: 4h is supportive-only and not a primary entry trigger."),
        ("BTC/USDT", "8h"): ("PARTIALLY_VALIDATED", True, "OOS gate: 8h is supportive-only and not a primary entry trigger."),
        ("BTC/USDT", "12h"): ("PARTIALLY_VALIDATED", False, "OOS gate: 12h remains contextual only and is not a primary entry trigger."),
        ("BTC/USDT", "1d"): ("PARTIALLY_VALIDATED", False, "OOS gate: 1d remains contextual only and is not a primary entry trigger."),
        ("ETH/USDT", "1h"): ("DEGRADED", False, "OOS gate: 1h is degraded and must never trigger a decision."),
        ("ETH/USDT", "4h"): ("PARTIALLY_VALIDATED", True, "OOS gate: 4h is supportive-only and not a primary entry trigger."),
        ("ETH/USDT", "8h"): ("PARTIALLY_VALIDATED", True, "OOS gate: 8h is supportive-only and not a primary entry trigger."),
        ("ETH/USDT", "12h"): ("PARTIALLY_VALIDATED", False, "OOS gate: 12h remains contextual only and is not a primary entry trigger."),
        ("ETH/USDT", "1d"): ("PARTIALLY_VALIDATED", False, "OOS gate: 1d remains contextual only and is not a primary entry trigger."),
    }.get((asset, horizon))
    if gate is not None:
        return gate
    return ("PARTIALLY_VALIDATED", False, "OOS gate: unsupported public horizon remains non-primary context only.")


def apply_oos_quality_gate(forecast: AssistantForecast) -> AssistantForecast:
    """Expose OOS quality metadata without hiding the forecast or promoting it to VALIDATED."""
    if forecast.forecast_source == "UNAVAILABLE":
        return replace(
            forecast,
            quality_status="UNAVAILABLE",
            usable_for_decision=False,
            freshness="UNKNOWN",
            forecast_age_minutes=None,
            reason=f"{forecast.reason} OOS gate: unavailable forecast cannot be used in the decision loop.".strip(),
        )

    quality_status, usable_for_decision, gate_reason = _asset_horizon_oos_gate(forecast.asset, forecast.horizon)
    if forecast.freshness in {"FRESH", "STALE"} and forecast.forecast_age_minutes is not None:
        freshness = forecast.freshness
        age_minutes = forecast.forecast_age_minutes
    else:
        freshness, age_minutes = _parse_forecast_age_minutes(forecast.forecast_timestamp)
    if freshness == "STALE":
        usable_for_decision = False
        gate_reason = f"{gate_reason} Forecast age is stale ({age_minutes:.1f} min); stale inputs are non-primary.".strip()
    if forecast.forecast_status in {"VALIDATED", "MODEL_AGREEMENT", "MODEL_DISAGREEMENT", "MODEL_PARTIAL_AGREEMENT"}:
        reduced_status = forecast.forecast_status
    else:
        reduced_status = quality_status if quality_status in {"DEGRADED", "PARTIALLY_VALIDATED"} else forecast.forecast_status

    return replace(
        forecast,
        forecast_status=reduced_status,
        quality_status=quality_status,
        usable_for_decision=usable_for_decision,
        freshness=freshness,
        forecast_age_minutes=age_minutes,
        reason=f"{forecast.reason} {gate_reason}".strip(),
    )


def unavailable_forecast(asset: str, asset_class: str, horizon: str, *, reason: str) -> AssistantForecast:
    return AssistantForecast(
        asset=asset,
        asset_class=asset_class,
        timeframe="1h" if horizon.endswith("h") else "1d",
        horizon=horizon,
        forecast_source="UNAVAILABLE",
        model=None,
        model_version=None,
        direction=None,
        probability_short=None,
        probability_neutral=None,
        probability_long=None,
        expected_return=None,
        expected_mfe=None,
        expected_mae=None,
        expected_duration=None,
        opportunity_score=None,
        confidence="UNKNOWN",
        forecast_timestamp=None,
        target_timestamp=None,
        data_quality="UNAVAILABLE",
        forecast_status="UNAVAILABLE",
        reason=reason,
        quality_status="UNAVAILABLE",
        usable_for_decision=False,
        freshness="UNKNOWN",
        forecast_age_minutes=None,
    )


def _valid_probabilities(values: tuple[float, float, float]) -> bool:
    return all(np.isfinite(values)) and abs(sum(values) - 1.0) <= 0.01


def direct_tcn_forecasts(
    *,
    asset: str,
    asset_class: str,
    model: str,
    model_version: str,
    timestamp: str,
    outputs: Mapping[int, Mapping[str, Any]],
) -> list[AssistantForecast]:
    """Normalize exact 1h TCN heads; all other public horizons stay unavailable."""
    by_public_horizon = {1: "1h", 4: "4h", 8: "8h", 12: "12h", 24: "1d"} if asset_class == "crypto" else {1: "1h", 4: "4h", 8: "8h", 12: "12h"}
    forecasts = {horizon: unavailable_forecast(asset, asset_class, horizon, reason="no exact registered model horizon") for horizon in PUBLIC_HORIZONS}
    for bars, public_horizon in by_public_horizon.items():
        output = outputs.get(bars)
        if output is None:
            continue
        probabilities = tuple(float(value) for value in output["probabilities"])
        numeric = (float(output["expected_return"]), float(output["expected_mfe"]), float(output["expected_mae"]), float(output["expected_duration"]))
        if not _valid_probabilities(probabilities) or not all(np.isfinite(numeric)):
            forecasts[public_horizon] = unavailable_forecast(asset, asset_class, public_horizon, reason="NUMERICAL_WARNING: invalid model output")
            continue
        direction = str(output["direction"])
        forecasts[public_horizon] = AssistantForecast(
            asset=asset,
            asset_class=asset_class,
            timeframe="1h",
            horizon=public_horizon,
            forecast_source="DIRECT_MODEL",
            model=model,
            model_version=model_version,
            direction=direction,
            probability_short=probabilities[0],
            probability_neutral=probabilities[1],
            probability_long=probabilities[2],
            expected_return=numeric[0],
            expected_mfe=numeric[1],
            expected_mae=numeric[2],
            expected_duration=numeric[3],
            opportunity_score=float(output["opportunity_score"]),
            confidence=str(output.get("confidence", "UNKNOWN")),
            forecast_timestamp=timestamp,
            target_timestamp=None,
            data_quality="AVAILABLE",
            forecast_status="PARTIALLY_VALIDATED",
            quality_status="PARTIALLY_VALIDATED",
            usable_for_decision=public_horizon in {"4h", "8h"},
            freshness="FRESH",
            forecast_age_minutes=0.0,
        )
    return [apply_oos_quality_gate(forecasts[horizon]) for horizon in PUBLIC_HORIZONS]


def direct_swing_forecasts(
    *,
    asset: str,
    model: str,
    model_version: str,
    timestamp: str,
    outputs: Mapping[int, Mapping[str, float]],
) -> list[AssistantForecast]:
    """Normalize exact daily swing heads without inventing intraday outputs."""
    forecasts = {horizon: unavailable_forecast(asset, "equity", horizon, reason="no exact registered model horizon") for horizon in PUBLIC_HORIZONS}
    for days, output in outputs.items():
        horizon = f"{days}d"
        if horizon not in forecasts:
            continue
        expected_return = float(output["expected_return"])
        if not np.isfinite(expected_return):
            forecasts[horizon] = unavailable_forecast(asset, "equity", horizon, reason="NUMERICAL_WARNING: invalid model output")
            continue
        direction = "LONG" if expected_return > 0 else "SHORT" if expected_return < 0 else "NEUTRAL"
        forecasts[horizon] = AssistantForecast(
            asset=asset, asset_class="equity", timeframe="1d", horizon=horizon,
            forecast_source="DIRECT_MODEL", model=model, model_version=model_version,
            direction=direction, probability_short=None, probability_neutral=None, probability_long=None,
            expected_return=expected_return, expected_mfe=float(output["expected_mfe"]), expected_mae=float(output["expected_mae"]),
            expected_duration=float(output["expected_duration"]), opportunity_score=float(output["opportunity_score"]),
            confidence="UNKNOWN", forecast_timestamp=timestamp, target_timestamp=None, data_quality="AVAILABLE",
            forecast_status="PARTIALLY_VALIDATED",
            quality_status="PARTIALLY_VALIDATED",
            usable_for_decision=False,
            freshness="FRESH",
            forecast_age_minutes=0.0,
        )
    return [apply_oos_quality_gate(forecasts[horizon]) for horizon in PUBLIC_HORIZONS]


def chronos2_forecast(*, asset: str, asset_class: str, horizon: str, output: Any) -> AssistantForecast:
    """Normalize a dedicated Chronos-2 result without treating it as a TCN fallback."""
    if output.status != "AVAILABLE":
        return unavailable_forecast(asset, asset_class, horizon, reason=f"Chronos-2 unavailable: {output.reason}")
    values = (output.expected_return, output.expected_favorable_move, output.expected_adverse_move)
    if not all(value is not None and np.isfinite(value) for value in values):
        return unavailable_forecast(asset, asset_class, horizon, reason="NUMERICAL_WARNING: invalid Chronos-2 output")
    forecast = AssistantForecast(
        asset=asset, asset_class=asset_class, timeframe=output.input_timeframe, horizon=horizon,
        forecast_source="CHRONOS_2", model=output.model_id, model_version=output.model_version,
        direction=output.direction, probability_short=None, probability_neutral=None, probability_long=None,
        expected_return=float(output.expected_return), expected_mfe=float(output.expected_favorable_move),
        expected_mae=float(output.expected_adverse_move), expected_duration=None,
        opportunity_score=float(output.model_score) if output.model_score is not None else None,
        confidence=output.confidence, forecast_timestamp=output.data_timestamp, target_timestamp=None,
        data_quality=output.data_quality, forecast_status="UNVALIDATED", reason=output.reason,
        quality_status="UNVALIDATED", usable_for_decision=False, freshness="UNKNOWN", forecast_age_minutes=None,
    )
    return apply_oos_quality_gate(forecast)


def combine_model_forecasts(forecasts: list[AssistantForecast]) -> AssistantForecast:
    """Combine only agreeing models; surface partial agreement and conflicts.

    Equal weighting is intentionally not used.  Without comparable OOS quality
    metadata, an agreement uses the median return as a robust summary while a
    disagreement emits no actionable combined direction or numeric forecast.
    """
    if not forecasts:
        raise ValueError("at least one forecast is required")
    first = forecasts[0]
    if any((item.asset, item.horizon) != (first.asset, first.horizon) for item in forecasts):
        raise ValueError("only forecasts for the same asset and horizon can be combined")
    available = [item for item in forecasts if item.forecast_status not in {"UNAVAILABLE", "MODEL_UNAVAILABLE"}]
    sources = tuple(item.model or "unknown" for item in forecasts)
    details = tuple(item.as_dict() for item in forecasts)
    if len(available) == 1:
        return apply_oos_quality_gate(replace(available[0], model_sources=sources, individual_forecasts=details, combination_method="SINGLE_AVAILABLE_MODEL", combined_confidence=available[0].confidence))
    if not available:
        return apply_oos_quality_gate(replace(first, model_sources=sources, individual_forecasts=details, combination_method="NO_AVAILABLE_MODEL", combined_confidence="UNKNOWN"))

    directions = {item.direction for item in available}
    directional = directions & {"LONG", "SHORT"}
    if len(directional) > 1:
        return apply_oos_quality_gate(replace(first, forecast_source="ENSEMBLE", model=" + ".join(sources), direction=None, probability_short=None, probability_neutral=None, probability_long=None, expected_return=None, expected_mfe=None, expected_mae=None, expected_duration=None, opportunity_score=None, confidence="MODEL_DISAGREEMENT", forecast_status="MODEL_DISAGREEMENT", reason="Available models disagree on direction; no combined numeric forecast.", model_sources=sources, individual_forecasts=details, combination_method="DISAGREEMENT_NO_AGGREGATION", combined_confidence="MODEL_DISAGREEMENT"))
    if len(directional) == 1 and len(directions) > 1:
        return apply_oos_quality_gate(replace(first, forecast_source="ENSEMBLE", model=" + ".join(sources), direction=None, probability_short=None, probability_neutral=None, probability_long=None, expected_return=None, expected_mfe=None, expected_mae=None, expected_duration=None, opportunity_score=None, confidence="MODEL_PARTIAL_AGREEMENT", forecast_status="MODEL_PARTIAL_AGREEMENT", reason="At least one available model is neutral while another is directional; no combined numeric forecast.", model_sources=sources, individual_forecasts=details, combination_method="PARTIAL_AGREEMENT_NO_AGGREGATION", combined_confidence="MODEL_PARTIAL_AGREEMENT"))

    probabilities = [[item.probability_short, item.probability_neutral, item.probability_long] for item in available]
    has_probabilities = all(all(value is not None and np.isfinite(value) for value in values) for values in probabilities)
    values = lambda name: [float(getattr(item, name)) for item in available if getattr(item, name) is not None and np.isfinite(getattr(item, name))]
    return apply_oos_quality_gate(replace(
        first,
        forecast_source="ENSEMBLE",
        model=" + ".join(sources),
        direction=next(iter(directional), first.direction),
        probability_short=float(np.median([value[0] for value in probabilities])) if has_probabilities else None,
        probability_neutral=float(np.median([value[1] for value in probabilities])) if has_probabilities else None,
        probability_long=float(np.median([value[2] for value in probabilities])) if has_probabilities else None,
        expected_return=float(np.median(values("expected_return"))) if values("expected_return") else None,
        expected_mfe=float(np.median(values("expected_mfe"))) if values("expected_mfe") else None,
        expected_mae=float(np.median(values("expected_mae"))) if values("expected_mae") else None,
        expected_duration=float(np.median(values("expected_duration"))) if values("expected_duration") else None,
        opportunity_score=float(np.median(values("opportunity_score"))) if values("opportunity_score") else None,
        confidence="MODEL_AGREEMENT",
        forecast_status="MODEL_AGREEMENT",
        reason="Available models agree on direction; median used only as a robust reporting summary.",
        model_sources=sources,
        individual_forecasts=details,
        combination_method="DIRECTION_AGREEMENT_MEDIAN",
        combined_confidence="MODEL_AGREEMENT",
    ))