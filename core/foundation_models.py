from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ForecastResult:
    model_id: str
    asset: str
    mode: str
    native_timeframe: str
    forecast_horizons: tuple[int, ...] = ()
    expected_returns: dict[int, float] = field(default_factory=dict)
    direction_probabilities: dict[int, dict[str, float]] = field(default_factory=dict)
    predicted_directions: dict[int, str] = field(default_factory=dict)
    expected_opportunities: dict[int, float] = field(default_factory=dict)
    expected_mfe: dict[int, float] = field(default_factory=dict)
    expected_mae: dict[int, float] = field(default_factory=dict)
    expected_durations: dict[int, float] = field(default_factory=dict)
    quantiles: dict[int, dict[str, float]] = field(default_factory=dict)
    volatility: dict[int, float] = field(default_factory=dict)
    uncertainty: dict[int, float] = field(default_factory=dict)
    score: float | None = None
    data_timestamp: str | None = None
    model_version: str | None = None

    def expected_return_for(self, horizon: int) -> float:
        return float(self.expected_returns.get(int(horizon), 0.0))

    def direction_probability(self, horizon: int, direction: str) -> float:
        return float(self.direction_probabilities.get(int(horizon), {}).get(direction.upper(), 0.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "asset": self.asset,
            "mode": self.mode,
            "native_timeframe": self.native_timeframe,
            "forecast_horizons": list(self.forecast_horizons),
            "expected_returns": dict(self.expected_returns),
            "direction_probabilities": {str(h): dict(v) for h, v in self.direction_probabilities.items()},
            "predicted_directions": {str(h): direction for h, direction in self.predicted_directions.items()},
            "expected_opportunities": {str(h): value for h, value in self.expected_opportunities.items()},
            "expected_mfe": {str(h): value for h, value in self.expected_mfe.items()},
            "expected_mae": {str(h): value for h, value in self.expected_mae.items()},
            "expected_durations": {str(h): value for h, value in self.expected_durations.items()},
            "quantiles": {str(h): dict(v) for h, v in self.quantiles.items()},
            "volatility": dict(self.volatility),
            "uncertainty": dict(self.uncertainty),
            "score": self.score,
            "data_timestamp": self.data_timestamp,
            "model_version": self.model_version,
        }


@dataclass
class BaseForecastModel:
    model_id: str
    asset: str
    mode: str
    native_timeframe: str
    forecast_horizons: tuple[int, ...]
    model_version: str | None = None
    data_source: str | None = None
    license_status: str = "RESEARCH"
    validation_status: str = "UNVALIDATED"
    production_status: str = "BLOCKED"

    def supports_horizon(self, horizon: int | str) -> bool:
        target = int(horizon)
        return target in self.forecast_horizons

    def as_result(self, *, expected_returns: dict[int, float] | None = None, direction_probabilities: dict[int, dict[str, float]] | None = None, predicted_directions: dict[int, str] | None = None, expected_opportunities: dict[int, float] | None = None, expected_mfe: dict[int, float] | None = None, expected_mae: dict[int, float] | None = None, expected_durations: dict[int, float] | None = None, score: float | None = None, data_timestamp: str | None = None) -> ForecastResult:
        return ForecastResult(
            model_id=self.model_id,
            asset=self.asset,
            mode=self.mode,
            native_timeframe=self.native_timeframe,
            forecast_horizons=self.forecast_horizons,
            expected_returns=expected_returns or {},
            direction_probabilities=direction_probabilities or {},
            predicted_directions=predicted_directions or {},
            expected_opportunities=expected_opportunities or {},
            expected_mfe=expected_mfe or {},
            expected_mae=expected_mae or {},
            expected_durations=expected_durations or {},
            score=score,
            data_timestamp=data_timestamp,
            model_version=self.model_version,
        )


@dataclass(frozen=True)
class ForecastConsensus:
    model_available: bool = False
    data_valid: bool = False
    context_valid: bool = False
    forecast_valid: bool = False
    model_version_registered: bool = False
    validation_status: str = "UNKNOWN"

    def is_valid(self) -> bool:
        status = str(self.validation_status).upper()
        return (
            self.model_available
            and self.data_valid
            and self.context_valid
            and self.forecast_valid
            and self.model_version_registered
            and status in {"ACCEPTABLE", "VALID", "OK", "PASS", "APPROVED"}
        )


__all__ = [
    "BaseForecastModel",
    "ForecastConsensus",
    "ForecastResult",
]
