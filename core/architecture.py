"""Versioned contracts for the multi-asset, multi-horizon paper platform."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml


class AssetRole(str, Enum):
    TARGET = "TARGET"
    FEATURE = "FEATURE"


class StrategyStatus(str, Enum):
    RESEARCH = "RESEARCH"
    VALIDATED = "VALIDATED"
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    LIVE = "LIVE"
    DISABLED = "DISABLED"
    REJECTED = "REJECTED"


class DecisionAction(str, Enum):
    NO_TRADE = "NO_TRADE"
    ALERT = "ALERT"
    PAPER_ENTRY = "PAPER_ENTRY"
    PAPER_EXIT = "PAPER_EXIT"
    HOLD = "HOLD"


class HorizonClass(str, Enum):
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"
    H4 = "H4"
    H5 = "H5"


HORIZON_RANGES_HOURS: dict[HorizonClass, tuple[float, float]] = {
    HorizonClass.H1: (0.25, 4.0),
    HorizonClass.H2: (4.0, 24.0),
    HorizonClass.H3: (24.0, 24.0 * 7),
    HorizonClass.H4: (24.0 * 7, 24.0 * 14),
    HorizonClass.H5: (24.0 * 14, 24.0 * 28),
}


def _utc(value: datetime | str) -> datetime:
    parsed = pd.Timestamp(value).to_pydatetime()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_value(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _json_value(asdict(value))
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return _utc(value).isoformat()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def to_payload(value: Any) -> dict[str, Any]:
    if not hasattr(value, "__dataclass_fields__"):
        raise TypeError("to_payload expects a dataclass instance")
    return _json_value(asdict(value))


def from_payload(cls: type[Any], payload: Mapping[str, Any]) -> Any:
    """Deserialize a supported architecture contract from JSON-compatible data."""
    data = dict(payload)
    if cls is HorizonSpec:
        return HorizonSpec(data["label"], float(data["hours"]), HorizonClass(data["horizon_class"]))
    if cls is AssetSpec:
        return AssetSpec(
            asset=data["asset"], asset_class=data["asset_class"], source=data["source"],
            provider_symbol=data["provider_symbol"], timeframes=tuple(data["timeframes"]),
            timezone=data["timezone"], trading_hours=data["trading_hours"],
            roles=tuple(AssetRole(role) for role in data["roles"]),
            availability_start=data.get("availability_start"), availability_end=data.get("availability_end"),
            expected_frequency=data.get("expected_frequency"), max_staleness_minutes=data.get("max_staleness_minutes"),
            quality_status=data.get("quality_status", "UNVERIFIED"),
        )
    if cls is DataUniverse:
        return DataUniverse(data["universe_id"], data["version"], tuple(from_payload(AssetSpec, item) for item in data["assets"]))
    if cls is FeatureSetSpec:
        return FeatureSetSpec(
            feature_set_id=data["feature_set_id"], version=data["version"], assets=tuple(data["assets"]),
            timeframes=tuple(data["timeframes"]), groups=tuple(data["groups"]), lookback=dict(data["lookback"]),
            alignment_rules=dict(data["alignment_rules"]), normalization=dict(data["normalization"]),
            label_version=data["label_version"], data_cutoff=data["data_cutoff"],
        )
    if cls is LabelSpec:
        return LabelSpec(
            label_version=data["label_version"],
            horizons=tuple(from_payload(HorizonSpec, item) for item in data["horizons"]),
            labels=tuple(data.get("labels", LabelSpec.__dataclass_fields__["labels"].default)),
            threshold_config=dict(data.get("threshold_config", {})),
        )
    if cls is StrategyConfig:
        data["horizon"] = HorizonClass(data["horizon"])
        data["status"] = StrategyStatus(data.get("status", StrategyStatus.RESEARCH.value))
        return StrategyConfig(**data)
    if cls is ExperimentManifest:
        return ExperimentManifest(**data)
    if cls is DecisionRecord:
        data["horizon"] = HorizonClass(data["horizon"])
        data["decision"] = DecisionAction(data["decision"])
        data["reason_codes"] = tuple(data["reason_codes"])
        return DecisionRecord(**data)
    if cls is BacktestRequest:
        data["strategy"] = from_payload(StrategyConfig, data["strategy"])
        return BacktestRequest(**data)
    if cls is BacktestResult:
        return BacktestResult(**data)
    raise TypeError(f"unsupported architecture contract: {cls!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


PROXY_ALIAS_LOOKUP: dict[str, str] = {
    "SP500_PROXY": "SPY",
    "NASDAQ100_PROXY": "QQQ",
    "SPY": "SPY",
    "QQQ": "QQQ",
}


@dataclass(frozen=True)
class ProxyTargetStatus:
    asset: str
    is_real_target: bool
    is_proxy_target: bool
    canonical_symbol: str | None = None


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_id: str
    asset: str
    asset_class: str
    mode: str
    native_timeframe: str
    forecast_horizons: tuple[int, ...] = ()
    data_source: str | None = None
    training_range: str | None = None
    effective_training_rows: int | None = None
    feature_version: str | None = None
    context_version: str | None = None
    model_version: str | None = None
    license_status: str = "RESEARCH"
    validation_status: str = "UNVALIDATED"
    production_status: str = "BLOCKED"


def _parse_numeric_horizon(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "")
        if not normalized:
            raise ValueError("horizon string is empty")
        for suffix in ("d", "h", "m"):
            if normalized.endswith(suffix):
                number = normalized[:-1]
                if number:
                    return float(number)
        return float(normalized)
    raise TypeError(f"Unsupported horizon value: {value!r}")


def horizon_to_bars(horizon: str | int | float, timeframe: str, *, trading_days: bool = False) -> int:
    """Convert a horizon label into bar count for the target sampling interval.

    Examples:
      - horizon_to_bars('1d', '4h') == 6
      - horizon_to_bars('14d', '4h') == 84
      - horizon_to_bars('4h', '1h') == 4
    """
    timeframe_key = str(timeframe).lower().replace(" ", "")
    if timeframe_key.endswith("d"):
        multiplier = 1
    else:
        minutes = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240, "6h": 360, "8h": 480, "12h": 720, "1d": 1440}.get(timeframe_key, None)
        if minutes is None:
            raise ValueError(f"Unsupported timeframe: {timeframe!r}")
        multiplier = minutes / 60.0 if not timeframe_key.endswith("m") else minutes
    raw = _parse_numeric_horizon(horizon)
    if trading_days:
        return max(1, int(round(raw)))
    if str(horizon).lower().replace(" ", "").endswith("d") and timeframe_key in {"4h", "1h", "15m", "30m", "1d"}:
        if timeframe_key == "1h":
            return max(1, int(round(raw * 24.0)))
        if timeframe_key == "4h":
            return max(1, int(round(raw * 6.0)))
        if timeframe_key == "15m":
            return max(1, int(round(raw * 96.0)))
        if timeframe_key == "30m":
            return max(1, int(round(raw * 48.0)))
        if timeframe_key == "1d":
            return max(1, int(round(raw * 1.0)))
    if str(horizon).lower().replace(" ", "").endswith("h") and timeframe_key in {"1h", "4h", "1d"}:
        hours = raw
        if timeframe_key == "1h":
            return max(1, int(round(hours)))
        if timeframe_key == "4h":
            return max(1, int(round(hours / 4.0)))
        if timeframe_key == "1d":
            return max(1, int(round(hours / 24.0)))
    if str(horizon).lower().replace(" ", "").endswith("m") and timeframe_key not in {"1d"}:
        minutes = raw
        return max(1, int(round(minutes / (float(minutes) if timeframe_key.endswith("m") else 1.0))))
    if timeframe_key.endswith("m"):
        period_minutes = float(timeframe_key.rstrip("m"))
        return max(1, int(round(raw / period_minutes)))
    return max(1, int(round(raw * multiplier)))


def trading_day_horizon_to_bars(horizon: str | int | float) -> int:
    """Convert a trading-day horizon to a count of trading sessions, preserving daily swing semantics."""
    value = _parse_numeric_horizon(horizon)
    return max(1, int(round(value)))


def proxy_target_status(asset: str) -> ProxyTargetStatus:
    """Return whether the asset is a genuine target or only a proxy alias."""
    cleaned = str(asset).strip()
    canonical = PROXY_ALIAS_LOOKUP.get(cleaned, cleaned)
    is_real_target = cleaned in {"SPY", "QQQ"}
    is_proxy = cleaned in {"SP500_PROXY", "NASDAQ100_PROXY"}
    return ProxyTargetStatus(
        asset=cleaned,
        is_real_target=is_real_target,
        is_proxy_target=is_proxy,
        canonical_symbol=canonical if is_real_target or is_proxy else None,
    )


def build_model_registry_entry(**kwargs: Any) -> ModelRegistryEntry:
    return ModelRegistryEntry(**kwargs)


@dataclass(frozen=True)
class HorizonSpec:
    label: str
    hours: float
    horizon_class: HorizonClass

    def __post_init__(self) -> None:
        if self.hours <= 0:
            raise ValueError("prediction horizon must be positive")
        lower, upper = HORIZON_RANGES_HOURS[self.horizon_class]
        if not lower <= self.hours <= upper:
            raise ValueError(f"{self.label} is outside {self.horizon_class.value} range")


@dataclass(frozen=True)
class AssetSpec:
    asset: str
    asset_class: str
    source: str
    provider_symbol: str
    timeframes: tuple[str, ...]
    timezone: str
    trading_hours: str
    roles: tuple[AssetRole, ...] = (AssetRole.FEATURE,)
    availability_start: str | None = None
    availability_end: str | None = None
    expected_frequency: str | None = None
    max_staleness_minutes: int | None = None
    quality_status: str = "UNVERIFIED"

    def __post_init__(self) -> None:
        if not self.asset or not self.timeframes:
            raise ValueError("asset and at least one timeframe are required")
        if not self.roles:
            raise ValueError("an asset must have at least one role")
        if self.max_staleness_minutes is not None and self.max_staleness_minutes < 0:
            raise ValueError("max_staleness_minutes cannot be negative")


@dataclass(frozen=True)
class DataUniverse:
    universe_id: str
    version: str
    assets: tuple[AssetSpec, ...]

    def __post_init__(self) -> None:
        names = [item.asset for item in self.assets]
        if len(names) != len(set(names)):
            raise ValueError("data universe assets must be unique")

    def asset(self, name: str) -> AssetSpec:
        for item in self.assets:
            if item.asset == name:
                return item
        raise KeyError(name)

    def targets(self) -> tuple[str, ...]:
        return tuple(item.asset for item in self.assets if AssetRole.TARGET in item.roles)

    def feature_assets(self) -> tuple[str, ...]:
        return tuple(item.asset for item in self.assets if AssetRole.FEATURE in item.roles)


@dataclass(frozen=True)
class FeatureSetSpec:
    feature_set_id: str
    version: str
    assets: tuple[str, ...]
    timeframes: tuple[str, ...]
    groups: tuple[str, ...]
    lookback: dict[str, int]
    alignment_rules: dict[str, Any]
    normalization: dict[str, Any]
    label_version: str
    data_cutoff: str


@dataclass(frozen=True)
class LabelSpec:
    label_version: str
    horizons: tuple[HorizonSpec, ...]
    labels: tuple[str, ...] = (
        "future_return",
        "direction",
        "probability_positive_return",
        "thresholded_return",
        "mae",
        "mfe",
        "stop_probability",
        "expected_holding_time",
    )
    threshold_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StrategyConfig:
    strategy_id: str
    version: str
    name: str
    asset: str
    direction: str
    asset_class: str
    horizon: HorizonClass
    observation_timeframe: str
    strategy_family: str
    entry_logic: dict[str, Any]
    filter_logic: dict[str, Any]
    model_id: str | None
    feature_set_id: str
    regime_filter: dict[str, Any]
    stop_logic: dict[str, Any]
    initial_stop: dict[str, Any]
    trailing_stop: dict[str, Any]
    exit_logic: dict[str, Any]
    partial_exit_config: dict[str, Any]
    risk_profile: str
    leverage_policy: dict[str, Any]
    status: StrategyStatus = StrategyStatus.RESEARCH

    def __post_init__(self) -> None:
        if self.direction not in {"LONG", "SHORT", "BOTH"}:
            raise ValueError("direction must be LONG, SHORT, or BOTH")
        if self.status == StrategyStatus.LIVE and not self.model_id and self.strategy_family == "ml_only":
            raise ValueError("ml_only live strategies require a model_id")


@dataclass(frozen=True)
class ExperimentManifest:
    experiment_id: str
    code_version: str
    data_universe_id: str
    feature_set_id: str
    label_version: str
    strategy_id: str
    strategy_version: str
    model: dict[str, Any]
    hyperparameters: dict[str, Any]
    train_period: dict[str, str]
    validation_period: dict[str, str]
    walk_forward: dict[str, Any]
    oos_period: dict[str, str]
    clean_holdout_period: dict[str, str] | None
    stress_settings: dict[str, Any]
    transaction_costs: dict[str, Any]
    slippage: dict[str, Any]
    funding_assumptions: dict[str, Any]
    result: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    timestamp: str
    asset: str
    direction: str
    strategy_id: str
    strategy_version: str
    horizon: HorizonClass
    timeframe: str
    regime: str
    model_id: str | None
    feature_set_id: str
    confidence: float
    expected_return: float
    expected_mae: float
    expected_mfe: float
    entry: float | None
    stop: float | None
    position_size: float | None
    risk_budget: float | None
    leverage: float | None
    portfolio_exposure: float | None
    drawdown: float | None
    decision: DecisionAction
    reason_codes: tuple[str, ...]
    inputs: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    risk_output: dict[str, Any] = field(default_factory=dict)
    trade_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        _utc(self.timestamp)


@dataclass(frozen=True)
class BacktestRequest:
    strategy: StrategyConfig
    dataset_id: str
    costs: dict[str, Any]
    slippage: dict[str, Any]
    execution_assumptions: dict[str, Any]


@dataclass(frozen=True)
class BacktestResult:
    strategy_id: str
    strategy_version: str
    dataset_id: str
    metrics: dict[str, float]
    trade_metrics: dict[str, float]
    cost_metrics: dict[str, float]
    exposure_metrics: dict[str, float]
    status: str = "COMPLETED"


def load_data_universe(path: str | Path) -> DataUniverse:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    assets = []
    for item in payload.get("assets", []):
        assets.append(
            AssetSpec(
                asset=item["asset"],
                asset_class=item["asset_class"],
                source=item["source"],
                provider_symbol=item.get("provider_symbol", item["asset"]),
                timeframes=tuple(item["timeframes"]),
                timezone=item["timezone"],
                trading_hours=item["trading_hours"],
                roles=tuple(AssetRole(role) for role in item.get("roles", ["FEATURE"])),
                availability_start=item.get("availability_start"),
                availability_end=item.get("availability_end"),
                expected_frequency=item.get("expected_frequency"),
                max_staleness_minutes=item.get("max_staleness_minutes"),
                quality_status=item.get("quality_status", "UNVERIFIED"),
            )
        )
    return DataUniverse(payload["universe_id"], str(payload["version"]), tuple(assets))


def align_asof(
    source: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    *,
    availability_lag: timedelta = timedelta(0),
    max_staleness: timedelta | None = None,
) -> pd.DataFrame:
    """Align observations using only values available at each target timestamp."""
    if not isinstance(source.index, pd.DatetimeIndex):
        raise TypeError("source must use a DatetimeIndex")
    target = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True)).sort_values()
    source_frame = source.copy()
    source_frame.index = pd.DatetimeIndex(pd.to_datetime(source_frame.index, utc=True))
    source_frame = source_frame.sort_index()
    source_frame = source_frame[~source_frame.index.duplicated(keep="last")]
    available = source_frame.reset_index(names="observed_at")
    available["available_at"] = available["observed_at"] + availability_lag
    query = pd.DataFrame({"target_at": target})
    aligned = pd.merge_asof(
        query.sort_values("target_at"),
        available.sort_values("available_at"),
        left_on="target_at",
        right_on="available_at",
        direction="backward",
        tolerance=max_staleness,
    ).set_index("target_at")
    aligned = aligned.drop(columns=["available_at"], errors="ignore")
    aligned.index = target
    return aligned.drop(columns=["observed_at"], errors="ignore")


def assert_no_future_features(features: pd.DataFrame, target_index: pd.DatetimeIndex) -> None:
    target = pd.DatetimeIndex(pd.to_datetime(target_index, utc=True))
    if not features.index.equals(target):
        raise ValueError("feature index must exactly match the prediction index")
    if "observed_at" in features.columns:
        observed = pd.DatetimeIndex(pd.to_datetime(features["observed_at"], utc=True))
        if (observed > target).any():
            raise ValueError("feature observation occurs after prediction timestamp")