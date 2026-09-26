"""Real-data decision pipeline for the order-free Trading Decision Assistant.

The pipeline reads an OHLCV frame supplied by the caller and uses the existing
causal indicator helpers. It does not fetch data, train models, or execute
orders. Missing model/data inputs remain explicit in the result.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.ml.features import compute_rsi
from core.strategy import compute_atr, compute_donchian_channels
from risk_engine import TradeSetup, compute_risk_parameters, max_leverage_for


SCORING_VERSION = "v2.0-trade-quality-gated"
RULE_SCORE_WEIGHT = 0.45
MODEL_SCORE_WEIGHT = 0.35
RISK_SCORE_WEIGHT = 0.20


class MarketRegime(str, Enum):
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNCLEAR = "UNCLEAR"


class SetupStatus(str, Enum):
    VALID = "VALID_SETUP"
    FORMING = "SETUP_FORMING"
    NONE = "NO_VALID_SETUP"
    STALE = "STALE"
    INSUFFICIENT = "INSUFFICIENT_DATA"
    ERROR = "ERROR"


@dataclass(frozen=True)
class IndicatorSnapshot:
    asset: str
    timeframe: str
    timestamp: str
    close: float
    atr: float
    atr_pct: float
    ema_fast: float
    ema_slow: float
    trend_strength: float
    momentum_lookback: float
    rsi: float
    bollinger_middle: float
    bollinger_upper: float
    bollinger_lower: float
    donchian_upper: float
    donchian_lower: float
    volatility_pct: float
    regime: str
    data_status: str


@dataclass(frozen=True)
class StrategyCandidate:
    strategy_id: str
    strategy_version: str
    category: str
    direction: str
    score: float
    valid: bool
    reason: tuple[str, ...]


@dataclass(frozen=True)
class ModelQuality:
    status: str
    model_type: str
    model_version: str
    score: float | None
    direction: str | None
    expected_return: float | None
    expected_mfe: float | None
    expected_mae: float | None
    reason: str


@dataclass(frozen=True)
class TradeQuality:
    status: str
    model_id: str
    model_version: str
    score: float | None
    probability: float | None
    label_version: str
    feature_version: str
    reason: str


@dataclass(frozen=True)
class IntegratedTradePlan:
    asset: str
    strategy_id: str
    strategy_version: str
    category: str
    timeframe: str
    direction: str
    setup_status: str
    signal_quality: float
    rule_score: float
    model_score: float | None
    final_score: float
    entry: float
    entry_low: float
    entry_high: float
    stop: float
    stop_distance: float
    take_profit_1: float
    take_profit_2: float
    trailing_method: str
    expected_holding_time: str
    risk_per_trade: float
    position_size: float
    notional: float
    margin: float
    recommended_leverage: float
    leverage_range: tuple[float, float, float]
    not_recommended_above: float
    risk_reward: float
    maximum_loss: float
    liquidation_estimate: float | None
    market_regime: str
    strategy_reason: tuple[str, ...]
    invalidated_below: float | None
    invalidated_above: float | None
    model: ModelQuality
    risk_score: float = 0.0
    scoring_version: str = SCORING_VERSION
    trade_quality_score: float | None = None


@dataclass(frozen=True)
class AnalysisResult:
    asset: str
    timeframe: str
    data_status: str
    regime: str
    indicators: IndicatorSnapshot | None
    candidates: tuple[StrategyCandidate, ...]
    best_candidate: StrategyCandidate | None
    model: ModelQuality
    plan: IntegratedTradePlan | None
    reason: str
    analysis_timestamp: str = ""
    data_timestamp: str | None = None
    data_age_seconds: float | None = None
    source: str = "local_cache"
    model_comparison: Any | None = None
    trade_quality: TradeQuality | None = None


def _normalise_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OHLCV missing columns: {sorted(missing)}")
    out = frame.copy()
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index, utc=True))
    out = out.sort_index().loc[~out.index.duplicated(keep="last")]
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=list(required))


def load_cached_ohlcv(asset: str, timeframe: str, data_dir: Path | str = "data") -> pd.DataFrame | None:
    """Load the newest matching cache without silently downloading anything."""
    symbol = asset.replace("/", "-")
    paths = sorted(Path(data_dir).glob(f"ohlcv_{symbol}_{timeframe}_*.csv"), key=lambda path: path.stat().st_mtime)
    if not paths:
        proxy = {"QQQ": "NASDAQ100_PROXY", "SPY": "SP500_PROXY"}.get(asset)
        if proxy:
            from core.ml.data import load_ohlcv_parquet
            parquet_paths = sorted(Path(data_dir).joinpath("market", proxy, timeframe).glob("*.parquet"), key=lambda path: path.stat().st_mtime)
            if parquet_paths:
                return _normalise_ohlcv(load_ohlcv_parquet(parquet_paths[-1]))
        return None
    frame = pd.read_csv(paths[-1], index_col=0, parse_dates=True)
    return _normalise_ohlcv(frame)


def _regime(ema_fast: float, ema_slow: float, trend_strength: float, atr_pct: float, volatility_pct: float) -> MarketRegime:
    if not np.isfinite([ema_fast, ema_slow, trend_strength, atr_pct, volatility_pct]).all():
        return MarketRegime.UNCLEAR
    if atr_pct >= volatility_pct * 1.5:
        return MarketRegime.HIGH_VOLATILITY
    if atr_pct <= volatility_pct * 0.6:
        return MarketRegime.LOW_VOLATILITY
    if trend_strength >= 1.0:
        return MarketRegime.TREND_UP
    if trend_strength <= -1.0:
        return MarketRegime.TREND_DOWN
    return MarketRegime.RANGE


def calculate_indicators(asset: str, timeframe: str, frame: pd.DataFrame, data_status: str = "FRESH") -> IndicatorSnapshot:
    ohlcv = _normalise_ohlcv(frame)
    if len(ohlcv) < 120:
        raise ValueError("at least 120 closed candles are required")
    close, high, low = ohlcv["close"], ohlcv["high"], ohlcv["low"]
    atr = compute_atr(high, low, close, 14)
    upper, lower, _, _ = compute_donchian_channels(high, low, 55, 20)
    ema_fast = close.ewm(span=20, adjust=False).mean()
    ema_slow = close.ewm(span=100, adjust=False).mean()
    returns = np.log(close / close.shift(1))
    volatility = returns.rolling(48).std()
    middle = close.rolling(20).mean()
    band_std = close.rolling(20).std()
    row = pd.DataFrame({
        "close": close, "atr": atr, "upper": upper, "lower": lower, "ema_fast": ema_fast, "ema_slow": ema_slow,
        "volatility": volatility, "middle": middle, "band_std": band_std, "rsi": compute_rsi(close, 14),
    }).dropna().iloc[-1]
    atr_pct = float(row["atr"] / row["close"])
    vol_pct = float(row["volatility"])
    strength = float((row["ema_fast"] - row["ema_slow"]) / max(row["atr"], 1e-12))
    regime = _regime(float(row["ema_fast"]), float(row["ema_slow"]), strength, atr_pct, float(volatility.dropna().median()))
    return IndicatorSnapshot(
        asset, timeframe, ohlcv.index[-1].isoformat(), float(row["close"]), float(row["atr"]), atr_pct,
        float(row["ema_fast"]), float(row["ema_slow"]), strength, float(close.pct_change(6).iloc[-1]),
        float(row["rsi"]), float(row["middle"]), float(row["middle"] + 2 * row["band_std"]), float(row["middle"] - 2 * row["band_std"]),
        float(row["upper"]), float(row["lower"]), vol_pct, regime.value, data_status,
    )


def select_strategy(indicators: IndicatorSnapshot) -> tuple[StrategyCandidate, ...]:
    """Score all families from the same observable indicator snapshot."""
    i = indicators
    long = i.direction if hasattr(i, "direction") else "LONG"
    trend_direction = "LONG" if i.trend_strength > 0 else "SHORT" if i.trend_strength < 0 else "UNCERTAIN"
    momentum_direction = "LONG" if i.momentum_lookback > 0 else "SHORT" if i.momentum_lookback < 0 else "UNCERTAIN"
    breakout_direction = "LONG" if i.close > i.donchian_upper else "SHORT" if i.close < i.donchian_lower else "UNCERTAIN"
    trend_score = 0.0
    trend_reasons: list[str] = []
    if i.regime in {MarketRegime.TREND_UP.value, MarketRegime.TREND_DOWN.value}:
        trend_score += 35.0; trend_reasons.append("EMA trend alignment")
    if breakout_direction == trend_direction and breakout_direction != "UNCERTAIN":
        trend_score += 40.0; trend_reasons.append("Donchian breakout")
    if momentum_direction == trend_direction and momentum_direction != "UNCERTAIN":
        trend_score += 25.0; trend_reasons.append("momentum agrees")
    momentum_score = 0.0
    momentum_reasons: list[str] = []
    if momentum_direction != "UNCERTAIN":
        momentum_score += 40.0; momentum_reasons.append("short-term momentum")
    if momentum_direction == trend_direction:
        momentum_score += 35.0; momentum_reasons.append("trend agrees")
    if i.regime not in {MarketRegime.HIGH_VOLATILITY.value, MarketRegime.LOW_VOLATILITY.value, MarketRegime.UNCLEAR.value}:
        momentum_score += 25.0; momentum_reasons.append("volatility acceptable")
    mean_direction = "SHORT" if i.rsi >= 70 or i.close >= i.bollinger_upper else "LONG" if i.rsi <= 30 or i.close <= i.bollinger_lower else "UNCERTAIN"
    mean_score = 0.0
    mean_reasons: list[str] = []
    if mean_direction != "UNCERTAIN":
        mean_score += 45.0; mean_reasons.append("RSI/Bollinger extension")
    if i.regime == MarketRegime.RANGE.value:
        mean_score += 40.0; mean_reasons.append("range regime")
    if i.regime not in {MarketRegime.TREND_UP.value, MarketRegime.TREND_DOWN.value}:
        mean_score += 15.0; mean_reasons.append("no strong trend")
    adaptive_direction = trend_direction if i.regime in {MarketRegime.TREND_UP.value, MarketRegime.TREND_DOWN.value} else momentum_direction
    adaptive_score = max(trend_score, momentum_score, mean_score) * 0.85 if adaptive_direction != "UNCERTAIN" else 0.0
    candidates = (
        StrategyCandidate("trend_breakout", "1.0.0", "TREND", trend_direction, min(trend_score, 100.0), trend_score >= 60 and trend_direction != "UNCERTAIN", tuple(trend_reasons)),
        StrategyCandidate("momentum", "1.0.0", "INTRADAY", momentum_direction, min(momentum_score, 100.0), momentum_score >= 60 and momentum_direction != "UNCERTAIN", tuple(momentum_reasons)),
        StrategyCandidate("mean_reversion", "1.0.0", "MEAN_REVERSION", mean_direction, min(mean_score, 100.0), mean_score >= 60 and mean_direction != "UNCERTAIN", tuple(mean_reasons)),
        StrategyCandidate("regime_adaptive", "1.0.0", "ADAPTIVE", adaptive_direction, min(adaptive_score, 100.0), adaptive_score >= 60 and adaptive_direction != "UNCERTAIN", (f"regime={i.regime}",)),
    )
    return tuple(sorted(candidates, key=lambda candidate: (-candidate.score, candidate.strategy_id)))


def model_artifact_status(asset: str, asset_class: str, model_type: str, model_version: str = "unknown", model_root: Path | str = "data/models") -> tuple[str, str, str]:
    """Validate the model family/artifact before accepting a persisted score."""
    if model_type == "swing":
        base = Path("models/checkpoints")
        stem = asset.lower()
        checkpoint = base / f"{stem}_swing.pt"
        metadata = base / f"{stem}_swing.json"
    else:
        family = "model1a_crypto" if asset_class == "crypto" else "model1b_equity"
        stem = asset.replace("/", "-") if asset_class == "crypto" else {"QQQ": "NASDAQ100_PROXY", "SPY": "SP500_PROXY"}.get(asset, asset)
        base = Path(model_root) / family
        checkpoint = base / f"{stem}_tcn.pt"
        metadata = base / f"{stem}_tcn_meta.json"
    if not checkpoint.exists() or not metadata.exists():
        return "MODEL_NOT_AVAILABLE", model_version, "Model checkpoint or metadata is missing."
    try:
        import json
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        supported = set(meta.get("timeframes", []))
        version = str(meta.get("model_version", model_version))
        return "AVAILABLE", version, f"Validated {version}; supported timeframes: {', '.join(sorted(supported))}."
    except (OSError, ValueError, TypeError) as exc:
        return "MODEL_NOT_AVAILABLE", model_version, f"Model metadata is unreadable: {exc}"


def model_quality(signal: dict[str, Any] | None = None, *, asset: str | None = None, asset_class: str = "crypto", model_version: str = "unknown", model_root: Path | str = "data/models") -> ModelQuality:
    if not signal:
        return ModelQuality("MODEL_NOT_AVAILABLE", "none", model_version, None, None, None, None, None, "No persisted model snapshot is available.")
    direction = str(signal.get("direction", "UNCERTAIN"))
    score = float(signal["score"]) if signal.get("score") is not None and np.isfinite(float(signal["score"])) else None
    model_type = str(signal.get("model_type", "unknown"))
    if score is None or asset is None:
        return ModelQuality("MODEL_NOT_AVAILABLE", model_type, model_version, None, direction, signal.get("expected_return"), signal.get("expected_mfe"), signal.get("expected_mae"), "Persisted snapshot has no usable model score.")
    status, resolved_version, reason = model_artifact_status(asset, asset_class, model_type, model_version, model_root)
    return ModelQuality(status, model_type, resolved_version, score if status == "AVAILABLE" else None, direction, signal.get("expected_return"), signal.get("expected_mfe"), signal.get("expected_mae"), reason)


def risk_setup_score(indicators: IndicatorSnapshot, risk: Any, model: ModelQuality) -> float:
    """Return a setup-only score; it never uses the rule score or model score."""
    stop_quality = np.clip(indicators.atr_pct / max(float(risk.stop_distance_pct), 1e-12), 0.0, 1.0)
    reward_quality = np.clip((float(risk.risk_reward_ratio) - 0.5) / 1.5, 0.0, 1.0)
    adverse_quality = np.clip(1.0 - abs(float(model.expected_mae or 0.0)) / max(abs(float(model.expected_mfe or 0.0)) + abs(float(model.expected_mae or 0.0)), 1e-12), 0.0, 1.0)
    exposure_quality = np.clip(1.0 - float(risk.margin_eur) / max(1.0, 500.0), 0.0, 1.0)
    return float(100.0 * (0.30 * stop_quality + 0.30 * reward_quality + 0.20 * adverse_quality + 0.20 * exposure_quality))


def build_integrated_plan(asset: str, timeframe: str, indicators: IndicatorSnapshot, candidate: StrategyCandidate, model: ModelQuality, *, capital: float, risk_pct: float = 0.005, asset_class: str = "crypto", instrument_type: str | None = None, trade_quality: TradeQuality | None = None) -> IntegratedTradePlan | None:
    if not candidate.valid or indicators.data_status not in {"LIVE", "FRESH", "AVAILABLE"}:
        return None
    direction = candidate.direction
    model_score = model.score if model.score is not None and model.direction == direction else None
    rule_score = candidate.score
    preliminary_score = RULE_SCORE_WEIGHT * rule_score + MODEL_SCORE_WEIGHT * (model_score or 0.0)
    if model.status == "AVAILABLE" and model.direction not in {direction, "UNCERTAIN"}:
        preliminary_score *= 0.5
    setup = TradeSetup(
        asset=asset, asset_class=asset_class, direction=direction, entry_price=indicators.close, atr=indicators.atr,
        expected_mae=float(model.expected_mae or -max(indicators.atr_pct, 0.001)), expected_mfe=float(model.expected_mfe or max(indicators.atr_pct * 2, 0.002)),
        score=preliminary_score, capital_eur=capital, risk_per_trade_pct=risk_pct, instrument_type=instrument_type,
        confidence=(model_score / 100.0 if model_score is not None else 1.0),
    )
    risk = compute_risk_parameters(setup)
    if risk.rejected:
        return None
    risk_score = risk_setup_score(indicators, risk, model)
    if trade_quality is not None and trade_quality.score is not None and trade_quality.status == "AVAILABLE":
        # Trade quality dominates forecast context only after explicit promotion.
        final_quality = float(np.clip(0.40 * rule_score + 0.30 * trade_quality.score + 0.15 * (model_score or 0.0) + 0.15 * risk_score, 0.0, 100.0))
    else:
        final_quality = float(np.clip(RULE_SCORE_WEIGHT * rule_score + MODEL_SCORE_WEIGHT * (model_score or 0.0) + RISK_SCORE_WEIGHT * risk_score, 0.0, 100.0))
    entry_offset = indicators.atr * 0.25
    tp1 = float(risk.take_profit_1_price)
    tp2 = float(risk.take_profit_2_price)
    max_leverage = max_leverage_for(asset_class, instrument_type)
    conservative = max(1.0, risk.leverage * 0.75)
    aggressive = min(max_leverage, risk.leverage * 1.25)
    plan_category = "SWING" if timeframe in {"4h", "1d"} else "INTRADAY"
    return IntegratedTradePlan(
        asset, candidate.strategy_id, candidate.strategy_version, plan_category, timeframe, direction, SetupStatus.VALID.value,
        final_quality, rule_score, model_score, final_quality, indicators.close, indicators.close - entry_offset, indicators.close + entry_offset, risk.stop_loss_price,
        risk.stop_distance_pct * indicators.close, tp1, tp2, f"{1.5:.1f} ATR ratchet; never widen", "strategy-defined bars", risk.risk_amount_eur,
        risk.position_size_eur / indicators.close, risk.position_size_eur, risk.margin_eur, risk.leverage, (conservative, risk.leverage, aggressive),
        aggressive, risk.risk_reward_ratio, risk.risk_amount_eur, risk.liquidation_price, indicators.regime, candidate.reason,
        risk.stop_loss_price if direction == "LONG" else None, risk.stop_loss_price if direction == "SHORT" else None, model, risk_score, SCORING_VERSION, trade_quality.score if trade_quality else None,
    )


def analyse_ohlcv(asset: str, timeframe: str, frame: pd.DataFrame | None, *, signal: dict[str, Any] | None = None, forecast: Any | None = None, model_comparison: Any | None = None, trade_quality: TradeQuality | None = None, capital: float = 500.0, asset_class: str = "crypto", data_status: str = "FRESH") -> AnalysisResult:
    if frame is None or frame.empty:
        model = model_quality(signal, asset=asset, asset_class=asset_class)
        return AnalysisResult(asset, timeframe, SetupStatus.INSUFFICIENT.value, MarketRegime.UNCLEAR.value, None, (), None, model, None, "No OHLCV cache is available; no setup is invented.", model_comparison=model_comparison, trade_quality=trade_quality)
    try:
        indicators = calculate_indicators(asset, timeframe, frame, data_status)
        candidates = select_strategy(indicators)
        best = next((candidate for candidate in candidates if candidate.valid), None)
        decision_reason = "Rule-based strategy selection with optional validated model quality filter."
        if asset in {"SPY", "QQQ"} and timeframe == "1h":
            trend_candidate = next(
                (candidate for candidate in candidates if candidate.strategy_id == "trend_breakout" and candidate.valid),
                None,
            )
            chronos = getattr(model_comparison, "chronos", None)
            if trend_candidate is None or getattr(chronos, "status", None) != "AVAILABLE" or getattr(chronos, "direction", None) != getattr(trend_candidate, "direction", None):
                best = None
                decision_reason = "No equity intraday recommendation: a valid trend-breakout rule and matching Chronos-2 direction are required."
            else:
                best = trend_candidate
                decision_reason = "Equity intraday recommendation gated by matching trend-breakout and Chronos-2 directions."
        if forecast is not None and getattr(forecast, "status", "") == "AVAILABLE":
            signal = {
                "model_type": getattr(forecast, "model_id", "model"),
                "score": getattr(forecast, "model_score", None),
                "direction": getattr(forecast, "direction", "UNCERTAIN"),
                "expected_return": getattr(forecast, "expected_return", None),
                "expected_mfe": getattr(forecast, "expected_favorable_move", None),
                "expected_mae": getattr(forecast, "expected_adverse_move", None),
            }
        model = model_quality(signal, asset=asset, asset_class=asset_class)
        plan = build_integrated_plan(asset, timeframe, indicators, best, model, capital=capital, asset_class=asset_class, trade_quality=trade_quality) if best else None
        status = SetupStatus.VALID.value if plan else SetupStatus.NONE.value
        data_timestamp = indicators.timestamp
        age = max(0.0, (pd.Timestamp.now(tz="UTC") - pd.Timestamp(data_timestamp)).total_seconds())
        return AnalysisResult(asset, timeframe, status, indicators.regime, indicators, candidates, best, model, plan, decision_reason, pd.Timestamp.now(tz="UTC").isoformat(), data_timestamp, age, "local_cache", model_comparison, trade_quality)
    except (ValueError, KeyError, TypeError) as exc:
        return AnalysisResult(asset, timeframe, SetupStatus.ERROR.value, MarketRegime.UNCLEAR.value, None, (), None, model_quality(signal, asset=asset, asset_class=asset_class), None, str(exc), pd.Timestamp.now(tz="UTC").isoformat(), model_comparison=model_comparison, trade_quality=trade_quality)


def result_dict(result: AnalysisResult) -> dict[str, Any]:
    return asdict(result)