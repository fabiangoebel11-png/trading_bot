"""Order-free bridge from closed OHLCV data to the existing trained Model 1 TCN.

This module deliberately does not train or fetch data. It loads a validated local
checkpoint, builds the same causal feature matrix used by the daemon, and infers
only on the latest closed sequence. Failure is explicit and non-fatal.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

from core.data_loader import drop_incomplete_last_candle
from core.ml.dataset import make_sequences
from core.ml.features import build_feature_matrix
from core.ml.inference import load_symbol_model
from core.ml.scoring import continuous_opportunity_score
from core.ml.swing_data import build_swing_dataset
from core.ml.swing_model import HORIZONS as SWING_HORIZONS, horizon_opportunity_score, load_swing_model, scores_from_predictions
from core.config import SwingMLConfig
from core.architecture import horizon_to_bars
from train import load_config


_COMPARISON_CACHE: dict[tuple[str, str, str, str, float, str], "ModelComparison"] = {}
_COMPARISON_CACHE_LOCK = Lock()
_COMPARISON_CACHE_LIMIT = 32


@dataclass(frozen=True)
class ModelForecast:
    status: str
    model_id: str
    model_version: str
    asset: str
    category: str
    input_timeframe: str
    forecast_horizon: str
    direction: str | None
    model_score: float | None
    expected_return: float | None
    expected_adverse_move: float | None
    expected_favorable_move: float | None
    uncertainty: str
    data_timestamp: str | None
    reason: str
    forecast_low: float | None = None
    forecast_median: float | None = None
    forecast_high: float | None = None
    horizon: str | None = None
    asset_class: str | None = None
    data_quality: str = "UNKNOWN"
    confidence: str = "UNKNOWN"
    expected_duration: float | None = None
    probability_short: float | None = None
    probability_neutral: float | None = None
    probability_long: float | None = None
    opportunity_score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset": self.asset,
            "asset_class": self.asset_class or ("crypto" if "USDT" in self.asset else "equity"),
            "timeframe": self.input_timeframe,
            "horizon": self.horizon or self.forecast_horizon,
            "direction": self.direction,
            "probability_short": self.probability_short,
            "probability_neutral": self.probability_neutral,
            "probability_long": self.probability_long,
            "expected_return": self.expected_return,
            "expected_mfe": self.expected_favorable_move,
            "expected_mae": self.expected_adverse_move,
            "expected_duration": self.expected_duration,
            "opportunity_score": self.opportunity_score if self.opportunity_score is not None else self.model_score,
            "confidence": self.confidence,
            "model": self.model_id,
            "model_version": self.model_version,
            "forecast_status": self.status,
            "data_quality": self.data_quality,
            "forecast_timestamp": self.data_timestamp,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ModelComparison:
    """Normalized closed-candle forecasts and their non-promotional selection."""

    tcn: ModelForecast
    chronos: ModelForecast
    agreement: str
    ensemble_score: float | None
    selected: ModelForecast
    ensemble_version: str = "v1.0-no-promotion"


def _closed_candles(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    closed = frame.copy()
    if not isinstance(closed.index, pd.DatetimeIndex):
        timestamp_column = next((column for column in ("timestamp", "datetime", "date", "time") if column in closed.columns), None)
        if timestamp_column is None:
            raise ValueError("OHLCV frame has no datetime index or timestamp column")
        closed = closed.set_index(timestamp_column)
    closed.index = pd.DatetimeIndex(pd.to_datetime(closed.index, utc=True))
    closed = drop_incomplete_last_candle(closed.sort_index(), timeframe)
    closed.index = closed.index.tz_localize(None)
    return closed


@lru_cache(maxsize=2)
def _load_chronos2_pipeline(model_name: str, device_map: str):
    """Load foundation weights once per process/model/device combination."""
    try:
        from chronos import Chronos2Pipeline
    except ImportError as exc:
        raise RuntimeError("Chronos-2 dependency is not installed; install the foundation extra.") from exc
    return Chronos2Pipeline.from_pretrained(model_name, device_map=device_map)


def _chronos2_quantile_forecast(frame: pd.DataFrame, horizon: int) -> tuple[float, float, float, str]:
    """Run Chronos-2 on only the latest closed target close series."""
    model_name = os.getenv("CHRONOS2_MODEL_ID", "amazon/chronos-2")
    pipeline = _load_chronos2_pipeline(model_name, os.getenv("CHRONOS2_DEVICE", "cpu"))
    import torch
    context = torch.as_tensor(frame["close"].astype(float).to_numpy(), dtype=torch.float32).reshape(1, 1, -1)
    forecasts = pipeline.predict(context, prediction_length=horizon)
    values = forecasts[0].detach().cpu().numpy() if hasattr(forecasts[0], "detach") else np.asarray(forecasts[0])
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 20 or values.shape[2] < horizon:
        raise ValueError(f"Chronos-2 returned unexpected forecast shape: {values.shape}")
    lower, median, upper = (float(values[0, index, -1]) for index in (3, 11, 19))
    return median, lower, upper, model_name


def _forecast_from_quantiles(close: float, median: float, lower: float, upper: float, *, asset: str, category: str, timeframe: str, horizon: int, model_name: str) -> ModelForecast:
    if close <= 0 or not np.isfinite([close, median, lower, upper]).all():
        raise ValueError("Chronos-2 returned non-finite prices")
    expected_return = (median / close) - 1.0
    lower_return = (lower / close) - 1.0
    upper_return = (upper / close) - 1.0
    direction = "LONG" if expected_return > 0 else "SHORT" if expected_return < 0 else "UNCERTAIN"
    spread = abs(upper_return - lower_return)
    model_score = float(np.clip(50.0 + 50.0 * abs(expected_return) / max(spread, 1e-12), 0.0, 100.0))
    uncertainty = "LOW" if spread < 0.01 else "MEDIUM" if spread < 0.03 else "HIGH"
    return ModelForecast(
        "AVAILABLE", "chronos2", model_name, asset, category, timeframe, f"next {horizon} {timeframe} bars",
        direction, model_score, expected_return, lower_return, upper_return, uncertainty, None,
        "Chronos-2 P10/P50/P90 output transformed deterministically into return and score.", lower, median, upper,
    )


def _config_for(asset: str, category: str, model_root: str | Path | None):
    path = "configs/training_crypto_intraday.yaml" if "USDT" in asset else "configs/training_equity_intraday.yaml"
    config = load_config(path)
    if model_root is not None:
        config.ml.model_dir = str(model_root)
    return config


def _infer_swing_forecast(
    asset: str,
    timeframe: str,
    frame: pd.DataFrame,
    *,
    horizon: str | int | float,
    model_root: str | Path | None,
) -> ModelForecast:
    model_id = f"model2_swing:{asset}"
    try:
        if asset not in {"SPY", "QQQ"} or timeframe != "1d":
            raise ValueError("registered swing model supports SPY/QQQ on 1d only")
        normalized = str(horizon).lower().replace("trading days", "d").replace(" ", "")
        horizon_days = int(float(normalized[:-1])) if normalized.endswith("d") else int(float(normalized))
        if horizon_days not in SWING_HORIZONS:
            raise ValueError(f"requested swing horizon {horizon!r} is unavailable; supported={SWING_HORIZONS}")
        config = load_config("configs/training_swing.yaml").swing
        if model_root is not None:
            config.model_dir = str(model_root)
        checkpoint = Path(config.model_dir) / f"{asset.lower()}_swing.pt"
        dataset = build_swing_dataset(config.market_cache_dir, asset, config.daily_lookback, list(config.target_horizons), config.macro_symbols)
        model = load_swing_model(checkpoint, config)
        outputs = model.predict(dataset.X_daily[-1:], dataset.X_entry[-1:], dataset.branch_mask[-1:])
        expected_return = float(outputs["returns"][0, list(SWING_HORIZONS).index(horizon_days)])
        expected_mfe = float(outputs["mfe"][0])
        expected_mae = float(outputs["mae"][0])
        expected_duration = float(outputs["duration"][0, list(SWING_HORIZONS).index(horizon_days)])
        entry_signal = float(outputs["entry"][0])
        expected_returns = {h: float(outputs["returns"][0, index]) for index, h in enumerate(SWING_HORIZONS)}
        score = horizon_opportunity_score(expected_return, expected_mfe, expected_mae, entry_signal, config)
        direction = "LONG" if expected_return > 0 else "SHORT" if expected_return < 0 else "UNCERTAIN"
        return ModelForecast("AVAILABLE", model_id, "swing-v1", asset, "SWING", timeframe, str(horizon), direction, score, expected_return, expected_mae, expected_mfe, "UNKNOWN", dataset.timestamps[-1].isoformat(), "Loaded registered SPY/QQQ swing checkpoint.", expected_duration=expected_duration, horizon=str(horizon), asset_class="equity", data_quality="AVAILABLE", confidence="UNKNOWN")
    except (OSError, ImportError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        return ModelForecast("MODEL_UNAVAILABLE", model_id, "unknown", asset, "SWING", timeframe, str(horizon), None, None, None, None, None, "UNKNOWN", None, str(exc), horizon=str(horizon), asset_class="equity", data_quality="UNAVAILABLE", confidence="UNKNOWN")


def _infer_registered_forecast(
    asset: str,
    timeframe: str,
    frame: pd.DataFrame,
    *,
    horizon: str | int | float,
    category: str,
    model_root: str | Path | None,
    chronos_reason: str,
) -> ModelForecast:
    if asset in {"SPY", "QQQ"} and category.upper() == "SWING":
        return _infer_swing_forecast(asset, timeframe, frame, horizon=horizon, model_root=model_root)
    return _infer_tcn_forecast(asset, timeframe, frame, horizon=horizon, category=category, model_root=model_root, chronos_reason=chronos_reason)


def _infer_tcn_forecast(
    asset: str,
    timeframe: str,
    frame: pd.DataFrame,
    *,
    horizon: str | int | float,
    category: str,
    model_root: str | Path | None,
    chronos_reason: str = "not attempted",
) -> ModelForecast:
    """Infer the local checkpoint without invoking a foundation model."""
    model_id = "model1a_crypto_tcn" if "USDT" in asset else "model1b_equity_tcn"
    if asset in {"SPY", "QQQ"} and category.upper() == "INTRADAY":
        return ModelForecast(
            "MODEL_UNAVAILABLE", model_id, "disabled-v3", asset, category, timeframe, str(horizon),
            None, None, None, None, None, "UNKNOWN", None,
            "Equity intraday TCN is disabled; use Chronos-2 with the rule-strategy agreement gate.",
            horizon=str(horizon), asset_class="equity", data_quality="UNAVAILABLE",
        )
    try:
        config = _config_for(asset, category, model_root)
        symbol = asset if "USDT" in asset else {"QQQ": "NASDAQ100_PROXY", "SPY": "SP500_PROXY"}.get(asset, asset)
        requested_horizon = horizon_to_bars(horizon, timeframe, trading_days=str(horizon).lower().endswith("trading days"))
        closed = _closed_candles(frame, timeframe)
        model, meta = load_symbol_model(symbol, config.ml)
        if requested_horizon not in model.forecast_horizons:
            raise ValueError(f"requested horizon {horizon!r} ({requested_horizon} bars) is unavailable; supported={model.forecast_horizons}")
        features = build_feature_matrix(closed, None, config.ml)
        features = features.reindex(columns=meta["feature_columns"])
        valid = features.dropna()
        sequence_length = int(meta["sequence_length"])
        if len(valid) < sequence_length:
            raise ValueError(f"insufficient feature warmup: {len(valid)} < {sequence_length}")
        window = valid.iloc[-sequence_length:]
        outputs = model.predict_trade_outputs_by_horizon(make_sequences(window.to_numpy(dtype=np.float32), sequence_length))[requested_horizon]
        quality = continuous_opportunity_score(outputs["opportunity"], outputs["expected_return"], outputs["expected_mfe"], outputs["expected_mae"])
        expected_return = float(outputs["expected_return"][-1])
        adverse = float(outputs["expected_mae"][-1])
        favorable = float(outputs["expected_mfe"][-1])
        direction = str(quality["direction"][-1])
        magnitude = abs(adverse) + abs(favorable)
        uncertainty = "LOW" if magnitude < 0.01 else "MEDIUM" if magnitude < 0.03 else "HIGH"
        internal_note = f" internal_model={symbol}" if symbol != asset else ""
        probabilities = outputs["probabilities"][-1]
        return ModelForecast("AVAILABLE", f"{model_id}:{symbol}", str(meta.get("model_version", "unknown")), asset, category, timeframe, str(horizon), direction, float(quality["score"][-1]), expected_return, adverse, favorable, uncertainty, window.index[-1].isoformat(), f"Loaded requested horizon head={requested_horizon}; Chronos-2 unavailable: {chronos_reason}.{internal_note}", horizon=str(horizon), asset_class="crypto" if "USDT" in asset else "equity", data_quality="AVAILABLE", confidence=uncertainty, expected_duration=float(outputs["expected_duration"][-1]), probability_short=float(probabilities[0]), probability_neutral=float(probabilities[1]), probability_long=float(probabilities[2]), opportunity_score=float(quality["score"][-1]))
    except (OSError, ImportError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        return ModelForecast("MODEL_UNAVAILABLE", model_id, "unknown", asset, category, timeframe, str(horizon), None, None, None, None, None, "UNKNOWN", None, str(exc), horizon=str(horizon), asset_class="crypto" if "USDT" in asset else "equity", data_quality="UNAVAILABLE", confidence="UNKNOWN")


def infer_chronos2_forecast(
    asset: str,
    timeframe: str,
    frame: pd.DataFrame,
    *,
    horizon: str | int | float,
    category: str = "INTRADAY",
    trading_days: bool = False,
) -> ModelForecast:
    """Run only the configured Chronos-2 source; never substitute a TCN."""
    try:
        horizon_bars = horizon_to_bars(horizon, timeframe, trading_days=trading_days)
        closed = _closed_candles(frame, timeframe)
        if len(closed) < max(64, horizon_bars * 4):
            raise ValueError("Chronos-2 context is too short for the requested horizon")
        median, lower, upper, model_name = _chronos2_quantile_forecast(closed, horizon_bars)
        forecast = _forecast_from_quantiles(
            float(closed["close"].iloc[-1]), median, lower, upper,
            asset=asset, category=category, timeframe=timeframe, horizon=horizon_bars, model_name=model_name,
        )
        return ModelForecast(
            forecast.status, forecast.model_id, forecast.model_version, forecast.asset, forecast.category,
            forecast.input_timeframe, str(horizon), forecast.direction, forecast.model_score,
            forecast.expected_return, forecast.expected_adverse_move, forecast.expected_favorable_move,
            forecast.uncertainty, closed.index[-1].isoformat(), forecast.reason, forecast.forecast_low,
            forecast.forecast_median, forecast.forecast_high, horizon=str(horizon),
            asset_class="crypto" if "USDT" in asset else "equity", data_quality="AVAILABLE",
            confidence=forecast.uncertainty,
        )
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        return ModelForecast(
            "MODEL_UNAVAILABLE", "chronos2", "unknown", asset, category, timeframe, str(horizon),
            None, None, None, None, None, "UNKNOWN", None, str(exc), horizon=str(horizon),
            asset_class="crypto" if "USDT" in asset else "equity", data_quality="UNAVAILABLE", confidence="UNKNOWN",
        )


def infer_model_forecast(
    asset: str,
    timeframe: str,
    frame: pd.DataFrame,
    *,
    horizon: str | int | float,
    category: str = "INTRADAY",
    model_root: str | Path | None = None,
) -> ModelForecast:
    """Prefer Chronos-2 zero-shot inference, then use the validated TCN fallback."""
    chronos = infer_chronos2_forecast(
        asset, timeframe, frame, horizon=horizon, category=category,
        trading_days=str(horizon).lower().endswith("trading days"),
    )
    if chronos.status == "AVAILABLE":
        return chronos
    return _infer_registered_forecast(asset, timeframe, frame, horizon=horizon, category=category, model_root=model_root, chronos_reason=chronos.reason)


def _model_agreement(tcn: ModelForecast, chronos: ModelForecast) -> tuple[str, float | None]:
    """Return a transparent consensus state without averaging unsupported models."""
    if tcn.status != "AVAILABLE" or chronos.status != "AVAILABLE":
        return "UNAVAILABLE", None
    if tcn.direction in {None, "UNCERTAIN"} or chronos.direction in {None, "UNCERTAIN"}:
        return "LOW", None
    if tcn.direction != chronos.direction:
        return "CONFLICT", None
    if tcn.model_score is None or chronos.model_score is None:
        return "LOW", None
    # This is a consensus confidence, not a promoted trading ensemble. A final
    # weighting requires chronological OOS and cost-aware evidence.
    return "HIGH", float(min(tcn.model_score, chronos.model_score))


def infer_model_comparison(
    asset: str,
    timeframe: str,
    frame: pd.DataFrame,
    *,
    horizon: str | int | float,
    category: str = "INTRADAY",
    model_root: str | Path | None = None,
) -> ModelComparison:
    """Infer Chronos-2 and TCN independently on the same latest closed candle."""
    try:
        closed_for_cache = _closed_candles(frame, timeframe)
        if closed_for_cache.empty:
            raise ValueError("no closed candles available")
        cache_key = (
            asset,
            timeframe,
            category.upper(),
            str(horizon),
            closed_for_cache.index[-1].isoformat(),
            float(closed_for_cache["close"].iloc[-1]),
            str(model_root or ""),
        )
        with _COMPARISON_CACHE_LOCK:
            cached = _COMPARISON_CACHE.get(cache_key)
        if cached is not None:
            return cached
    except (OSError, KeyError, TypeError, ValueError):
        cache_key = None

    chronos = ModelForecast("MODEL_UNAVAILABLE", "chronos2", "unknown", asset, category, timeframe, "unknown", None, None, None, None, None, "UNKNOWN", None, "not attempted")
    tcn = ModelForecast("MODEL_UNAVAILABLE", "model1a_crypto_tcn" if "USDT" in asset else "model1b_equity_tcn", "unknown", asset, category, timeframe, "unknown", None, None, None, None, None, "UNKNOWN", None, "not attempted")
    try:
        closed = _closed_candles(frame, timeframe)
        horizon_bars = horizon_to_bars(horizon, timeframe, trading_days=str(horizon).lower().endswith("trading days"))
        if len(closed) >= max(64, horizon_bars * 4):
            try:
                median, lower, upper, model_name = _chronos2_quantile_forecast(closed, horizon_bars)
                raw = _forecast_from_quantiles(float(closed["close"].iloc[-1]), median, lower, upper, asset=asset, category=category, timeframe=timeframe, horizon=horizon_bars, model_name=model_name)
                chronos = ModelForecast(raw.status, raw.model_id, raw.model_version, raw.asset, raw.category, raw.input_timeframe, raw.forecast_horizon, raw.direction, raw.model_score, raw.expected_return, raw.expected_adverse_move, raw.expected_favorable_move, raw.uncertainty, closed.index[-1].isoformat(), raw.reason, raw.forecast_low, raw.forecast_median, raw.forecast_high)
            except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                chronos = ModelForecast("MODEL_UNAVAILABLE", "chronos2", "unknown", asset, category, timeframe, str(horizon), None, None, None, None, None, "UNKNOWN", closed.index[-1].isoformat(), str(exc), horizon=str(horizon), data_quality="UNAVAILABLE")
        else:
            chronos = ModelForecast("MODEL_UNAVAILABLE", "chronos2", "unknown", asset, category, timeframe, str(horizon), None, None, None, None, None, "UNKNOWN", closed.index[-1].isoformat(), "Chronos-2 context is too short for the requested horizon", horizon=str(horizon), data_quality="UNAVAILABLE")
    except (OSError, KeyError, TypeError, ValueError) as exc:
        chronos = ModelForecast("MODEL_UNAVAILABLE", "chronos2", "unknown", asset, category, timeframe, "unknown", None, None, None, None, None, "UNKNOWN", None, str(exc))

    tcn = _infer_registered_forecast(asset, timeframe, frame, horizon=horizon, category=category, model_root=model_root, chronos_reason="not attempted")

    agreement, ensemble_score = _model_agreement(tcn, chronos)
    # No model or ensemble is active until fixed walk-forward and untouched
    # holdout evidence demonstrate incremental trading value. Keep the raw
    # forecasts visible, but withhold their score from the decision pipeline.
    selected = ModelForecast(
        "MODEL_UNAVAILABLE",
        "model_conflict" if agreement == "CONFLICT" else "model_observation_only",
        "v1.0-no-promotion",
        asset,
        category,
        timeframe,
        chronos.forecast_horizon if chronos.forecast_horizon != "unknown" else tcn.forecast_horizon,
        None,
        None,
        None,
        None,
        None,
        "HIGH" if agreement == "CONFLICT" else "UNKNOWN",
        chronos.data_timestamp or tcn.data_timestamp,
        "TCN and Chronos-2 disagree; model component withheld."
        if agreement == "CONFLICT"
        else "Model outputs are observation-only pending chronological OOS promotion.",
    )
    comparison = ModelComparison(tcn, chronos, agreement, ensemble_score, selected)
    if cache_key is not None:
        with _COMPARISON_CACHE_LOCK:
            if len(_COMPARISON_CACHE) >= _COMPARISON_CACHE_LIMIT:
                _COMPARISON_CACHE.pop(next(iter(_COMPARISON_CACHE)))
            _COMPARISON_CACHE[cache_key] = comparison
    return comparison
