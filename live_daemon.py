"""Live-Daemon: zieht Marktdaten (CCXT fuer Krypto, yfinance/TwelveData fuer
Equities ueber die bereits vorhandene ``core.ml.providers``-Infrastruktur),
inferiert Model 1A (Crypto Sniper), Model 1B (Equity Sniper) und Model 2
(Equity Swing) und schreibt Scores/MAE/MFE/Expected-Duration/Markt-Status
fortlaufend in ``trading_state.db``.

Bewusst getrennt vom Paper-Broker/GUI-Prozess (siehe Modul-Docstring in
``paper_broker.py``): dieser Prozess SCHREIBT nur ``market_state``/``signals``,
er liest/schreibt nie ``open_trades``/``paper_portfolio`` -- kann jederzeit
neu gestartet werden, ohne dass der Broker seinen State verliert.

Zwei Taktraten (Hybrid-Polling, analog zu ``execution/live_trader.py``):
  * ``PRICE_POLL_S`` (schnell): nur der aktuelle Preis + Session-Status.
  * ``SIGNAL_REFRESH_S`` (langsam): vollständiges Nachladen der historischen
    Kerzen + ML-Inferenz -- die Modelle aendern ihre Meinung ohnehin nur mit
    jeder neuen abgeschlossenen Kerze (5m fuer Model 1, 1d fuer Model 2).
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import wraps
from datetime import datetime, time as clock_time, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import state_db
from train import load_config
from core.config import TradingBotConfig, TrendMLConfig
from core.data_loader import drop_incomplete_last_candle
from core.market_state import DataHealth, FeedStatus, MarketStateService, PredictionSnapshot
from core.ml.data import load_ohlcv_parquet
from core.ml.dataset import make_sequences
from core.ml.device import get_device
from core.ml.features import build_feature_matrix, fetch_breadth_basket
from core.ml.assistant_forecasts import PUBLIC_HORIZONS, chronos2_forecast, combine_model_forecasts, combine_rule_chronos_forecast, direct_swing_forecasts, direct_tcn_forecasts, unavailable_forecast
from core.ml.forecast_contract import context_feature_frame
from core.ml.inference import load_symbol_model
from core.ml.providers import AssetSpec, asset_specs_from_config, canonical_cache_path, prepare_asset_specs
from core.ml.scoring import continuous_opportunity_score
from core.ml.swing_data import build_swing_dataset
from core.ml.swing_model import horizon_opportunity_score, load_swing_model, scores_from_predictions, HORIZONS
from core.signal_orchestrator import AlertGate, AlertGateConfig, intraday_snapshot
from core.strategy import compute_atr
from decision_pipeline import calculate_indicators, select_strategy
from risk_engine import TradeSetup, compute_strategy_profiles
from core.notifications.telegram_bot import TelegramBot
from model_integration import infer_chronos2_forecast

PRICE_POLL_S = 20.0
SIGNAL_REFRESH_S = 15 * 60.0
MIN_CHRONOS_SIGNAL_SCORE = 55.0
ASSISTANT_ASSETS = ("BTC/USDT", "ETH/USDT", "SPY", "QQQ")
FX_ASSET = "EUR/USD"
FX_TICKER = "EURUSD=X"
FX_REFRESH_S = 60.0 * 60.0
LIVE_MAX_CANDLES = 500
LIVE_DAILY_DONCHIAN_BARS = 55

# Model 2 (Swing) uses the plain ticker as its asset name; Model 1B (equity
# intraday) uses the "logical" proxy name from TrendMLConfig.market_assets.
# Everywhere in the DB/GUI/broker we standardize on the ticker (QQQ/SPY),
# since that's what a human operator / the paper broker actually trades.
EQUITY_NAME_TO_TICKER = {"NASDAQ100_PROXY": "QQQ", "SP500_PROXY": "SPY"}

CONFIG_PATHS = {
    "crypto": "configs/training_tcn_crypto_1h.yaml",
    "equity": "configs/training_tcn_equity_1h.yaml",
    "swing": "configs/training_swing.yaml",
}


def retry_with_backoff(max_retries: int = 3, base_delay_s: float = 1.0):
    """Retry transient provider failures without ever escaping the daemon loop."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return function(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 - provider-specific errors vary
                    last_error = exc
                    if attempt + 1 < max_retries:
                        time.sleep(base_delay_s * (2**attempt))
            raise last_error
        return wrapped
    return decorate


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _history_days_for_bars(bar_count: int, timeframe: str, session_timezone: str) -> int:
    normalized_timeframe = f"{timeframe[:-1]}D" if timeframe.lower().endswith("d") else timeframe
    duration = pd.Timedelta(normalized_timeframe)
    if duration <= pd.Timedelta(0):
        raise ValueError(f"Invalid live timeframe: {timeframe!r}")
    if session_timezone not in {"", "UTC", "Etc/UTC"}:
        bars_per_session = (
            1.0
            if duration >= pd.Timedelta(days=1)
            else 390.0 / (duration.total_seconds() / 60.0)
        )
        return max(1, math.ceil(bar_count / bars_per_session * 7.0 / 5.0))
    return max(1, math.ceil(bar_count * duration.total_seconds() / 86400.0))


def _base_candle_limit(ml_config: TrendMLConfig, asset_name: str) -> int:
    if (
        ml_config.base_timeframe == "1h"
        and "USDT" in asset_name.upper()
        and "1d" in (ml_config.higher_timeframes or [])
    ):
        sequence_length = max(1, int(ml_config.sequence_length))
        return max(
            LIVE_MAX_CANDLES,
            LIVE_DAILY_DONCHIAN_BARS * 24 + sequence_length + 24,
        )
    return LIVE_MAX_CANDLES


def _bounded_live_specs(config: TradingBotConfig, specs: list[AssetSpec]) -> list[AssetSpec]:
    bounded = []
    for spec in specs:
        definition = config.ml.market_assets.get(spec.name, {})
        if (
            spec.timeframe == "5m"
            and isinstance(definition, dict)
            and definition.get("derive_from") == "5m"
            and config.ml.base_timeframe == "1h"
        ):
            target_bars = LIVE_MAX_CANDLES
            target_timeframe = "1h"
        elif spec.timeframe == config.ml.base_timeframe and spec.name in config.ml.assets:
            target_bars = _base_candle_limit(config.ml, spec.name)
            target_timeframe = spec.timeframe
        else:
            target_bars = LIVE_MAX_CANDLES
            target_timeframe = spec.timeframe
        max_days = _history_days_for_bars(target_bars, target_timeframe, spec.session_timezone)
        bounded.append(replace(spec, history_days=max(1, min(int(spec.history_days), max_days)), history_start=None))
    return bounded


def _closed_candle_timestamp(frame: pd.DataFrame, timeframe: str) -> str | None:
    if frame.empty:
        return None
    timestamp = pd.Timestamp(frame.index[-1])
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return (timestamp + pd.Timedelta(timeframe)).isoformat()


def _load_base_ohlc(ml_config: TrendMLConfig, asset_name: str) -> pd.DataFrame | None:
    specs = [s for s in asset_specs_from_config(TradingBotConfig(ml=ml_config)) if s.name == asset_name and s.timeframe == ml_config.base_timeframe]
    if not specs:
        return None
    path = canonical_cache_path(ml_config.market_cache_dir, specs[0])
    if not path.exists():
        return None
    frame = load_ohlcv_parquet(path)
    if frame.empty:
        return frame
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, utc=True))
    frame = drop_incomplete_last_candle(frame.sort_index(), ml_config.base_timeframe)
    return frame.tail(_base_candle_limit(ml_config, asset_name))


def _resolve_training_timeframe(meta: dict, config: TrendMLConfig) -> str:
    """Accept the canonical field when present, otherwise infer from the
    existing ``timeframes`` list so older 1h artifacts still load with the
    correct production-timeframe contract while true mismatches remain hard
    errors."""
    if meta.get("training_timeframe") not in (None, ""):
        return str(meta["training_timeframe"])

    timeframes = meta.get("timeframes")
    if isinstance(timeframes, list) and timeframes:
        candidates = [str(value) for value in timeframes if value not in (None, "")]
        if config.base_timeframe in candidates:
            return config.base_timeframe
        if len(set(candidates)) == 1:
            return candidates[0]
        raise ValueError(
            f"model metadata is missing a unique training timeframe; timeframes={candidates!r}, "
            f"required={config.base_timeframe!r}"
        )
    raise ValueError("model metadata is missing training_timeframe and has no usable timeframes contract")


def _require_chronos_directional_edge(forecast):
    if forecast.forecast_status != "MODEL_AGREEMENT":
        return forecast
    chronos = next(
        (
            item for item in forecast.individual_forecasts
            if isinstance(item, dict)
            and ("CHRONOS" in str(item.get("forecast_source", "")).upper()
                 or "CHRONOS" in str(item.get("model", "")).upper())
        ),
        None,
    )
    try:
        chronos_score = float(chronos.get("opportunity_score")) if chronos is not None else float("nan")
    except (TypeError, ValueError):
        chronos_score = float("nan")
    if not np.isfinite(chronos_score) or not MIN_CHRONOS_SIGNAL_SCORE <= chronos_score <= 100.0:
        return replace(
            forecast,
            forecast_status="DEGRADED",
            quality_status="DEGRADED",
            usable_for_decision=False,
            combined_confidence="WEAK_CHRONOS_SIGNAL",
            reason=(
                f"{forecast.reason} Chronos-2 directional score must be at least "
                f"{MIN_CHRONOS_SIGNAL_SCORE:.0f}; got {chronos_score if np.isfinite(chronos_score) else 'unavailable'}."
            ).strip(),
        )
    return forecast


def _crypto_signal_from_consensus(forecasts, now: datetime) -> dict[str, object]:
    eligible = [
        forecast for forecast in forecasts
        if forecast.horizon in {"4h", "8h"}
        and forecast.forecast_status == "MODEL_AGREEMENT"
        and forecast.usable_for_decision
        and forecast.direction in {"LONG", "SHORT"}
        and forecast.opportunity_score is not None
    ]
    if not eligible:
        return {
            "timestamp": now.isoformat(), "score": 0.0, "direction": "UNCERTAIN",
            "expected_return": 0.0, "expected_duration": 0.0, "expected_mfe": 0.0,
            "expected_mae": 0.0, "entry_score": 0.0, "forecast_score": 0.0,
            "model_version": "no_valid_ensemble", "context_status": "UNAVAILABLE",
        }
    selected = max(eligible, key=lambda item: float(item.opportunity_score))
    score = float(selected.opportunity_score)
    return {
        "timestamp": selected.forecast_timestamp or now.isoformat(),
        "score": score,
        "direction": selected.direction,
        "expected_return": float(selected.expected_return or 0.0),
        "expected_duration": float(selected.expected_duration or 0.0),
        "expected_mfe": float(selected.expected_mfe or 0.0),
        "expected_mae": float(selected.expected_mae or 0.0),
        "entry_score": score,
        "forecast_score": score,
        "model_version": selected.model_version,
        "context_status": "OK",
    }


def _equity_intraday_signal_from_consensus(forecasts, now: datetime) -> dict[str, object]:
    candidates = []
    for forecast in forecasts:
        if (
            forecast.horizon not in {"1h", "4h", "8h", "12h"}
            or forecast.forecast_status != "MODEL_AGREEMENT"
            or not forecast.usable_for_decision
            or forecast.direction not in {"LONG", "SHORT"}
        ):
            continue
        rule = next(
            (item for item in forecast.individual_forecasts if isinstance(item, dict) and "RULE_STRATEGY" in str(item.get("model", "")).upper()),
            None,
        )
        chronos = next(
            (item for item in forecast.individual_forecasts if isinstance(item, dict) and "CHRONOS" in str(item.get("model", "")).upper()),
            None,
        )
        try:
            rule_score = float(rule["score"]) if rule is not None else float("nan")
            chronos_score = float(chronos["opportunity_score"]) if chronos is not None else float("nan")
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite([rule_score, chronos_score]).all():
            continue
        consensus_score = min(rule_score, chronos_score)
        candidates.append((
            (consensus_score, (rule_score + chronos_score) / 2.0, -int(forecast.horizon[:-1])),
            forecast,
            rule_score,
            consensus_score,
        ))
    if not candidates:
        return {
            "timestamp": now.isoformat(), "score": 0.0, "direction": "UNCERTAIN",
            "expected_return": 0.0, "expected_duration": 0.0, "expected_mfe": 0.0,
            "expected_mae": 0.0, "entry_score": 0.0, "forecast_score": 0.0,
            "model_version": "no_valid_chronos_rule_ensemble", "context_status": "UNAVAILABLE",
        }
    _, selected, rule_score, consensus_score = max(candidates, key=lambda item: item[0])
    return {
        "timestamp": selected.forecast_timestamp or now.isoformat(),
        "score": consensus_score,
        "direction": selected.direction,
        "expected_return": float(selected.expected_return or 0.0),
        "expected_duration": float(selected.expected_duration or int(selected.horizon[:-1])),
        "expected_mfe": float(selected.expected_mfe or 0.0),
        "expected_mae": float(selected.expected_mae or 0.0),
        "entry_score": consensus_score,
        "forecast_score": consensus_score,
        "rule_score": rule_score,
        "model_version": selected.model_version,
        "context_status": "OK",
    }


class Model1Runner:
    """Loads and caches one Model 1 (TCN multitask) checkpoint per asset for
    the lifetime of the daemon process -- avoids re-loading torch weights
    from disk every single refresh cycle."""

    def __init__(self, ml_config: TrendMLConfig) -> None:
        self.config = ml_config
        self._models: dict[str, tuple] = {}
        self._failed: set[str] = set()

    def _model(self, asset_name: str):
        if asset_name in self._failed:
            return None
        if asset_name not in self._models:
            try:
                model, meta = load_symbol_model(asset_name, self.config)
                device = get_device()
                model_module = getattr(model, "model", None)
                if device.type == "cuda" and model_module is not None:
                    try:
                        model_module.to(device)
                        model.device = device
                        model.use_amp = bool(model.config.use_amp)
                    except Exception as exc:  # noqa: BLE001 - keep inference available if CUDA allocation fails
                        print(f"[model1] CUDA unavailable for {asset_name}; retaining CPU model: {exc}")
                        model.to_cpu()
                if self.config.context_required:
                    training_timeframe = _resolve_training_timeframe(meta, self.config)
                    if training_timeframe != self.config.base_timeframe:
                        raise ValueError(
                            f"TCN artifact training timeframe {training_timeframe!r} does not match production timeframe {self.config.base_timeframe!r}"
                        )
                    if meta.get("context_feature_version") is not None and meta.get("context_feature_version") != self.config.context_feature_version:
                        raise ValueError("TCN artifact lacks the required context feature contract")
                self._models[asset_name] = (model, meta)
            except ValueError as exc:  # metadata contract mismatch or stale artifact -- surface it
                print(f"[model1] could not load checkpoint for {asset_name}: {exc}")
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"[model1] could not load checkpoint for {asset_name}: {exc}")
                self._failed.add(asset_name)
                return None
        return self._models[asset_name]

    def infer(self, asset_name: str, ohlc: pd.DataFrame, macro_prices: pd.DataFrame | None, breadth_return: pd.Series | None) -> dict | None:
        loaded = self._model(asset_name)
        if loaded is None:
            return None
        model, meta = loaded
        features = build_feature_matrix(ohlc, macro_prices, self.config, breadth_return)
        features = features.reindex(columns=meta["feature_columns"])
        valid = features.dropna()
        seq_len = int(meta["sequence_length"])
        if len(valid) < seq_len:
            return None
        # Only the most recent sequence is needed for live inference.
        window = valid.iloc[-seq_len:]
        X_seq = make_sequences(window.to_numpy(dtype=np.float32), seq_len)
        outputs = model.predict_trade_outputs(X_seq)
        quality = continuous_opportunity_score(
            outputs["opportunity"], outputs["expected_return"], outputs["expected_mfe"], outputs["expected_mae"],
        )
        return {
            "model_id": self.config.model_id,
            "model_version": str(meta.get("model_version", "unknown")),
            "timestamp": window.index[-1],
            "opportunity": float(outputs["opportunity"][0]),
            "expected_return": float(outputs["expected_return"][0]),
            "expected_duration": float(outputs["expected_duration"][0]),
            "expected_mfe": float(outputs["expected_mfe"][0]),
            "expected_mae": float(outputs["expected_mae"][0]),
            "entry_score": float(outputs["entry_score"][0]),
            "score": float(quality["score"][0]),
            "forecast_score": float(quality["score"][0]),
            "direction_probability_short": float(outputs["probabilities"][0, 0]),
            "direction_probability_long": float(outputs["probabilities"][0, 2]),
            "lower_quantile": None,
            "median_quantile": None,
            "upper_quantile": None,
            "expected_volatility": None,
            "forecast_uncertainty": "UNKNOWN",
            "context_status": "OK" if macro_prices is not None and not macro_prices.empty else "MISSING",
            "direction": str(quality["direction"][0]),
        }

    def infer_by_horizon(self, asset_name: str, ohlc: pd.DataFrame, macro_prices: pd.DataFrame | None, breadth_return: pd.Series | None) -> tuple[str, str, dict[int, dict]] | None:
        """Return each available trained head for the public assistant contract."""
        loaded = self._model(asset_name)
        if loaded is None:
            return None
        model, meta = loaded
        features = build_feature_matrix(ohlc, macro_prices, self.config, breadth_return).reindex(columns=meta["feature_columns"])
        valid = features.dropna()
        sequence_length = int(meta["sequence_length"])
        if len(valid) < sequence_length:
            return None
        window = valid.iloc[-sequence_length:]
        outputs = model.predict_trade_outputs_by_horizon(make_sequences(window.to_numpy(dtype=np.float32), sequence_length))
        normalized: dict[int, dict] = {}
        for horizon, output in outputs.items():
            quality = continuous_opportunity_score(output["opportunity"], output["expected_return"], output["expected_mfe"], output["expected_mae"])
            normalized[int(horizon)] = {
                "probabilities": output["probabilities"][0],
                "expected_return": output["expected_return"][0],
                "expected_mfe": output["expected_mfe"][0],
                "expected_mae": output["expected_mae"][0],
                "expected_duration": output["expected_duration"][0],
                "opportunity_score": quality["score"][0],
                "direction": quality["direction"][0],
                "confidence": "UNKNOWN",
            }
        return window.index[-1].isoformat(), str(meta.get("model_version", "unknown")), normalized


class LiveDaemon:
    def __init__(self) -> None:
        self.crypto_cfg = load_config(CONFIG_PATHS["crypto"])
        self.equity_cfg = load_config(CONFIG_PATHS["equity"])
        self.swing_cfg = load_config(CONFIG_PATHS["swing"])
        self._print_inference_device()
        state_db.init_db()
        self.market_states = MarketStateService()
        self.model1_crypto = Model1Runner(self.crypto_cfg.ml)
        self._swing_models: dict[str, object] = {}
        self._last_signal_refresh = 0.0
        self._signal_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="live-model-refresh")
        self._signal_future = None
        self._exchange = None
        self.telegram = TelegramBot()
        self._notified_swing: set[tuple[str, str, str]] = set()
        self._pending_notifications: list[dict[str, object]] = []

    @staticmethod
    def _print_inference_device() -> None:
        try:
            device = get_device()
            chronos_device = os.getenv("CHRONOS2_DEVICE", "").strip().lower()
            if chronos_device not in {"cpu", "cuda"}:
                chronos_device = device.type
                os.environ["CHRONOS2_DEVICE"] = chronos_device
            print(f"Inference Device: TCN={device.type.upper()}, Chronos-2={chronos_device.upper()}")
        except ImportError:
            print("Inference Device: CPU (PyTorch not installed; ML inference unavailable)")
        except Exception as exc:  # noqa: BLE001 - startup telemetry must not block the daemon
            print(f"Inference Device: CPU (fallback: {type(exc).__name__}: {exc})")

    # --- ccxt (nur oeffentliche Endpunkte: Preis + Funding-Rate) ---------------
    def _get_exchange(self):
        if self._exchange is None:
            import ccxt

            self._exchange = ccxt.binance({"options": {"defaultType": "future"}})
        return self._exchange

    @retry_with_backoff()
    def _fetch_crypto_price_once(self, symbol: str) -> float:
        ticker = self._get_exchange().fetch_ticker(symbol)
        price = float(ticker["last"])
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"invalid live price returned for {symbol}")
        return price

    def _fetch_crypto_price(self, symbol: str) -> float | None:
        try:
            return self._fetch_crypto_price_once(symbol)
        except Exception as exc:  # noqa: BLE001 - fail closed after retries
            print(f"[daemon] crypto price fetch failed for {symbol}: {exc}")
            return None

    @retry_with_backoff()
    def _fetch_crypto_funding_rate_once(self, symbol: str) -> float:
        data = self._get_exchange().fetch_funding_rate(symbol)
        return float(data.get("fundingRate") or 0.0)

    def _fetch_crypto_funding_rate(self, symbol: str) -> float:
        try:
            return self._fetch_crypto_funding_rate_once(symbol)
        except Exception:  # noqa: BLE001 - funding is optional telemetry
            return 0.0

    @retry_with_backoff()
    def _fetch_equity_price_once(self, ticker: str) -> float:
        import yfinance as yf

        info = yf.Ticker(ticker).fast_info
        price = info.get("last_price") if isinstance(info, dict) else info.last_price
        if not price:
            raise RuntimeError(f"no live price returned for {ticker}")
        price = float(price)
        if not np.isfinite(price) or price <= 0:
            raise RuntimeError(f"invalid live price returned for {ticker}")
        return price

    def _fetch_equity_price(self, ticker: str) -> float | None:
        try:
            return self._fetch_equity_price_once(ticker)
        except Exception as exc:  # noqa: BLE001 - fail closed after retries
            print(f"[daemon] equity price fetch failed for {ticker}: {exc}")
            return None

    def _fetch_eur_usd_rate(self) -> float | None:
        import yfinance as yf

        info = yf.Ticker(FX_TICKER).fast_info
        price = info.get("last_price") if isinstance(info, dict) else info.last_price
        if price is None:
            raise RuntimeError("no EUR/USD quote returned")
        rate = float(price)
        if not np.isfinite(rate) or rate <= 0:
            raise RuntimeError("invalid EUR/USD quote returned")
        return rate

    def _refresh_eur_usd_rate(self, conn, now: datetime) -> None:
        existing = conn.execute(
            "SELECT timestamp, last_price, updated_at FROM market_state WHERE asset = ?",
            (FX_ASSET,),
        ).fetchone()
        if existing is not None and existing["updated_at"]:
            try:
                last_attempt = datetime.fromisoformat(str(existing["updated_at"]).replace("Z", "+00:00"))
                if last_attempt.tzinfo is None:
                    last_attempt = last_attempt.replace(tzinfo=timezone.utc)
                current_time = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
                if (current_time - last_attempt).total_seconds() < FX_REFRESH_S:
                    return
            except (TypeError, ValueError):
                pass

        try:
            rate = self._fetch_eur_usd_rate()
        except Exception as exc:  # noqa: BLE001 - FX failure falls back to the last stored quote
            print(f"[daemon] EUR/USD fetch failed: {exc}")
            rate = None
        fresh_quote = rate is not None and np.isfinite(rate) and rate > 0
        quote_timestamp = now.isoformat() if fresh_quote else (existing["timestamp"] if existing else None)
        quote_price = float(rate) if fresh_quote else None
        if fresh_quote:
            data_age_seconds = 0.0
            reason = ""
        elif quote_timestamp:
            try:
                previous_quote = datetime.fromisoformat(str(quote_timestamp).replace("Z", "+00:00"))
                if previous_quote.tzinfo is None:
                    previous_quote = previous_quote.replace(tzinfo=timezone.utc)
                current_time = now if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
                data_age_seconds = max(0.0, (current_time - previous_quote).total_seconds())
            except (TypeError, ValueError):
                data_age_seconds = None
            reason = "FX_STALE: EUR/USD refresh failed; retaining the last successful quote"
        else:
            data_age_seconds = None
            reason = "FX_UNAVAILABLE: no successful EUR/USD quote has been stored"
        has_stored_quote = quote_price is not None or (existing is not None and existing["last_price"] is not None)
        conn.execute(
            "INSERT INTO market_state (asset, asset_class, timestamp, market_open, session_type, is_trading_day, "
            "is_holiday, session_progress, last_price, atr, atr_pct, funding_rate, data_age_seconds, feed_healthy, "
            "freshness_ok, warmup_ready, reason, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(asset) DO UPDATE SET timestamp=CASE WHEN excluded.freshness_ok=1 THEN excluded.timestamp ELSE market_state.timestamp END, "
            "last_price=COALESCE(excluded.last_price, market_state.last_price), data_age_seconds=excluded.data_age_seconds, "
            "feed_healthy=excluded.feed_healthy, freshness_ok=excluded.freshness_ok, warmup_ready=excluded.warmup_ready, "
            "reason=excluded.reason, updated_at=excluded.updated_at",
            (
                FX_ASSET, "fx", quote_timestamp or now.isoformat(), 1, "FX", 1, 0, None,
                quote_price, None, None, None, data_age_seconds, int(fresh_quote), int(fresh_quote),
                int(has_stored_quote or fresh_quote), reason, now.isoformat(),
            ),
        )

    # --- schneller Takt: nur Preis + Session --------------------------------------
    def price_tick(self) -> None:
        now = _now()
        with state_db.connect() as conn:
            self._refresh_eur_usd_rate(conn, now)
            for symbol in self.crypto_cfg.ml.assets:
                price = self._fetch_crypto_price(symbol)
                self._upsert_market_state(conn, symbol, "crypto", price, now)
                self._persist_strategy_profiles(conn, symbol, "crypto", now)
            for name in self.equity_cfg.ml.assets:
                ticker = EQUITY_NAME_TO_TICKER.get(name, name)
                market = self.market_states.state_at(ticker, now)
                regular_session = market.market_open and market.session_type == "REGULAR"
                price = self._fetch_equity_price(ticker) if regular_session else None
                self._upsert_market_state(conn, ticker, "equity", price, now, market_snapshot=market)
                self._persist_strategy_profiles(conn, ticker, "equity", now)
            conn.commit()

    def _upsert_market_state(
        self, conn, asset: str, asset_class: str, price: float | None, now: datetime, *, market_snapshot=None,
    ) -> None:
        market = market_snapshot or self.market_states.state_at(asset, now)
        existing = conn.execute(
            "SELECT timestamp, atr, atr_pct, funding_rate FROM market_state WHERE asset = ?", (asset,),
        ).fetchone()
        atr = existing["atr"] if existing else None
        atr_pct = existing["atr_pct"] if existing else None
        funding_rate = existing["funding_rate"] if existing else None
        market_open = bool(market.market_open) and (asset_class != "equity" or market.session_type == "REGULAR")
        fresh_quote = price is not None and np.isfinite(price) and price > 0 and market_open
        quote_price = float(price) if fresh_quote else None
        if asset_class == "crypto" and fresh_quote:
            funding_rate = self._fetch_crypto_funding_rate(asset)
        if fresh_quote:
            source_timestamp = now.isoformat()
            data_age_seconds = 0.0
        elif existing is not None and existing["timestamp"]:
            source_timestamp = existing["timestamp"]
            try:
                previous = datetime.fromisoformat(str(source_timestamp).replace("Z", "+00:00"))
                if previous.tzinfo is None:
                    previous = previous.replace(tzinfo=timezone.utc)
                data_age_seconds = max(0.0, (now - previous).total_seconds())
            except (TypeError, ValueError):
                data_age_seconds = None
        else:
            source_timestamp = now.isoformat()
            data_age_seconds = None
        if asset_class == "equity" and not market_open:
            reason = "MARKET_CLOSED: last regular-session price retained for reference only"
        elif not fresh_quote:
            reason = "DATA_STALE: no fresh price tick"
        else:
            reason = ""
        conn.execute(
            "INSERT INTO market_state (asset, asset_class, timestamp, market_open, session_type, is_trading_day, "
            "is_holiday, session_progress, last_price, atr, atr_pct, funding_rate, data_age_seconds, feed_healthy, "
            "freshness_ok, warmup_ready, reason, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(asset) DO UPDATE SET timestamp=CASE WHEN excluded.freshness_ok=1 THEN excluded.timestamp ELSE market_state.timestamp END, market_open=excluded.market_open, "
            "session_type=excluded.session_type, is_trading_day=excluded.is_trading_day, is_holiday=excluded.is_holiday, "
            "session_progress=excluded.session_progress, last_price=COALESCE(excluded.last_price, market_state.last_price), "
            "funding_rate=COALESCE(excluded.funding_rate, market_state.funding_rate), data_age_seconds=excluded.data_age_seconds, "
            "feed_healthy=excluded.feed_healthy, freshness_ok=excluded.freshness_ok, warmup_ready=excluded.warmup_ready, "
            "reason=excluded.reason, updated_at=excluded.updated_at",
            (
                asset, asset_class, source_timestamp, int(market_open), market.session_type, int(market.is_trading_day),
                int(market.is_holiday), market.session_progress, quote_price, atr, atr_pct, funding_rate,
                data_age_seconds, int(fresh_quote), int(fresh_quote), int(bool(atr)), reason,
                now.isoformat(),
            ),
        )

    def _persist_strategy_profiles(self, conn, asset: str, asset_class: str, now: datetime) -> None:
        market = conn.execute("SELECT * FROM market_state WHERE asset = ?", (asset,)).fetchone()
        if market is None or market["last_price"] is None:
            return
        fx = conn.execute("SELECT last_price FROM market_state WHERE asset = ?", (FX_ASSET,)).fetchone()
        usd_per_eur = float(fx["last_price"]) if fx and fx["last_price"] is not None else 0.0
        if not np.isfinite(usd_per_eur) or usd_per_eur <= 0:
            return
        signal_type = "crypto_sniper" if asset_class == "crypto" else "swing"
        signal = conn.execute("SELECT * FROM signals WHERE asset = ? AND model_type = ?", (asset, signal_type)).fetchone()
        direction = signal["direction"] if signal and signal["direction"] in {"LONG", "SHORT"} else "LONG"
        setup = TradeSetup(
            asset=asset, asset_class=asset_class, direction=direction,
            entry_price=float(market["last_price"]), atr=float(market["atr"] or 0.0),
            expected_mae=float(signal["expected_mae"] if signal else 0.0),
            expected_mfe=float(signal["expected_mfe"] if signal else 0.0),
            score=float(signal["score"] if signal else 0.0),
            capital_eur=float(self.swing_cfg.swing.get("initial_capital_eur", 500.0)) if isinstance(self.swing_cfg.swing, dict) else 500.0,
            instrument_type="crypto_perpetual" if asset_class == "crypto" else "equity_underlying",
            usd_per_eur=usd_per_eur,
        )
        profiles = compute_strategy_profiles(setup)
        conn.execute(
            "INSERT INTO strategy_state (asset, asset_class, timestamp, direction, entry_price, aggressive_json, conservative_json, updated_at) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(asset) DO UPDATE SET asset_class=excluded.asset_class, timestamp=excluded.timestamp, direction=excluded.direction, entry_price=excluded.entry_price, aggressive_json=excluded.aggressive_json, conservative_json=excluded.conservative_json, updated_at=excluded.updated_at",
            (asset, asset_class, now.isoformat(), direction, setup.entry_price, json.dumps(profiles["aggressive"].__dict__), json.dumps(profiles["conservative"].__dict__), now.isoformat()),
        )

    # --- langsamer Takt: volle Kerzen-Aktualisierung + ML-Inferenz -----------------
    def signal_refresh(self, *, force_refresh: bool = False) -> None:
        now = _now()
        self._refresh_crypto(now, force_refresh=force_refresh)
        self._refresh_equity_and_swing(now, force_refresh=force_refresh)

    def _persist_context_health(self, conn, asset: str, ohlc: pd.DataFrame, macro_df: pd.DataFrame | None) -> None:
        if macro_df is None or macro_df.empty:
            state_db.save_context_snapshot(state_db.DB_PATH, asset, ohlc.index[-1].isoformat(), "MISSING", {"reason": "context matrix unavailable"}, conn=conn)
            return
        source_names = {"GC=F": "GOLD", "^VIX": "VIX", "^TNX": "US10Y", "EURUSD=X": "EURUSD", "CL=F": "WTI", "^GDAXI": "DAX", "ES=F": "ES", "NQ=F": "NQ"}
        context = {source_names[column]: pd.DataFrame({"close": macro_df[column]}) for column in macro_df.columns if column in source_names}
        _, snapshot = context_feature_frame(context, ohlc.index)
        state_db.save_context_snapshot(state_db.DB_PATH, asset, snapshot.timestamp, snapshot.status, snapshot.as_dict(), conn=conn)

    @staticmethod
    def _chronos_forecasts(
        asset: str,
        asset_class: str,
        ohlc: pd.DataFrame,
        source_timestamp: str | None = None,
    ) -> dict[str, object]:
        """Use Chronos-2 only on horizons with matching bar semantics."""
        horizons = PUBLIC_HORIZONS if asset_class == "crypto" else ("1h", "4h", "8h", "12h")
        forecasts = {}
        for horizon in horizons:
            output = infer_chronos2_forecast(asset, "1h", ohlc, horizon=horizon)
            if source_timestamp and getattr(output, "status", None) == "AVAILABLE":
                output = replace(output, data_timestamp=source_timestamp)
            forecasts[horizon] = chronos2_forecast(
                asset=asset,
                asset_class=asset_class,
                horizon=horizon,
                output=output,
            )
        return forecasts

    def _prepare(self, config: TradingBotConfig, *, force_refresh: bool = False) -> None:
        specs = _bounded_live_specs(config, asset_specs_from_config(config))
        try:
            self._prepare_once(config, specs, force_refresh=force_refresh)
        except Exception as exc:  # noqa: BLE001
            print(f"[daemon] data refresh failed (using stale cache if present): {exc}")

    @retry_with_backoff()
    def _prepare_once(self, config: TradingBotConfig, specs, *, force_refresh: bool = False) -> None:
        prepare_asset_specs(
            specs,
            cache_root=config.ml.market_cache_dir,
            cache_dir=config.data.cache_dir,
            exchange_id=config.data.exchange_id,
            default_market_type=config.data.market_type,
            twelve_data_settings=config.data.twelve_data,
            force_refresh=force_refresh,
        )

    def _refresh_crypto(self, now: datetime, *, force_refresh: bool = False) -> None:
        self._prepare(self.crypto_cfg, force_refresh=force_refresh)
        macro_df = None
        try:
            from core.ml.train import load_prepared_macro_matrix

            macro_df = load_prepared_macro_matrix(self.crypto_cfg.ml)
            macro_df = macro_df.tail(LIVE_MAX_CANDLES)
        except Exception as exc:  # noqa: BLE001 - missing context must remain explicit
            print(f"[daemon] crypto context matrix unavailable: {exc}")
        breadth = None
        try:
            breadth_history_days = _history_days_for_bars(
                _base_candle_limit(self.crypto_cfg.ml, self.crypto_cfg.ml.assets[0]),
                self.crypto_cfg.ml.base_timeframe,
                "UTC",
            )
            breadth_config = replace(self.crypto_cfg.ml, history_days=breadth_history_days)
            breadth = fetch_breadth_basket(self.crypto_cfg.data, breadth_config, use_cache=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[daemon] breadth basket fetch failed: {exc}")

        with state_db.connect() as conn:
            for symbol in self.crypto_cfg.ml.assets:
                try:
                    ohlc = _load_base_ohlc(self.crypto_cfg.ml, symbol)
                    if ohlc is None or ohlc.empty:
                        conn.execute(
                            "UPDATE assistant_forecasts SET forecast_status='DEGRADED', quality_status='DEGRADED', "
                            "usable_for_decision=0, freshness='UNKNOWN', reason=? WHERE asset=?",
                            ("Data unavailable during refresh; previous forecasts are not actionable.", symbol),
                        )
                        self._write_signal(
                            conn, symbol, "crypto_sniper", _crypto_signal_from_consensus([], now), now,
                        )
                        conn.commit()
                        continue
                    self._persist_context_health(conn, symbol, ohlc, macro_df)
                    atr = compute_atr(ohlc["high"], ohlc["low"], ohlc["close"], self.crypto_cfg.ml.atr_window)
                    last_close = float(ohlc["close"].iloc[-1])
                    last_atr = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else 0.0
                    conn.execute(
                        "UPDATE market_state SET atr = ?, atr_pct = ?, warmup_ready = 1 WHERE asset = ?",
                        (last_atr, last_atr / last_close if last_close else 0.0, symbol),
                    )
                    conn.commit()
                    source_timestamp = _closed_candle_timestamp(ohlc, self.crypto_cfg.ml.base_timeframe)
                    horizon_outputs = self.model1_crypto.infer_by_horizon(symbol, ohlc, macro_df, breadth)
                    if horizon_outputs is not None:
                        _, version, outputs = horizon_outputs
                        forecasts = direct_tcn_forecasts(
                            asset=symbol,
                            asset_class="crypto",
                            model=self.crypto_cfg.ml.model_id,
                            model_version=version,
                            timestamp=source_timestamp,
                            outputs=outputs,
                        )
                    else:
                        forecasts = direct_tcn_forecasts(
                            asset=symbol,
                            asset_class="crypto",
                            model=self.crypto_cfg.ml.model_id,
                            model_version="unknown",
                            timestamp=source_timestamp or now.isoformat(),
                            outputs={},
                        )
                    chronos = self._chronos_forecasts(symbol, "crypto", ohlc, source_timestamp)
                    forecasts = [
                        _require_chronos_directional_edge(combine_model_forecasts([forecast, chronos[forecast.horizon]]))
                        for forecast in forecasts
                    ]
                    state_db.replace_assistant_forecasts(conn, symbol, [item.as_dict() for item in forecasts])
                    self._write_signal(conn, symbol, "crypto_sniper", _crypto_signal_from_consensus(forecasts, now), now)
                    self._persist_strategy_profiles(conn, symbol, "crypto", now)
                except Exception:  # noqa: BLE001
                    print(f"[daemon] crypto inference failed for {symbol}:\n{traceback.format_exc()}")
                    try:
                        conn.execute(
                            "UPDATE assistant_forecasts SET forecast_status='DEGRADED', quality_status='DEGRADED', "
                            "usable_for_decision=0, freshness='UNKNOWN', reason=? WHERE asset=?",
                            ("Model refresh failed; previous forecasts are not actionable.", symbol),
                        )
                        self._write_signal(
                            conn, symbol, "crypto_sniper", _crypto_signal_from_consensus([], now), now,
                        )
                    except Exception:  # noqa: BLE001 - preserve the daemon loop if even state invalidation fails
                        print(f"[daemon] could not invalidate stale crypto signal for {symbol}:\n{traceback.format_exc()}")
                conn.commit()
            conn.commit()

    def _refresh_equity_and_swing(self, now: datetime, *, force_refresh: bool = False) -> None:
        self._prepare(self.equity_cfg, force_refresh=force_refresh)
        macro_df = None
        try:
            from core.ml.train import load_prepared_macro_matrix

            macro_df = load_prepared_macro_matrix(self.equity_cfg.ml)
            macro_df = macro_df.tail(LIVE_MAX_CANDLES)
        except Exception as exc:  # noqa: BLE001
            print(f"[daemon] equity macro matrix unavailable: {exc}")

        with state_db.connect() as conn:
            for name in self.equity_cfg.ml.assets:
                ticker = EQUITY_NAME_TO_TICKER.get(name, name)
                horizon_outputs = None
                ohlc: pd.DataFrame | None = None
                trend_rule = None
                conn.execute("DELETE FROM signals WHERE asset = ? AND model_type IN ('intraday', 'swing')", (ticker,))
                try:
                    ohlc = _load_base_ohlc(self.equity_cfg.ml, name)
                    if ohlc is None or ohlc.empty:
                        conn.execute(
                            "UPDATE assistant_forecasts SET forecast_status='DEGRADED', quality_status='DEGRADED', "
                            "usable_for_decision=0, freshness='UNKNOWN', reason=? WHERE asset=?",
                            ("Equity data unavailable during refresh; previous forecasts are not actionable.", ticker),
                        )
                        conn.commit()
                        continue
                    self._persist_context_health(conn, ticker, ohlc, macro_df)
                    atr = compute_atr(ohlc["high"], ohlc["low"], ohlc["close"], self.equity_cfg.ml.atr_window)
                    last_close = float(ohlc["close"].iloc[-1])
                    last_atr = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else 0.0
                    conn.execute(
                        "UPDATE market_state SET atr = ?, atr_pct = ?, warmup_ready = 1 WHERE asset = ?",
                        (last_atr, last_atr / last_close if last_close else 0.0, ticker),
                    )
                    indicators = calculate_indicators(ticker, "1h", ohlc, "FRESH")
                    trend_rule = next(
                        (
                            candidate
                            for candidate in select_strategy(indicators)
                            if candidate.strategy_id == "trend_breakout" and candidate.valid
                        ),
                        None,
                    )
                except Exception:  # noqa: BLE001
                    print(f"[daemon] equity rule analysis failed for {name}:\n{traceback.format_exc()}")
                    trend_rule = None

                conn.commit()
                try:
                    swing_outputs = self._refresh_swing(conn, ticker, now)
                    conn.commit()
                    source_timestamp = (
                        _closed_candle_timestamp(ohlc, self.equity_cfg.ml.base_timeframe)
                        if ohlc is not None and not ohlc.empty else None
                    )
                    chronos = (
                        self._chronos_forecasts(ticker, "equity", ohlc, source_timestamp)
                        if ohlc is not None and not ohlc.empty else {}
                    )
                    rule_direction = trend_rule.direction if trend_rule is not None else None
                    rule_score = trend_rule.score if trend_rule is not None else None
                    swing_by_horizon = {}
                    if swing_outputs is not None:
                        swing_timestamp, outputs = swing_outputs
                        daily = direct_swing_forecasts(asset=ticker, model="model2_swing", model_version="swing-v1", timestamp=swing_timestamp, outputs=outputs)
                        swing_by_horizon = {forecast.horizon: forecast for forecast in daily}
                    forecasts = []
                    for horizon in PUBLIC_HORIZONS:
                        if horizon in {"1h", "4h", "8h", "12h"}:
                            forecasts.append(
                                _require_chronos_directional_edge(combine_rule_chronos_forecast(
                                    asset=ticker,
                                    horizon=horizon,
                                    rule_direction=rule_direction,
                                    rule_score=rule_score,
                                    chronos=chronos.get(horizon),
                                ))
                            )
                        elif horizon in swing_by_horizon:
                            forecasts.append(swing_by_horizon[horizon])
                        else:
                            forecasts.append(unavailable_forecast(ticker, "equity", horizon, reason="No exact registered equity model horizon."))
                    state_db.replace_assistant_forecasts(conn, ticker, [item.as_dict() for item in forecasts])
                    self._write_signal(
                        conn, ticker, "intraday", _equity_intraday_signal_from_consensus(forecasts, now), now,
                    )
                except Exception:  # noqa: BLE001
                    print(f"[daemon] equity forecast refresh failed for {ticker}:\n{traceback.format_exc()}")
                    conn.execute(
                        "UPDATE assistant_forecasts SET forecast_status='DEGRADED', quality_status='DEGRADED', "
                        "usable_for_decision=0, freshness='UNKNOWN', reason=? WHERE asset=?",
                        ("Equity model refresh failed; previous forecasts are not actionable.", ticker),
                    )
                    conn.commit()
            conn.commit()
        self._flush_notifications()

    def _refresh_swing(self, conn, ticker: str, now: datetime) -> tuple[str, dict[int, dict]] | None:
        swing = self.swing_cfg.swing
        checkpoint = Path(swing.model_dir) / f"{ticker.lower()}_swing.pt"
        if not checkpoint.exists():
            return None
        if ticker not in self._swing_models:
            self._swing_models[ticker] = load_swing_model(checkpoint, swing)
        model = self._swing_models[ticker]
        dataset = build_swing_dataset(swing.market_cache_dir, ticker, swing.daily_lookback, swing.target_horizons, swing.macro_symbols)
        if len(dataset.timestamps) == 0:
            return None
        source_timestamp = pd.Timestamp(dataset.metadata.get("daily_data_end", dataset.timestamps[-1]))
        if source_timestamp.tzinfo is None:
            source_timestamp = source_timestamp.tz_localize("UTC")
        raw = model.predict(dataset.X_daily[-1:], dataset.X_entry[-1:], dataset.branch_mask[-1:])
        predictions = {f"expected_return_{h}d": float(raw["returns"][0, i]) for i, h in enumerate(HORIZONS)}
        predictions["expected_mfe"] = float(raw["mfe"][0])
        predictions["expected_mae"] = float(raw["mae"][0])
        predictions["entry_signal"] = float(raw["entry"][0])
        predictions["expected_duration_days"] = float(raw["duration"][0, -1])
        from core.ml.swing_model import swing_snapshot

        snapshot = swing_snapshot(ticker, source_timestamp, predictions, swing)
        self._write_snapshot(conn, ticker, "swing", snapshot, now)
        self._persist_strategy_profiles(conn, ticker, "equity", now)
        public_outputs = {}
        for index, horizon in enumerate(HORIZONS):
            expected_return = float(raw["returns"][0, index])
            public_outputs[horizon] = {
                "expected_return": expected_return,
                "expected_mfe": float(raw["mfe"][0]),
                "expected_mae": float(raw["mae"][0]),
                "expected_duration": float(raw["duration"][0, index]),
                "opportunity_score": float(horizon_opportunity_score(expected_return, float(raw["mfe"][0]), float(raw["mae"][0]), float(raw["entry"][0]), swing)),
            }
        should_alert = snapshot.swing_score is not None and snapshot.entry_score is not None and snapshot.swing_score > 80 and snapshot.entry_score > 80
        if not should_alert:
            return source_timestamp.isoformat(), public_outputs
        if should_alert:
            key = (ticker, snapshot.direction, f"{snapshot.swing_score:.2f}:{snapshot.entry_score:.2f}")
            if key not in self._notified_swing:
                profile_row = conn.execute("SELECT aggressive_json FROM strategy_state WHERE asset = ?", (ticker,)).fetchone()
                aggressive = json.loads(profile_row["aggressive_json"]) if profile_row else {}
                self._pending_notifications.append({
                    "event": "High-quality swing setup",
                    "asset": ticker,
                    "direction": snapshot.direction,
                    "swing_score": round(snapshot.swing_score, 2),
                    "entry_score": round(snapshot.entry_score, 2),
                    "entry": aggressive.get("entry_price"),
                    "stop": aggressive.get("stop_loss_price"),
                    "take_profit": aggressive.get("take_profit_price"),
                    "knockout": aggressive.get("knockout_barrier_price") or None,
                    "protection_model": aggressive.get("protection_model"),
                    "leverage": aggressive.get("leverage"),
                })
                self._notified_swing.add(key)
            forecast_timestamp = source_timestamp.isoformat()
            return forecast_timestamp, public_outputs

    def _flush_notifications(self) -> None:
        pending, self._pending_notifications = self._pending_notifications, []
        for event in pending:
            event_name = str(event.pop("event"))
            if not self.telegram.send_event(event_name, **event):
                print(f"[daemon] Telegram notification not delivered for {event.get('asset')}")

    def _write_snapshot(self, conn, asset: str, model_type: str, snapshot: PredictionSnapshot, now: datetime) -> None:
        alert_allowed, alert_reason = 0, ""
        if model_type == "intraday":
            swing_row = conn.execute("SELECT * FROM signals WHERE asset = ? AND model_type = 'swing'", (asset,)).fetchone()
            if swing_row is not None:
                swing_snap = PredictionSnapshot(
                    score=swing_row["score"], direction=swing_row["direction"], expected_return=swing_row["expected_return"],
                    expected_duration_bars=swing_row["expected_duration_bars"], expected_mfe=swing_row["expected_mfe"],
                    expected_mae=swing_row["expected_mae"], timestamp=now, model_type="swing", asset=asset,
                    swing_score=swing_row["swing_score"], entry_score=swing_row["entry_score"],
                    swing_opportunity_score=swing_row["swing_opportunity_score"],
                )
                market = conn.execute("SELECT * FROM market_state WHERE asset = ?", (asset,)).fetchone()
                if market is not None:
                    market_obj = self.market_states.state_at(asset, now)
                    health = DataHealth(now, now, 0.0, True, True, FeedStatus.AVAILABLE, True, True, FeedStatus.AVAILABLE, True, True)
                    decision = AlertGate(AlertGateConfig()).evaluate(swing_snap, snapshot, market_obj, health)
                    alert_allowed, alert_reason = int(decision.allowed), decision.reason
        conn.execute(
            "INSERT INTO signals (asset, model_type, timestamp, score, direction, expected_return, expected_duration_bars, "
            "expected_duration_days, expected_mfe, expected_mae, swing_score, entry_score, swing_opportunity_score, "
            "alert_allowed, alert_reason, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(asset, model_type) DO UPDATE SET timestamp=excluded.timestamp, score=excluded.score, "
            "direction=excluded.direction, expected_return=excluded.expected_return, "
            "expected_duration_bars=excluded.expected_duration_bars, expected_duration_days=excluded.expected_duration_days, "
            "expected_mfe=excluded.expected_mfe, expected_mae=excluded.expected_mae, swing_score=excluded.swing_score, "
            "entry_score=excluded.entry_score, swing_opportunity_score=excluded.swing_opportunity_score, "
            "alert_allowed=excluded.alert_allowed, alert_reason=excluded.alert_reason, updated_at=excluded.updated_at",
            (
                asset, model_type, snapshot.timestamp.isoformat() if hasattr(snapshot.timestamp, "isoformat") else str(snapshot.timestamp),
                snapshot.score, snapshot.direction, snapshot.expected_return, snapshot.expected_duration_bars,
                snapshot.expected_duration_days, snapshot.expected_mfe, snapshot.expected_mae, snapshot.swing_score,
                snapshot.entry_score, snapshot.swing_opportunity_score, alert_allowed, alert_reason, now.isoformat(),
            ),
        )

    def _write_signal(self, conn, asset: str, model_type: str, result: dict, now: datetime) -> None:
        conn.execute(
            "INSERT INTO signals (asset, model_type, timestamp, score, direction, expected_return, expected_duration_bars, "
            "expected_duration_days, expected_mfe, expected_mae, swing_score, entry_score, swing_opportunity_score, "
            "alert_allowed, alert_reason, updated_at, model_version, data_timestamp, rule_score, forecast_score, "
            "trade_quality_score, context_score, risk_score, combined_opportunity_score, strategy_state, context_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(asset, model_type) DO UPDATE SET timestamp=excluded.timestamp, score=excluded.score, "
            "direction=excluded.direction, expected_return=excluded.expected_return, "
            "expected_duration_bars=excluded.expected_duration_bars, expected_mfe=excluded.expected_mfe, "
            "expected_mae=excluded.expected_mae, entry_score=excluded.entry_score, alert_allowed=excluded.alert_allowed, "
            "alert_reason=excluded.alert_reason, updated_at=excluded.updated_at, model_version=excluded.model_version, "
            "data_timestamp=excluded.data_timestamp, rule_score=excluded.rule_score, forecast_score=excluded.forecast_score, "
            "trade_quality_score=excluded.trade_quality_score, context_score=excluded.context_score, risk_score=excluded.risk_score, "
            "combined_opportunity_score=excluded.combined_opportunity_score, strategy_state=excluded.strategy_state, context_status=excluded.context_status",
            (
                asset, model_type, str(result["timestamp"]), result["score"],
                result["direction"] if result["direction"] in {"LONG", "SHORT"} else "UNCERTAIN",
                result["expected_return"], result["expected_duration"], None, result["expected_mfe"], result["expected_mae"],
                None, result["entry_score"], None,
                int(result["score"] >= 80.0 and result["direction"] in {"LONG", "SHORT"}),
                "directional ensemble score >= 80" if result["score"] >= 80.0 and result["direction"] in {"LONG", "SHORT"} else "",
                now.isoformat(), result.get("model_version"), str(result.get("timestamp")) if result.get("timestamp") is not None else None, result.get("rule_score"),
                result.get("forecast_score", result.get("score")), result.get("trade_quality_score"), result.get("context_score"),
                result.get("risk_score"), result.get("combined_opportunity_score"), result.get("strategy_state"), result.get("context_status"),
            ),
        )

    # --- Hauptschleife -------------------------------------------------------------
    def _latest_equity_session_date(self, now: pd.Timestamp) -> object:
        """Return the newest completed/current regular US session date."""
        ny = ZoneInfo("America/New_York")
        market = self.market_states.state_at("SPY", now.to_pydatetime())
        local = now.tz_convert(ny)
        if market.session_type in {"REGULAR", "AFTER_HOURS"}:
            return local.date()
        candidate = local.date()
        if market.is_trading_day and local.time() >= clock_time(16):
            return candidate
        candidate -= pd.Timedelta(days=1)
        while not self.market_states.state_at("SPY", datetime.combine(candidate, clock_time(12), tzinfo=ny)).is_trading_day:
            candidate -= pd.Timedelta(days=1)
        return candidate

    def _startup_freshness(self, forecast_rows, now: pd.Timestamp) -> tuple[list[str], set[str]]:
        """Return stale rows and valid non-regular equity markets separately."""
        ny = ZoneInfo("America/New_York")
        latest_equity_session = self._latest_equity_session_date(now)
        stale: list[str] = []
        market_closed: set[str] = set()
        for row in forecast_rows:
            asset, horizon = row["asset"], row["horizon"]
            if not row["forecast_timestamp"]:
                stale.append(f"{asset} {horizon}: missing source timestamp")
                continue
            timestamp = pd.Timestamp(row["forecast_timestamp"])
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            if asset in {"BTC/USDT", "ETH/USDT"}:
                if now - timestamp > pd.Timedelta(hours=2):
                    stale.append(f"{asset} {horizon}: source age {(now - timestamp).round('min')}")
                continue
            market = self.market_states.state_at(asset, now.to_pydatetime())
            source_local = timestamp.tz_convert(ny)
            source_session = timestamp.date() if row["timeframe"] == "1d" or str(horizon).lower().endswith("d") else source_local.date()
            if source_session != latest_equity_session:
                stale.append(f"{asset} {horizon}: source session {source_session} != {latest_equity_session}")
                continue
            if market.session_type == "REGULAR":
                if row["timeframe"] == "1d" or str(horizon).lower().endswith("d"):
                    continue
                if now - timestamp > pd.Timedelta(hours=2):
                    stale.append(f"{asset} {horizon}: source age {(now - timestamp).round('min')}")
            else:
                market_closed.add(asset)
        return stale, market_closed

    def _refresh_runtime_status(self) -> str:
        """Evaluate persisted matrices after startup or a completed signal refresh."""
        with state_db.connect() as conn:
            counts = dict(conn.execute("SELECT asset, COUNT(*) FROM assistant_forecasts GROUP BY asset").fetchall())
            forecast_rows = conn.execute("SELECT asset, horizon, timeframe, forecast_timestamp FROM assistant_forecasts").fetchall()
        missing = [asset for asset in ASSISTANT_ASSETS if counts.get(asset, 0) != 9]
        if missing:
            detail = f"Missing complete forecast matrices: {', '.join(missing)}"
            state_db.set_runtime_status("DEGRADED", detail)
            return "DEGRADED"
        stale, market_closed = self._startup_freshness(forecast_rows, pd.Timestamp.now(tz="UTC"))
        if stale:
            stale_assets = {item.split(" ", 1)[0] for item in stale}
            status = "PARTIAL_DEGRADATION" if stale_assets and len(stale_assets) < len(ASSISTANT_ASSETS) else "DEGRADED"
            detail = "STALE_SOURCE_DATA: " + "; ".join(stale[:4])
            state_db.set_runtime_status(status, detail)
            return status
        detail = "Live data, context, risk profiles, and forecast matrices initialized."
        if market_closed:
            detail += f" MARKET_CLOSED: {', '.join(sorted(market_closed))}."
        state_db.set_runtime_status("READY", detail)
        return "READY"

    def initialize(self, *, background_signals: bool = False) -> str:
        """Build current market state before polling, optionally refreshing signals in the background."""
        state_db.set_runtime_status("INITIALIZING", "Refreshing market history, context, models, and forecasts.")
        try:
            self.price_tick()
            if background_signals:
                self._start_background_signal_refresh(force_refresh=True)
                return "INITIALIZING"
            self.signal_refresh(force_refresh=True)
            status = self._refresh_runtime_status()
            self._last_signal_refresh = time.monotonic()
            return status
        except Exception as exc:  # noqa: BLE001 - startup must expose failure state without crashing the loop
            detail = f"DATA_SOURCE_ERROR: {type(exc).__name__}: {exc}"
            state_db.set_runtime_status("DATA_UNAVAILABLE", detail)
            return "DATA_UNAVAILABLE"

    def _run_background_signal_refresh(self, *, force_refresh: bool) -> str:
        try:
            self.signal_refresh(force_refresh=force_refresh)
            return self._refresh_runtime_status()
        except Exception as exc:  # noqa: BLE001 - the price loop continues while model refresh degrades
            detail = f"DATA_SOURCE_ERROR: signal refresh failed: {type(exc).__name__}: {exc}"
            state_db.set_runtime_status("DEGRADED", detail)
            print(f"[daemon] {detail}")
            raise

    def _start_background_signal_refresh(self, *, force_refresh: bool) -> bool:
        future = self._signal_future
        if future is not None and not future.done():
            return False
        self._signal_future = self._signal_executor.submit(
            self._run_background_signal_refresh,
            force_refresh=force_refresh,
        )
        self._last_signal_refresh = time.monotonic()
        return True

    def _collect_background_signal_refresh(self) -> str | None:
        future = self._signal_future
        if future is None or not future.done():
            return None
        self._signal_future = None
        self._last_signal_refresh = time.monotonic()
        try:
            return future.result()
        except Exception:  # noqa: BLE001 - the worker already persisted the degraded state
            return "DEGRADED"

    def run_cycle(self, *, force_signal: bool = False, background_signals: bool = False) -> str:
        """Run one order-free live cycle; used by the loop and accelerated tests."""
        try:
            self.price_tick()
        except Exception:  # noqa: BLE001 - preserve loop resilience per cycle
            state_db.set_runtime_status("DEGRADED", "DATA_SOURCE_ERROR: price refresh failed")
            return "DEGRADED"
        if background_signals:
            completed_status = self._collect_background_signal_refresh()
            if force_signal or time.monotonic() - self._last_signal_refresh >= SIGNAL_REFRESH_S:
                self._start_background_signal_refresh(force_refresh=force_signal)
            if self._signal_future is not None:
                return "INITIALIZING"
            return completed_status or "READY"
        if force_signal or time.monotonic() - self._last_signal_refresh >= SIGNAL_REFRESH_S:
            try:
                self.signal_refresh(force_refresh=force_signal)
                self._last_signal_refresh = time.monotonic()
                return self._refresh_runtime_status()
            except Exception:  # noqa: BLE001 - provider/model failure must remain explicit
                state_db.set_runtime_status("DEGRADED", "DATA_SOURCE_ERROR: signal refresh failed")
                return "DEGRADED"
        return "READY"

    def run_forever(self) -> None:
        print(f"Live daemon starting, DB={state_db.DB_PATH}")
        print(f"Startup status: {self.initialize(background_signals=True)}")
        try:
            while True:
                loop_start = time.monotonic()
                self.run_cycle(background_signals=True)

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, PRICE_POLL_S - elapsed))
        except KeyboardInterrupt:
            print("Daemon gracefully stopped")


def main() -> None:
    LiveDaemon().run_forever()


if __name__ == "__main__":
    main()
