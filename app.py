"""Streamlit dashboard for the read-only trading decision assistant.

The app intentionally reads the local SQLite state only; it never submits orders,
starts training jobs, or executes broker actions.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math

import pandas as pd
import streamlit as st

import state_db
from core.ml.assistant_forecasts import PUBLIC_HORIZONS
from risk_engine import AGGRESSIVE_PROFILE, CONSERVATIVE_PROFILE, RECOMMENDED_PROFILE, TradeSetup, compute_risk_parameters, max_leverage_for

st.set_page_config(page_title="Trading Decision Assistant", layout="wide")
st.title("Trading Decision Assistant")
st.caption("Read-only decision support. No execution, no automatic training.")
try:
    state_db.init_db()
    _DB_INIT_ERROR = None
except Exception as exc:  # noqa: BLE001 - the UI must survive an unavailable database
    _DB_INIT_ERROR = f"{type(exc).__name__}: {exc}"

REFRESH_SECONDS = 15
MAX_FORECAST_AGE_SECONDS = 75 * 60
MAX_MARKET_STATE_AGE_SECONDS = 2 * 60
FX_ASSET = "EUR/USD"
FX_FRESHNESS_MAX_AGE_SECONDS = 65 * 60
POSITION_STOP_ATR_MULTIPLE = 1.5
POSITION_TP1_R = 1.5
POSITION_TP2_R = 2.5
POSITION_TP1_CLOSE_PCT = 50.0
POSITION_TP2_CLOSE_PCT = 25.0
REALTIME_MIN_EXPECTED_R = 1.0
MIN_CHRONOS_SIGNAL_SCORE = 55.0
KO_CERTIFICATE_WARNING = (
    "⚠️ Achtung bei Hebel-Zertifikaten/Knock-Outs: Das berechnete Risiko gilt für lineare Margin. "
    "Stelle manuell sicher, dass die KO-Schwelle des Emittenten weiter entfernt ist als der hier berechnete Stop-Loss!"
)
RISK_PROFILE_RATES = {
    "conservative": float(CONSERVATIVE_PROFILE.risk_per_trade_pct),
    "recommended": float(RECOMMENDED_PROFILE.risk_per_trade_pct),
    "aggressive": 0.25,
}


@st.cache_data(ttl=REFRESH_SECONDS)
def load_table(table: str) -> pd.DataFrame:
    try:
        with state_db.connect() as conn:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    except Exception:  # noqa: BLE001 - partial startup/SQLite lock must not crash the UI
        return pd.DataFrame()


def _has_columns(frame: pd.DataFrame, *columns: str) -> bool:
    return all(column in frame.columns for column in columns)


def _rows_for_asset(frame: pd.DataFrame, asset: str) -> pd.DataFrame:
    if frame.empty or not _has_columns(frame, "asset"):
        return pd.DataFrame()
    return frame.loc[frame["asset"].fillna("").astype(str) == asset]


def _age_seconds(timestamp: str | None) -> float | None:
    if timestamp is None:
        return None
    try:
        value = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - value).total_seconds()
    except (TypeError, ValueError):
        return None


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    try:
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return float(default)
        return numeric
    except (TypeError, ValueError):
        return float(default)


def _optional_float(value) -> float | None:
    numeric = _safe_float(value, math.nan)
    return numeric if math.isfinite(numeric) else None


def _safe_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "ready", "fresh"}:
            return True
        if normalized in {"0", "false", "no", "stale", "unavailable", ""}:
            return False
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return default
        return bool(value)
    except (TypeError, ValueError):
        return default


def _safe_text(value, default: str = "") -> str:
    if value is None:
        return default
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def _json_load(value, fallback=None):
    if isinstance(value, (dict, list)):
        return value
    if value is None:
        return fallback
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return fallback
    except (TypeError, ValueError):
        pass
    try:
        return json.loads(str(value))
    except (TypeError, ValueError):
        return fallback


def _display_float(value, fallback: float = 0.0) -> float:
    numeric = _safe_float(value, fallback)
    return float(numeric)


def _trade_direction_excursions(direction: str, raw_mfe: float, raw_mae: float) -> tuple[float, float]:
    """Convert raw price-path excursions into favorable/adverse trade direction."""
    raw_mfe = _safe_float(raw_mfe)
    raw_mae = _safe_float(raw_mae)
    if direction == "SHORT":
        return max(-raw_mae, 0.0), min(-raw_mfe, 0.0)
    if direction == "LONG":
        return max(raw_mfe, 0.0), min(raw_mae, 0.0)
    raise ValueError(f"Unsupported trade direction: {direction!r}")


def _load_assistant_forecast_row(asset: str, horizon: str) -> pd.Series | None:
    forecast_df = load_table("assistant_forecasts")
    if not _has_columns(forecast_df, "asset", "horizon"):
        return None
    rows = forecast_df[(forecast_df["asset"] == asset) & (forecast_df["horizon"] == horizon)]
    return None if rows.empty else rows.iloc[0]


def _infer_asset_class(asset: str) -> str:
    return "crypto" if any(token in asset.upper() for token in ("USDT", "BTC", "ETH")) else "equity"


def _market_snapshot(asset: str) -> pd.Series | None:
    market_df = load_table("market_state")
    rows = _rows_for_asset(market_df, asset)
    if rows.empty:
        return None
    return rows.iloc[0]


def _usd_per_eur_context() -> dict | None:
    market = _market_snapshot(FX_ASSET)
    if market is None:
        return None
    rate = _optional_float(market.get("last_price"))
    if rate is None or rate <= 0:
        return None
    timestamp = _safe_text(market.get("timestamp"))
    age_seconds = _age_seconds(timestamp)
    fresh = (
        _safe_bool(market.get("freshness_ok"), False)
        and age_seconds is not None
        and 0 <= age_seconds <= FX_FRESHNESS_MAX_AGE_SECONDS
    )
    return {
        "usd_per_eur": rate,
        "age_seconds": age_seconds,
        "fresh": fresh,
        "timestamp": timestamp,
    }


def _account_equity() -> float:
    portfolio = load_table("paper_portfolio")
    if not portfolio.empty and _has_columns(portfolio, "equity_eur"):
        equity = _optional_float(portfolio.iloc[0].get("equity_eur"))
        if equity is not None and equity > 0.0:
            return equity
    return 0.0


def _available_margin() -> float:
    portfolio = load_table("paper_portfolio")
    if not portfolio.empty and _has_columns(portfolio, "cash_eur"):
        available = _optional_float(portfolio.iloc[0].get("cash_eur"))
        if available is not None:
            return max(available, 0.0)
    return 0.0


def _risk_profile_risk_fraction(profile: str) -> float:
    return RISK_PROFILE_RATES.get(str(profile).lower(), RISK_PROFILE_RATES["recommended"])


def _risk_profile_label(profile: str) -> str:
    return {
        "conservative": "Konservativ",
        "recommended": "Recommended",
        "aggressive": "Aggressiv",
    }.get(str(profile).lower(), "Recommended")


def _max_leverage_for_asset(asset: str) -> float:
    if _infer_asset_class(asset) == "crypto":
        return max_leverage_for("crypto", "crypto_perpetual", 10.0)
    return max_leverage_for("equity", "equity_margin", 100.0)


def _quote_currency(asset: str) -> str:
    return "USDT" if str(asset).upper().endswith("/USDT") else "USD"


def _select_position_model_context(asset: str) -> tuple[str, dict | None, float | None, str]:
    candidates = []
    for horizon in _realtime_horizons(asset):
        forecast = _load_assistant_forecast_row(asset, horizon)
        if forecast is None:
            continue
        if (
            _safe_text(forecast.get("forecast_status")).upper() != "MODEL_AGREEMENT"
            or not _safe_bool(forecast.get("usable_for_decision"), False)
            or _safe_text(forecast.get("freshness")).upper() != "FRESH"
        ):
            continue
        forecast_age = _age_seconds(forecast.get("forecast_timestamp"))
        if forecast_age is None or not 0 <= forecast_age <= MAX_FORECAST_AGE_SECONDS:
            continue
        votes = _forecast_votes(forecast)
        if _agreement_label(votes) != "Ja":
            continue
        scores = [vote["score"] for vote in votes.values() if vote["score"] is not None]
        if len(scores) != 2:
            continue
        direction = _safe_text(forecast.get("direction")).upper()
        htf_trend = _htf_trend(_market_snapshot(asset), forecast)
        htf_aligned = htf_trend == ("UP" if direction == "LONG" else "DOWN")
        rank = (min(scores), int(htf_aligned), sum(scores) / len(scores), -_horizon_hours(horizon))
        candidates.append((rank, horizon, forecast, min(scores)))
    if not candidates:
        return "1h", None, None, "No fresh directional model agreement; stop falls back to the stored 1h ATR."
    _, horizon, forecast, score = max(candidates, key=lambda item: item[0])
    return horizon, forecast, score, "Fresh model agreement selected the volatility horizon."


def _runtime_health(runtime_df: pd.DataFrame, forecast_df: pd.DataFrame) -> tuple[str, str]:
    if runtime_df.empty or not _has_columns(runtime_df, "status"):
        return "NO_DATA", "No runtime snapshot yet."
    runtime = runtime_df.iloc[0]
    status = str(runtime.get("status", "UNKNOWN")).upper()
    detail = str(runtime.get("detail", "")).strip()
    stale = 0
    if _has_columns(forecast_df, "forecast_status"):
        stale = int((forecast_df["forecast_status"].fillna("").str.upper() == "STALE").sum())
    if status == "READY":
        if stale:
            return "PARTIAL_DEGRADATION", f"Core runtime is healthy, but {stale} stale forecast row(s) remain."
        if "MARKET_CLOSED" in detail:
            return "MARKET_CLOSED", detail
        return "READY", detail or "Runtime is healthy."
    if status == "DEGRADED" and stale:
        return "PARTIAL_DEGRADATION", f"System degraded only for {stale} stale forecast row(s); other assets remain usable."
    if status == "DEGRADED":
        return "DEGRADED", detail or "Some data paths are degraded."
    return status, detail or "Runtime status is unavailable."


def _market_state_issue(asset: str, market: pd.Series | None) -> str:
    if market is None:
        return "Data warming up... No market snapshot is stored for this asset yet."
    if _infer_asset_class(asset) == "equity":
        market_keys = market.index if hasattr(market, "index") else market.keys()
        if "market_open" not in market_keys:
            return "Data warming up... Equity session status is not available."
        session_type = _safe_text(market.get("session_type")).upper()
        if not _safe_bool(market.get("market_open"), False) or session_type != "REGULAR":
            return f"MARKET_CLOSED: {asset} is outside the regular US session; the last close is reference-only."
    if not _safe_bool(market.get("feed_healthy"), False) or not _safe_bool(market.get("freshness_ok"), False):
        return f"DATA_STALE: {asset} has no fresh live price; no setup can be sized."
    if not _safe_bool(market.get("warmup_ready"), False):
        return f"Data warming up... ATR is not ready for {asset}."
    data_age = _optional_float(market.get("data_age_seconds"))
    if data_age is None or data_age > MAX_MARKET_STATE_AGE_SECONDS:
        return f"DATA_STALE: {asset} quote age is unknown or exceeds {MAX_MARKET_STATE_AGE_SECONDS}s."
    market_timestamp = _safe_text(market.get("updated_at")) or _safe_text(market.get("timestamp"))
    market_age = _age_seconds(market_timestamp)
    if market_age is None or not 0 <= market_age <= MAX_MARKET_STATE_AGE_SECONDS:
        return "Data warming up... The market snapshot is stale or its timestamp is invalid."
    return ""


def _build_planner_summary(
    asset: str,
    capital: float,
    risk_pct: float,
    direction: str,
    entry_price: float,
    atr: float,
    selected_leverage: float | None,
    profile: str = "recommended",
    available_margin_eur: float | None = None,
    usd_per_eur: float = 1.0,
) -> dict:
    profile_config = {
        "conservative": CONSERVATIVE_PROFILE,
        "aggressive": AGGRESSIVE_PROFILE,
        "recommended": RECOMMENDED_PROFILE,
    }.get(str(profile).lower(), RECOMMENDED_PROFILE)
    asset_class = _infer_asset_class(asset)
    instrument_type = "crypto_perpetual" if asset_class == "crypto" else "equity_margin"
    leverage_ceiling = _max_leverage_for_asset(asset)
    available_margin = capital if available_margin_eur is None else max(_safe_float(available_margin_eur), 0.0)
    setup = TradeSetup(
        asset=asset,
        asset_class=asset_class,
        instrument_type=instrument_type,
        exchange_max_leverage=leverage_ceiling,
        direction=direction,
        entry_price=entry_price,
        atr=atr,
        expected_mae=0.0,
        expected_mfe=0.0,
        score=100.0,
        capital_eur=capital,
        risk_per_trade_pct=risk_pct,
        max_position_fraction=1.0 if profile_config.name == "aggressive" else profile_config.max_position_fraction,
        max_portfolio_risk_pct=min(max(risk_pct, 0.06), 1.0),
        atr_buffer_multiple=profile_config.atr_buffer_multiple,
        selected_leverage=selected_leverage,
        profile=profile_config.name,
        available_margin_eur=available_margin,
        usd_per_eur=usd_per_eur,
    )
    params = compute_risk_parameters(setup)
    requested_risk_eur = capital * risk_pct
    margin_exceeded = not params.rejected and params.margin_eur > available_margin + 1e-8
    risk_shortfall_eur = max(requested_risk_eur - params.risk_amount_eur, 0.0)
    risk_budget_shortfall = risk_shortfall_eur > max(1e-8, abs(requested_risk_eur) * 1e-6)
    return {
        "rejected": params.rejected,
        "margin_exceeded": margin_exceeded,
        "risk_budget_shortfall": risk_budget_shortfall,
        "risk_shortfall_eur": risk_shortfall_eur,
        "reason": params.rejection_reason or (
            "Required margin exceeds available margin; reduce portfolio risk or choose higher leverage."
            if margin_exceeded else ""
        ),
        "available_margin_eur": available_margin,
        "max_allowed_leverage": params.max_allowed_leverage,
        "selected_leverage": params.leverage,
        "position_size_eur": params.position_size_eur,
        "quantity": params.quantity,
        "margin_eur": params.margin_eur,
        "risk_amount_eur": params.risk_amount_eur,
        "requested_risk_eur": requested_risk_eur,
        "stop_loss_price": params.stop_loss_price,
        "take_profit_1_price": params.take_profit_1_price,
        "take_profit_2_price": params.take_profit_2_price,
        "trailing_stop_price": params.trailing_stop_price,
        "trailing_stop_trigger_pct": params.trailing_stop_trigger_pct,
        "stop_distance_pct": params.stop_distance_pct,
        "risk_reward_ratio": params.risk_reward_ratio,
    }


def _model_vote(label: str, item: dict | None) -> dict:
    if not item:
        return {"label": label, "direction": "-", "score": None, "raw": {}}
    direction = _safe_text(item.get("direction"), "-").upper()
    component_status = _safe_text(item.get("forecast_status")).upper()
    if component_status in {"UNAVAILABLE", "MODEL_UNAVAILABLE", "STALE"}:
        direction = "-"
    if label != "Rule-Strategy":
        component_age = _age_seconds(item.get("forecast_timestamp"))
        if (
            _safe_text(item.get("freshness")).upper() != "FRESH"
            or component_age is None
            or not 0 <= component_age <= MAX_FORECAST_AGE_SECONDS
        ):
            direction = "-"
    if label == "Chronos-2" and not _valid_chronos_quantiles(item):
        direction = "-"
    if direction not in {"LONG", "SHORT", "NEUTRAL"}:
        direction = "-"
    score = _optional_float(item.get("score"))
    if score is None:
        score = _optional_float(item.get("opportunity_score"))
    if label == "Chronos-2" and (score is None or score < MIN_CHRONOS_SIGNAL_SCORE):
        direction = "-"
    return {"label": label, "direction": direction, "score": score, "raw": item}


def _valid_chronos_quantiles(item: dict | None) -> bool:
    if not isinstance(item, dict):
        return False
    aliases = (
        ("p10_price", "forecast_low", "p10", "lower_quantile"),
        ("p50_price", "forecast_median", "p50", "median_quantile"),
        ("p90_price", "forecast_high", "p90", "upper_quantile"),
    )
    values = []
    for names in aliases:
        value = next((item.get(name) for name in names if item.get(name) is not None), None)
        numeric = _optional_float(value)
        if numeric is None:
            return False
        values.append(numeric)
    return all(value > 0 for value in values) and values[0] <= values[1] <= values[2]


def _forecast_votes(forecast: pd.Series | dict | None) -> dict:
    empty = _model_vote("-", None)
    if forecast is None:
        return {"model_1": empty, "model_2": empty}
    details = _json_load(forecast.get("individual_forecasts_json"), [])
    if isinstance(details, dict):
        details = [details]
    if not isinstance(details, list):
        details = []

    chronos = None
    rule = None
    direct = None
    for item in details:
        if not isinstance(item, dict):
            continue
        source = str(item.get("forecast_source") or "").upper()
        model = str(item.get("model") or "").upper()
        if "CHRONOS" in source or "CHRONOS" in model:
            chronos = item
        elif "RULE_STRATEGY" in source or "RULE_STRATEGY" in model:
            rule = item
        elif source == "DIRECT_MODEL" or "TCN" in model:
            direct = item

    asset = _safe_text(forecast.get("asset"))
    asset_class = _safe_text(forecast.get("asset_class")).lower() or _infer_asset_class(asset)
    if asset_class == "equity":
        model_1 = _model_vote("Chronos-2", chronos) if chronos is not None else empty
        model_2 = _model_vote("Rule-Strategy", rule) if rule is not None else empty
        if chronos is None and rule is None and direct is not None:
            model_name = str(direct.get("model") or "Equity model").replace("_", " ")
            model_1 = _model_vote(model_name, direct)
        return {"model_1": model_1, "model_2": model_2}

    model_1 = _model_vote("TCN", direct) if direct is not None else empty
    model_2 = _model_vote("Chronos-2", chronos) if chronos is not None else empty
    return {"model_1": model_1, "model_2": model_2}


def _agreement_label(votes: dict) -> str:
    first = votes["model_1"]["direction"]
    second = votes["model_2"]["direction"]
    if first not in {"LONG", "SHORT"} or second not in {"LONG", "SHORT"}:
        return "Neutral"
    return "Ja" if first == second else "Nein"


def _signal_cell(vote: dict) -> str:
    direction = vote["direction"]
    if direction == "-":
        return "N/A"
    score = vote["score"]
    score_text = f" · {score:.0f}" if score is not None else ""
    return f"{vote['label']}: {direction}{score_text}"


def _normalize_trend(value) -> str | None:
    normalized = _safe_text(value).upper()
    if normalized in {"UP", "UPTREND", "BULLISH", "LONG"}:
        return "UP"
    if normalized in {"DOWN", "DOWNTREND", "BEARISH", "SHORT"}:
        return "DOWN"
    if normalized in {"FLAT", "NEUTRAL", "SIDEWAYS"}:
        return "FLAT"
    return None


def _htf_trend(market: pd.Series | None, forecast: pd.Series | None) -> str:
    sources = [source for source in (market, forecast) if source is not None]
    for source in sources:
        for key in ("htf_trend", "trend_4h", "higher_timeframe_trend"):
            trend = _normalize_trend(source.get(key))
            if trend is not None:
                return trend
    for source in sources:
        for key in ("htf_trend_strength", "trend_strength_4h"):
            strength = _optional_float(source.get(key))
            if strength is not None:
                return "UP" if strength > 0 else "DOWN" if strength < 0 else "FLAT"
    for item in _json_load(forecast.get("individual_forecasts_json"), []) if forecast is not None else []:
        if isinstance(item, dict):
            for key in ("htf_trend", "trend_4h", "higher_timeframe_trend"):
                trend = _normalize_trend(item.get(key))
                if trend is not None:
                    return trend
    return "N/A"


def _database_setup_summary(
    asset: str,
    horizon: str,
    profile: str,
    capital: float,
    risk_pct: float,
    selected_leverage: float | None,
    available_margin_eur: float,
) -> tuple[dict | None, str]:
    if capital <= 0 or available_margin_eur <= 0:
        return None, "Data warming up... A valid portfolio equity and free-margin snapshot are required before sizing."
    runtime_df = load_table("runtime_status")
    if runtime_df.empty or not _has_columns(runtime_df, "status"):
        return None, "Data warming up... The live daemon has not published a runtime-ready state."
    runtime_status = _safe_text(runtime_df.iloc[0].get("status")).upper()
    if runtime_status not in {"READY", "PARTIAL_DEGRADATION"}:
        return None, f"Data warming up... Live runtime status is {runtime_status or 'UNKNOWN'}."
    market = _market_snapshot(asset)
    market_issue = _market_state_issue(asset, market)
    if market_issue:
        return None, market_issue

    forecast = _load_assistant_forecast_row(asset, horizon)
    if forecast is None:
        return None, "Data warming up... No forecast is stored for this asset and horizon yet."
    votes = _forecast_votes(forecast)
    if not any(
        vote["label"] == "Chronos-2" and _valid_chronos_quantiles(vote["raw"])
        for vote in votes.values()
    ):
        return None, "No valid setup: fresh Chronos-2 P10/P50/P90 quantiles are missing or invalid."
    if _agreement_label(votes) != "Ja":
        chronos_vote = next((vote for vote in votes.values() if vote["label"] == "Chronos-2"), None)
        if chronos_vote is not None and chronos_vote["score"] is not None and chronos_vote["score"] < MIN_CHRONOS_SIGNAL_SCORE:
            return None, f"No valid setup: Chronos-2 is neutral/too weak (score < {MIN_CHRONOS_SIGNAL_SCORE:.0f})."
        return None, "No valid setup: the two model directions do not agree or one model is unavailable."
    if _safe_text(forecast.get("forecast_status")).upper() != "MODEL_AGREEMENT":
        return None, "No valid setup: the stored ensemble status is not MODEL_AGREEMENT."
    if not _safe_bool(forecast.get("usable_for_decision"), False):
        return None, "No valid setup: this forecast is not marked usable for a decision."
    if _safe_text(forecast.get("freshness")).upper() != "FRESH":
        return None, "Data warming up... The forecast is stale or its freshness is unknown."
    forecast_age = _age_seconds(forecast.get("forecast_timestamp"))
    if forecast_age is None or not 0 <= forecast_age <= MAX_FORECAST_AGE_SECONDS:
        return None, "Data warming up... The persisted forecast timestamp is stale, invalid, or in the future."

    entry_price = _optional_float(market.get("last_price"))
    atr = _optional_float(market.get("atr"))
    if entry_price is None or entry_price <= 0 or atr is None or atr <= 0:
        return None, "Data warming up... A finite live price and positive ATR are required before sizing."
    direction = _safe_text(forecast.get("direction")).upper()
    if direction not in {"LONG", "SHORT"}:
        return None, "No valid setup: the agreed direction is not directional."
    try:
        horizon_hours = _horizon_hours(horizon)
    except ValueError as exc:
        return None, str(exc)
    fx_context = _usd_per_eur_context()
    if fx_context is None:
        return None, (
            "FX_UNAVAILABLE: Kein EUR/USD-Kurs gespeichert. Live-Daemon starten/neustarten; "
            "nach dem ersten erfolgreichen Abruf wird das Sizing freigeschaltet."
        )
    horizon_volatility_scale = math.sqrt(horizon_hours)
    sizing_atr = atr * horizon_volatility_scale

    summary = _build_planner_summary(
        asset=asset,
        capital=capital,
        risk_pct=risk_pct,
        direction=direction,
        entry_price=entry_price,
        atr=sizing_atr,
        selected_leverage=selected_leverage,
        profile=profile,
        available_margin_eur=available_margin_eur,
        usd_per_eur=fx_context["usd_per_eur"],
    )
    model_1 = votes["model_1"]
    excursions = None
    if model_1["label"] == "TCN":
        raw_mfe = _optional_float(model_1["raw"].get("expected_mfe"))
        raw_mae = _optional_float(model_1["raw"].get("expected_mae"))
        if raw_mfe is not None and raw_mae is not None:
            excursions = _trade_direction_excursions(direction, raw_mfe, raw_mae)
    return {
        "forecast": forecast,
        "market": market,
        "votes": votes,
        "summary": summary,
        "entry_price": entry_price,
        "atr": atr,
        "sizing_atr": sizing_atr,
        "horizon_volatility_scale": horizon_volatility_scale,
        "direction": direction,
        "excursions": excursions,
        "fx_context": fx_context,
    }, ""


def _build_position_management_summary(
    asset: str,
    capital: float,
    direction: str,
    entry_price: float,
    allocated_margin_eur: float,
    selected_leverage: float,
) -> tuple[dict | None, str]:
    if capital <= 0:
        return None, "Data warming up... A positive global trading-capital value is required."
    if direction not in {"LONG", "SHORT"}:
        return None, "Choose LONG or SHORT for the existing trade."
    if not math.isfinite(entry_price) or entry_price <= 0:
        return None, "Enter a positive trade entry price."
    if not math.isfinite(allocated_margin_eur) or not 0 < allocated_margin_eur <= capital:
        return None, "Allocated trade margin must be positive and no greater than global trading capital."
    tp1_close_pct = POSITION_TP1_CLOSE_PCT
    tp2_close_pct = POSITION_TP2_CLOSE_PCT
    leverage_cap = _max_leverage_for_asset(asset)
    if not math.isfinite(selected_leverage) or not 1 <= selected_leverage <= leverage_cap:
        return None, f"Leverage must be between 1x and the {leverage_cap:.0f}x asset-class limit."

    horizon, forecast, consensus_score, horizon_reason = _select_position_model_context(asset)
    fx_context = _usd_per_eur_context()
    if fx_context is None:
        return None, (
            "FX_UNAVAILABLE: Kein EUR/USD-Kurs gespeichert. Live-Daemon starten/neustarten; "
            "nach dem ersten erfolgreichen Abruf wird das Sizing freigeschaltet."
        )
    usd_per_eur = fx_context["usd_per_eur"]
    market = _market_snapshot(asset)
    market_issue = _market_state_issue(asset, market) if market is not None else "No market/ATR context is stored."
    market_closed = bool(market is not None and _infer_asset_class(asset) == "equity" and not _safe_bool(market.get("market_open"), False))
    if market is None:
        return None, "Data warming up... A current 1h ATR snapshot is required to derive a volatility stop."
    if market_issue and not market_closed:
        return None, market_issue
    entry_atr = _optional_float(market.get("atr")) if market is not None else None
    current_price = _optional_float(market.get("last_price")) if market is not None else None
    if entry_atr is None or entry_atr <= 0 or current_price is None or current_price <= 0:
        return None, "Data warming up... A positive stored 1h ATR and basis price are required."

    # ATR is stored for the underlying at 1h resolution. Scale volatility by
    # sqrt(horizon), then express the resulting return against the user's own
    # leveraged-product entry price; never mix the underlying's absolute price.
    horizon_hours = _horizon_hours(horizon)
    underlying_stop_fraction = (
        entry_atr / current_price
        * math.sqrt(horizon_hours)
        * POSITION_STOP_ATR_MULTIPLE
    )
    stop_fraction = underlying_stop_fraction
    if not math.isfinite(stop_fraction) or not 0 < stop_fraction < 0.40:
        return None, "The automatic ATR/horizon stop is too wide for valid positive TP levels; no levels were generated."
    requested_notional_usd = allocated_margin_eur * usd_per_eur * selected_leverage
    requested_notional_eur = requested_notional_usd / usd_per_eur
    requested_stop_loss_eur = requested_notional_usd * stop_fraction / usd_per_eur
    if requested_stop_loss_eur > capital:
        return None, (
            f"This margin × leverage × stop combination risks €{requested_stop_loss_eur:,.2f}, "
            f"more than the full portfolio (€{capital:,.2f}). Reduce margin, leverage, or stop distance."
        )

    asset_class = _infer_asset_class(asset)
    instrument_type = "crypto_perpetual" if asset_class == "crypto" else "equity_margin"
    try:
        params = compute_risk_parameters(TradeSetup(
            asset=asset,
            asset_class=asset_class,
            instrument_type=instrument_type,
            exchange_max_leverage=leverage_cap,
            direction=direction,
            entry_price=entry_price,
            atr=entry_price * stop_fraction,
            expected_mae=0.0,
            expected_mfe=0.0,
            score=100.0,
            capital_eur=capital,
            risk_per_trade_pct=requested_stop_loss_eur / capital,
            max_position_fraction=1.0,
            max_portfolio_risk_pct=1.0,
            atr_buffer_multiple=1.0,
            selected_leverage=selected_leverage,
            profile="manual_position",
            available_margin_eur=allocated_margin_eur,
            usd_per_eur=usd_per_eur,
        ))
    except (TypeError, ValueError, OverflowError) as exc:
        return None, f"Could not calculate the existing-position plan: {exc}"
    if params.rejected:
        return None, params.rejection_reason or "Risk engine rejected this existing-position plan."

    votes = _forecast_votes(forecast)
    agreement = (
        forecast is not None
        and _safe_text(forecast.get("forecast_status")).upper() == "MODEL_AGREEMENT"
        and _safe_bool(forecast.get("usable_for_decision"), False)
        and _agreement_label(votes) == "Ja"
    )
    model_direction = _safe_text(forecast.get("direction")).upper() if agreement else None
    forecast_timestamp = _safe_text(forecast.get("forecast_timestamp")) if forecast is not None else ""
    forecast_age = _age_seconds(forecast_timestamp) if forecast_timestamp else None
    agreement = bool(agreement and forecast_age is not None and 0 <= forecast_age <= MAX_FORECAST_AGE_SECONDS)
    if not agreement:
        model_direction = None

    summary = {
        "stop_loss_price": params.stop_loss_price,
        "take_profit_1_price": params.take_profit_1_price,
        "take_profit_2_price": params.take_profit_2_price,
        "trailing_stop_activation_price": params.take_profit_1_price,
        "trailing_stop_price": (
            entry_price * (1.0 + 0.5 * stop_fraction)
            if direction == "LONG"
            else entry_price * (1.0 - 0.5 * stop_fraction)
        ),
        "trailing_stop_trigger_pct": POSITION_TP1_R * stop_fraction,
        "trailing_stop_distance_pct": stop_fraction,
        "stop_distance_pct": params.stop_distance_pct,
        "risk_reward_ratio": params.risk_reward_ratio,
        "position_size_eur": params.position_size_eur,
        "quantity": params.quantity,
        "margin_eur": params.margin_eur,
        "selected_leverage": params.leverage,
        "risk_amount_eur": params.risk_amount_eur,
        "requested_risk_eur": requested_stop_loss_eur,
        "maximum_allowed_leverage": params.max_allowed_leverage,
        "tp1_close_pct": tp1_close_pct,
        "tp2_close_pct": tp2_close_pct,
        "trailing_close_pct": 100.0 - tp1_close_pct - tp2_close_pct,
        "tp1_estimated_profit_eur": requested_notional_eur * stop_fraction * POSITION_TP1_R * tp1_close_pct / 100.0,
        "tp2_estimated_profit_eur": requested_notional_eur * stop_fraction * POSITION_TP2_R * tp2_close_pct / 100.0,
    }
    actual_stop_risk_eur = params.risk_amount_eur
    return {
        "asset": asset,
        "horizon": horizon,
        "horizon_hours": horizon_hours,
        "horizon_reason": horizon_reason,
        "consensus_score": consensus_score,
        "direction": direction,
        "entry_price": entry_price,
        "current_price": current_price,
        "atr": entry_atr,
        "underlying_stop_pct": underlying_stop_fraction * 100.0,
        "product_stop_pct": stop_fraction * 100.0,
        "allocated_margin_eur": allocated_margin_eur,
        "requested_notional_eur": requested_notional_eur,
        "fx_context": fx_context,
        "market": market,
        "market_closed": market_closed,
        "market_issue": market_issue,
        "summary": summary,
        "actual_stop_risk_eur": actual_stop_risk_eur,
        "model_direction": model_direction,
        "model_agreement": agreement,
        "model_status": _safe_text(forecast.get("forecast_status")) if forecast is not None else "UNAVAILABLE",
    }, ""


def _realtime_horizons(asset: str) -> tuple[str, ...]:
    return ("4h", "8h") if _infer_asset_class(asset) == "crypto" else ("1h", "4h", "8h", "12h")


def _horizon_hours(horizon: str) -> int:
    normalized = str(horizon).strip().lower()
    if normalized.endswith("h"):
        return int(float(normalized[:-1]))
    if normalized.endswith("d"):
        return int(float(normalized[:-1]) * 24)
    raise ValueError(f"Unsupported forecast horizon: {horizon!r}")


def _select_realtime_setup(asset: str, profile: str, capital: float) -> tuple[dict | None, str]:
    risk_pct = _risk_profile_risk_fraction(profile)
    candidates = []
    failures = []
    for horizon in _realtime_horizons(asset):
        payload, error = _database_setup_summary(
            asset=asset,
            horizon=horizon,
            profile=profile,
            capital=capital,
            risk_pct=risk_pct,
            selected_leverage=None,
            available_margin_eur=capital,
        )
        if payload is None:
            failures.append(error)
            continue
        summary = payload["summary"]
        if summary["rejected"] or summary["margin_exceeded"] or summary["risk_budget_shortfall"]:
            failures.append(summary.get("reason") or "Risk budget or available margin is insufficient.")
            continue

        votes = payload["votes"]
        scores = [
            vote["score"] for vote in votes.values()
            if vote["score"] is not None and math.isfinite(vote["score"])
        ]
        if len(scores) != 2:
            failures.append(f"{horizon}: both ensemble component scores are required for ranking.")
            continue
        expected_return = _optional_float(payload["forecast"].get("expected_return"))
        stop_fraction = _optional_float(summary.get("stop_distance_pct"))
        if expected_return is None or stop_fraction is None or stop_fraction <= 0:
            failures.append(f"{horizon}: expected move or ATR stop distance is unavailable.")
            continue
        signed_expected_move = expected_return if payload["direction"] == "LONG" else -expected_return
        expected_r = signed_expected_move / stop_fraction
        if not math.isfinite(expected_r) or expected_r < REALTIME_MIN_EXPECTED_R:
            failures.append(
                f"{horizon}: forecast move is only {expected_r:.2f}R; at least "
                f"{REALTIME_MIN_EXPECTED_R:.1f}R versus the ATR stop is required."
            )
            continue
        htf_trend = _htf_trend(payload["market"], payload["forecast"])
        aligns_with_htf = htf_trend == ("UP" if payload["direction"] == "LONG" else "DOWN")
        rank = (expected_r, min(scores), int(aligns_with_htf), sum(scores) / len(scores), -_horizon_hours(horizon))
        strategy = "TCN + Chronos Consensus" if votes["model_1"]["label"] == "TCN" else "Chronos + Rule-Strategy Trend"
        payload.update({
            "horizon": horizon,
            "strategy_label": strategy,
            "htf_trend": htf_trend,
            "ensemble_score_floor": min(scores),
            "expected_move_pct": signed_expected_move,
            "expected_reward_r": expected_r,
        })
        candidates.append((rank, payload))

    if candidates:
        return max(candidates, key=lambda item: item[0])[1], ""
    if failures:
        if any("MARKET_CLOSED" in failure for failure in failures):
            return None, next(failure for failure in failures if "MARKET_CLOSED" in failure)
        if any("DATA_STALE" in failure for failure in failures):
            return None, next(failure for failure in failures if "DATA_STALE" in failure)
        return None, failures[0]
    return None, "No eligible realtime horizon is available for this asset."


def _format_strategy_memo(profile: str, direction: str, entry_price: float, summary: dict) -> str:
    profile_label = _risk_profile_label(profile)
    direction_label = "Long" if direction == "LONG" else "Short"
    entry = _display_float(entry_price)
    stop = _display_float(summary.get("stop_loss_price"))
    target_1 = _display_float(summary.get("take_profit_1_price"))
    target_2 = _display_float(summary.get("take_profit_2_price"))
    return (
        f"Profil {profile_label}: {direction_label}-Einstieg bei {entry:.6g}. "
        f"Stop-Loss zwingend bei {stop:.6g} absichern. Erster Teilgewinn bei "
        f"{target_1:.6g}. Trailing-Stop beobachten ab {target_2:.6g}."
    )


def _render_fx_status(payload: dict) -> None:
    fx = payload.get("fx_context")
    if not fx:
        return
    age_seconds = fx.get("age_seconds")
    age_label = "Alter unbekannt" if age_seconds is None else f"Alter {age_seconds / 60.0:.0f} Min."
    st.caption(f"EUR/USD: 1 € = {fx['usd_per_eur']:.5f} $ · Kurszeit {fx.get('timestamp') or 'unbekannt'} · {age_label}")
    if not fx.get("fresh"):
        st.warning(
            f"EUR/USD-Abruf ist nicht frisch; Berechnung nutzt den letzten gespeicherten Kurs "
            f"1 € = {fx['usd_per_eur']:.5f} $."
        )


def _render_time_stop(horizon: str) -> None:
    hours = 2 * _horizon_hours(horizon)
    st.info(
        f"Time-Stop: {hours} Stunden "
        "(Trade zum Marktpreis schließen, falls TP1 nicht erreicht wurde)"
    )


def _render_setup_result(
    payload: dict | None,
    error: str,
    profile: str,
    horizon: str,
    capital: float,
    *,
    realtime: bool = False,
) -> None:
    if payload is None:
        st.warning(error)
        return

    summary = payload["summary"]
    if summary["rejected"]:
        st.error(summary["reason"] or "Risk engine rejected this setup. No trade memo is available.")
        return
    if summary["margin_exceeded"] or summary["risk_budget_shortfall"]:
        issues = []
        if summary["margin_exceeded"]:
            issues.append("Required margin exceeds the entered trading capital.")
        if summary["risk_budget_shortfall"]:
            issues.append(f"Risk budget shortfall: €{summary['risk_shortfall_eur']:,.2f}.")
        st.error("No recommended trade: " + " ".join(issues))
        st.caption("Sizing diagnostics only; trade levels are blocked until the full profile risk fits.")
        diagnostic_cols = st.columns(4)
        diagnostic_cols[0].metric("Calculated position", f"€{summary['position_size_eur']:,.2f}")
        diagnostic_cols[1].metric("Effective risk", f"€{summary['risk_amount_eur']:,.2f}")
        diagnostic_cols[2].metric("Required margin", f"€{summary['margin_eur']:,.2f}")
        diagnostic_cols[3].metric("Available capital", f"€{capital:,.2f}")
        return

    _render_fx_status(payload)
    if realtime:
        quote_currency = _quote_currency(payload["market"].get("asset", ""))
        st.success(
            f"Vorgeschlagene Strategie {payload['strategy_label']} | Horizont: {horizon} | "
            f"Entry: {payload['entry_price']:.6g} {quote_currency} | "
            f"Stop-Loss: {summary['stop_loss_price']:.6g} {quote_currency} | "
            f"Take-Profit: {summary['take_profit_1_price']:.6g} / {summary['take_profit_2_price']:.6g} {quote_currency} | "
            f"Empfohlener Hebel: {summary['selected_leverage']:.2f}x | "
            f"Positionsgröße: €{summary['position_size_eur']:,.2f} EUR-Notional | "
            f"Forecast: +{payload['expected_move_pct']:.2%} ≈ {payload['expected_reward_r']:.2f}R"
        )
        st.caption(
            f"Auswahlscore-Untergrenze {payload['ensemble_score_floor']:.1f} · "
            f"Forecast-Move/Stop-Risiko {payload['expected_reward_r']:.2f}R · "
            f"HTF-Trend {payload['htf_trend']} · Hebel automatisch aus Stop und Margin berechnet."
        )
    else:
        st.markdown("### Trading Memo")
        st.info(_format_strategy_memo(profile, payload["direction"], payload["entry_price"], summary))
    setup_cols = st.columns(4)
    setup_cols[0].metric("Direction", payload["direction"])
    setup_cols[1].metric("Horizont", horizon)
    setup_cols[2].metric("Entry · DB-Kurs", f"{payload['entry_price']:.6g}")
    setup_cols[3].metric(
        "Sizing-ATR",
        f"{payload.get('sizing_atr', payload['atr']):.6g}",
        help="Für den gewählten Mehrstunden-Horizont aus der 1h-ATR mit √Zeit skaliert.",
    )
    risk_cols = st.columns(5)
    risk_cols[0].metric("Absolute Positionsgröße", f"€{summary['position_size_eur']:,.2f}")
    risk_cols[1].metric("Berechnete Menge", f"{summary['quantity']:.8g}")
    risk_cols[2].metric("Effektives Risiko", f"€{summary['risk_amount_eur']:,.2f}")
    risk_cols[3].metric("Benötigte Margin", f"€{summary['margin_eur']:,.2f}")
    risk_cols[4].metric("Gewählter Hebel", f"{summary['selected_leverage']:.2f}x")
    level_cols = st.columns(4)
    level_cols[0].metric("Stop-Loss", f"{summary['stop_loss_price']:.6g}")
    level_cols[1].metric("Take-Profit 1", f"{summary['take_profit_1_price']:.6g}")
    level_cols[2].metric("Take-Profit 2", f"{summary['take_profit_2_price']:.6g}")
    level_cols[3].metric("Eingegebenes Trading-Kapital", f"€{capital:,.2f}")
    st.caption(
        f"{_risk_profile_label(profile)} · Risikobudget €{summary['requested_risk_eur']:,.2f} · "
        f"Stop-Distanz {summary['stop_distance_pct']:.2%} · "
        f"R:R {summary['risk_reward_ratio']:.1f}:1"
    )
    if payload["excursions"] is not None:
        favorable, adverse = payload["excursions"]
        st.caption(f"TCN-Pfadschätzung · günstig +{favorable:.2%} · ungünstig {adverse:.2%}")
    _render_time_stop(horizon)


def _render_position_management(payload: dict | None, error: str, capital: float) -> None:
    if payload is None:
        st.warning(error)
        return
    quote_currency = _quote_currency(payload["asset"])
    summary = payload["summary"]
    if payload["market_closed"]:
        st.warning(
            f"MARKET_CLOSED: Volatilitäts-Stop nutzt die letzte gespeicherte ATR als Referenz; "
            f"dein Produkteinstieg in {quote_currency} bleibt die Preisbasis."
        )
    elif payload["market_issue"]:
        st.warning(f"Markt-/ATR-Kontext nicht frisch: {payload['market_issue']}")
    if payload["model_direction"] is None:
        st.info(payload["horizon_reason"])
    elif payload["model_direction"] != payload["direction"]:
        st.warning(f"Modell-Consensus {payload['model_direction']} widerspricht deiner bestehenden {payload['direction']}-Position; Level bleiben auf deinen Trade berechnet.")
    else:
        st.caption(f"Frischer Consensus {payload['model_direction']} wählt automatisch den {payload['horizon']}-Volatilitätshorizont.")
    _render_fx_status(payload)

    st.success(
        f"{payload['direction']} ab {payload['entry_price']:.6g} {quote_currency} · "
        f"automatischer SL {summary['stop_loss_price']:.6g} ({payload['product_stop_pct']:.2f}%) · "
        f"TP1 {summary['take_profit_1_price']:.6g} (+{summary['stop_distance_pct'] * POSITION_TP1_R:.2%}): "
        f"{summary['tp1_close_pct']:.0f}% verkaufen · "
        f"TP2 {summary['take_profit_2_price']:.6g} (+{summary['stop_distance_pct'] * POSITION_TP2_R:.2%}): "
        f"{summary['tp2_close_pct']:.0f}% verkaufen"
    )
    level_cols = st.columns(5)
    level_cols[0].metric("Einstieg", f"{payload['entry_price']:.6g} {quote_currency}")
    level_cols[1].metric("Stop-Loss", f"{summary['stop_loss_price']:.6g} {quote_currency}")
    level_cols[2].metric("Take-Profit 1", f"{summary['take_profit_1_price']:.6g} {quote_currency}")
    level_cols[3].metric("Take-Profit 2", f"{summary['take_profit_2_price']:.6g} {quote_currency}")
    level_cols[4].metric(
        "Trail startet ab TP1",
        f"{summary['trailing_stop_activation_price']:.6g} {quote_currency}",
    )
    risk_cols = st.columns(5)
    risk_cols[0].metric("Eingesetzte Margin", f"€{payload['allocated_margin_eur']:,.2f}")
    risk_cols[1].metric("Impliziertes Notional", f"€{summary['position_size_eur']:,.2f}")
    risk_cols[2].metric("Berechnete Menge", f"{summary['quantity']:.8g}")
    risk_cols[3].metric("Verlust am SL", f"€{payload['actual_stop_risk_eur']:,.2f}")
    risk_cols[4].metric("Anteil Gesamtkapital", f"{payload['actual_stop_risk_eur'] / capital:.1%}")
    if payload["actual_stop_risk_eur"] >= capital * 0.8:
        st.error("Extrem hohes Risiko: Ein Stop-Aus kann den Großteil des globalen Kapitals aufbrauchen.")
    elif payload["actual_stop_risk_eur"] >= capital * 0.5:
        st.warning("Sehr hohes Risiko: Ein Stop-Aus führt rechnerisch zu mindestens 50% Verlust des globalen Kapitals.")

    sale_cols = st.columns(3)
    sale_cols[0].metric("Bei TP1 verkaufen", f"{summary['tp1_close_pct']:.0f}% der Position")
    sale_cols[1].metric("Bei TP2 verkaufen", f"{summary['tp2_close_pct']:.0f}% der Position")
    sale_cols[2].metric("Im Trailing lassen", f"{summary['trailing_close_pct']:.0f}% der Position")
    profit_cols = st.columns(2)
    profit_cols[0].metric("Geschätzter TP1-Gewinn", f"€{summary['tp1_estimated_profit_eur']:,.2f}")
    profit_cols[1].metric("Geschätzter TP2-Gewinn", f"€{summary['tp2_estimated_profit_eur']:,.2f}")
    st.caption(
        f"Stop automatisch aus Basiswert-1h-ATR {payload['atr']:.6g} × 1.5 × √{payload['horizon_hours']}h "
        f"= {payload['product_stop_pct']:.2f}% vom Produkteinstieg · "
        f"TP1 +{POSITION_TP1_R:.1f}R · TP2 +{POSITION_TP2_R:.1f}R · "
        f"Trailing startet bei TP1, folgt mit 1R Abstand; initialer Trail-Level "
        f"{summary['trailing_stop_price']:.6g} {quote_currency}. "
        f"Hebel {summary['selected_leverage']:.2f}x (stop-sicheres Maximum {summary['maximum_allowed_leverage']:.2f}x). "
        "√Zeit ist eine Volatilitätsnäherung. Lineares Modell; Produkt-Delta, täglicher Reset, Knock-out, "
        "Finanzierung, Gebühren und FX können abweichen."
    )
    _render_time_stop(payload["horizon"])


@st.fragment(run_every=f"{REFRESH_SECONDS}s")
def _render_realtime_strategy(capital_eur: float) -> None:
    st.subheader("Realtime Strategy")
    asset_col, profile_col = st.columns([1.2, 1.0])
    asset = asset_col.selectbox("Asset", ASSETS, key="realtime_asset")
    profile_label = profile_col.selectbox(
        "Risikoprofil", ["Konservativ", "Recommended", "Aggressiv"], key="realtime_profile",
    )
    profile = {
        "Konservativ": "conservative",
        "Recommended": "recommended",
        "Aggressiv": "aggressive",
    }[profile_label]
    risk_pct = _risk_profile_risk_fraction(profile)
    st.caption(
        f"Kapitalbasis €{capital_eur:,.2f} · Risiko am Stop {risk_pct:.1%} "
        f"(€{capital_eur * risk_pct:,.2f}) · horizontübergreifende Suche alle {REFRESH_SECONDS}s."
    )
    st.caption("Der Zeitrahmen wird aus frischen, übereinstimmenden Modellprognosen ausgewählt; keine Order wird ausgeführt.")
    st.caption(
        f"Horizont-Auswahl: höchste erwartete Bewegung relativ zum ATR-Stop; "
        f"Kandidaten unter {REALTIME_MIN_EXPECTED_R:.1f}R werden verworfen. Chronos-Endpreis-Quantile werden nicht als MAE/Stop missbraucht."
    )
    if profile == "aggressive":
        st.warning(
            f"Aggressiv: bis zu 25% des eingegebenen Kapitals (€{capital_eur * risk_pct:,.2f}) stehen am Stop auf dem Spiel; "
            "Gaps und Slippage können den Verlust erhöhen."
        )

    payload, error = _select_realtime_setup(asset, profile, capital_eur)
    if payload is not None:
        snapshot_time = _safe_text(payload["market"].get("updated_at")) or _safe_text(payload["market"].get("timestamp"))
        st.caption(f"Letzter Markt-Snapshot: {snapshot_time or 'unbekannt'} · read-only, keine Orderausführung.")
    _render_setup_result(
        payload, error, profile, payload.get("horizon", "auto") if payload else "auto", capital_eur,
        realtime=True,
    )
    st.warning(KO_CERTIFICATE_WARNING)


def _json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if value is None:
        return None
    try:
        missing = pd.isna(value)
        if not hasattr(missing, "__len__") and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return _json_compatible(value.item())
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _collect_quantile_rows(value, path: str = "") -> list[dict]:
    aliases = {
        "p10": "P10", "p_10": "P10", "p10_price": "P10", "q10": "P10", "lower_quantile": "P10", "forecast_low": "P10",
        "p50": "P50", "p_50": "P50", "p50_price": "P50", "q50": "P50", "median_quantile": "P50", "forecast_median": "P50",
        "p90": "P90", "p_90": "P90", "p90_price": "P90", "q90": "P90", "upper_quantile": "P90", "forecast_high": "P90",
    }
    rows = []
    if isinstance(value, dict):
        row = {"Source": path or "forecast"}
        for key, item in value.items():
            label = aliases.get(str(key).strip().lower().replace("-", "_"))
            if label is not None:
                row[label] = _json_compatible(item)
        if any(row.get(key) is not None for key in ("P10", "P50", "P90")):
            rows.append(row)
        for key, item in value.items():
            if str(key).strip().lower().replace("-", "_") not in aliases:
                child_path = f"{path}.{key}" if path else str(key)
                rows.extend(_collect_quantile_rows(item, child_path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_collect_quantile_rows(item, f"{path}[{index}]"))
    return rows


ASSETS = ("BTC/USDT", "ETH/USDT", "QQQ", "SPY")
runtime_df = load_table("runtime_status")
market_df = load_table("market_state")
forecast_df = load_table("assistant_forecasts")
signals_df = load_table("signals")
strategy_df = load_table("strategy_state")
context_df = load_table("context_state")
runtime_status, runtime_text = _runtime_health(runtime_df, forecast_df)

if _DB_INIT_ERROR:
    st.warning("Data warming up... SQLite is temporarily unavailable; views are fail-closed.")

capital_default = _available_margin()
if capital_default <= 0:
    capital_default = _account_equity()
if capital_default <= 0:
    capital_default = 500.0
capital_col, capital_note_col = st.columns([1, 3])
global_capital_eur = capital_col.number_input(
    "Verfügbares Trading-Kapital (€)",
    min_value=1.0,
    max_value=1_000_000_000.0,
    value=float(capital_default),
    step=50.0,
    key="global_trading_capital_eur",
)
capital_note_col.caption(
    "Gemeinsame Rechenbasis für Risiko und Margin in beiden Strategie-Tabs; manuelle Eingabe, keine Broker-Kontoverbindung."
)

strategy_tab, realtime_tab, radar_tab, raw_tab = st.tabs(
    ["Strategy Assistant", "Realtime Strategy", "Radar", "Raw Data / Deep Dive"]
)

with strategy_tab:
    st.subheader("Strategy Assistant")
    asset_col, direction_col, leverage_col = st.columns([1.2, 1.0, 1.0])
    asset = asset_col.selectbox("Asset", ASSETS, key="assistant_asset")
    direction = direction_col.selectbox("Bestehender Trade", ["LONG", "SHORT"], key="assistant_trade_direction")
    capital = float(global_capital_eur)
    market = _market_snapshot(asset)
    quote_currency = _quote_currency(asset)
    market_entry = _optional_float(market.get("last_price")) if market is not None else None
    entry_default = market_entry if market_entry is not None and market_entry > 0 else 100.0
    leverage_cap = _max_leverage_for_asset(asset)
    leverage = leverage_col.number_input(
        f"Hebel (max. {leverage_cap:.0f}x)",
        min_value=1.0,
        max_value=leverage_cap,
        value=min(5.0, leverage_cap),
        step=1.0,
        key=f"assistant_leverage_{asset.replace('/', '_')}",
    )
    entry_col = st.container()
    entry_price = entry_col.number_input(
        f"Dein Produkt-Einstieg ({quote_currency})",
        min_value=0.000001,
        value=float(entry_default),
        step=max(float(entry_default) * 0.001, 0.01),
        key=f"assistant_entry_{asset.replace('/', '_')}",
        help="Den tatsächlichen Preis des gehaltenen Produkts eingeben. Der gespeicherte Basiswertkurs ist bei Hebelprodukten nicht derselbe Preis.",
    )
    st.caption(
        f"Wichtig: Der Einstieg in {quote_currency} muss der Kurs deines Hebelprodukts sein. "
        "Der Datenbankkurs ist nur die Basiswert-Referenz und kann einen deutlich anderen Preis haben."
    )
    margin_col, capital_col = st.columns(2)
    allocated_margin_eur = margin_col.number_input(
        "Für diesen Trade eingesetzte Margin (€)",
        min_value=0.01,
        max_value=capital,
        value=float(capital),
        step=50.0,
        key=f"assistant_margin_{asset.replace('/', '_')}",
        help="Der Standard setzt das gesamte globale Trading-Kapital als Margin an. Für deine reale Position den tatsächlich eingesetzten Betrag eingeben.",
    )
    capital_col.metric("Globales Kapital", f"€{capital:,.2f}")
    st.caption(
        f"Notional-Schätzung €{allocated_margin_eur * leverage:,.2f} · "
        "Horizont, ATR-Stop, TP-Levels und Trailing-Aktivierung/Teilverkäufe werden automatisch berechnet."
    )
    payload, error = _build_position_management_summary(
        asset=asset,
        capital=capital,
        direction=direction,
        entry_price=float(entry_price),
        allocated_margin_eur=float(allocated_margin_eur),
        selected_leverage=float(leverage),
    )
    _render_position_management(payload, error, capital)
    st.warning(KO_CERTIFICATE_WARNING)

with realtime_tab:
    _render_realtime_strategy(float(global_capital_eur))

with radar_tab:
    st.subheader("Radar")
    if runtime_status == "READY":
        st.success(f"Runtime: READY · {runtime_text}")
    elif runtime_status == "MARKET_CLOSED":
        st.warning(runtime_text)
    elif runtime_status in {"PARTIAL_DEGRADATION", "DEGRADED"}:
        st.warning(f"Runtime: {runtime_status} · {runtime_text}")
    else:
        st.info(f"Runtime: {runtime_status} · {runtime_text}")
    if forecast_df.empty or not _has_columns(forecast_df, "asset", "horizon"):
        st.warning("Data warming up... The persisted forecast matrix is empty or incomplete.")

    forecast_by_key = {}
    if _has_columns(forecast_df, "asset", "horizon"):
        for _, row in forecast_df.iterrows():
            forecast_by_key[(str(row.get("asset") or ""), str(row.get("horizon") or ""))] = row
    market_by_asset = {}
    if _has_columns(market_df, "asset"):
        for _, row in market_df.iterrows():
            market_by_asset[str(row.get("asset") or "")] = row
    for asset_name in ASSETS:
        market = market_by_asset.get(asset_name)
        issue = _market_state_issue(asset_name, market) if market is not None else ""
        if issue:
            st.warning(issue)

    radar_rows = []
    for asset_name in ASSETS:
        market = market_by_asset.get(asset_name)
        for horizon_name in PUBLIC_HORIZONS:
            forecast = forecast_by_key.get((asset_name, horizon_name))
            votes = _forecast_votes(forecast)
            atr = _optional_float(market.get("atr")) if market is not None else None
            radar_rows.append({
                "Asset": asset_name,
                "Horizont": horizon_name,
                "Model 1 Signal": _signal_cell(votes["model_1"]),
                "Model 2 Signal": _signal_cell(votes["model_2"]),
                "Model-Agreement": _agreement_label(votes),
                "Aktuelle ATR": f"{atr:.6g}" if atr is not None and atr > 0 else "N/A",
                "HTF-Trend": _htf_trend(market, forecast),
            })
    radar = pd.DataFrame(radar_rows)
    styled = radar.style.map(
        lambda value: (
            "background-color: #d9f2e6; color: #145a32; font-weight: 700" if value == "Ja"
            else "background-color: #f8dddd; color: #8f1d1d; font-weight: 700" if value == "Nein"
            else "background-color: #eceff1; color: #4b5563"
        ),
        subset=["Model-Agreement"],
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.caption("HTF-Trend wird nur aus persistierten Feldern angezeigt; die UI lädt oder berechnet keine OHLCV-Indikatoren.")

with raw_tab:
    st.subheader("Raw Data / Deep Dive")
    if _DB_INIT_ERROR:
        st.error(f"Database initialization error: {_DB_INIT_ERROR}")
    st.markdown("### Forecast matrix")
    if forecast_df.empty:
        st.warning("Data warming up... No forecast rows are stored yet.")
    else:
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        if _has_columns(forecast_df, "asset", "horizon"):
            raw_options = [
                f"{row.get('asset', 'unknown')} · {row.get('horizon', 'unknown')}"
                for _, row in forecast_df.iterrows()
            ]
            selected_raw = st.selectbox("Raw forecast record", raw_options, key="raw_forecast_record")
            selected_position = raw_options.index(selected_raw)
            raw_record = forecast_df.iloc[selected_position].to_dict()
            individual = _json_load(raw_record.get("individual_forecasts_json"), raw_record.get("individual_forecasts_json"))
            sources = _json_load(raw_record.get("model_sources_json"), raw_record.get("model_sources_json"))
            raw_record["individual_forecasts"] = individual
            raw_record["model_sources"] = sources
            st.json(_json_compatible(raw_record))
            quantile_rows = _collect_quantile_rows(individual)
            if quantile_rows:
                st.dataframe(pd.DataFrame(quantile_rows), use_container_width=True, hide_index=True)
            else:
                st.info("P10/P50/P90 are not present in the persisted forecast payload; they are not inferred from expected return.")

    raw_tables = (
        ("Market state / backend indicators", market_df),
        ("Signals / model outputs", signals_df),
        ("Strategy state / risk profiles", strategy_df),
        ("Context state / raw JSON", context_df),
        ("Runtime status", runtime_df),
        ("Paper portfolio", load_table("paper_portfolio")),
    )
    for table_label, table_frame in raw_tables:
        with st.expander(table_label, expanded=False):
            if table_frame.empty:
                st.info("No rows stored yet.")
            else:
                st.dataframe(table_frame, use_container_width=True, hide_index=True)

st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} · cache {REFRESH_SECONDS}s")
