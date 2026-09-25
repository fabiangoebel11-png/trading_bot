"""Streamlit-Dashboard: reines Read-Only-Frontend, das ausschliesslich die
SQLite-DB liest (``trading_state.db``) -- laedt selbst NIE Marktdaten und
sendet nie Orders. Kann unabhaengig vom Daemon/Broker gestartet/gestoppt
werden; zeigt einfach "keine Daten" an, wenn die anderen Prozesse noch nicht
liefen.

Start: ``streamlit run app.py``
"""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import asdict
import hashlib
import json

import pandas as pd
import streamlit as st

import state_db
from decision_engine import analyze_position, calculate_position, scenario_analysis
from decision_pipeline import TradeQuality, analyse_ohlcv, load_cached_ohlcv
from model_integration import infer_model_comparison
from model_schedule import get_task_schedule

st.set_page_config(page_title="Trading Bot Cockpit", layout="wide")
st.title("Trading Decision Assistant")
st.caption("Informations- und Analyseoberfläche. Diese GUI führt keine Trades aus.")
state_db.init_db()

REFRESH_SECONDS = 15


@st.cache_data(ttl=REFRESH_SECONDS)
def load_table(table: str) -> pd.DataFrame:
    with state_db.connect() as conn:
        try:
            return pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception:
            return pd.DataFrame()


def _age_seconds(iso_timestamp: str | None) -> float | None:
    if not iso_timestamp:
        return None
    try:
        ts = datetime.fromisoformat(iso_timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except ValueError:
        return None


@st.cache_data(ttl=REFRESH_SECONDS)
def load_real_analysis(asset: str, timeframe: str, capital: float, horizon: str):
    frame = load_cached_ohlcv(asset, timeframe, "data")
    with state_db.connect() as conn:
        market = conn.execute("SELECT * FROM market_state WHERE asset = ?", (asset,)).fetchone()
        signal = None
        if market is not None:
            model_type = "crypto_sniper" if market["asset_class"] == "crypto" else ("swing" if timeframe in {"4h", "1d"} else "intraday")
            signal = conn.execute("SELECT * FROM signals WHERE asset = ? AND model_type = ?", (asset, model_type)).fetchone()
        signal_payload = dict(signal) if signal is not None else None
        asset_class = market["asset_class"] if market is not None else ("crypto" if "USDT" in asset else "equity")
    data_status = "FRESH" if market is not None and bool(market["feed_healthy"]) and bool(market["freshness_ok"]) else "STALE"
    comparison = None
    if frame is not None:
        comparison = infer_model_comparison(asset, timeframe, frame, horizon=horizon, category="SWING" if timeframe in {"4h", "1d"} else "INTRADAY")
    trade_quality = TradeQuality(
        status="MODEL_UNAVAILABLE",
        model_id="trade_quality_hgb",
        model_version="hgb-trade-quality-v1",
        score=None,
        probability=None,
        label_version="directional-triple-barrier-v1",
        feature_version="trade-entry-v1",
        reason="No promoted Trade-Quality artifact is enabled; observation-only.",
    )
    return analyse_ohlcv(asset, timeframe, frame, signal=signal_payload, forecast=comparison.selected if comparison else None, model_comparison=comparison, trade_quality=trade_quality, capital=capital, asset_class=asset_class, data_status=data_status)


def persist_trade_idea(plan, comparison=None) -> None:
    payload = asdict(plan)
    idea_id = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]
    state_db.save_assistant_trade_idea(state_db.DB_PATH, {
        "idea_id": idea_id,
        "asset": plan.asset,
        "strategy_id": plan.strategy_id,
        "strategy_version": plan.strategy_version,
        "model": plan.model.model_type,
        "model_version": plan.model.model_version,
        "status": plan.setup_status,
        "plan": payload,
        "input_snapshot": {
            "asset": plan.asset,
            "timeframe": plan.timeframe,
            "regime": plan.market_regime,
            "scoring_version": plan.scoring_version,
            "ensemble_version": comparison.ensemble_version if comparison else None,
            "model_agreement": comparison.agreement if comparison else None,
            "tcn": asdict(comparison.tcn) if comparison else None,
            "chronos": asdict(comparison.chronos) if comparison else None,
            "ensemble_score": comparison.ensemble_score if comparison else None,
            "trade_quality_score": plan.trade_quality_score,
            "trade_quality_status": "AVAILABLE" if plan.trade_quality_score is not None else "MODEL_UNAVAILABLE",
        },
        "recommendation": " · ".join(plan.strategy_reason),
    })


tab_radar, tab_ideas, tab_positions, tab_assistant, tab_strategy, tab_portfolio = st.tabs(
    ["Radar", "Trade Ideas", "My Positions", "Trade Assistant", "Strategies", "Paper Simulator"]
)

# =============================================================================
# Ansicht 1: Radar -- Live-Scores aller Assets, Market-State, Signale
# =============================================================================
with tab_radar:
    runtime_df = load_table("runtime_status")
    market_df = load_table("market_state")
    context_df = load_table("context_state")
    signals_df = load_table("signals")
    strategy_df = load_table("strategy_state")
    forecast_df = load_table("assistant_forecasts")

    if runtime_df.empty:
        st.warning("Runtime-Status unbekannt. Der Live-Daemon wurde noch nicht initialisiert.")
    else:
        runtime = runtime_df.iloc[0]
        status = str(runtime["status"])
        if status == "READY":
            st.success(f"Runtime: READY · {runtime['updated_at']}")
        elif status == "INITIALIZING":
            st.info(f"Runtime: INITIALIZING · {runtime['detail']}")
        else:
            st.warning(f"Runtime: {status} · {runtime['detail']}")

    if market_df.empty:
        st.info("Keine Marktdaten. Läuft `live_daemon.py`?")
    else:
        st.subheader("Markt-Status")
        display = market_df.copy()
        display["data_age_s"] = display["updated_at"].apply(_age_seconds)
        display["stale"] = display["data_age_s"].apply(lambda s: s is not None and s > 120)
        st.dataframe(
            display[["asset", "asset_class", "last_price", "market_open", "session_type", "atr_pct", "funding_rate", "feed_healthy", "data_age_s"]]
            .rename(columns={"last_price": "Preis", "market_open": "Offen", "session_type": "Session", "atr_pct": "ATR%", "feed_healthy": "Feed OK", "data_age_s": "Alter (s)"}),
            use_container_width=True,
        )
        if not context_df.empty:
            st.subheader("Kontext-Status")
            st.dataframe(
                context_df[["asset", "timestamp", "status", "updated_at"]],
                use_container_width=True,
            )

        st.subheader("Signale")
        if signals_df.empty:
            st.info("Noch keine Signale berechnet (erster Signal-Refresh-Zyklus läuft noch).")
        else:
            pretty = signals_df.copy()
            pretty["alert_allowed"] = pretty["alert_allowed"].astype(bool)
            st.dataframe(
                pretty[["asset", "model_type", "score", "direction", "expected_mfe", "expected_mae", "expected_duration_days", "alert_allowed", "alert_reason", "timestamp"]],
                use_container_width=True,
            )
            triggered = pretty[pretty["alert_allowed"]]
            if not triggered.empty:
                st.success(f"🔔 {len(triggered)} aktives Alert-Gate: {', '.join(triggered['asset'].unique())}")

    st.subheader("Live Forecasts")
    if forecast_df.empty:
        st.info("Noch keine Forecast-Matrix. Der nächste Signal-Refresh des Live-Daemons erstellt sie.")
    else:
        forecast_df = forecast_df.copy()
        forecast_df["forecast_age_min"] = forecast_df["forecast_timestamp"].apply(_age_seconds).div(60.0)
        forecast_df["forecast_age_min"] = forecast_df["forecast_age_min"].round(1)
        forecast_df["stale_forecast"] = forecast_df["forecast_age_min"].apply(lambda age: age is not None and age > 75)
        order = {horizon: index for index, horizon in enumerate(("1h", "4h", "8h", "12h", "1d", "3d", "5d", "10d", "20d"))}
        forecast_df["_order"] = forecast_df["horizon"].map(order)
        asset = st.selectbox("Asset detail", ["BTC/USDT", "ETH/USDT", "SPY", "QQQ"], key="forecast_asset")
        asset_forecasts = forecast_df[forecast_df["asset"] == asset].sort_values("_order")
        if asset_forecasts.empty:
            st.warning(f"Keine Forecasts für {asset} gespeichert.")
        else:
            st.dataframe(
                asset_forecasts[["horizon", "direction", "opportunity_score", "expected_return", "forecast_source", "forecast_status", "model_sources_json", "combination_method", "combined_confidence", "data_quality", "forecast_age_min", "stale_forecast", "reason"]]
                .rename(columns={"horizon": "Horizont", "direction": "Richtung", "opportunity_score": "Score", "expected_return": "Erwartete Rendite", "forecast_source": "Quelle", "forecast_status": "Status", "model_sources_json": "Modellquellen", "combination_method": "Kombinationsmethode", "combined_confidence": "Model Agreement", "data_quality": "Datenqualität", "forecast_age_min": "Forecast-Alter (min)", "stale_forecast": "STALE", "reason": "Hinweis"}),
                use_container_width=True,
            )

    st.subheader("Aktuelle Decision Summary")
    capital = st.number_input("Analysekapital (€)", min_value=1.0, value=500.0, step=50.0, key="dashboard_capital")
    summary_rows = []
    for asset in ("BTC/USDT", "ETH/USDT", "QQQ", "SPY"):
        for category, timeframe, horizon in (("Intraday", "1h", "4h"), ("Swing", "1d", "5d")):
            result = load_real_analysis(asset, timeframe, capital, horizon)
            plan = result.plan
            schedule = get_task_schedule(asset, timeframe)
            summary_rows.append({
                "Asset": asset, "Category": category, "Timeframe": timeframe, "Data": result.data_status,
                "Inference": schedule.model_inference,
                "Regime": result.regime, "Strategy": plan.strategy_id if plan else "-",
                "Direction": plan.direction if plan else "-", "Rule Score": round(plan.rule_score, 1) if plan else None,
                "Model Score": round(plan.model_score, 1) if plan and plan.model_score is not None else "NOT_AVAILABLE",
                "Model Agreement": result.model_comparison.agreement if result.model_comparison else "NOT_AVAILABLE",
                "Trade Quality": round(result.trade_quality.score, 1) if result.trade_quality and result.trade_quality.score is not None else "NOT_AVAILABLE",
                "Final Score": round(plan.final_score, 1) if plan else None,
                "Status": plan.setup_status if plan else result.data_status,
                "Data Age (s)": round(result.data_age_seconds, 0) if result.data_age_seconds is not None else None,
            })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    st.caption("Rule Score, Model Score und Final Score bleiben getrennt. Fehlende Modelle oder Daten erzeugen keinen erfundenen Score.")

# =============================================================================
# Ansicht 2: Trade Ideas -- deterministic plans from persisted snapshots
# =============================================================================
with tab_ideas:
    st.subheader("Trade Ideas")
    account = st.number_input("Analysekapital (€)", min_value=1.0, value=500.0, step=50.0, key="idea_account")
    asset_filter = st.multiselect("Asset", ["BTC/USDT", "ETH/USDT", "QQQ", "SPY"], default=["BTC/USDT", "ETH/USDT", "QQQ", "SPY"])
    direction_filter = st.multiselect("Direction", ["LONG", "SHORT"], default=["LONG", "SHORT"])
    timeframe_filter = st.selectbox("Timeframe", ["1h", "4h", "1d"])
    horizon_filter = st.selectbox("Forecast horizon", {"1h": ["1h", "4h", "8h", "12h", "24h"], "4h": ["1d", "3d", "5d", "7d", "14d"], "1d": ["1d", "3d", "5d", "10d", "15d"]}[timeframe_filter])
    plans = []
    for asset in asset_filter:
        result = load_real_analysis(asset, timeframe_filter, account, horizon_filter)
        if result.plan is not None and result.plan.direction in direction_filter:
            persist_trade_idea(result.plan, result.model_comparison)
            plans.append((result.plan, result))
        elif result.plan is None:
            st.info(f"{asset} / {timeframe_filter}: {result.data_status} · {result.reason}")
    for plan, result in plans:
        with st.expander(f"{plan.asset} · {plan.timeframe} · {plan.direction} · Qualität {plan.signal_quality:.1f}/100"):
            left, right = st.columns(2)
            left.write(f"Strategie: {plan.strategy_id} v{plan.strategy_version} ({plan.category})")
            left.write(f"Rule Score: {plan.rule_score:.1f} · Model Score: {plan.model_score if plan.model_score is not None else 'NOT_AVAILABLE'} · Final Score: {plan.final_score:.1f}")
            left.write(f"Regime: {plan.market_regime} · Entry: {plan.entry_low:.6f} - {plan.entry_high:.6f}")
            left.write(f"Stop: {plan.stop:.6f} · TP1: {plan.take_profit_1:.6f} · TP2: {plan.take_profit_2:.6f}")
            left.write(f"Trailing: {plan.trailing_method} · Haltedauer: {plan.expected_holding_time}")
            right.write(f"Notional: {plan.notional:.2f} € · Position Size: {plan.position_size:.8f}")
            right.write(f"Risiko / maximaler Verlust: {plan.risk_per_trade:.2f} €")
            right.write(f"Hebel: konservativ {plan.leverage_range[0]:.2f}x · empfohlen {plan.recommended_leverage:.2f}x · aggressiv {plan.leverage_range[2]:.2f}x")
            right.write(f"Nicht empfohlen über: {plan.not_recommended_above:.2f}x · R:R {plan.risk_reward:.2f}")
            right.write(f"Model: {plan.model.status} ({plan.model.model_type})")
            comparison = result.model_comparison
            if comparison is not None:
                right.write(f"TCN: {comparison.tcn.direction or comparison.tcn.status} ({comparison.tcn.model_score if comparison.tcn.model_score is not None else 'NOT_AVAILABLE'})")
                right.write(f"Chronos-2: {comparison.chronos.direction or comparison.chronos.status} ({comparison.chronos.model_score if comparison.chronos.model_score is not None else 'NOT_AVAILABLE'})")
                right.write(f"Model Agreement: {comparison.agreement} · Ensemble: {comparison.ensemble_score if comparison.ensemble_score is not None else 'NOT_ACTIVE'}")
            if result.trade_quality is not None:
                right.write(f"Trade Quality: {result.trade_quality.score if result.trade_quality.score is not None else 'NOT_AVAILABLE'} · {result.trade_quality.status}")
            right.write(f"Data: {result.data_timestamp or '-'} · Age: {result.data_age_seconds:.0f}s · Source: {result.source}")
            st.write("Warum: " + " · ".join(plan.strategy_reason))
            st.write(f"Invalidierung: unter {plan.invalidated_below or '-'} / über {plan.invalidated_above or '-'}")
            st.write("Candidates: " + " · ".join(f"{candidate.strategy_id} {candidate.score:.1f}" for candidate in result.candidates))

# =============================================================================
# Ansicht 3: My Positions -- manual input and read-only analysis
# =============================================================================
with tab_positions:
    position_df = load_table("assistant_positions")
    market_df = load_table("market_state")
    st.subheader("Manuelle Positionen")
    with st.form("manual_position"):
        col1, col2, col3 = st.columns(3)
        position_id = col1.text_input("Positions-ID")
        asset = col1.text_input("Asset", value="BTC/USDT")
        direction = col1.selectbox("Richtung", ["LONG", "SHORT"])
        entry = col2.number_input("Entry", min_value=0.000001, value=100.0)
        quantity = col2.number_input("Menge", min_value=0.000001, value=1.0)
        margin = col2.number_input("Margin (€)", min_value=0.000001, value=100.0)
        stop = col3.number_input("Stop", min_value=0.000001, value=95.0)
        take_profit = col3.number_input("Take Profit", min_value=0.000001, value=110.0)
        leverage = col3.number_input("Hebel", min_value=0.01, value=1.0, step=0.25)
        timeframe = col3.selectbox("Timeframe", ["15m", "1h", "4h", "1d"], key="position_timeframe")
        strategy = st.text_input("Strategie (optional)", key="position_strategy")
        notes = st.text_input("Notiz")
        submitted = st.form_submit_button("Position speichern")
    if submitted:
        if not position_id.strip():
            st.error("Positions-ID ist erforderlich.")
        else:
            state_db.init_db()
            state_db.save_assistant_position(state_db.DB_PATH, {
                "position_id": position_id.strip(), "asset": asset.strip(), "direction": direction,
                "entry_price": entry, "quantity": quantity, "margin": margin, "stop_price": stop,
                "take_profit_price": take_profit, "notes": f"{notes} | timeframe={timeframe} | strategy={strategy}".strip(" |"),
            })
            st.cache_data.clear()
            st.success("Position gespeichert. Es wurde kein Trade ausgeführt.")
    if position_df.empty:
        st.info("Keine manuell erfassten Positionen.")
    else:
        rows = []
        for _, position in position_df[position_df["status"] == "OPEN"].iterrows():
            market = market_df[market_df["asset"] == position["asset"]] if not market_df.empty else pd.DataFrame()
            current = float(market.iloc[0]["last_price"]) if not market.empty and pd.notna(market.iloc[0]["last_price"]) else float(position["entry_price"])
            analysis = analyze_position(
                position_id=str(position["position_id"]), asset=str(position["asset"]), direction=str(position["direction"]),
                entry=float(position["entry_price"]), current=current, quantity=float(position["quantity"]), margin=float(position["margin"]),
                stop=float(position["stop_price"]), take_profit=float(position["take_profit_price"]),
            )
            position_math = calculate_position(account=float(position["margin"]), entry=float(position["entry_price"]), current=current, stop=float(position["stop_price"]), take_profit=float(position["take_profit_price"]), leverage=leverage if leverage > 0 else 1.0, direction=str(position["direction"]))
            rows.append({"ID": analysis.position_id, "Asset": analysis.asset, "State": analysis.state, "PnL": analysis.current_pnl, "PnL % Margin": analysis.current_pnl_pct, "Risk bis Stop": analysis.risk_to_stop, "Liquidation": position_math.liquidation_estimate, "R:R": position_math.risk_reward, "Empfehlung": analysis.recommendation})
            with st.expander(f"MY {analysis.asset} {analysis.direction}"):
                st.write(f"Entry: {position['entry_price']} · Current: {current} · Leverage: {leverage:.2f}x · PnL: {analysis.current_pnl:.2f} € ({analysis.current_pnl_pct:.2%} Margin Return)")
                st.write(f"Strategy: {strategy or 'not specified'} · State: {analysis.state} · Recommendation: {analysis.recommendation}")
                st.write(f"Important levels: Break-even {position['entry_price']:.6f} · Trailing {current - (current - float(position['entry_price'])) * 0.5:.6f} · TP1 {take_profit:.6f} · TP2 {take_profit:.6f} · Invalidation {stop:.6f}")
                scenario_rows = [vars(item) for item in scenario_analysis(entry=float(position["entry_price"]), current=current, quantity=float(position["quantity"]), margin=float(position["margin"]), stop=float(position["stop_price"]), take_profit=float(position["take_profit_price"]), direction=str(position["direction"]))]
                scenario_frame = pd.DataFrame(scenario_rows)
                scenario_frame["change_pct"] *= 100.0
                st.dataframe(scenario_frame, use_container_width=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =============================================================================
# Ansicht 4: Trade Assistant -- scenarios only
# =============================================================================
with tab_assistant:
    st.subheader("Trade Assistant")
    st.caption("Was-wäre-wenn-Rechnung auf Basis deiner Eingaben; keine Order- oder Brokerfunktion.")
    live_asset = st.selectbox("Live Asset", ["BTC/USDT", "ETH/USDT", "SPY", "QQQ"], key="assistant_live_asset")
    live_market = load_table("market_state")
    live_context = load_table("context_state")
    live_strategy = load_table("strategy_state")
    live_forecasts = load_table("assistant_forecasts")
    live_signals = load_table("signals")
    market_row = live_market[live_market["asset"] == live_asset] if not live_market.empty else pd.DataFrame()
    context_row = live_context[live_context["asset"] == live_asset] if not live_context.empty else pd.DataFrame()
    strategy_row = live_strategy[live_strategy["asset"] == live_asset] if not live_strategy.empty else pd.DataFrame()
    if market_row.empty:
        st.warning("LIVE_DATA_UNAVAILABLE: Kein aktueller Marktstatus gespeichert.")
    else:
        market = market_row.iloc[0]
        price_col, health_col, session_col = st.columns(3)
        price_col.metric("Current Price", f"{float(market['last_price']):.6f}" if pd.notna(market["last_price"]) else "UNAVAILABLE")
        health_col.metric("Data", "LIVE" if bool(market["feed_healthy"]) and bool(market["freshness_ok"]) else "STALE")
        session_col.metric("Session", str(market["session_type"]))
    assistant_forecasts = live_forecasts[live_forecasts["asset"] == live_asset].copy() if not live_forecasts.empty else pd.DataFrame()
    if not assistant_forecasts.empty:
        assistant_forecasts["forecast_age_min"] = assistant_forecasts["forecast_timestamp"].apply(_age_seconds).div(60.0).round(1)
        order = {horizon: index for index, horizon in enumerate(("1h", "4h", "8h", "12h", "1d", "3d", "5d", "10d", "20d"))}
        assistant_forecasts["_order"] = assistant_forecasts["horizon"].map(order)
        st.dataframe(assistant_forecasts.sort_values("_order")[["horizon", "direction", "probability_long", "probability_short", "expected_return", "forecast_source", "forecast_status", "model_sources_json", "combination_method", "combined_confidence", "forecast_age_min"]], use_container_width=True)
    else:
        st.info("FORECAST_UNAVAILABLE: Die Live-Forecast-Matrix wurde noch nicht erstellt.")
    context_status = context_row.iloc[0]["status"] if not context_row.empty else "MISSING"
    rule_state = strategy_row.iloc[0]["direction"] if not strategy_row.empty else "UNAVAILABLE"
    signal_rows = live_signals[live_signals["asset"] == live_asset] if not live_signals.empty else pd.DataFrame()
    decision_state = "DATA_INSUFFICIENT" if market_row.empty or assistant_forecasts.empty else ("SETUP_FORMING" if not signal_rows.empty else "WAIT")
    status_col, rule_col, context_col = st.columns(3)
    status_col.metric("Decision State", decision_state)
    rule_col.metric("Rule Strategy", rule_state)
    context_col.metric("Market Context", context_status)
    col1, col2 = st.columns(2)
    scenario_entry = col1.number_input("Entry", min_value=0.000001, value=100.0, key="scenario_entry")
    scenario_current = col1.number_input("Aktueller Preis", min_value=0.000001, value=100.0, key="scenario_current")
    scenario_stop = col2.number_input("Stop", min_value=0.000001, value=95.0, key="scenario_stop")
    scenario_tp = col2.number_input("Take Profit", min_value=0.000001, value=110.0, key="scenario_tp")
    scenario_quantity = st.number_input("Menge", min_value=0.000001, value=1.0, key="scenario_quantity")
    scenario_margin = st.number_input("Margin (€)", min_value=0.000001, value=100.0, key="scenario_margin")
    scenario_direction = st.selectbox("Richtung", ["LONG", "SHORT"], key="scenario_direction")
    try:
        scenario_df = pd.DataFrame([vars(result) for result in scenario_analysis(entry=scenario_entry, current=scenario_current, quantity=scenario_quantity, margin=scenario_margin, stop=scenario_stop, take_profit=scenario_tp, direction=scenario_direction)])
        scenario_df["change_pct"] = scenario_df["change_pct"] * 100.0
        st.dataframe(scenario_df, use_container_width=True)
    except ValueError as exc:
        st.error(str(exc))

# =============================================================================
# Ansicht 5: Paper Portfolio
# =============================================================================
with tab_portfolio:
    portfolio_df = load_table("paper_portfolio")
    open_trades_df = load_table("open_trades")
    history_df = load_table("trade_history")
    equity_curve_df = load_table("equity_curve")

    if portfolio_df.empty:
        st.info("Portfolio noch nicht initialisiert. Läuft `paper_broker.py`?")
    else:
        p = portfolio_df.iloc[0]
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Equity", f"{p['equity_eur']:.2f} €", f"{p['equity_eur'] - p['initial_capital_eur']:+.2f} €")
        col2.metric("Cash (frei)", f"{p['cash_eur']:.2f} €")
        col3.metric("Realisiertes PnL", f"{p['realized_pnl_eur']:.2f} €")
        col4.metric("Gebühren gesamt", f"-{p['fees_paid_eur']:.2f} €")
        col5.metric("Funding gesamt", f"{-p['funding_paid_eur']:.2f} €")

        if not equity_curve_df.empty:
            curve = equity_curve_df.copy()
            curve["timestamp"] = pd.to_datetime(curve["timestamp"])
            st.line_chart(curve.set_index("timestamp")["equity_eur"])

        st.subheader(f"Offene Trades ({len(open_trades_df)})")
        if open_trades_df.empty:
            st.caption("Keine offenen Positionen.")
        else:
            st.dataframe(
                open_trades_df[["asset", "direction", "entry_price", "current_price", "quantity", "leverage", "unrealized_pnl_eur", "stop_loss_price", "take_profit_price", "knockout_barrier_price", "opened_reason"]],
                use_container_width=True,
            )

        st.subheader("Trade-Historie")
        if history_df.empty:
            st.caption("Noch keine geschlossenen Trades.")
        else:
            wins = (history_df["realized_pnl_eur"] > 0).sum()
            total = len(history_df)
            win_rate = wins / total if total else 0.0
            net_pnl = history_df["realized_pnl_eur"].sum()
            hcol1, hcol2, hcol3 = st.columns(3)
            hcol1.metric("Trades gesamt", total)
            hcol2.metric("Win-Rate", f"{win_rate * 100:.1f}%")
            hcol3.metric("Netto-PnL (nach Gebühren)", f"{net_pnl:.2f} €")
            st.dataframe(
                history_df.sort_values("exit_time", ascending=False)[["asset", "direction", "entry_price", "exit_price", "realized_pnl_eur", "fees_eur", "funding_eur", "exit_reason", "exit_time"]],
                use_container_width=True,
            )

# =============================================================================
# Ansicht 6: Strategy Expander -- Telemetrie + Risk-Engine-Stufenplan pro Asset
# =============================================================================
with tab_strategy:
    market_df = load_table("market_state")
    signals_df = load_table("signals")
    open_trades_df = load_table("open_trades")

    if market_df.empty:
        st.info("Keine Daten.")
    else:
        for asset in sorted(market_df["asset"].unique()):
            market_row = market_df[market_df["asset"] == asset].iloc[0]
            asset_signals = signals_df[signals_df["asset"] == asset] if not signals_df.empty else pd.DataFrame()
            open_trade = open_trades_df[open_trades_df["asset"] == asset] if not open_trades_df.empty else pd.DataFrame()
            strategy_row = strategy_df[strategy_df["asset"] == asset].iloc[0] if not strategy_df.empty and asset in set(strategy_df["asset"]) else None

            with st.expander(f"{asset} ({market_row['asset_class']}) — Preis {market_row['last_price']}"):
                if strategy_row is not None:
                    try:
                        aggressive = json.loads(strategy_row["aggressive_json"])
                        conservative = json.loads(strategy_row["conservative_json"])
                        st.caption(f"What-if-Berechnung · Richtung {strategy_row['direction']} · Entry {float(strategy_row['entry_price']):.6f}")
                        left, right = st.columns(2)
                        for column, title, profile in ((left, "Aggressiv", aggressive), (right, "Konservativ", conservative)):
                            with column:
                                st.markdown(f"**{title}**")
                                st.metric("Effektiver Hebel", f"{float(profile.get('leverage', 0.0)):.2f}x")
                                st.caption(
                                    f"Instrument: {profile.get('instrument_type', 'unbekannt')} · "
                                    f"Max.: {float(profile.get('max_allowed_leverage', 0.0)):.2f}x · "
                                    f"Schutz: {profile.get('protection_model', 'unbekannt')}"
                                )
                                st.write(
                                    f"Risk-Budget: {float(profile.get('allowed_risk_fraction', 0.0)):.2%} "
                                    f"({float(profile.get('risk_amount_eur', 0.0)):.2f} €) · "
                                    f"Risk-Notional: {float(profile.get('risk_based_notional_eur', 0.0)):.2f} € · "
                                    f"Finales Notional: {float(profile.get('position_size_eur', 0.0)):.2f} € · "
                                    f"Margin: {float(profile.get('margin_eur', 0.0)):.2f} €"
                                )
                                st.write(
                                    f"Entry: {float(profile.get('entry_price', 0.0)):.6f} · "
                                    f"SL: {float(profile.get('stop_loss_price', 0.0)):.6f} · "
                                    f"TP: {float(profile.get('take_profit_price', 0.0)):.6f} · "
                                    f"Liquidation: {profile.get('liquidation_price') or 'nicht anwendbar'} · "
                                    f"Safety Barrier: {profile.get('safety_barrier_price') or 'nicht anwendbar'}"
                                )
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        st.warning(f"Strategy-State nicht lesbar: {exc}")
                else:
                    st.caption("Noch keine What-if-Profile gespeichert.")

                if asset_signals.empty:
                    st.caption("Keine Signale.")
                    continue

                for _, sig in asset_signals.iterrows():
                    st.markdown(f"**{sig['model_type']}** — Score {sig['score']:.1f}, Richtung {sig['direction']}")
                    horizon = (
                        f"{sig['expected_duration_days']:.1f} trading days"
                        if pd.notna(sig['expected_duration_days'])
                        else f"{sig['expected_duration_bars']:.1f} bars"
                    )
                    st.write(
                        f"MFE (Take-Profit-Potenzial): {sig['expected_mfe']:.2%} · "
                        f"MAE (erwarteter Rücksetzer): {sig['expected_mae']:.2%} · "
                        f"Horizont: {horizon}"
                    )

                if not open_trade.empty:
                    t = open_trade.iloc[0]
                    st.success(
                        f"**Aktiver Paper-Trade**: {t['direction']} @ {t['entry_price']:.4f}, "
                        f"aktueller Kurs {t['current_price']:.4f} — "
                        f"TP bei {t['take_profit_price']:.4f}, SL (nachgezogen) bei {t['stop_loss_price']:.4f}, "
                        f"Hebel {t['leverage']:.2f}x, Schutzmodell {t.get('protection_model', 'unbekannt')}, "
                        f"unrealisiert {t['unrealized_pnl_eur']:+.2f} €."
                    )
                else:
                    lead = asset_signals[asset_signals["model_type"].isin(["crypto_sniper", "swing"])]
                    if not lead.empty:
                        score = lead.iloc[0]["score"]
                        gate_state = "🟢 über Schwelle -- Einstieg möglich" if score >= 80 else "⚪ unter Schwelle -- kein Einstieg"
                        st.info(f"Kein aktiver Trade. Lead-Score {score:.1f}/100 ({gate_state}).")

st.caption(f"Zuletzt aktualisiert (Client-Cache {REFRESH_SECONDS}s) · DB: {state_db.DB_PATH}")
