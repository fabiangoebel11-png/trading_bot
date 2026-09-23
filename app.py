"""Streamlit-Dashboard: reines Read-Only-Frontend, das ausschliesslich die
SQLite-DB liest (``trading_state.db``) -- laedt selbst NIE Marktdaten und
sendet nie Orders. Kann unabhaengig vom Daemon/Broker gestartet/gestoppt
werden; zeigt einfach "keine Daten" an, wenn die anderen Prozesse noch nicht
liefen.

Start: ``streamlit run app.py``
"""
from __future__ import annotations

from datetime import datetime, timezone
import json

import pandas as pd
import streamlit as st

import state_db

st.set_page_config(page_title="Trading Bot Cockpit", layout="wide")
st.title("📡 Trading Bot Cockpit (Paper Trading)")

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


tab_radar, tab_portfolio, tab_strategy = st.tabs(["🎯 Radar", "💼 Paper Portfolio", "🔬 Strategy Expander"])

# =============================================================================
# Ansicht 1: Radar -- Live-Scores aller Assets, Market-State, Signale
# =============================================================================
with tab_radar:
    market_df = load_table("market_state")
    signals_df = load_table("signals")
    strategy_df = load_table("strategy_state")

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

# =============================================================================
# Ansicht 2: Paper Portfolio
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
# Ansicht 3: Strategy Expander -- Telemetrie + Risk-Engine-Stufenplan pro Asset
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
                                st.metric("Hebel", f"{float(profile.get('leverage', 0.0)):.2f}x")
                                st.write(
                                    f"Entry: {float(profile.get('entry_price', 0.0)):.6f} · "
                                    f"SL: {float(profile.get('stop_loss_price', 0.0)):.6f} · "
                                    f"TP: {float(profile.get('take_profit_price', 0.0)):.6f} · "
                                    f"KO: {float(profile.get('knockout_barrier_price', 0.0)):.6f}"
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
                    st.write(
                        f"MFE (Take-Profit-Potenzial): {sig['expected_mfe']:.2%} · "
                        f"MAE (erwarteter Rücksetzer): {sig['expected_mae']:.2%} · "
                        f"Horizont: {sig['expected_duration_days'] if pd.notna(sig['expected_duration_days']) else sig['expected_duration_bars']:.1f}"
                    )

                if not open_trade.empty:
                    t = open_trade.iloc[0]
                    st.success(
                        f"**Aktiver Paper-Trade**: {t['direction']} @ {t['entry_price']:.4f}, "
                        f"aktueller Kurs {t['current_price']:.4f} — "
                        f"KO bei {t['knockout_barrier_price']:.4f}, TP bei {t['take_profit_price']:.4f}, "
                        f"SL (nachgezogen) bei {t['stop_loss_price']:.4f}, Hebel {t['leverage']:.2f}x, "
                        f"unrealisiert {t['unrealized_pnl_eur']:+.2f} €."
                    )
                else:
                    lead = asset_signals[asset_signals["model_type"].isin(["crypto_sniper", "swing"])]
                    if not lead.empty:
                        score = lead.iloc[0]["score"]
                        gate_state = "🟢 über Schwelle -- Einstieg möglich" if score >= 80 else "⚪ unter Schwelle -- kein Einstieg"
                        st.info(f"Kein aktiver Trade. Lead-Score {score:.1f}/100 ({gate_state}).")

st.caption(f"Zuletzt aktualisiert (Client-Cache {REFRESH_SECONDS}s) · DB: {state_db.DB_PATH}")
