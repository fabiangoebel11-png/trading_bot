"""Paper-Trading-Simulator: hört auf die SQLite-DB (geschrieben von
``live_daemon.py``), eröffnet/verwaltet/schließt virtuelle Trades mit
realistischer Kosten-Simulation je Asset-Klasse.

Bewusst komplett getrennt vom Daemon-Prozess: kann unabhängig neu gestartet
werden, verwaltet offene Trades ausschließlich über die DB (``open_trades``)
-- kein In-Memory-State, der einen Neustart nicht überleben würde.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

import state_db
from risk_engine import TradeSetup, compute_risk_parameters, ratchet_trailing_stop
from core.notifications.telegram_bot import TelegramBot

# --- Kosten-Simulation --------------------------------------------------------
# Equities (QQQ/SPY) ueber Trade-Republic-artige KO-Zertifikate:
EQUITY_ORDER_FEE_EUR = 1.00
EQUITY_SPREAD_PCT = 0.0015           # 0.15%
EQUITY_SLIPPAGE_PCT = 0.0005         # zusätzliche Markt-Order-Slippage
EQUITY_EXECUTION_PENALTY_PCT = 0.0015  # zusätzliche pessimistische Gap-/Broker-Abweichung

# Crypto (BTC/ETH) ueber professionelle Hebel-Boersen (Bybit/Binance-Stil):
CRYPTO_TAKER_FEE_PCT = 0.0005         # 0.05%
CRYPTO_BASE_SPREAD_PCT = 0.0002       # Floor bei sehr ruhigem Markt
CRYPTO_SPREAD_ATR_COEFFICIENT = 0.05  # dynamischer Spread skaliert mit ATR%
CRYPTO_SLIPPAGE_PCT = 0.0003
CRYPTO_EXECUTION_PENALTY_PCT = 0.0005  # zusätzliche pessimistische Gap-/Orderbuch-Abweichung
FUNDING_INTERVAL_HOURS = 8            # Binance/Bybit-Standard: 00/08/16 UTC

# --- Entry-/Exit-Gates ---------------------------------------------------------
CRYPTO_ENTRY_SCORE_THRESHOLD = 80.0
EQUITY_SWING_SCORE_THRESHOLD = 80.0
EQUITY_ENTRY_SCORE_THRESHOLD = 80.0
EXIT_SCORE_FLOOR = 40.0                # Regime-Dreher: Score faellt unter diese Schwelle -> Exit
MAX_CONCURRENT_TRADES = 3              # Anlehnung an DataConfig-Hardcap (max 3 Symbole gleichzeitig)


@dataclass
class Fees:
    entry_fee_eur: float
    exit_fee_eur: float


def _is_funding_hour(now: datetime) -> bool:
    """True genau in der Minute, in der eine der Binance/Bybit-Standard-
    Funding-Stunden (00/08/16 UTC) beginnt -- wird vom Broker mit einer
    Vorbar-gegen-neu-Uhrzeit-Prüfung aufgerufen (siehe ``_apply_funding``)."""
    return now.hour % FUNDING_INTERVAL_HOURS == 0


def equity_entry_cost(notional_eur: float) -> float:
    return EQUITY_ORDER_FEE_EUR + notional_eur * (EQUITY_SPREAD_PCT + EQUITY_SLIPPAGE_PCT)


def equity_exit_cost(notional_eur: float) -> float:
    return EQUITY_ORDER_FEE_EUR + notional_eur * (EQUITY_SPREAD_PCT + EQUITY_SLIPPAGE_PCT)


def crypto_dynamic_spread_pct(atr_pct: float) -> float:
    """Spread waechst mit der aktuellen Volatilitaet (ATR%) -- in ruhigen
    Maerkten eng (Floor), in hektischen Phasen breiter, wie am echten
    Orderbuch einer Perp-Boerse."""
    if not np.isfinite(atr_pct):
        atr_pct = 0.0
    return float(max(CRYPTO_BASE_SPREAD_PCT, abs(atr_pct) * CRYPTO_SPREAD_ATR_COEFFICIENT))


def crypto_entry_cost(notional_eur: float, atr_pct: float) -> float:
    spread_pct = crypto_dynamic_spread_pct(atr_pct)
    return notional_eur * (CRYPTO_TAKER_FEE_PCT + spread_pct + CRYPTO_SLIPPAGE_PCT)


def crypto_exit_cost(notional_eur: float, atr_pct: float) -> float:
    return crypto_entry_cost(notional_eur, atr_pct)


class PaperBroker:
    def __init__(self, db_path=state_db.DB_PATH, initial_capital_eur: float = 500.0) -> None:
        self.db_path = db_path
        state_db.init_db(db_path)
        state_db.ensure_portfolio(db_path, initial_capital_eur)
        self.telegram = TelegramBot()
        self.recovered_trade_ids: list[str] = []
        self._recover_open_trades()

    def _recover_open_trades(self) -> None:
        """Validate that restart recovery sees persisted trades and stop levels."""
        with state_db.connect(self.db_path) as conn:
            rows = conn.execute("SELECT trade_id, stop_loss_price FROM open_trades").fetchall()
            self.recovered_trade_ids = [str(row["trade_id"]) for row in rows if row["stop_loss_price"] is not None]
        if self.recovered_trade_ids:
            print(f"[paper_broker] recovered {len(self.recovered_trade_ids)} open trade(s) from SQLite")

    # --- Portfolio helpers ----------------------------------------------------
    def _portfolio(self, conn) -> dict:
        row = conn.execute("SELECT * FROM paper_portfolio WHERE id = 1").fetchone()
        return dict(row)

    @staticmethod
    def _drawdown_pct(portfolio: dict) -> float:
        initial = float(portfolio["initial_capital_eur"])
        equity = float(portfolio["equity_eur"])
        if initial <= 0.0:
            return 1.0
        return max((initial - equity) / initial, 0.0)

    def _open_trades(self, conn) -> list[dict]:
        return [dict(row) for row in conn.execute("SELECT * FROM open_trades").fetchall()]

    def _market_state(self, conn, asset: str) -> dict | None:
        row = conn.execute("SELECT * FROM market_state WHERE asset = ?", (asset,)).fetchone()
        return dict(row) if row is not None else None

    def _signal(self, conn, asset: str, model_type: str) -> dict | None:
        row = conn.execute("SELECT * FROM signals WHERE asset = ? AND model_type = ?", (asset, model_type)).fetchone()
        return dict(row) if row is not None else None

    def _open_risk_eur(self, open_trades: list[dict]) -> float:
        """Summe des noch ausstehenden Verlust-Risikos aller offenen Trades
        (Notional * Stop-Distanz), fuer die geteilte 500-EUR-Wallet-Budgetierung."""
        total = 0.0
        for trade in open_trades:
            entry_price = float(trade["entry_price"])
            stop_loss_price = float(trade["stop_loss_price"])
            notional_eur = float(trade["notional_eur"])
            if (
                not np.isfinite(entry_price)
                or entry_price <= 0.0
                or not np.isfinite(stop_loss_price)
                or not np.isfinite(notional_eur)
                or notional_eur < 0.0
            ):
                return float("inf")
            stop_distance_pct = abs(entry_price - stop_loss_price) / entry_price
            total += notional_eur * stop_distance_pct
        return total

    # --- Hauptzyklus ------------------------------------------------------------
    def run_cycle(self) -> None:
        now = datetime.now(timezone.utc)
        with state_db.connect(self.db_path) as conn:
            open_trades = self._open_trades(conn)
            self._update_marks_and_exits(conn, open_trades, now)
            open_trades = self._open_trades(conn)  # re-read after exits
            self._apply_funding(conn, open_trades, now)
            open_trades = self._open_trades(conn)
            self._evaluate_entries(conn, open_trades, now)
            self._write_equity_curve(conn, now)
            conn.commit()

    @staticmethod
    def _execution_price(price: float, direction: str, asset_class: str, *, entry: bool) -> float:
        penalty = (EQUITY_SLIPPAGE_PCT + EQUITY_EXECUTION_PENALTY_PCT) if asset_class == "equity" else (CRYPTO_SLIPPAGE_PCT + CRYPTO_EXECUTION_PENALTY_PCT)
        adverse_sign = 1.0 if direction == "LONG" else -1.0
        if not entry:
            adverse_sign *= -1.0
        return float(price * (1.0 + adverse_sign * penalty))

    # --- Mark-to-Market + Exit-Bedingungen --------------------------------------
    def _update_marks_and_exits(self, conn, open_trades: list[dict], now: datetime) -> None:
        for trade in open_trades:
            market = self._market_state(conn, trade["asset"])
            if market is None or market["last_price"] is None:
                continue  # kein frischer Preis diesen Zyklus -- Position bleibt unangetastet (z.B. Wochenende bei Equities)
            price = float(market["last_price"])
            direction = trade["direction"]
            quantity = trade["quantity"]
            unrealized = (price - trade["entry_price"]) * quantity if direction == "LONG" else (trade["entry_price"] - price) * quantity

            stop_distance_pct = abs(trade["entry_price"] - trade["initial_stop_loss_price"]) / trade["entry_price"]
            new_stop = ratchet_trailing_stop(direction, trade["stop_loss_price"], price, stop_distance_pct)

            exit_reason = None
            if direction == "LONG":
                if price <= new_stop:
                    exit_reason = "stop_loss"
                elif price >= trade["take_profit_price"]:
                    exit_reason = "take_profit"
                elif trade.get("protection_model") != "underlying_no_liquidation" and price <= trade["knockout_barrier_price"]:
                    exit_reason = "knockout"
            else:
                if price >= new_stop:
                    exit_reason = "stop_loss"
                elif price <= trade["take_profit_price"]:
                    exit_reason = "take_profit"
                elif trade.get("protection_model") != "underlying_no_liquidation" and price >= trade["knockout_barrier_price"]:
                    exit_reason = "knockout"

            # Regime-Dreh-Erkennung: nutzt das jeweils fuehrende Signal
            # (crypto_sniper fuer Krypto, swing fuer Equities -- Model 2 ist
            # der strategische Rahmen, ein Dreh dort ist wichtiger als ein
            # kurzfristiger Entry-Wackler).
            lead_model = "crypto_sniper" if trade["asset_class"] == "crypto" else "swing"
            signal = self._signal(conn, trade["asset"], lead_model)
            if exit_reason is None and signal is not None:
                score = float(signal["swing_opportunity_score"] or signal["score"])
                if score < EXIT_SCORE_FLOOR or (signal["direction"] in {"LONG", "SHORT"} and signal["direction"] != direction):
                    exit_reason = "regime_flip"

            if exit_reason is not None:
                self._close_trade(conn, trade, price, exit_reason, now)
            else:
                conn.execute(
                    "UPDATE open_trades SET current_price = ?, unrealized_pnl_eur = ?, stop_loss_price = ? WHERE trade_id = ?",
                    (price, unrealized, new_stop, trade["trade_id"]),
                )

    def _close_trade(self, conn, trade: dict, exit_price: float, reason: str, now: datetime) -> None:
        exit_price = self._execution_price(exit_price, trade["direction"], trade["asset_class"], entry=False)
        notional_eur = trade["quantity"] * exit_price
        if trade["asset_class"] == "equity":
            fee = equity_exit_cost(notional_eur)
        else:
            market = self._market_state(conn, trade["asset"])
            atr_pct = float(market["atr_pct"]) if market and market["atr_pct"] is not None else 0.0
            fee = crypto_exit_cost(notional_eur, atr_pct)

        direction = trade["direction"]
        quantity = trade["quantity"]
        gross_pnl = (exit_price - trade["entry_price"]) * quantity if direction == "LONG" else (trade["entry_price"] - exit_price) * quantity
        realized_pnl = gross_pnl - fee - trade["entry_fees_eur"]

        conn.execute(
            "INSERT INTO trade_history (trade_id, asset, asset_class, direction, model_source, entry_price, entry_time, "
            "exit_price, exit_time, quantity, notional_eur, leverage, realized_pnl_eur, fees_eur, funding_eur, exit_reason, opened_reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?, "
            "(SELECT COALESCE(SUM(0),0)), ?, ?)".replace(
                "(SELECT COALESCE(SUM(0),0))", "0.0"
            ),
            (
                trade["trade_id"], trade["asset"], trade["asset_class"], direction, trade["model_source"],
                trade["entry_price"], trade["entry_time"], exit_price, now.isoformat(), quantity,
                notional_eur, trade["leverage"], realized_pnl, fee + trade["entry_fees_eur"], reason, trade["opened_reason"],
            ),
        )
        conn.execute("DELETE FROM open_trades WHERE trade_id = ?", (trade["trade_id"],))
        event = {"asset": trade["asset"], "direction": direction, "exit_price": round(exit_price, 6), "pnl_eur": round(realized_pnl, 2), "trade_id": trade["trade_id"]}
        event_name = {"stop_loss": "Trailing Stop Hit", "take_profit": "Take Profit Hit", "knockout": "Knock-Out Hit"}.get(reason, f"Trade Closed ({reason})")
        self.telegram.send_event(event_name, **event)

        portfolio = self._portfolio(conn)
        conn.execute(
            "UPDATE paper_portfolio SET cash_eur = cash_eur + ?, realized_pnl_eur = realized_pnl_eur + ?, "
            "fees_paid_eur = fees_paid_eur + ?, open_trade_count = open_trade_count - 1, updated_at = ? WHERE id = 1",
            (realized_pnl + trade["margin_eur"], realized_pnl, fee, now.isoformat()),
        )

    # --- Funding (Krypto, alle 8h) -----------------------------------------------
    def _apply_funding(self, conn, open_trades: list[dict], now: datetime) -> None:
        if not _is_funding_hour(now):
            return
        for trade in open_trades:
            if trade["asset_class"] != "crypto":
                continue
            last_funding = trade["last_funding_time"]
            if last_funding is not None:
                last_dt = datetime.fromisoformat(last_funding)
                if last_dt.hour == now.hour and last_dt.date() == now.date():
                    continue  # already charged this exact funding window
            market = self._market_state(conn, trade["asset"])
            funding_rate = float(market["funding_rate"]) if market and market["funding_rate"] is not None else 0.0
            notional_eur = trade["quantity"] * trade["current_price"]
            # Long zahlt bei positivem Funding, Short bekommt (Standard-Perp-Konvention).
            funding_cost = notional_eur * funding_rate if trade["direction"] == "LONG" else -notional_eur * funding_rate
            conn.execute(
                "UPDATE open_trades SET unrealized_pnl_eur = unrealized_pnl_eur - ?, last_funding_time = ? WHERE trade_id = ?",
                (funding_cost, now.isoformat(), trade["trade_id"]),
            )
            conn.execute(
                "UPDATE paper_portfolio SET funding_paid_eur = funding_paid_eur + ?, updated_at = ? WHERE id = 1",
                (funding_cost, now.isoformat()),
            )

    # --- Neue Entries -------------------------------------------------------------
    def _evaluate_entries(self, conn, open_trades: list[dict], now: datetime) -> None:
        open_assets = {trade["asset"] for trade in open_trades}
        if len(open_trades) >= MAX_CONCURRENT_TRADES:
            return
        portfolio = self._portfolio(conn)
        open_risk_eur = self._open_risk_eur(open_trades)

        for row in conn.execute("SELECT DISTINCT asset FROM market_state").fetchall():
            asset = row["asset"]
            if asset in open_assets:
                continue
            market = self._market_state(conn, asset)
            if market is None or not market["market_open"] or market["last_price"] is None:
                continue
            if not market["feed_healthy"] or not market["freshness_ok"] or not market["warmup_ready"]:
                continue

            asset_class = market["asset_class"]
            setup = self._build_setup(conn, asset, asset_class, market, portfolio, open_risk_eur)
            if setup is None:
                continue

            params = compute_risk_parameters(setup)
            if params.rejected or params.position_size_eur <= 0:
                continue

            self._open_trade(conn, setup, params, market, now)
            open_risk_eur += params.risk_amount_eur
            portfolio = self._portfolio(conn)
            open_trades = self._open_trades(conn)
            if len(open_trades) >= MAX_CONCURRENT_TRADES:
                break

    def _build_setup(self, conn, asset: str, asset_class: str, market: dict, portfolio: dict, open_risk_eur: float) -> TradeSetup | None:
        if asset_class == "crypto":
            signal = self._signal(conn, asset, "crypto_sniper")
            if signal is None or signal["score"] < CRYPTO_ENTRY_SCORE_THRESHOLD:
                return None
            if signal["direction"] not in {"LONG", "SHORT"}:
                return None
            reason = f"crypto_sniper score={signal['score']:.1f} >= {CRYPTO_ENTRY_SCORE_THRESHOLD}"
            return self._finalize_setup(asset, asset_class, signal["direction"], market, signal, portfolio, open_risk_eur, reason)

        # Equity: braucht BEIDE Modelle uebereinstimmend (Model 2 Swing-Setup UND Model 1B Entry-Timing)
        swing = self._signal(conn, asset, "swing")
        intraday = self._signal(conn, asset, "intraday")
        if swing is None or intraday is None:
            return None
        swing_score = float(swing["swing_opportunity_score"] or swing["score"])
        entry_score = float(intraday["score"])
        if swing_score < EQUITY_SWING_SCORE_THRESHOLD or entry_score < EQUITY_ENTRY_SCORE_THRESHOLD:
            return None
        if swing["direction"] not in {"LONG", "SHORT"}:
            return None
        reason = f"swing={swing_score:.1f}>=80 & entry={entry_score:.1f}>=80"
        return self._finalize_setup(asset, asset_class, swing["direction"], market, swing, portfolio, open_risk_eur, reason)

    def _finalize_setup(self, asset, asset_class, direction, market, lead_signal, portfolio, open_risk_eur, reason) -> TradeSetup:
        setup = TradeSetup(
            asset=asset, asset_class=asset_class, direction=direction, entry_price=float(market["last_price"]),
            atr=float(market["atr"] or 0.0), expected_mae=float(lead_signal["expected_mae"]), expected_mfe=float(lead_signal["expected_mfe"]),
            score=float(lead_signal["swing_opportunity_score"] or lead_signal["score"]), capital_eur=float(portfolio["equity_eur"]),
            open_risk_eur=open_risk_eur,
            instrument_type="crypto_perpetual" if asset_class == "crypto" else "equity_underlying",
            available_margin_eur=max(float(portfolio["cash_eur"]), 0.0),
            margin_in_use_eur=max(float(portfolio["equity_eur"]) - float(portfolio["cash_eur"]), 0.0),
            drawdown_pct=self._drawdown_pct(portfolio),
        )
        object.__setattr__(setup, "_reason", reason)  # stash for _open_trade without changing the dataclass shape
        return setup

    def _open_trade(self, conn, setup: TradeSetup, params, market: dict, now: datetime) -> None:
        executed_entry_price = self._execution_price(params.entry_price, setup.direction, setup.asset_class, entry=True)
        notional_eur = params.position_size_eur
        if setup.asset_class == "equity":
            entry_fee = equity_entry_cost(notional_eur)
        else:
            atr_pct = float(market["atr_pct"]) if market["atr_pct"] is not None else 0.0
            entry_fee = crypto_entry_cost(notional_eur, atr_pct)

        trade_id = str(uuid.uuid4())
        reason = getattr(setup, "_reason", "signal")
        conn.execute(
            "INSERT INTO open_trades (trade_id, asset, asset_class, direction, model_source, entry_price, entry_time, "
            "quantity, notional_eur, margin_eur, leverage, stop_loss_price, take_profit_price, knockout_barrier_price, "
            "initial_stop_loss_price, current_price, unrealized_pnl_eur, entry_score, swing_score, expected_mfe, expected_mae, "
            "expected_duration_bars, opened_reason, last_funding_time, entry_fees_eur, instrument_type, protection_model, "
            "liquidation_price, safety_barrier_price) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trade_id, setup.asset, setup.asset_class, setup.direction,
                "crypto_sniper" if setup.asset_class == "crypto" else "model2_swing+model1b_entry",
                executed_entry_price, now.isoformat(), params.quantity, notional_eur, params.margin_eur, params.leverage,
                params.stop_loss_price, params.take_profit_price, params.knockout_barrier_price, params.stop_loss_price,
                executed_entry_price, 0.0, None, None, setup.expected_mfe, setup.expected_mae, None, reason, None, entry_fee,
                params.instrument_type, params.protection_model, params.liquidation_price, params.safety_barrier_price,
            ),
        )
        conn.execute(
            "UPDATE paper_portfolio SET cash_eur = cash_eur - ?, fees_paid_eur = fees_paid_eur + ?, "
            "open_trade_count = open_trade_count + 1, updated_at = ? WHERE id = 1",
            (params.margin_eur + entry_fee, entry_fee, now.isoformat()),
        )
        self.telegram.send_event(
            "Trade Opened", asset=setup.asset, direction=setup.direction,
            entry_price=round(executed_entry_price, 6), pnl_start_eur=0.0,
            leverage=round(params.leverage, 2), stop=round(params.stop_loss_price, 6),
            take_profit=round(params.take_profit_price, 6),
        )

    # --- Reporting --------------------------------------------------------------
    def _write_equity_curve(self, conn, now: datetime) -> None:
        portfolio = self._portfolio(conn)
        open_trades = self._open_trades(conn)
        unrealized_total = sum(t["unrealized_pnl_eur"] or 0.0 for t in open_trades)
        margin_locked = sum(t["margin_eur"] for t in open_trades)
        equity_eur = portfolio["cash_eur"] + margin_locked + unrealized_total
        conn.execute(
            "UPDATE paper_portfolio SET equity_eur = ?, updated_at = ? WHERE id = 1",
            (equity_eur, now.isoformat()),
        )
        conn.execute(
            "INSERT OR REPLACE INTO equity_curve (timestamp, equity_eur, cash_eur, open_positions) VALUES (?,?,?,?)",
            (now.isoformat(), equity_eur, portfolio["cash_eur"], len(open_trades)),
        )


def main() -> None:
    import time

    broker = PaperBroker()
    print(f"Paper broker running against {broker.db_path}")
    try:
        while True:
            try:
                broker.run_cycle()
            except Exception as exc:  # noqa: BLE001 -- never let one bad cycle kill the process
                print(f"[paper_broker] cycle error: {type(exc).__name__}: {exc}")
            time.sleep(15)
    except KeyboardInterrupt:
        print("Broker gracefully stopped")


if __name__ == "__main__":
    main()
