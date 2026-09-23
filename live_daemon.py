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

import time
import traceback
import json
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import state_db
from train import load_config
from core.config import TradingBotConfig, TrendMLConfig
from core.market_state import DataHealth, FeedStatus, MarketStateService, PredictionSnapshot
from core.ml.data import load_ohlcv_parquet
from core.ml.dataset import make_sequences
from core.ml.device import get_device
from core.ml.features import build_feature_matrix, fetch_breadth_basket
from core.ml.inference import load_symbol_model
from core.ml.providers import AssetSpec, asset_specs_from_config, canonical_cache_path, prepare_asset_specs
from core.ml.scoring import continuous_opportunity_score
from core.ml.swing_data import build_swing_dataset
from core.ml.swing_model import load_swing_model, scores_from_predictions, HORIZONS
from core.signal_orchestrator import AlertGate, AlertGateConfig, intraday_snapshot
from core.strategy import compute_atr
from risk_engine import TradeSetup, compute_strategy_profiles
from core.notifications.telegram_bot import TelegramBot

PRICE_POLL_S = 20.0
SIGNAL_REFRESH_S = 15 * 60.0

# Model 2 (Swing) uses the plain ticker as its asset name; Model 1B (equity
# intraday) uses the "logical" proxy name from TrendMLConfig.market_assets.
# Everywhere in the DB/GUI/broker we standardize on the ticker (QQQ/SPY),
# since that's what a human operator / the paper broker actually trades.
EQUITY_NAME_TO_TICKER = {"NASDAQ100_PROXY": "QQQ", "SP500_PROXY": "SPY"}

CONFIG_PATHS = {
    "crypto": "configs/training_crypto_intraday.yaml",
    "equity": "configs/training_equity_intraday.yaml",
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


def _load_base_ohlc(ml_config: TrendMLConfig, asset_name: str) -> pd.DataFrame | None:
    specs = [s for s in asset_specs_from_config(TradingBotConfig(ml=ml_config)) if s.name == asset_name and s.timeframe == ml_config.base_timeframe]
    if not specs:
        return None
    path = canonical_cache_path(ml_config.market_cache_dir, specs[0])
    if not path.exists():
        return None
    return load_ohlcv_parquet(path)


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
                self._models[asset_name] = load_symbol_model(asset_name, self.config)
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
            "timestamp": window.index[-1],
            "opportunity": float(outputs["opportunity"][0]),
            "expected_return": float(outputs["expected_return"][0]),
            "expected_duration": float(outputs["expected_duration"][0]),
            "expected_mfe": float(outputs["expected_mfe"][0]),
            "expected_mae": float(outputs["expected_mae"][0]),
            "entry_score": float(outputs["entry_score"][0]),
            "score": float(quality["score"][0]),
            "direction": str(quality["direction"][0]),
        }


class LiveDaemon:
    def __init__(self) -> None:
        self.crypto_cfg = load_config(CONFIG_PATHS["crypto"])
        self.equity_cfg = load_config(CONFIG_PATHS["equity"])
        self.swing_cfg = load_config(CONFIG_PATHS["swing"])
        self._print_inference_device()
        state_db.init_db()
        self.market_states = MarketStateService()
        self.model1_crypto = Model1Runner(self.crypto_cfg.ml)
        self.model1_equity = Model1Runner(self.equity_cfg.ml)
        self._swing_models: dict[str, object] = {}
        self._last_signal_refresh = 0.0
        self._exchange = None
        self.telegram = TelegramBot()
        self._notified_swing: set[tuple[str, str, str]] = set()
        self._pending_notifications: list[dict[str, object]] = []

    @staticmethod
    def _print_inference_device() -> None:
        try:
            device = get_device()
            print(f"Inference Device: {device.type.upper()}")
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
        return float(ticker["last"])

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
        return float(price)

    def _fetch_equity_price(self, ticker: str) -> float | None:
        try:
            return self._fetch_equity_price_once(ticker)
        except Exception as exc:  # noqa: BLE001 - fail closed after retries
            print(f"[daemon] equity price fetch failed for {ticker}: {exc}")
            return None

    # --- schneller Takt: nur Preis + Session --------------------------------------
    def price_tick(self) -> None:
        now = _now()
        with state_db.connect() as conn:
            for symbol in self.crypto_cfg.ml.assets:
                price = self._fetch_crypto_price(symbol)
                self._upsert_market_state(conn, symbol, "crypto", price, now)
                self._persist_strategy_profiles(conn, symbol, "crypto", now)
            for name in self.equity_cfg.ml.assets:
                ticker = EQUITY_NAME_TO_TICKER.get(name, name)
                price = self._fetch_equity_price(ticker)
                self._upsert_market_state(conn, ticker, "equity", price, now)
                self._persist_strategy_profiles(conn, ticker, "equity", now)
            conn.commit()

    def _upsert_market_state(self, conn, asset: str, asset_class: str, price: float | None, now: datetime) -> None:
        market = self.market_states.state_at(asset, now)
        existing = conn.execute("SELECT atr, atr_pct, funding_rate FROM market_state WHERE asset = ?", (asset,)).fetchone()
        atr = existing["atr"] if existing else None
        atr_pct = existing["atr_pct"] if existing else None
        funding_rate = existing["funding_rate"] if existing else None
        if asset_class == "crypto" and price is not None:
            funding_rate = self._fetch_crypto_funding_rate(asset)
        feed_healthy = price is not None
        conn.execute(
            "INSERT INTO market_state (asset, asset_class, timestamp, market_open, session_type, is_trading_day, "
            "is_holiday, session_progress, last_price, atr, atr_pct, funding_rate, data_age_seconds, feed_healthy, "
            "freshness_ok, warmup_ready, reason, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0.0,?,?,?,?,?) "
            "ON CONFLICT(asset) DO UPDATE SET timestamp=excluded.timestamp, market_open=excluded.market_open, "
            "session_type=excluded.session_type, is_trading_day=excluded.is_trading_day, is_holiday=excluded.is_holiday, "
            "session_progress=excluded.session_progress, last_price=COALESCE(excluded.last_price, market_state.last_price), "
            "funding_rate=COALESCE(excluded.funding_rate, market_state.funding_rate), feed_healthy=excluded.feed_healthy, "
            "freshness_ok=excluded.freshness_ok, updated_at=excluded.updated_at",
            (
                asset, asset_class, now.isoformat(), int(market.market_open), market.session_type, int(market.is_trading_day),
                int(market.is_holiday), market.session_progress, price, atr, atr_pct, funding_rate, int(feed_healthy),
                int(feed_healthy), int(bool(atr)), "" if feed_healthy else "no live price this tick",
                now.isoformat(),
            ),
        )

    def _persist_strategy_profiles(self, conn, asset: str, asset_class: str, now: datetime) -> None:
        market = conn.execute("SELECT * FROM market_state WHERE asset = ?", (asset,)).fetchone()
        if market is None or market["last_price"] is None:
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
        )
        profiles = compute_strategy_profiles(setup)
        conn.execute(
            "INSERT INTO strategy_state (asset, asset_class, timestamp, direction, entry_price, aggressive_json, conservative_json, updated_at) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(asset) DO UPDATE SET asset_class=excluded.asset_class, timestamp=excluded.timestamp, direction=excluded.direction, entry_price=excluded.entry_price, aggressive_json=excluded.aggressive_json, conservative_json=excluded.conservative_json, updated_at=excluded.updated_at",
            (asset, asset_class, now.isoformat(), direction, setup.entry_price, json.dumps(profiles["aggressive"].__dict__), json.dumps(profiles["conservative"].__dict__), now.isoformat()),
        )

    # --- langsamer Takt: volle Kerzen-Aktualisierung + ML-Inferenz -----------------
    def signal_refresh(self) -> None:
        now = _now()
        self._refresh_crypto(now)
        self._refresh_equity_and_swing(now)

    def _prepare(self, config: TradingBotConfig) -> None:
        specs = asset_specs_from_config(config)
        try:
            self._prepare_once(config, specs)
        except Exception as exc:  # noqa: BLE001
            print(f"[daemon] data refresh failed (using stale cache if present): {exc}")

    @retry_with_backoff()
    def _prepare_once(self, config: TradingBotConfig, specs) -> None:
        prepare_asset_specs(
            specs,
            cache_root=config.ml.market_cache_dir,
            cache_dir=config.data.cache_dir,
            exchange_id=config.data.exchange_id,
            default_market_type=config.data.market_type,
            twelve_data_settings=config.data.twelve_data,
        )

    def _refresh_crypto(self, now: datetime) -> None:
        self._prepare(self.crypto_cfg)
        breadth = None
        try:
            breadth = fetch_breadth_basket(self.crypto_cfg.data, self.crypto_cfg.ml, use_cache=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[daemon] breadth basket fetch failed: {exc}")

        with state_db.connect() as conn:
            for symbol in self.crypto_cfg.ml.assets:
                try:
                    ohlc = _load_base_ohlc(self.crypto_cfg.ml, symbol)
                    if ohlc is None or ohlc.empty:
                        continue
                    atr = compute_atr(ohlc["high"], ohlc["low"], ohlc["close"], self.crypto_cfg.ml.atr_window)
                    last_close = float(ohlc["close"].iloc[-1])
                    last_atr = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else 0.0
                    conn.execute(
                        "UPDATE market_state SET atr = ?, atr_pct = ?, warmup_ready = 1 WHERE asset = ?",
                        (last_atr, last_atr / last_close if last_close else 0.0, symbol),
                    )
                    result = self.model1_crypto.infer(symbol, ohlc, None, breadth)
                    if result is None:
                        continue
                    self._write_signal(conn, symbol, "crypto_sniper", result, now)
                    self._persist_strategy_profiles(conn, symbol, "crypto", now)
                except Exception:  # noqa: BLE001
                    print(f"[daemon] crypto inference failed for {symbol}:\n{traceback.format_exc()}")
            conn.commit()

    def _refresh_equity_and_swing(self, now: datetime) -> None:
        self._prepare(self.equity_cfg)
        macro_df = None
        try:
            from core.ml.train import load_prepared_macro_matrix

            macro_df = load_prepared_macro_matrix(self.equity_cfg.ml)
        except Exception as exc:  # noqa: BLE001
            print(f"[daemon] equity macro matrix unavailable: {exc}")

        with state_db.connect() as conn:
            for name in self.equity_cfg.ml.assets:
                ticker = EQUITY_NAME_TO_TICKER.get(name, name)
                try:
                    ohlc = _load_base_ohlc(self.equity_cfg.ml, name)
                    if ohlc is None or ohlc.empty:
                        continue
                    atr = compute_atr(ohlc["high"], ohlc["low"], ohlc["close"], self.equity_cfg.ml.atr_window)
                    last_close = float(ohlc["close"].iloc[-1])
                    last_atr = float(atr.iloc[-1]) if np.isfinite(atr.iloc[-1]) else 0.0
                    conn.execute(
                        "UPDATE market_state SET atr = ?, atr_pct = ?, warmup_ready = 1 WHERE asset = ?",
                        (last_atr, last_atr / last_close if last_close else 0.0, ticker),
                    )
                    result = self.model1_equity.infer(name, ohlc, macro_df, None)
                    if result is not None:
                        snapshot = intraday_snapshot(ticker, result["timestamp"], result)
                        self._write_snapshot(conn, ticker, "intraday", snapshot, now)
                except Exception:  # noqa: BLE001
                    print(f"[daemon] equity model1b inference failed for {name}:\n{traceback.format_exc()}")

                try:
                    self._refresh_swing(conn, ticker, now)
                except Exception:  # noqa: BLE001
                    print(f"[daemon] swing model2 inference failed for {ticker}:\n{traceback.format_exc()}")
            conn.commit()
        self._flush_notifications()

    def _refresh_swing(self, conn, ticker: str, now: datetime) -> None:
        swing = self.swing_cfg.swing
        checkpoint = Path(swing.model_dir) / f"{ticker.lower()}_swing.pt"
        if not checkpoint.exists():
            return
        if ticker not in self._swing_models:
            self._swing_models[ticker] = load_swing_model(checkpoint, swing)
        model = self._swing_models[ticker]
        dataset = build_swing_dataset(swing.market_cache_dir, ticker, swing.daily_lookback, swing.target_horizons, swing.macro_symbols)
        if len(dataset.timestamps) == 0:
            return
        raw = model.predict(dataset.X_daily[-1:], dataset.X_entry[-1:], dataset.branch_mask[-1:])
        predictions = {f"expected_return_{h}d": float(raw["returns"][0, i]) for i, h in enumerate(HORIZONS)}
        predictions["expected_mfe"] = float(raw["mfe"][0])
        predictions["expected_mae"] = float(raw["mae"][0])
        predictions["entry_signal"] = float(raw["entry"][0])
        predictions["expected_duration_days"] = float(raw["duration"][0, -1])
        from core.ml.swing_model import swing_snapshot

        snapshot = swing_snapshot(ticker, dataset.timestamps[-1], predictions, swing)
        self._write_snapshot(conn, ticker, "swing", snapshot, now)
        self._persist_strategy_profiles(conn, ticker, "equity", now)
        if snapshot.swing_score is not None and snapshot.entry_score is not None and snapshot.swing_score > 80 and snapshot.entry_score > 80:
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
                    "knockout": aggressive.get("knockout_barrier_price"),
                    "leverage": aggressive.get("leverage"),
                })
                self._notified_swing.add(key)

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
            "alert_allowed, alert_reason, updated_at) VALUES (?,?,?,?,?,?,?,NULL,?,?,NULL,?,NULL,?,?,?) "
            "ON CONFLICT(asset, model_type) DO UPDATE SET timestamp=excluded.timestamp, score=excluded.score, "
            "direction=excluded.direction, expected_return=excluded.expected_return, "
            "expected_duration_bars=excluded.expected_duration_bars, expected_mfe=excluded.expected_mfe, "
            "expected_mae=excluded.expected_mae, entry_score=excluded.entry_score, alert_allowed=excluded.alert_allowed, "
            "alert_reason=excluded.alert_reason, updated_at=excluded.updated_at",
            (
                asset, model_type, str(result["timestamp"]), result["score"],
                result["direction"] if result["direction"] in {"LONG", "SHORT"} else "UNCERTAIN",
                result["expected_return"], result["expected_duration"], result["expected_mfe"], result["expected_mae"],
                result["entry_score"], int(result["score"] >= 80.0), "score >= 80" if result["score"] >= 80.0 else "",
                now.isoformat(),
            ),
        )

    # --- Hauptschleife -------------------------------------------------------------
    def run_forever(self) -> None:
        print(f"Live daemon starting, DB={state_db.DB_PATH}")
        try:
            while True:
                loop_start = time.monotonic()
                try:
                    self.price_tick()
                except Exception:  # noqa: BLE001
                    print(f"[daemon] price_tick failed:\n{traceback.format_exc()}")

                if time.monotonic() - self._last_signal_refresh >= SIGNAL_REFRESH_S:
                    try:
                        self.signal_refresh()
                    except Exception:  # noqa: BLE001
                        print(f"[daemon] signal_refresh failed:\n{traceback.format_exc()}")
                    self._last_signal_refresh = time.monotonic()

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, PRICE_POLL_S - elapsed))
        except KeyboardInterrupt:
            print("Daemon gracefully stopped")


def main() -> None:
    LiveDaemon().run_forever()


if __name__ == "__main__":
    main()
