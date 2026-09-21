"""Live trading loop skeleton: polls candles, reuses the exact same signal logic
as the backtester, and routes orders through the maker-first execution client.

Intended to run headless on a VPS. Defaults to DRY_RUN=true (see .env.example)
so it can be exercised safely before real capital is at risk.
"""
from __future__ import annotations

import time

import pandas as pd

from core.config import TradingBotConfig
from core.data_loader import fetch_ohlcv_history, resample_ohlcv
from core.hybrid_strategy import apply_ml_gate, train_ml_gate
from core.macro import apply_macro_gate, fetch_macro_data
from core.risk import apply_dynamic_stop, dynamic_stop_distance
from core.strategy import generate_signals
from execution.exchange_client import BinanceFuturesClient


class LiveTrader:
    def __init__(self, config: TradingBotConfig, retrain_every_polls: int = 1440) -> None:
        self.config = config
        self.client = BinanceFuturesClient()
        self.current_position = 0
        self.retrain_every_polls = retrain_every_polls
        self._polls_since_retrain = 0
        self._ml_model = None
        self._macro_df = None

    def _load_recent_data(self) -> pd.DataFrame:
        cfg = self.config.data
        lookback_days = max(3, cfg.history_days // 100)  # small rolling window, warm-up only
        raw_a = fetch_ohlcv_history(
            cfg.symbol_a, cfg.base_timeframe, lookback_days, cache_dir=cfg.cache_dir, use_cache=False
        )
        raw_b = fetch_ohlcv_history(
            cfg.symbol_b, cfg.base_timeframe, lookback_days, cache_dir=cfg.cache_dir, use_cache=False
        )
        res_a = resample_ohlcv(raw_a, cfg.resample_to) if cfg.resample_to else raw_a
        res_b = resample_ohlcv(raw_b, cfg.resample_to) if cfg.resample_to else raw_b
        return pd.DataFrame({cfg.symbol_a: res_a["close"], cfg.symbol_b: res_b["close"]}).dropna()

    def poll_once(self) -> None:
        df = self._load_recent_data()
        signals = generate_signals(df, self.config.data.symbol_a, self.config.data.symbol_b, self.config.strategy)
        if signals.empty:
            return

        if self.config.ml.enabled:
            if self._ml_model is None or self._polls_since_retrain >= self.retrain_every_polls:
                # Retrain on everything up to (but excluding) the freshest bar, so the
                # model is always evaluated out-of-sample on the newest data point.
                self._ml_model = train_ml_gate(signals, signals.index[-1], self.config)
                self._polls_since_retrain = 0
            signals = apply_ml_gate(signals, self._ml_model, self.config)
            self._polls_since_retrain += 1

        if self.config.macro.enabled:
            if self._macro_df is None or self._polls_since_retrain == 0:
                self._macro_df = fetch_macro_data(self.config.macro, use_cache=False)
            signals = apply_macro_gate(signals, self.config, self._macro_df)

        spread = signals["spread"]
        entry_spread = spread.where(signals["position"].diff().fillna(signals["position"]) != 0).ffill()
        stop_distance = dynamic_stop_distance(spread, self.config.risk)
        position = apply_dynamic_stop(signals["position"], spread, entry_spread, stop_distance)

        target_position = int(position.iloc[-1])
        if target_position != self.current_position:
            self._rebalance(target_position, signals.iloc[-1])
            self.current_position = target_position

    def _rebalance(self, target_position: int, latest: pd.Series) -> None:
        symbol_a = self.config.data.symbol_a
        symbol_b = self.config.data.symbol_b
        hedge_ratio = latest["hedge_ratio"]

        side_a = "buy" if target_position == 1 else "sell"
        side_b = "sell" if target_position == 1 else "buy"

        if target_position == 0:
            print("Flattening position (maker-first, taker fallback on failure)...")
        else:
            print(f"Opening position {target_position} on {symbol_a}/{symbol_b}, hedge_ratio={hedge_ratio:.4f}")

        self.client.place_maker_order(symbol_a, side_a, amount=1.0)
        self.client.place_maker_order(symbol_b, side_b, amount=abs(hedge_ratio))

    def run_forever(self, poll_interval_s: float = 60.0) -> None:
        while True:
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive on transient errors
                print(f"[LiveTrader] error during poll: {exc}")
            time.sleep(poll_interval_s)


if __name__ == "__main__":
    trader = LiveTrader(TradingBotConfig())
    trader.run_forever()
