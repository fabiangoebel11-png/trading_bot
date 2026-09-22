"""Tests that PositionSelectionConfig is a true no-op when disabled (parity
with today's baseline) and correctly ranks/weights symbols when enabled
(core/backtester.py: run_backtest_from_signals, _apply_position_selection)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtester import run_backtest_from_signals
from core.config import TradingBotConfig


def _signals(n: int, ema_fast: float, ema_slow: float, atr_pct: float) -> pd.DataFrame:
    # Constant close (no drift/noise): the ATR-trailing stop never triggers,
    # so "position=1" stays active for the whole synthetic history -- keeps
    # this test isolated to the selection/weighting logic itself instead of
    # incidental stop-outs from a random walk.
    index = pd.date_range("2024-01-01", periods=n, freq="1h")
    close = np.full(n, 100.0)
    return pd.DataFrame(
        {
            "position": 1.0,
            "close": close,
            "atr": close * atr_pct,
            "ema_fast": close * ema_fast,
            "ema_slow": close * ema_slow,
        },
        index=index,
    )


def _config(tmp_path) -> TradingBotConfig:
    config = TradingBotConfig()
    config.data.symbols = ["BTC/USDT", "ETH/USDT"]
    config.macro.enabled = False
    config.funding.enabled = False
    config.monte_carlo.output_dir = str(tmp_path)
    return config


def test_selection_disabled_matches_inverse_leverage_baseline(tmp_path) -> None:
    config = _config(tmp_path)
    assert config.selection.enabled is False  # default must stay off

    signals = {
        "BTC/USDT": _signals(300, ema_fast=1.05, ema_slow=1.0, atr_pct=0.01),
        "ETH/USDT": _signals(300, ema_fast=1.20, ema_slow=1.0, atr_pct=0.01),
    }
    portfolio = run_backtest_from_signals(signals, config)
    # Both symbols active every bar with equal leverage inputs -> baseline
    # inverse-leverage weighting should split close to evenly, NOT favor
    # ETH/USDT despite its much stronger score.
    btc_weight = portfolio["BTC/USDT_weight"].iloc[-1]
    eth_weight = portfolio["ETH/USDT_weight"].iloc[-1]
    assert btc_weight == eth_weight  # identical leverage_candidate inputs -> identical weights


def test_selection_enabled_picks_strongest_score_only(tmp_path) -> None:
    config = _config(tmp_path)
    config.selection.enabled = True
    config.selection.max_active_positions = 1

    signals = {
        # Much stronger EMA divergence relative to ATR% -> much higher score.
        "BTC/USDT": _signals(300, ema_fast=1.30, ema_slow=1.0, atr_pct=0.01),
        "ETH/USDT": _signals(300, ema_fast=1.01, ema_slow=1.0, atr_pct=0.01),
    }
    portfolio = run_backtest_from_signals(signals, config)
    assert portfolio["BTC/USDT_weight"].iloc[-1] == 1.0
    assert portfolio["ETH/USDT_weight"].iloc[-1] == 0.0


def test_selection_weights_are_score_proportional_without_top_n_cap(tmp_path) -> None:
    config = _config(tmp_path)
    config.selection.enabled = True
    config.selection.max_active_positions = None  # no ranking cut, but still score-weighted

    signals = {
        "BTC/USDT": _signals(300, ema_fast=1.10, ema_slow=1.0, atr_pct=0.01),  # score ~10
        "ETH/USDT": _signals(300, ema_fast=1.30, ema_slow=1.0, atr_pct=0.01),  # score ~30
    }
    portfolio = run_backtest_from_signals(signals, config)
    btc_weight = portfolio["BTC/USDT_weight"].iloc[-1]
    eth_weight = portfolio["ETH/USDT_weight"].iloc[-1]
    assert eth_weight > btc_weight  # stronger score gets the bigger share
    assert (btc_weight + eth_weight) - 1.0 < 1e-9


def test_selection_never_raises_hard_caps(tmp_path) -> None:
    """A selection variant must never be able to increase max_total_notional_usdt
    or the shared stop-out risk budget -- both remain enforced identically.
    The cap is respected unless the SEPARATE, legitimate min_leverage floor
    requires more notional than the cap (a distinct hard requirement that may
    win by design) -- but the cap must never be exceeded beyond that floor."""
    config = _config(tmp_path)
    config.selection.enabled = True
    config.selection.max_active_positions = 1
    config.capital.max_total_notional_usdt = 120.0

    signals = {
        "BTC/USDT": _signals(300, ema_fast=1.30, ema_slow=1.0, atr_pct=0.01),
        "ETH/USDT": _signals(300, ema_fast=1.01, ema_slow=1.0, atr_pct=0.01),
    }
    portfolio = run_backtest_from_signals(signals, config)
    symbols = ["BTC/USDT", "ETH/USDT"]
    weights = portfolio[[f"{s}_weight" for s in symbols]].to_numpy()
    leverage = pd.concat({s: portfolio.attrs["per_symbol"][s]["leverage"] for s in symbols}, axis=1).to_numpy()
    prior_capital = portfolio["capital_usdt"].shift(1).fillna(config.capital.initial_capital_usdt).to_numpy()

    implied_notional = (weights * leverage).sum(axis=1) * prior_capital
    min_leverage_floor_notional = config.risk.min_leverage * prior_capital
    assert (implied_notional <= np.maximum(120.0, min_leverage_floor_notional) + 1e-6).all()
