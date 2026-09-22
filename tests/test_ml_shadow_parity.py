"""Tests that the shadow PnL comparison reuses the EXACT same macro/stop/
cooldown/latency/funding/shared-wallet logic for both the rule and ML
variant (core.backtester.run_rule_vs_ml_backtest, core.ml.shadow.
build_shadow_log_from_backtest)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtester import run_rule_vs_ml_backtest
from core.config import TradingBotConfig
from core.ml.shadow import SHADOW_LOG_COLUMNS, build_shadow_log_from_backtest


def _synthetic_ohlc(n: int, seed: int, drift: float = 0.0002) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n, freq="1h")
    returns = rng.normal(loc=drift, scale=0.01, size=n)
    close = 100 * np.cumprod(1 + returns)
    high = close * 1.001
    low = close * 0.999
    open_ = close
    volume = rng.uniform(10, 100, size=n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def _config(tmp_path) -> TradingBotConfig:
    config = TradingBotConfig()
    config.data.symbols = ["BTC/USDT", "ETH/USDT"]
    config.macro.enabled = False  # no network fetch in tests
    config.funding.enabled = False
    config.monte_carlo.output_dir = str(tmp_path)
    config.ml.model_dir = str(tmp_path / "models")
    return config


def test_rule_and_ml_runs_share_identical_downstream_pipeline(tmp_path) -> None:
    """With no trained model (no OOS confidence available), ML confirmation
    is a no-op -- so the rule and ML runs must be byte-identical, proving
    both go through the exact same macro/stop/cooldown/latency/funding/
    shared-wallet code path (only the ML confirmation *block* can differ)."""
    config = _config(tmp_path)
    multi_ohlc = {
        "BTC/USDT": _synthetic_ohlc(400, seed=1),
        "ETH/USDT": _synthetic_ohlc(400, seed=2),
    }

    portfolio_rule, portfolio_ml = run_rule_vs_ml_backtest(multi_ohlc, config)

    pd.testing.assert_series_equal(
        portfolio_rule["strategy_return"], portfolio_ml["strategy_return"], check_names=False
    )
    per_symbol_rule = portfolio_rule.attrs["per_symbol"]
    per_symbol_ml = portfolio_ml.attrs["per_symbol"]
    for symbol in config.data.symbols:
        pd.testing.assert_series_equal(
            per_symbol_rule[symbol]["execution_position"],
            per_symbol_ml[symbol]["execution_position"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            per_symbol_rule[symbol]["leverage"], per_symbol_ml[symbol]["leverage"], check_names=False
        )


def test_build_shadow_log_from_backtest_uses_true_net_pnl_and_costs(tmp_path) -> None:
    config = _config(tmp_path)
    multi_ohlc = {
        "BTC/USDT": _synthetic_ohlc(400, seed=3),
        "ETH/USDT": _synthetic_ohlc(400, seed=4),
    }
    portfolio_rule, portfolio_ml = run_rule_vs_ml_backtest(multi_ohlc, config)
    per_symbol_rule = portfolio_rule.attrs["per_symbol"]
    per_symbol_ml = portfolio_ml.attrs["per_symbol"]

    log = build_shadow_log_from_backtest("BTC/USDT", per_symbol_rule["BTC/USDT"], per_symbol_ml["BTC/USDT"])

    expected_columns = set(SHADOW_LOG_COLUMNS) - {"timestamp"}
    assert expected_columns <= set(log.columns) | {"symbol"}
    # rule_pnl/ml_pnl must be the actual net strategy_return (fees+funding
    # already deducted), not a simplified leverage=1 recomputation.
    pd.testing.assert_series_equal(
        log["rule_pnl"], per_symbol_rule["BTC/USDT"]["strategy_return"].loc[log.index], check_names=False
    )
    pd.testing.assert_series_equal(
        log["ml_pnl"], per_symbol_ml["BTC/USDT"]["strategy_return"].loc[log.index], check_names=False
    )
    assert (log["rule_cost"] >= 0).all()
    assert (log["ml_cost"] >= 0).all()
