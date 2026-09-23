from __future__ import annotations

import pandas as pd

from core.config import TradingBotConfig
from core.ml.features import build_feature_matrix
from core.ml.features import build_breadth_features, build_higher_timeframe_features
from core.ml.train import load_prepared_macro_matrix
from train import load_config


def test_training_config_uses_four_primary_assets_and_5m_target() -> None:
    config = load_config("configs/training.yaml")
    assert config.ml.base_timeframe == "5m"
    assert config.ml.assets == ["NASDAQ100_PROXY", "SP500_PROXY", "BTC/USDT", "ETH/USDT"]
    assert config.ml.higher_timeframes == ["15m", "1h", "4h", "1d"]


def test_prepared_context_symbols_are_configured_for_ml() -> None:
    config = load_config("configs/training.yaml")
    assert {"^VIX", "^TNX", "EURUSD=X", "GC=F", "CL=F", "^GDAXI"} <= set(config.ml.macro_symbols)


def test_5m_feature_matrix_includes_all_higher_timeframes_and_macro() -> None:
    config = TradingBotConfig()
    config.ml.base_timeframe = "5m"
    config.ml.higher_timeframes = ["15m", "1h", "4h", "1d"]
    index = pd.date_range("2024-01-01", periods=2000, freq="5min", tz="UTC")
    close = pd.Series(range(100, 2100), index=index, dtype=float)
    ohlc = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0})
    macro_index = pd.date_range("2023-12-01", periods=60, freq="D", tz="UTC")
    macro = pd.DataFrame({symbol: 100.0 for symbol in config.ml.macro_symbols}, index=macro_index)
    features = build_feature_matrix(ohlc, macro, config.ml)
    for timeframe in config.ml.higher_timeframes:
        assert f"htf_{timeframe}_ema_regime" in features.columns
    for symbol in config.ml.macro_symbols:
        safe = symbol.replace("^", "").replace("=", "").replace("/", "-")
        assert f"{safe}_ret" in features.columns
    assert "macro_data_age_hours" in features.columns


def test_15m_resampling_uses_minutes_not_months() -> None:
    config = TradingBotConfig().ml
    index = pd.date_range("2024-01-01", periods=100, freq="5min", tz="UTC")
    close = pd.Series(range(100, 200), index=index, dtype=float)
    ohlc = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0})
    features = build_higher_timeframe_features(ohlc, "15m", config)
    assert len(features) >= 30


def test_breadth_naive_index_aligns_to_utc_intraday_index() -> None:
    config = TradingBotConfig().ml
    close_index = pd.date_range("2024-01-01", periods=400, freq="5min", tz="UTC")
    close = pd.Series(100.0, index=close_index)
    breadth = pd.Series(0.001, index=pd.date_range("2024-01-01", periods=400, freq="5min"))
    features = build_breadth_features(close, breadth, config)
    assert features["breadth_ret_1"].notna().sum() > 0
