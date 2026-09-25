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


def test_training_tcn_crypto_1h_yaml_loads_conservative_cuda_diagnostics() -> None:
    config = load_config("configs/training_tcn_crypto_1h.yaml")
    assert config.training_device == "cuda"
    assert config.inference_device == "cpu"
    assert config.ml.model_id == "tcn_crypto_1h"
    assert config.ml.batch_size == 2048
    assert config.ml.learning_rate == 0.0003
    assert config.ml.use_amp is False
    assert config.ml.gradient_clip_norm == 1.0
    assert tuple(config.ml.forecast_horizons) == (1, 4, 8, 12, 24)
    assert config.ml.sequence_length == 128
    assert config.ml.hidden_channels == 96
    assert config.ml.num_layers == 6
    assert config.ml.dropout == 0.35


def test_training_crypto_intraday_yaml_uses_multi_horizon_contract() -> None:
    config = load_config("configs/training_crypto_intraday.yaml")
    assert config.ml.label_horizon == 1
    assert tuple(config.ml.forecast_horizons) == (1, 4, 8, 12, 24)


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
