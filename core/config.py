"""Central configuration for the stat-arb pairs trading system.

All tunable parameters live here so that backtests, walk-forward validation
and live execution use exactly the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    symbol_a: str = "BTC/USDT"
    symbol_b: str = "ETH/USDT"
    exchange_id: str = "binance"
    market_type: str = "swap"  # USDT-M perpetual futures
    base_timeframe: str = "5m"  # smallest timeframe supported by Binance we fetch
    resample_to: str = "10min"  # target trading timeframe (pandas offset alias)
    history_days: int = 1095  # >= 3 years
    cache_dir: str = "data"


@dataclass
class StrategyConfig:
    hedge_ratio_window: int = 144  # rolling OLS window (candles) -> 24h @ 10m
    zscore_window: int = 72  # rolling mean/std window -> 12h @ 10m
    entry_z: float = 2.5
    exit_z: float = 0.25
    stop_loss_z: float = 4.5
    trend_filter_window: int = 360  # SMA window used to detect trending regimes
    trend_threshold: float = 0.05  # relative distance from SMA that disables entries


@dataclass
class RiskConfig:
    base_leverage: float = 2.0
    max_leverage: float = 4.0
    min_leverage: float = 2.0
    vol_target_annualized: float = 0.20  # target annualized spread volatility
    atr_window: int = 48
    atr_stop_multiplier: float = 3.0  # dynamic stop distance = multiplier * ATR(spread)


@dataclass
class ExecutionConfig:
    maker_fee: float = 0.0002  # 0.02%, realistic Binance USDT-M maker fee w/ BNB discount
    taker_fee: float = 0.0005  # 0.05%, used as fallback if maker fill fails
    latency_candles: int = 1  # worst-case: signal computed on candle close, filled next candle
    maker_fill_probability: float = 0.85  # probability a resting limit order gets filled
    slippage_bps_on_taker_fallback: float = 2.0


@dataclass
class WalkForwardConfig:
    train_size: int = 4320  # ~30 days @ 10m
    test_size: int = 1440  # ~10 days @ 10m
    step_size: int = 1440  # slide forward by the test window (no overlap)
    min_folds: int = 3
    n_jobs: int = 1  # 1 = sequential; -1 = use all Ryzen 7 cores (ProcessPoolExecutor)


@dataclass
class MLConfig:
    """Config for the ML gate that filters the raw stat-arb signal by predicted
    mean-reversion probability. Model is retrained per walk-forward fold on the
    fold's training segment only, so it never sees test-segment labels."""

    enabled: bool = True
    model_type: str = "gbm"  # "gbm" (CPU, sklearn HistGradientBoosting) | "lstm" (GPU, torch)
    feature_lookback: int = 12  # candles used for momentum/volatility features
    label_horizon: int = 12  # candles ahead used to define the reversion label
    reversion_shrink: float = 0.5  # label=1 if |z_t+h| <= shrink * |z_t|
    min_abs_zscore_for_label: float = 1.0  # only label bars with a meaningful excursion
    proba_threshold: float = 0.55  # gate: only trade if P(reversion) >= threshold
    random_state: int = 42
    # LSTM-specific (only used when model_type == "lstm", trained on RTX 3070 if available)
    lstm_sequence_length: int = 24
    lstm_hidden_size: int = 32
    lstm_epochs: int = 15
    lstm_learning_rate: float = 1e-3
    lstm_batch_size: int = 512  # RTX 3070 (8GB VRAM) comfortably fits this for small feature counts
    lstm_use_amp: bool = True  # mixed-precision training via torch.cuda.amp on CUDA devices


@dataclass
class MacroConfig:
    """Optional multi-asset regime filter: blocks/pauses trading when external markets
    (equity indices, volatility index) signal a systemic risk-off regime. Disabled by
    default so the core crypto pipeline has no hard dependency on external data feeds."""

    enabled: bool = False
    symbols: list[str] = field(default_factory=lambda: ["^GSPC", "^VIX"])  # yfinance tickers
    lookback_days: int = 730
    drawdown_window: int = 20  # trading days
    drawdown_threshold: float = -0.08  # equity index drawdown from rolling high that triggers risk-off
    vix_zscore_window: int = 60
    vix_zscore_threshold: float = 2.0  # VIX rolling z-score spike that triggers risk-off
    reporting_lag_days: int = 1  # daily close only usable from the next day onward (no look-ahead)
    cache_dir: str = "data"


@dataclass
class TradingBotConfig:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)
