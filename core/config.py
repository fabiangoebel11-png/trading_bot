"""Central configuration for the crypto trend-following / breakout system.

All tunable parameters live here so that backtests, walk-forward validation
and live execution use exactly the same numbers.

Note: the original mean-reversion pairs-trading config (hedge ratio / z-score
based ``StrategyConfig``) was retired after a Monte Carlo random-start stress
test falsified it (0% profitable windows). ``TrendConfig`` replaces it; see
``core/strategy.py`` for the rationale.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repo root (core/config.py -> core/ -> repo root), so cache/model paths are
# resolved robustly regardless of the process's current working directory
# (cron/systemd/Task Scheduler on a freshly cloned machine may launch from a
# different cwd than an interactive shell).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_data_dir() -> str:
    return str(_PROJECT_ROOT / "data")


def _default_model_dir() -> str:
    return str(_PROJECT_ROOT / "data" / "models")


def _default_market_cache_dir() -> str:
    return str(_PROJECT_ROOT / "data" / "market")


def _default_monte_carlo_dir() -> str:
    return str(_PROJECT_ROOT / "data" / "monte_carlo")


def _default_live_state_path() -> str:
    return str(_PROJECT_ROOT / "data" / "live_state.json")


def _default_audit_log_path() -> str:
    return str(_PROJECT_ROOT / "data" / "live_audit.jsonl")


def _default_shadow_log_path() -> str:
    return str(_PROJECT_ROOT / "data" / "ml_shadow_log.csv")


def _default_portfolio_shadow_log_path() -> str:
    return str(_PROJECT_ROOT / "data" / "ml_portfolio_shadow_log.csv")


@dataclass
class DataConfig:
    # Universe strictly capped at 3 crypto majors (BTC/ETH + one broad-market proxy,
    # SOL/USDT) -- enough for cross-asset diversification/risk-parity without
    # diluting focus or ballooning correlated-tail risk across a long tail of alts.
    symbols: list[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    exchange_id: str = "binance"
    market_type: str = "swap"  # USDT-M perpetual futures
    base_timeframe: str = "1h"  # stabler, less noisy timeframe than 10m for trend-following
    resample_to: str | None = None  # optional further downsample, e.g. "4h"
    history_days: int = 2500  # >= 3 years
    cache_dir: str = field(default_factory=_default_data_dir)
    twelve_data: dict[str, object] = field(default_factory=lambda: {
        "enabled": True,
        "api_key_env": "TWELVE_DATA_API_KEY",
        "max_requests_per_day": 800,
        "max_requests_per_minute": 8,
        "safe_requests_per_minute": 7,
        "primary_interval": "5min",
        "derive_higher_timeframes": True,
        "max_retries": 3,
        "request_timeout_s": 30,
    })

    def __post_init__(self) -> None:
        if len(self.symbols) > 3:
            raise ValueError(
                f"Universe strictly limited to at most 3 symbols, got {self.symbols!r}. "
                "Widening the book dilutes the risk-parity weighting and macro-filter design."
            )


@dataclass
class TrendConfig:
    """Donchian-channel breakout with an EMA regime filter: only take breakouts in
    the direction of the prevailing trend. Classic managed-futures/CTA-style setup,
    designed to profit from (rather than fight) crypto's strong directional moves."""

    fast_ma_window: int = 20  # EMA regime filter, fast leg
    slow_ma_window: int = 100  # EMA regime filter, slow leg
    donchian_entry_window: int = 55  # breakout channel for new entries (Turtle-style)
    donchian_exit_window: int = 20  # tighter channel for exits (locks in trend reversals)
    atr_window: int = 14  # classic ATR period, used for the volatility filter below
    min_atr_pct: float = 0.0015  # skip near-zero-volatility chop (avoid overtrading dead markets)
    allow_short: bool = True


@dataclass
class RiskConfig:
    base_leverage: float = 2.0
    max_leverage: float = 4.0
    min_leverage: float = 2.0
    vol_target_annualized: float = 0.35  # trend systems run hotter vol targets than mean-reversion
    atr_window: int = 24  # ~1 trading day of hourly bars
    atr_stop_multiplier: float = 3.0  # trailing stop distance = multiplier * ATR(price)


@dataclass
class ExecutionConfig:
    maker_fee: float = 0.0002  # 0.02%, realistic Binance USDT-M maker fee w/ BNB discount
    taker_fee: float = 0.0005  # 0.05%, used as fallback if maker fill fails
    latency_candles: int = 1  # worst-case: signal computed on candle close, filled next candle
    maker_fill_probability: float = 0.85  # probability a resting limit order gets filled
    slippage_bps_on_taker_fallback: float = 2.0

    # --- Exchange leverage/margin-mode setup (live only, no-op in backtests) ---
    # Set ONCE per symbol at process startup, never per order: the exchange's
    # leverage setting is just a ceiling on what the account is *allowed* to
    # hold, not the actual leverage used. Real exposure is controlled
    # exclusively through order ``amount`` (see ``core.risk``/``core.
    # backtester`` vol-targeted + risk-capped leverage, always well below this
    # ceiling by construction: ``RiskConfig.max_leverage`` defaults to 4x).
    # Cross-margin: losses/margin are shared across the whole wallet (matches
    # this system's shared-wallet risk model, see ``CapitalConfig``) instead
    # of isolated per-symbol margin silently liquidating one position while
    # the rest of the wallet sits untouched.
    exchange_margin_mode: str = "cross"
    exchange_leverage_ceiling: int = 10  # safe, generous headroom above RiskConfig.max_leverage (4x default)

    # Real exchanges (Bybit/Binance) only allow a discrete set of leverage
    # values per symbol, not an arbitrary float like the vol-targeted "2.37x"
    # the risk engine computes -- the actually *set* exchange leverage must be
    # read from that symbol's real market metadata (``CCXTExchangeClient.
    # fetch_leverage_brackets``) and rounded to the nearest allowed step
    # (``core.risk.select_discrete_exchange_leverage``). This list is ONLY a
    # degrade-gracefully fallback for when the exchange doesn't expose
    # brackets via ccxt (or a network hiccup) -- never hardcode a specific
    # exchange's real tiers here.
    fallback_leverage_steps: list[float] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]
    )

    # Account-level safety controls. ``trading_enabled`` is the manual kill
    # switch: existing positions remain protected/managed, but no new
    # exposure may be opened while it is false.
    trading_enabled: bool = True
    max_daily_loss_pct: float = 0.03
    max_account_drawdown_pct: float = 0.10
    reconciliation_interval_s: float = 60.0
    state_path: str = field(default_factory=_default_live_state_path)
    audit_log_path: str = field(default_factory=_default_audit_log_path)


@dataclass
class PositionSelectionConfig:
    """Opt-in alternative to trading every currently active (non-flat) signal
    at once: rank active symbols by a fixed, causal conviction score and only
    trade the strongest ``max_active_positions`` of them, weighted
    proportional to score instead of inverse-leverage.

    ``score_i = |ema_fast_i/ema_slow_i - 1| / max(atr_pct_i, score_epsilon)``
    -- trend-regime strength (how far the fast/slow EMA regime filter has
    diverged, both already computed causally by
    ``core.strategy.generate_trend_signals``) divided by ATR% (already
    computed there too) -- no new indicators, purely a ranking of existing,
    already-vetted causal columns.

    Disabled by default: this is a NEW strategy variant that must be
    validated against the current all-active-signals baseline on identical
    Monte Carlo windows (``research/compare_position_selection.py``) before a
    human manually enables it -- mirrors exactly the same
    validate-before-promote pattern already used for the ML shadow layer
    (``core/config.py: TrendMLConfig.production_enabled``).
    """

    enabled: bool = False
    max_active_positions: int | None = None  # None = no cap (identical to current behavior even if enabled)
    score_epsilon: float = 1e-9  # avoid divide-by-zero on a near-zero ATR%


@dataclass
class FundingConfig:
    """Historical perpetual-swap funding-rate costs, applied only on actual funding
    event bars (~every 8h) using the position/leverage held at that moment."""

    enabled: bool = True
    cache_dir: str = field(default_factory=_default_data_dir)


@dataclass
class WalkForwardConfig:
    train_size: int = 2160  # ~90 days @ 1h
    test_size: int = 720  # ~30 days @ 1h
    step_size: int = 720  # slide forward by the test window (no overlap)
    min_folds: int = 3
    n_jobs: int = 1  # 1 = sequential; -1 = use all Ryzen 7 cores (ProcessPoolExecutor)


@dataclass
class MacroConfig:
    """Macro regime filter: uses liquid, near-24h-tradable index futures (e.g. S&P
    500 E-mini) as a directional trend/regime confirmation and systemic risk-off
    trade-blocker for the crypto book. Enabled by default -- unlike the retired
    mean-reversion pipeline, macro regime confirmation is now a core part of the
    trend-following edge, not an optional add-on."""

    enabled: bool = True
    symbols: list[str] = field(default_factory=lambda: ["ES=F", "^VIX"])  # yfinance tickers
    fallback_symbol: str = "^GSPC"  # used if the E-mini futures ticker has no/insufficient data
    # Broad, low-turnover world-equity proxy (iShares MSCI World ETF) -- a second,
    # independent read on the global risk-on/risk-off climate beyond the US-centric
    # S&P E-mini, blended as a *soft* confirmation (see core/macro.py) rather than
    # a hard gate, to avoid overfitting the filter to one index's quirks.
    world_symbol: str = "URTH"
    # Must cover at least as much history as DataConfig.history_days (now a
    # dynamic-universe outer join reaching back to BTC/ETH's earliest available
    # data, not just the youngest listed symbol) -- otherwise apply_macro_gate's
    # neutral-default fill (risk_off=False/trend_bias=0) silently disables the
    # macro filter for the entire older stretch of the backtest, which would go
    # unnoticed since it's not an error, just weaker regime confirmation.
    # ES=F/^VIX/URTH all have decades of yfinance history, so this is cheap.
    lookback_days: int = 2600
    trend_fast_window: int = 20  # daily-bar EMA regime filter on the macro asset
    trend_slow_window: int = 100
    counter_trend_scale: float = 0.5  # scale crypto position size when it opposes the macro trend
    drawdown_window: int = 20  # trading days
    drawdown_threshold: float = -0.08  # equity index drawdown from rolling high that triggers risk-off
    vix_zscore_window: int = 60
    vix_zscore_threshold: float = 2.0  # VIX rolling z-score spike that triggers risk-off
    reporting_lag_days: int = 1  # daily close only usable from the next day onward (no look-ahead)
    cache_dir: str = field(default_factory=_default_data_dir)


@dataclass
class MonteCarloConfig:
    """Random-start-date stress test: resamples many random contiguous windows from
    a single full-history backtest to check whether performance depends on when you
    happen to start counting (an overlapping block-bootstrap, not an IID Monte Carlo
    -- see core/monte_carlo.py for the caveat)."""

    n_runs: int = 1000
    window_days: int = 365  # length of each simulated window
    random_seed: int = 42
    use_gpu: bool = True  # batched tensor computation on the RTX 3070 if available
    output_dir: str = field(default_factory=_default_monte_carlo_dir)


@dataclass
class TrendMLConfig:
    """Trend-continuation confirmation model (GPU, RTX 3070): a *new*, independent
    config from the retired mean-reversion ``MLConfig`` (removed from this file;
    see ``core/hybrid_strategy.py`` / ``core/ml/*`` for the dead reversion-gate
    code that still references it and is not runnable as-is).

    Runs on the *same* 1h timeframe as the rule-based Donchian/EMA execution
    logic (see ``DataConfig.base_timeframe``) -- originally this trained on
    raw 5m candles instead, on the theory that unsmoothed noise would produce
    a more robust confirmation signal. Three independent Monte Carlo stress
    tests (Session 2026-09-22: standard 1h, deep-history 1h, and a dedicated
    5m high-resolution run) falsified that: even after correctly time-scaling
    every bar-count window by 12x and volatility-scaling the ATR stop/filter
    by sqrt(12), the 5m execution timeframe was catastrophic (median Sharpe
    -0.94, -66% mean return) purely from whipsaw/fee/slippage death-by-a-
    thousand-cuts, while 1h execution was consistently profitable (median
    Sharpe +0.79..+0.82) across both the standard and deep-history (2017+)
    windows. A confirmation model trained on 5m candles while the only
    tradable execution timeframe is 1h is an internal contradiction (the
    model's noise-level features can never be acted on profitably anyway) --
    unifying both onto 1h resolves it. ``atr_window`` (14) is reused unchanged
    (now literally matches ``TrendConfig.atr_window`` at the same 14h window).
    ``label_horizon``/``sequence_length``/the network depth *were* explicitly
    retuned for 1h afterwards (Session 2026-09-22, Teil "ML-Optimierung fuer
    1h"): a 5m-era 64-bar/4-layer network only had a ~3-day effective horizon
    at 5m resolution but a mere ~2.7-day one at 1h once the timeframe changed
    -- deliberately widened rather than left as coincidentally-reused numbers
    that happened to look unchanged (see the field comments below for the
    exact new values and rationale).

    Overfitting to that noise is fought on three independent fronts:
      1. Labels are a triple-barrier touch (take-profit/stop-loss/time), not a
         simple fixed-horizon return, and cross-validation is *purged* (drop
         training rows whose label window overlaps the test fold) with an
         *embargo* gap after each test fold -- both prevent the leakage that a
         naive walk-forward split gets wrong with overlapping-horizon labels.
      2. The model itself (``core/ml/tcn_model.py``) is a dilated causal CNN
         with dropout + weight decay (AdamW), gradient clipping, label
         smoothing, early stopping on a held-out validation slice, and a
         causal attention-pooling head (still no future leakage: every pooled
         timestep only ever encoded its own causal past).
      3. Features include cross-asset macro context (S&P 500 / Nasdaq 100 / MSCI
         World / VIX) causally lagged, so the model has more to work with than
         BTC/ETH price action alone and is less likely to just memorize it.
    """

    enabled: bool = False
    base_timeframe: str = "5m"  # primary prediction timeframe; higher frames are context
    # Raised from 1095 to match DataConfig's 2500-day cap: ``core/ml/train.py:
    # load_ml_ohlc`` intersects all 3 symbols' timestamps, so this is only an
    # upper bound -- it naturally truncates to SOL/USDT's real listing date
    # (~2020-08-11), giving the TCN the 2020/2021 bull, the 2022 bear and the
    # 2023-24 recovery instead of only the last 3 years. No fabricated
    # pre-listing data for any symbol. At 1h (vs. the previous 5m) this is also
    # ~12x fewer candles per symbol to fetch/train on.
    history_days: int = 2500

    # Cross-asset macro context, causally lagged (see core/ml/features.py).
    macro_symbols: list[str] = field(
        default_factory=lambda: ["^VIX", "^TNX", "EURUSD=X", "GC=F", "CL=F", "^GDAXI"]
    )
    # Must cover at least as much history as ``history_days`` above -- was
    # left at 1095 while ``history_days`` was raised to 2500 (Teil 5),
    # exactly the same class of bug already fixed once for the rule-based
    # ``MacroConfig.lookback_days`` (Teil 6). The consequence was worse here
    # than a "silently weaker filter": ``core/ml/train.py: load_ml_ohlc`` +
    # ``build_row_level_dataset`` join technical features with macro features
    # via ``dropna()`` -- every row older than the macro lookback has NaN
    # macro columns and gets dropped entirely, silently truncating the
    # claimed 2500-day training set down to ~1095 days (no 2020/2021 bull
    # regime in training data despite the docstring above claiming otherwise).
    # Raised to match ``DataConfig.history_days``'s buffer (2600, same value
    # already used for the rule-based macro filter) -- ES=F/VIX/URTH have
    # decades of yfinance history, so this is cheap.
    macro_lookback_days: int = 2600
    macro_reporting_lag_days: int = 1
    macro_cache_dir: str = field(default_factory=_default_data_dir)

    # Multi-timeframe regime context (Session 2026-09-22, Teil 11): higher-
    # timeframe candles resampled from the SAME 1h OHLC already loaded/fetched
    # for this symbol -- zero extra data feeds, so this is directly
    # implementable live (``core/ml/features.py:
    # build_higher_timeframe_features`` just resamples whatever 1h history
    # ``LiveTrader``/the training pipeline already has in memory). Deliberately
    # only *higher* timeframes (4h, 1d), never lower ones (e.g. 5m): the
    # 2026-09-22 Monte Carlo stress test already falsified 5m as a source of
    # tradeable signal (whipsaw/noise/fee death) for THIS strategy family, and
    # reintroducing it here as a feature input would (a) risk teaching the
    # model to key off the same noise, and (b) require a second, independently
    # fetched/cached live data stream just for feature-building -- extra
    # infra/fragility for dubious benefit. Higher timeframes are the opposite:
    # strictly coarser/less noisy, add genuinely new multi-scale trend
    # information (a classic CTA technique -- "check the daily chart before
    # trusting the hourly breakout"), and are free to compute from data
    # already in hand. Reuses the exact same EMA/Donchian/ATR/RSI window
    # lengths as the 1h technical features (no new tunable numbers, just new
    # resolutions of already-vetted ones -- see the feature-builder docstring).
    higher_timeframes: list[str] = field(default_factory=lambda: ["15m", "1h", "4h", "1d"])

    # Cross-coin systemic-noise context ("rest of the crypto market" proxy).
    # TOTAL2/TOTAL3 market-cap indices are not available via ccxt/yfinance, so
    # a basket of liquid majors *outside* the traded BTC/ETH/SOL universe is
    # fetched instead (same ccxt/cache path as the OHLCV loader) and reduced to
    # a single equal-weight breadth return series. Deliberately NOT fed into
    # the model as raw per-coin prices/returns (those would be near-duplicates
    # of BTC/ETH/SOL and destabilize the TCN with collinear inputs) -- only a
    # handful of already-decorrelating derived features are used (the basket's
    # own return/vol regime, and this symbol's rolling correlation with it),
    # see core/ml/features.py:build_breadth_features.
    breadth_enabled: bool = True
    breadth_symbols: list[str] = field(
        default_factory=lambda: ["BNB/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "LINK/USDT"]
    )
    breadth_corr_window: int = 60  # now 60h (2.5 days) instead of 5h at 5m
    breadth_vol_window: int = 288  # now 288h (12 days) instead of 24h at 5m
    breadth_zscore_window: int = 288  # now 288h (12 days) instead of 24h at 5m

    # Triple-barrier labeling (López de Prado style): label = which barrier is
    # touched first within `label_horizon` bars -- up (+1), down (-1) or neither
    # (0, time barrier). Barrier distance scales with ATR so it adapts to
    # regime volatility instead of using a fixed % target.
    atr_window: int = 14
    # 72h (3 days) horizon: widened from 48h now that the model is native 1h
    # (deeper context/receptive field below can actually inform a further-out
    # touch outcome; compute cost is irrelevant per user direction).
    label_horizon: int = 72
    barrier_atr_multiple: float = 2.0
    stop_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 2.0
    opportunity_loss_weight: float = 1.0
    return_loss_weight: float = 1.0
    duration_loss_weight: float = 0.25
    mfe_loss_weight: float = 0.25
    mae_loss_weight: float = 0.25
    direction_loss_weight: float = 1.0
    return_target_scale: float = 1.0
    excursion_target_scale: float = 1.0
    collapse_warning_fraction: float = 0.98
    assets: list[str] = field(default_factory=lambda: ["NASDAQ100_PROXY", "SP500_PROXY", "BTC/USDT", "ETH/USDT"])
    optional_assets: list[str] = field(default_factory=lambda: ["DAX"])
    asset_providers: dict[str, str] = field(
        default_factory=lambda: {
            "NASDAQ-100": "yfinance",
            "S&P 500": "yfinance",
            "DAX": "yfinance",
            "BTC/USDT": "ccxt",
            "ETH/USDT": "ccxt",
        }
    )
    market_assets: dict[str, dict[str, object]] = field(
        default_factory=lambda: {
            "NASDAQ-100": {"provider": "yfinance", "symbol": "^NDX", "provider_symbols": {"eodhd": "NDX.INDX"}, "market_type": "equity_index"},
            "S&P 500": {"provider": "yfinance", "symbol": "^GSPC", "provider_symbols": {"eodhd": "GSPC.INDX"}, "market_type": "equity_index"},
            "DAX": {"provider": "yfinance", "symbol": "^GDAXI", "provider_symbols": {"eodhd": "GDAXI.INDX"}, "market_type": "equity_index"},
            "BTC/USDT": {"provider": "ccxt", "symbol": "BTC/USDT", "market_type": "swap"},
            "ETH/USDT": {"provider": "ccxt", "symbol": "ETH/USDT", "market_type": "swap"},
        }
    )
    context_assets: dict[str, dict[str, object]] = field(
        default_factory=lambda: {
            "VIX": {"provider": "yfinance", "symbol": "^VIX", "market_type": "equity_index"},
            "US10Y": {"provider": "yfinance", "symbol": "^TNX", "market_type": "rate"},
            "EURUSD": {"provider": "yfinance", "symbol": "EURUSD=X", "market_type": "fx"},
            "GOLD": {"provider": "yfinance", "symbol": "GC=F", "market_type": "commodity"},
            "WTI": {"provider": "yfinance", "symbol": "CL=F", "market_type": "commodity"},
        }
    )
    market_cache_dir: str = field(default_factory=_default_market_cache_dir)
    timeframe_history_days: dict[str, int] = field(
        default_factory=lambda: {"5m": 1095, "15m": 1825, "1h": 3650, "4h": 5475, "1d": 14600}
    )
    context_timeframes: list[str] = field(default_factory=lambda: ["1d"])
    allow_history_shortfall: bool = True
    timeframes: list[str] = field(default_factory=lambda: ["5m", "15m", "1h", "4h", "1d"])
    feature_columns: list[str] | None = None
    mfe_loss_weight: float = 1.0
    mae_loss_weight: float = 1.0
    class_weight_mode: str = "balanced"
    focal_gamma: float = 2.0
    entry_quality_mode: str = "payoff"
    entry_quality_loss_weight: float = 1.0
    optimizer: str = "adamw"
    scheduler: str | None = None
    kernel_size: int = 3
    calibration_method: str | None = "temperature"
    model_profile: str = "mixed_asset_intraday"
    relative_return_centering: bool = True
    baseline_logistic_enabled: bool = True
    baseline_gradient_boosting_enabled: bool = True
    score_thresholds: list[float] = field(default_factory=lambda: [80.0, 90.0])
    train_end: str = "2024-12-31 23:59:59"
    validation_start: str = "2025-01-01 00:00:00"
    validation_end: str = "2025-12-31 23:59:59"
    test_start: str = "2026-01-01 00:00:00"
    test_end: str | None = None
    purge_hours: int = 6
    use_asset_specific_train_start: bool = True
    use_latest_available_test_end: bool = True
    minimum_train_samples: int = 1000
    minimum_validation_samples: int = 100
    minimum_test_samples: int = 100
    shadow_start: str = "2026-01-01"
    walk_forward_windows: list[dict[str, str]] = field(default_factory=list)

    # Model / training -- retuned specifically for 1h-native execution
    # (previously 5m-era defaults, see module docstring in core/ml/tcn_model.py).
    # 128 bars (~5.3 days) lookback, wide enough to cover the deeper network's
    # receptive field below (~253h/~10.5 days theoretical max, most of the
    # attention weight in practice concentrates on the more recent portion).
    sequence_length: int = 128
    hidden_channels: int = 96
    num_layers: int = 6  # dilations 1,2,4,8,16,32 -> receptive field ~253 bars (~10.5 days at 1h)
    dropout: float = 0.35  # slightly higher than the 5m-era 0.3: a deeper/wider net needs more regularization
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 4096
    max_epochs: int = 100  # training time is not a constraint (RTX 3070); early stopping still governs actual length
    early_stopping_patience: int = 10  # a deeper network converges/plateaus more slowly than the old 4-layer net
    val_fraction: float = 0.15  # tail slice of each fold's train split, used only for early stopping
    label_smoothing: float = 0.05
    grad_clip_norm: float = 1.0
    use_amp: bool = True  # mixed precision on the RTX 3070; no-op on CPU

    # Purged walk-forward cross-validation.
    n_splits: int = 5
    embargo_fraction: float = 0.01  # fraction of total bars purged/embargoed around each test fold

    random_state: int = 42
    model_dir: str = field(default_factory=_default_model_dir)

    # Fraction of the ML sizing adjustment applied to the raw rule position:
    # 0.0 leaves the rule signal untouched; 1.0 is the original full ML
    # confirmation behavior. Keep ML disabled by default; candidate weights
    # are evaluated by research/compare_monte_carlo_ml.py before any use.
    confirmation_weight: float = 1.0

    # Experimental deployment gate for ML confirmation. It reuses the fixed,
    # causal regime definitions evaluated in research: only crash or strong-
    # trend bars may be ML-scaled; sideways bars retain the raw rule signal.
    # Disabled until the paired out-of-sample comparison validates it.
    regime_gate_enabled: bool = False
    regime_ema_span: int = 100
    regime_trend_strength: float = 0.05
    regime_drawdown_lookback_hours: int = 24 * 30
    regime_crash_drawdown: float = -0.15

    # Exponential recency weighting of the training loss (standard
    # quant/HFT practice for a non-stationary market): a bar's weight is
    # ``0.5 ** (age_days / sample_weight_half_life_days)``, so the 2020/2021
    # data extended above still teaches the model the big-picture regime
    # shapes without letting that era's now-stale microstructure dominate
    # gradients over the current regime. Purely a function of *time*, never
    # of the realized label/return/direction, so bull and bear bars of the
    # same age are weighted identically -- no directional/profit bias. Set to
    # ``None`` or ``0`` to disable (uniform weighting, prior behavior).
    sample_weight_half_life_days: float | None = 365.0

    # --- Shadow-mode / manual production gate ---
    # ``enabled`` (above) only controls whether the ML confirmation layer is
    # *computed and logged* at all -- it never by itself lets ML touch real
    # order sizing. Real money is only ever affected when BOTH ``enabled``
    # and ``production_enabled`` are True. While ``production_enabled`` is
    # False, the rule strategy alone drives live orders and the ML layer runs
    # purely as an audited shadow (see ``core/ml/shadow.py``): every bar's
    # rule position, ML position, confidence, regime, ATR, price, cost and
    # PnL are persisted to ``shadow_log_path`` for later comparison.
    #
    # This flag must ONLY ever be flipped by a human, manually, after
    # ``core/ml/promotion.py: evaluate_promotion`` has returned
    # ``PromotionResult.passed == True`` on a genuine chronological holdout
    # (real, previously-unseen data collected *after* the model/hyperparameters
    # were frozen -- see ``core/ml/holdout.py``). No code in this repository
    # is permitted to set this programmatically; ``evaluate_promotion`` is
    # read-only and returns a recommendation, never a side effect.
    production_enabled: bool = False
    shadow_log_path: str = field(default_factory=_default_shadow_log_path)
    # Portfolio-level (shared-wallet) shadow log: one row per poll, each side
    # (rule/ML) driven by its OWN hypothetical capital/cold-start/absolute-cap
    # trajectory (see core/ml/shadow.py: PortfolioShadowSimulator) rather than
    # the real account's fetched equity, which only ever reflects whichever
    # variant is actually being traded. This -- not the per-symbol
    # shadow_log_path above -- is what a genuine promotion decision must use,
    # since real money trades as ONE shared wallet, not N independent books.
    portfolio_shadow_log_path: str = field(default_factory=_default_portfolio_shadow_log_path)


@dataclass
class SwingMLConfig:
    """Independent daily-primary multi-resolution swing model configuration."""

    enabled: bool = True
    assets: list[str] = field(default_factory=lambda: ["QQQ", "SPY"])
    market_cache_dir: str = field(default_factory=_default_market_cache_dir)
    model_dir: str = field(default_factory=lambda: str(_PROJECT_ROOT / "models" / "checkpoints"))
    daily_lookback: int = 90
    intraday_lookback_1h: int = 48
    intraday_lookback_5m: int = 48
    target_horizons: list[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 30
    dropout: float = 0.2
    hidden_dimensions: list[int] = field(default_factory=lambda: [128, 64])
    swing_score_weight: float = 0.70
    entry_score_weight: float = 0.30
    max_model2_leverage: float = 10.0
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    purge_days: int = 20
    feature_version: str = "swing-v1"
    target_version: str = "swing-targets-v1"
    macro_reporting_lag_days: int = 1
    macro_symbols: list[str] = field(default_factory=lambda: ["^VIX", "^TNX", "EURUSD=X", "GC=F", "CL=F", "^GDAXI"])


@dataclass
class CapitalConfig:
    """Money management for a single **shared** wallet traded across up to 3
    coins simultaneously (not 3 independent sub-accounts): simulates trading a
    fixed real-money account instead of abstract percentage returns, so
    drawdowns/position sizes can be reasoned about in EUR/USDT before any
    capital is actually put live.

    Position sizing is risk-based, not just vol-targeted: leverage is capped so
    that if *every currently open position's* ATR-trailing stop were hit on the
    same bar, the combined loss cannot exceed ``max_portfolio_risk_pct`` of the
    *current* shared equity (compounds losses down, not a fixed-euro risk --
    classic anti-ruin sizing, but budgeted across the whole wallet rather than
    per-symbol). Capital allocation across simultaneously active coins is
    inverse-volatility weighted and renormalized to sum to 1 across only the
    currently active symbols, so 1 active signal gets the full wallet, 2 or 3
    split it proportionally -- see ``core.risk.shared_wallet_risk_scale`` /
    ``core.backtester``.
    """

    initial_capital_usdt: float = 500.0
    max_portfolio_risk_pct: float = 0.01  # <= 1% of shared equity (5 EUR of 500 EUR) if ALL open positions stop out together
    cooldown_candles: int = 6  # bars to sit out (per symbol) after any exit before a new entry is allowed

    # --- Cold-start / equity-cushion protection (sequence-of-returns risk) ---
    # A loss right at the start hits the initial 500 USDT hardest (nothing to
    # absorb it yet). Until the account has ever closed at least
    # ``cold_start_buffer_pct`` above ``initial_capital_usdt``, every symbol's
    # final leverage is additionally scaled by ``cold_start_risk_scale``. Once
    # that cushion has been built even once, the scale-down is permanently
    # lifted (ratchet) -- normal drawdown behavior afterwards is completely
    # unaffected, even if capital later falls back toward/below the start.
    cold_start_buffer_pct: float = 0.10  # need +10% above the starting balance once, ever
    cold_start_risk_scale: float = 0.5  # half leverage/risk while still in the cold-start phase

    # --- Absolute margin ceiling (Konzept 2) ---
    # Hard USDT notional cap per symbol, independent of how large the shared
    # wallet grows via profits or deposits (percent-of-equity sizing alone
    # would scale the position without bound). ``None`` disables the cap.
    # Default is a generous multiple of the 500 USDT starting capital so it
    # does not bind today, but stops any single symbol's exposure from
    # growing unboundedly as the account compounds; tune to taste/exchange
    # liquidity before scaling capital up materially.
    max_absolute_position_size_usdt: float | None = 2000.0

    # Hard ceiling on the SUM of every symbol's notional exposure at once
    # (independent of ``max_absolute_position_size_usdt``, which only bounds
    # each symbol individually -- 3 symbols each just under their individual
    # cap could otherwise still add up to an unbounded total). ``None``
    # disables it. A position-selection/ranking pass
    # (``PositionSelectionConfig``) must never be able to raise this or any
    # other hard limit, only ever reduce how many symbols compete for the
    # same already-capped budget.
    max_total_notional_usdt: float | None = None


@dataclass
class TradingBotConfig:
    data: DataConfig = field(default_factory=DataConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    funding: FundingConfig = field(default_factory=FundingConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)
    capital: CapitalConfig = field(default_factory=CapitalConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    ml: TrendMLConfig = field(default_factory=TrendMLConfig)
    swing: SwingMLConfig = field(default_factory=SwingMLConfig)
    selection: PositionSelectionConfig = field(default_factory=PositionSelectionConfig)

