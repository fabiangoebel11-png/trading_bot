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


def _default_monte_carlo_dir() -> str:
    return str(_PROJECT_ROOT / "data" / "monte_carlo")


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
    lookback_days: int = 730
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

    Deliberately trains on raw, unsmoothed intraday noise (default 5m candles,
    the finest granularity already cached in ``data/``) instead of the 1h bars
    used by the rule-based trend strategy: the goal is a model that has learned
    to survive real market noise, not one that only works on a denoised proxy.
    Overfitting to that noise is fought on three independent fronts:
      1. Labels are a triple-barrier touch (take-profit/stop-loss/time), not a
         simple fixed-horizon return, and cross-validation is *purged* (drop
         training rows whose label window overlaps the test fold) with an
         *embargo* gap after each test fold -- both prevent the leakage that a
         naive walk-forward split gets wrong with overlapping-horizon labels.
      2. The model itself (``core/ml/tcn_model.py``) is a small dilated causal
         CNN with dropout + weight decay (AdamW), gradient clipping, label
         smoothing and early stopping on a held-out validation slice.
      3. Features include cross-asset macro context (S&P 500 / Nasdaq 100 / MSCI
         World / VIX) causally lagged, so the model has more to work with than
         BTC/ETH price action alone and is less likely to just memorize it.
    """

    enabled: bool = False
    base_timeframe: str = "5m"  # intentionally noisy/unsmoothed, unlike the 1h rule-based strategy
    history_days: int = 1095

    # Cross-asset macro context, causally lagged (see core/ml/features.py).
    macro_symbols: list[str] = field(default_factory=lambda: ["ES=F", "NQ=F", "^VIX", "URTH"])
    macro_lookback_days: int = 1095
    macro_reporting_lag_days: int = 1
    macro_cache_dir: str = field(default_factory=_default_data_dir)

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
    breadth_corr_window: int = 60
    breadth_vol_window: int = 288
    breadth_zscore_window: int = 288

    # Triple-barrier labeling (López de Prado style): label = which barrier is
    # touched first within `label_horizon` bars -- up (+1), down (-1) or neither
    # (0, time barrier). Barrier distance scales with ATR so it adapts to
    # regime volatility instead of using a fixed % target.
    atr_window: int = 14
    label_horizon: int = 48  # bars ahead; 48 * 5m = 4h
    barrier_atr_multiple: float = 2.0

    # Model / training.
    sequence_length: int = 64  # lookback bars fed to the TCN per sample
    hidden_channels: int = 64
    num_layers: int = 4
    dropout: float = 0.3
    weight_decay: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 60
    early_stopping_patience: int = 6
    val_fraction: float = 0.15  # tail slice of each fold's train split, used only for early stopping
    label_smoothing: float = 0.05
    grad_clip_norm: float = 1.0
    use_amp: bool = True  # mixed precision on the RTX 3070; no-op on CPU

    # Purged walk-forward cross-validation.
    n_splits: int = 5
    embargo_fraction: float = 0.01  # fraction of total bars purged/embargoed around each test fold

    random_state: int = 42
    model_dir: str = field(default_factory=_default_model_dir)


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

