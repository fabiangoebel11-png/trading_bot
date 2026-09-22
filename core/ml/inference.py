"""Load a trained trend-continuation model and turn it into a position-sizing
confirmation signal for ``core/strategy.py``'s rule-based Donchian/EMA output.

Deliberately mirrors ``core/macro.py``'s ``apply_macro_gate`` pattern: the ML
signal *scales* the rule-based position rather than replacing it outright --
the base strategy stays the primary edge, the model just says "how much do I
believe this breakout" (a meta-labeling style veto/sizing layer, not a
standalone signal generator).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import TrendMLConfig
from core.ml.dataset import make_sequences
from core.ml.features import build_feature_matrix
from core.ml.regime import classify_regimes, ml_regime_active
from core.ml.tcn_model import TCNTrendModel


def load_symbol_model(symbol: str, config: TrendMLConfig) -> tuple[TCNTrendModel, dict]:
    import dataclasses

    import torch  # local import: optional dependency

    model_dir = Path(config.model_dir)
    safe_symbol = symbol.replace("/", "-")
    meta = json.loads((model_dir / f"{safe_symbol}_tcn_meta.json").read_text())
    # weights_only=False: trusted, locally produced checkpoint containing numpy
    # arrays (feature mean/std) alongside the model state dict.
    state = torch.load(model_dir / f"{safe_symbol}_tcn.pt", map_location="cpu", weights_only=False)

    # Reconstruct the architecture from what was ACTUALLY used at training
    # time (persisted in meta by core/ml/train.py), not from whatever the
    # caller's current ``config`` happens to be -- otherwise any later change
    # to ``TrendMLConfig.hidden_channels``/``num_layers``/``dropout`` (e.g.
    # the 1h retuning in this same session) makes every previously trained
    # checkpoint's ``load_state_dict`` fail with a shape mismatch. Falls back
    # to the passed-in config's values for checkpoints saved before this fix
    # (best-effort only -- old artifacts trained before the retuning are still
    # only safely loadable if the config hasn't actually changed since).
    arch_config = dataclasses.replace(
        config,
        hidden_channels=meta.get("hidden_channels", config.hidden_channels),
        num_layers=meta.get("num_layers", config.num_layers),
        dropout=meta.get("dropout", config.dropout),
    )
    model = TCNTrendModel(arch_config, n_features=state["n_features"])
    model.load_state_dict(state)
    return model, meta


def load_oos_confidence(symbol: str, config: TrendMLConfig) -> pd.Series | None:
    """Load the stitched, genuinely out-of-sample purged-walk-forward
    confidence series saved by ``core/ml/train.py: train_symbol_model``.

    This is the ONLY confidence series that is valid to use for historical
    performance measurement (backtester/Monte Carlo/walk-forward validation):
    every value in it came from a fold whose model was trained strictly on
    data *before* (purged/embargoed against) the bar it scores. The final
    model persisted alongside it (``load_symbol_model``) is fit on the FULL
    history and is only valid for genuine prospective (live) inference --
    using it to score historical bars during a backtest would let bars from
    e.g. 2020 be scored by a model that has already seen 2024-2026 data, an
    in-sample leak that silently inflates any backtested Sharpe/Monte-Carlo
    result (see ``core/strategy.py: generate_portfolio_signals``, which uses
    this function instead of the live model for exactly that reason).

    Returns ``None`` if the file doesn't exist (e.g. an older model artifact
    trained before this fix, or training was skipped) -- callers must treat
    that as "no valid historical ML signal available", never fall back to
    the leaky final-model path.
    """
    path = Path(config.model_dir) / f"{symbol.replace('/', '-')}_tcn_oos_confidence.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)["oos_confidence"]


def predict_trend_confidence(
    ohlc: pd.DataFrame,
    macro_prices: pd.DataFrame | None,
    model: TCNTrendModel,
    meta: dict,
    config: TrendMLConfig,
    breadth_return: pd.Series | None = None,
) -> pd.Series:
    """Returns a series aligned to ``ohlc.index``, in [-1, 1]: ``P(up) -
    P(down)`` from the model, 0.0 wherever a full lookback sequence/feature
    warm-up isn't yet available (neutral -- never blocks the base strategy).

    Uses ``model`` as-is (typically the full-history-trained final model from
    ``load_symbol_model``) -- valid ONLY for genuine prospective inference on
    bars at or after the model's training cutoff (i.e. live trading,
    ``execution/live_trader.py``). Do NOT use this to score historical bars
    for backtesting/Monte Carlo -- see ``load_oos_confidence`` for the
    leak-free alternative used by ``core/strategy.py:
    generate_portfolio_signals``."""
    features = build_feature_matrix(ohlc, macro_prices, config, breadth_return)
    features = features.reindex(columns=meta["feature_columns"])
    valid = features.dropna()

    seq_len = meta["sequence_length"]
    result = pd.Series(0.0, index=ohlc.index)
    if len(valid) < seq_len:
        return result

    X_seq = make_sequences(valid.to_numpy(dtype=np.float32), seq_len)
    proba = model.predict_proba(X_seq)
    confidence = proba[:, 2] - proba[:, 0]

    aligned_index = valid.index[seq_len - 1 :]
    result.loc[aligned_index] = confidence
    return result


def apply_ml_confirmation(
    signals: pd.DataFrame,
    confidence: pd.Series,
    min_confidence: float = 0.15,
    disagreement_scale: float = 0.3,
) -> pd.DataFrame:
    """Scale ``signals["position"]`` by model confidence: full size when the
    model agrees with the rule-based direction with at least
    ``min_confidence``, damped by ``disagreement_scale`` when the model
    disagrees or is unconvinced. Never flips the sign or adds new trades the
    base strategy didn't already generate."""
    out = signals.copy()
    position = out["position"].astype(float)
    conf = confidence.reindex(out.index).fillna(0.0)

    agrees = (np.sign(position) == np.sign(conf)) & (conf.abs() >= min_confidence)
    scaled = np.where(agrees, position, position * disagreement_scale)

    out["ml_confidence"] = conf
    out["position"] = np.where(position == 0, 0.0, scaled)
    return out


def apply_regime_gated_ml_confirmation(
    signals: pd.DataFrame, confidence: pd.Series, config: TrendMLConfig
) -> pd.DataFrame:
    """Apply ML sizing only in the configured causal crash/trend regimes.

    Missing OOS confidence and inactive sideways bars retain the raw signal
    exactly. This is shared by backtesting and live inference to avoid a
    train/serve regime-gate mismatch.
    """
    confidence = confidence.reindex(signals.index)
    active = ml_regime_active(signals["close"], config)
    available = confidence.notna()
    out = apply_ml_confirmation(signals, confidence)
    weight = config.confirmation_weight
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"confirmation_weight must be within [0, 1], got {weight}")
    # Blend position sizes, never directions: a fractional weight limits how
    # strongly a model disagreement can reduce the base strategy exposure.
    out["position"] = signals["position"] + weight * (out["position"] - signals["position"])
    passthrough = ~(active & available)
    out.loc[passthrough, "position"] = signals.loc[passthrough, "position"]
    out["ml_regime"] = classify_regimes(signals["close"], config)
    out["ml_regime_active"] = active
    out["ml_confirmation_weight"] = weight
    return out
