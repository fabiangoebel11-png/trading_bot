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
from core.ml.tcn_model import TCNTrendModel


def load_symbol_model(symbol: str, config: TrendMLConfig) -> tuple[TCNTrendModel, dict]:
    import torch  # local import: optional dependency

    model_dir = Path(config.model_dir)
    safe_symbol = symbol.replace("/", "-")
    meta = json.loads((model_dir / f"{safe_symbol}_tcn_meta.json").read_text())
    # weights_only=False: trusted, locally produced checkpoint containing numpy
    # arrays (feature mean/std) alongside the model state dict.
    state = torch.load(model_dir / f"{safe_symbol}_tcn.pt", map_location="cpu", weights_only=False)

    model = TCNTrendModel(config, n_features=state["n_features"])
    model.load_state_dict(state)
    return model, meta


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
    warm-up isn't yet available (neutral -- never blocks the base strategy)."""
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
