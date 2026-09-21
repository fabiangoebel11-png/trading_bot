"""Hybrid strategy: gate the rule-based stat-arb position with an ML-predicted
mean-reversion probability, trained strictly on past (train-fold) data only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import MLConfig, TradingBotConfig
from core.macro import apply_macro_gate
from core.ml.features import build_feature_matrix, build_training_dataset
from core.ml.gbm_model import GradientBoostingReversionModel
from core.strategy import generate_signals


def build_model(ml_config: MLConfig):
    if ml_config.model_type == "gbm":
        return GradientBoostingReversionModel(ml_config)
    if ml_config.model_type == "lstm":
        from core.ml.lstm_model import LSTMReversionModel  # lazy: torch is optional

        return LSTMReversionModel(ml_config)
    raise ValueError(f"Unknown ml.model_type: {ml_config.model_type}")


def train_ml_gate(signals: pd.DataFrame, train_end_ts, config: TradingBotConfig):
    """Fit an ML model on the portion of ``signals`` strictly before ``train_end_ts``.

    Purge/embargo: the reversion label at bar t is built from data at t + horizon
    (see ``build_reversion_labels``). Without purging, training rows within
    ``label_horizon`` bars of ``train_end_ts`` would have labels computed from data
    that lies inside the test segment -- a subtle look-ahead leak across the
    walk-forward boundary (Lopez de Prado's "purged cross-validation" problem).
    Those rows are therefore excluded from training entirely.
    """
    symbol_a, symbol_b = config.data.symbol_a, config.data.symbol_b
    X, y = build_training_dataset(signals, symbol_a, symbol_b, config.ml)

    bar_delta = signals.index.to_series().diff().median()
    purge_cutoff = train_end_ts - bar_delta * config.ml.label_horizon
    train_mask = X.index <= purge_cutoff

    model = build_model(config.ml)
    model.fit(X.loc[train_mask], y.loc[train_mask])
    return model


def apply_ml_gate(signals: pd.DataFrame, model, config: TradingBotConfig) -> pd.DataFrame:
    """Predict reversion probability for every bar and zero out positions where the
    model is not confident enough that the excursion will actually mean-revert."""
    symbol_a, symbol_b = config.data.symbol_a, config.data.symbol_b
    features = build_feature_matrix(signals, symbol_a, symbol_b, config.ml)
    valid = features.dropna()

    proba = model.predict_proba(valid)

    out = signals.copy()
    out["ml_proba"] = np.nan
    out.loc[proba.index, "ml_proba"] = proba
    out["ml_proba"] = out["ml_proba"].fillna(0.5)  # neutral (no gating) where features are missing

    gate_pass = out["ml_proba"] >= config.ml.proba_threshold
    out["position"] = np.where(gate_pass, out["position"], 0)
    return out


def generate_hybrid_signals(
    df: pd.DataFrame,
    config: TradingBotConfig,
    train_end_ts=None,
    model=None,
    macro_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, object]:
    """End-to-end: rolling OLS + z-score signal, ML gate, then macro risk-off gate.
    If ``model`` is not provided, one is trained on data strictly before
    ``train_end_ts`` (or on the full frame if ``train_end_ts`` is None -- only
    appropriate for exploratory, non-walk-forward runs)."""
    signals = generate_signals(df, config.data.symbol_a, config.data.symbol_b, config.strategy)

    if model is None:
        cutoff = train_end_ts if train_end_ts is not None else signals.index[-1]
        model = train_ml_gate(signals, cutoff, config)

    gated = apply_ml_gate(signals, model, config)
    gated = apply_macro_gate(gated, config, macro_df)
    return gated, model
