"""Deterministic trade-quality scoring and high-confidence metrics."""
from __future__ import annotations

import numpy as np


def validate_model1_prediction_outputs(outputs: dict[str, np.ndarray], quality: dict[str, np.ndarray] | None = None) -> None:
    """Reject invalid public Model 1 predictions before they reach callers."""
    required = ("opportunity", "expected_return", "expected_duration", "expected_mfe", "expected_mae")
    missing = [name for name in required if name not in outputs]
    if missing:
        raise ValueError(f"Model 1 prediction is missing public outputs: {missing}")
    lengths = {len(np.asarray(outputs[name])) for name in required}
    if len(lengths) != 1:
        raise ValueError("Model 1 public outputs have inconsistent lengths")
    for name in required:
        value = np.asarray(outputs[name], dtype=float)
        if not np.isfinite(value).all():
            raise ValueError(f"Model 1 public output {name!r} is non-finite")
    if "entry_score" in outputs:
        entry_score = np.asarray(outputs["entry_score"], dtype=float)
        if not np.isfinite(entry_score).all() or np.any((entry_score < 0) | (entry_score > 100)):
            raise ValueError("Model 1B entry_score must be finite and within [0, 100]")
    if np.any(np.asarray(outputs["expected_duration"], dtype=float) < 0):
        raise ValueError("Model 1 expected duration must be non-negative")
    if quality is not None:
        score = np.asarray(quality["score"], dtype=float)
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 100)):
            raise ValueError("Model 1 public score must be finite and within [0, 100]")


def continuous_opportunity_score(
    opportunity: np.ndarray,
    expected_return: np.ndarray,
    expected_favorable: np.ndarray,
    expected_adverse: np.ndarray,
    *,
    return_scale: float = 0.01,
) -> dict[str, np.ndarray]:
    """Convert continuous model heads into an always-defined 0-100 score."""
    opportunity = np.clip(np.asarray(opportunity, dtype=float), 0.0, 1.0)
    expected_return = np.asarray(expected_return, dtype=float)
    favorable = np.maximum(np.asarray(expected_favorable, dtype=float), 0.0)
    adverse = np.maximum(np.abs(np.asarray(expected_adverse, dtype=float)), 0.0)
    if any(not np.isfinite(value).all() for value in (opportunity, expected_return, favorable, adverse)):
        raise ValueError("Model 1 score conversion requires finite prediction heads")
    payoff = favorable / np.maximum(favorable + adverse, 1e-8)
    directional_confidence = np.clip(np.abs(expected_return) / max(return_scale, 1e-8), 0.0, 1.0)
    score = 100.0 * (0.55 * opportunity + 0.25 * payoff + 0.20 * directional_confidence)
    if not np.isfinite(score).all() or np.any((score < 0) | (score > 100)):
        raise ValueError("Model 1 score conversion produced an invalid score")
    direction = np.where(np.isnan(expected_return), "UNKNOWN", np.where(expected_return > 0, "LONG", np.where(expected_return < 0, "SHORT", "UNCERTAIN")))
    return {
        "score": np.clip(score, 0.0, 100.0),
        "direction": direction,
        "probability": opportunity,
        "expected_favorable": favorable,
        "expected_adverse": adverse,
        "payoff_component": payoff,
    }


def score_trade_quality(
    probabilities: np.ndarray,
    expected_favorable: np.ndarray,
    expected_adverse: np.ndarray,
    *,
    probability_weight: float = 0.6,
    payoff_weight: float = 0.4,
) -> dict[str, np.ndarray]:
    """Map model outputs to transparent 0-100 setup scores.

    The model reports a 3-class distribution over ``[SHORT, NO_TRADE, LONG]``.
    A high score must reflect a credible opportunity, not merely a large
    NO_TRADE probability. The tradeability gate therefore uses the probability
    of taking a trade, and only then scores the selected direction.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    favorable = np.maximum(np.asarray(expected_favorable, dtype=float), 0.0)
    adverse = np.maximum(np.abs(np.asarray(expected_adverse, dtype=float)), 0.0)

    if probabilities.ndim != 2 or probabilities.shape[1] < 3:
        raise ValueError("probabilities must have shape (n_samples, 3)")

    no_trade_probability = probabilities[:, 1]
    directional_probability = np.maximum(probabilities[:, 0], probabilities[:, 2])
    trade_probability = 1.0 - no_trade_probability
    direction_index = np.argmax(probabilities[:, [0, 2]], axis=1)
    direction = np.where(direction_index == 0, "SHORT", "LONG")

    is_tradeable = trade_probability >= 0.35
    direction = np.where(is_tradeable, direction, "NO_TRADE")

    payoff = favorable / np.maximum(favorable + adverse, 1e-12)
    directional_strength = directional_probability / np.maximum(trade_probability, 1e-12)
    directional_strength = np.clip(directional_strength, 0.0, 1.0)
    trade_score = 100.0 * (probability_weight * directional_strength + payoff_weight * payoff)
    # NO_TRADE should never receive a high score: the model is only rewarded for
    # high-quality setups that have a statistically meaningful probability of a
    # directional move. Keeping the flat branch near zero ensures the score is a
    # trade-quality signal rather than a hidden "P(NO_TRADE)" ranking.
    no_trade_score = 100.0 * np.clip(0.5 * trade_probability, 0.0, 1.0)
    score = np.where(is_tradeable, trade_score, no_trade_score)

    return {
        "direction": direction,
        "probability": np.where(is_tradeable, directional_probability, no_trade_probability),
        "score": np.clip(score, 0.0, 100.0),
        "expected_favorable": favorable,
        "expected_adverse": adverse,
        "payoff_component": payoff,
    }


def threshold_metrics(
    quality: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    thresholds: tuple[float, ...] = (80.0, 90.0),
    realized_r: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Report precision, recall, count, expectancy and excursion by score gate."""
    labels = np.asarray(labels, dtype=int)
    direction = np.asarray(quality["direction"])
    score = np.asarray(quality["score"], dtype=float)
    expected_favorable = np.asarray(quality["expected_favorable"], dtype=float)
    expected_adverse = np.asarray(quality["expected_adverse"], dtype=float)
    realized_r = None if realized_r is None else np.asarray(realized_r, dtype=float)
    result: dict[str, float | int] = {}
    for threshold in thresholds:
        key = str(int(threshold))
        selected = score >= threshold
        predicted = np.where(direction == "LONG", 1, np.where(direction == "SHORT", -1, 0))
        correct = selected & (predicted == labels) & (predicted != 0)
        signal_count = int(selected.sum())
        result[f"precision_at_{key}"] = float(correct.sum() / signal_count) if signal_count else float("nan")
        eligible = labels != 0
        result[f"recall_at_{key}"] = float(correct.sum() / eligible.sum()) if eligible.sum() else float("nan")
        result[f"signals_at_{key}"] = signal_count
        result[f"expectancy_at_{key}"] = float(expected_favorable[selected].mean() + expected_adverse[selected].mean()) if signal_count else float("nan")
        result[f"mfe_at_{key}"] = float(expected_favorable[selected].mean()) if signal_count else float("nan")
        result[f"mae_at_{key}"] = float(expected_adverse[selected].mean()) if signal_count else float("nan")
        if realized_r is not None:
            selected_r = realized_r[selected]
            result[f"average_r_at_{key}"] = float(selected_r.mean()) if signal_count else float("nan")
            result[f"win_rate_at_{key}"] = float((selected_r > 0).mean()) if signal_count else float("nan")
    return result


def score_bucket_metrics(
    quality: dict[str, np.ndarray],
    realized_return: np.ndarray,
    realized_mfe: np.ndarray,
    realized_mae: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    """Evaluate continuous score separation without converting it to a class gate."""
    score = np.asarray(quality["score"], dtype=float)
    realized_return = np.asarray(realized_return, dtype=float)
    realized_mfe = np.asarray(realized_mfe, dtype=float)
    realized_mae = np.asarray(realized_mae, dtype=float)
    direction = np.where(realized_return > 0, 1, np.where(realized_return < 0, -1, 0))
    predicted = np.where(np.asarray(quality["direction"]) == "LONG", 1, np.where(np.asarray(quality["direction"]) == "SHORT", -1, 0))
    edges = (0.0, 20.0, 40.0, 60.0, 80.0, 90.0, 100.000001)
    result: dict[str, dict[str, float | int]] = {}
    for left, right in zip(edges, edges[1:]):
        selected = (score >= left) & (score < right)
        key = f"{int(left)}-{min(int(right), 100)}"
        result[key] = {
            "count": int(selected.sum()),
            "mean_future_return": float(realized_return[selected].mean()) if selected.any() else float("nan"),
            "mean_mfe": float(realized_mfe[selected].mean()) if selected.any() else float("nan"),
            "mean_mae": float(realized_mae[selected].mean()) if selected.any() else float("nan"),
            "directional_hit_rate": float((predicted[selected] == direction[selected]).mean()) if selected.any() else float("nan"),
        }
    return result
