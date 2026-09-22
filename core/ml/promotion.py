"""Manual-only promotion gate for the ML shadow layer.

This module NEVER sets ``TrendMLConfig.production_enabled`` -- it only
computes a read-only recommendation from chronological holdout data (see
``core/ml/holdout.py``). A human must review ``PromotionResult`` and edit the
config by hand; nothing in this codebase is permitted to flip that flag
automatically.

Promotion requires ALL of:
  1. positive delta-Sharpe over the full holdout period,
  2. a better (higher) median per-window return,
  3. a not-worse (equal-or-better) median per-window max drawdown,
  4. improvement in more than 55% of identical (paired) holdout windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from core.metrics import sharpe_ratio
from core.ml.holdout import paired_window_metrics


@dataclass
class PromotionCriteria:
    min_window_improvement_rate: float = 0.55  # strictly more than 55% of windows must improve
    window_days: int = 30
    min_windows: int = 4  # too few windows -> not enough holdout evidence yet


@dataclass
class PromotionResult:
    passed: bool
    n_windows: int
    delta_sharpe: float
    delta_sharpe_positive: bool
    median_return_delta_pct: float
    median_return_better: bool
    median_max_drawdown_delta_pct: float
    max_drawdown_not_worse: bool
    window_improvement_rate: float
    window_improvement_ok: bool
    windows: pd.DataFrame
    criteria: PromotionCriteria
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "BESTANDEN" if self.passed else "NICHT BESTANDEN"
        lines = [
            f"Holdout-Promotion-Gate: {status} ({self.n_windows} Fenster). "
            "ml.production_enabled bleibt ausschliesslich manuell zu setzen.",
            f"  Delta-Sharpe (gesamter Holdout): {self.delta_sharpe:+.3f} "
            f"[{'OK' if self.delta_sharpe_positive else 'fehlt'}, muss > 0]",
            f"  Median-Return-Delta je Fenster: {self.median_return_delta_pct:+.2f}pp "
            f"[{'OK' if self.median_return_better else 'fehlt'}, muss > 0]",
            f"  Median-Max-Drawdown-Delta je Fenster: {self.median_max_drawdown_delta_pct:+.2f}pp "
            f"[{'OK' if self.max_drawdown_not_worse else 'fehlt'}, darf nicht schlechter sein]",
            f"  Fenster mit Verbesserung: {self.window_improvement_rate:.1%} "
            f"[{'OK' if self.window_improvement_ok else 'fehlt'}, muss > "
            f"{self.criteria.min_window_improvement_rate:.0%}]",
        ]
        if self.reasons:
            lines.append("  Hinweise: " + "; ".join(self.reasons))
        return "\n".join(lines)


def evaluate_promotion(
    rule_returns: pd.Series,
    ml_returns: pd.Series,
    holdout_start: pd.Timestamp,
    periods_per_year: int,
    criteria: PromotionCriteria | None = None,
) -> PromotionResult:
    """Read-only holdout evaluation. ``rule_returns``/``ml_returns`` must be
    bar-level strategy returns (e.g. ``core.ml.shadow.build_shadow_log``'s
    ``rule_pnl``/``ml_pnl`` columns, or a genuine live shadow log) covering at
    least the period from ``holdout_start`` onward. Data before
    ``holdout_start`` is ignored -- it may have informed model/hyperparameter
    selection and is therefore not valid evidence for promotion."""
    criteria = criteria or PromotionCriteria()
    reasons: list[str] = []

    rule_holdout = rule_returns[rule_returns.index >= holdout_start]
    ml_holdout = ml_returns[ml_returns.index >= holdout_start]
    if rule_holdout.empty or ml_holdout.empty:
        reasons.append("keine Daten im Holdout-Zeitraum")
        return PromotionResult(
            passed=False,
            n_windows=0,
            delta_sharpe=0.0,
            delta_sharpe_positive=False,
            median_return_delta_pct=0.0,
            median_return_better=False,
            median_max_drawdown_delta_pct=0.0,
            max_drawdown_not_worse=False,
            window_improvement_rate=0.0,
            window_improvement_ok=False,
            windows=pd.DataFrame(),
            criteria=criteria,
            reasons=reasons,
        )

    delta_sharpe = sharpe_ratio(ml_holdout, periods_per_year) - sharpe_ratio(rule_holdout, periods_per_year)
    delta_sharpe_positive = delta_sharpe > 0.0

    windows = paired_window_metrics(rule_returns, ml_returns, holdout_start, periods_per_year, criteria.window_days)
    n_windows = len(windows)
    if n_windows < criteria.min_windows:
        reasons.append(
            f"nur {n_windows} vollstaendige Holdout-Fenster (< {criteria.min_windows}) -- "
            "zu wenig Evidenz fuer eine Promotion-Entscheidung"
        )

    if n_windows == 0:
        median_return_delta = 0.0
        median_dd_delta = 0.0
        window_improvement_rate = 0.0
    else:
        median_return_delta = float(windows["delta_return_pct"].median())
        median_dd_delta = float(windows["delta_max_drawdown_pct"].median())
        window_improvement_rate = float((windows["delta_sharpe"] > 0).mean())

    median_return_better = median_return_delta > 0.0
    max_drawdown_not_worse = median_dd_delta >= 0.0
    window_improvement_ok = window_improvement_rate > criteria.min_window_improvement_rate

    passed = (
        n_windows >= criteria.min_windows
        and delta_sharpe_positive
        and median_return_better
        and max_drawdown_not_worse
        and window_improvement_ok
    )

    return PromotionResult(
        passed=passed,
        n_windows=n_windows,
        delta_sharpe=delta_sharpe,
        delta_sharpe_positive=delta_sharpe_positive,
        median_return_delta_pct=median_return_delta,
        median_return_better=median_return_better,
        median_max_drawdown_delta_pct=median_dd_delta,
        max_drawdown_not_worse=max_drawdown_not_worse,
        window_improvement_rate=window_improvement_rate,
        window_improvement_ok=window_improvement_ok,
        windows=windows,
        criteria=criteria,
        reasons=reasons,
    )
