"""DEPRECATED: entry point for the retired ML-gated mean-reversion pairs strategy.

A Monte Carlo random-start stress test (see run_monte_carlo_stress_test.py)
falsified the underlying mean-reversion strategy this script gated (0%
profitable windows). The project has since pivoted to a trend-following /
breakout approach -- see run_production_strategy.py and core/strategy.py.

This script is kept only as a historical reference and is not runnable as-is:
``core.hybrid_strategy`` depends on ``core.config.MLConfig`` and
``core.strategy.generate_signals``, both removed as part of the pivot.
"""
from __future__ import annotations


def main() -> None:
    raise SystemExit(
        "run_hybrid_strategy.py is deprecated (mean-reversion strategy retired). "
        "Use research/run_production_strategy.py (trend-following) instead."
    )


if __name__ == "__main__":
    main()

