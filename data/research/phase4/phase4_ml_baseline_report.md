# Phase 4 Controlled ML Baseline Evaluation

Research-only diagnostic. No runtime, execution, strategy, risk, leverage, paper, or live code was changed.

## Protocol

- 24 frozen BTC/ETH tasks; 2 feature sets; 3 fixed baseline models.
- WF validation: 2022, 2023, and 2024 with expanding training windows.
- Purge and embargo: one complete target horizon for every task.
- StandardScaler is fitted separately on each training fold only.
- Thresholds 0.50/0.55/0.60/0.65/0.70 are descriptive only.
- Final holdout 2025-01-01 through 2026-09-22 is evaluation-only and never enters selection.

## Validation Summary

                        feature_set_id               model_id  metric_roc_auc  metric_pr_auc  metric_log_loss  metric_brier_score  metric_balanced_accuracy
                        crypto_core_v1 hist_gradient_boosting        0.529292       0.538731         0.715255            0.258984                  0.521115
                        crypto_core_v1    logistic_regression        0.518444       0.536112         0.694786            0.250802                  0.514344
                        crypto_core_v1          random_forest        0.534264       0.543365         0.700784            0.253321                  0.525170
crypto_core_v1+crypto_context_proxy_v1 hist_gradient_boosting        0.538431       0.542899         0.888883            0.303996                  0.526213
crypto_core_v1+crypto_context_proxy_v1    logistic_regression        0.538937       0.554878         0.738896            0.268483                  0.517954
crypto_core_v1+crypto_context_proxy_v1          random_forest        0.538542       0.546027         0.766129            0.278003                  0.524339

## Core vs Context

The comparison is reported by feature set and walk-forward period. No overall best-model ranking or promotion decision is produced.

## Holdout

Holdout records: 144. These values are stored separately as untouched out-of-sample diagnostics and were not used for feature/model/threshold selection.

## Interpretation

Results describe directional information value and temporal generalization only. They do not establish a trading strategy and do not authorize promotion.
