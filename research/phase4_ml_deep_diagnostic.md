# Phase 4.2 Deep Diagnostic of Frozen ML Baselines

Read-only diagnostic. No training, tuning, feature changes, promotion, strategy, risk, paper, or live changes were performed.

## A. Temporal Stability

All 31 Phase 4.1 ROBUST CANDIDATE combinations were expanded into WF1/WF2/WF3 dossiers. Descriptive temporal categories: {"A": 28, "B": 3}.
A = stable across all WFs with positive AUC/PR-AUC and tight AUC/Brier ranges; B = mostly stable with one weak metric/fold; C = aggregate-positive but visibly variable; D = one WF dominates the positive AUC excess or combines a strong and weak period.
These categories are diagnostic labels, not rankings or selection rules.

## B. Asset Patterns
{
  "BTC/USDT": 0.555977,
  "ETH/USDT": 0.549038
}
Differences are reported by asset and do not imply that one asset is preferable.

## C. Horizon Patterns
{
  "12h": 0.542262,
  "1h": 0.557467,
  "24h": 0.537535,
  "2h": 0.554265,
  "4h": 0.551079,
  "8h": 0.54547
}
Horizon groups remain frozen; no horizon was selected.

## D. Model Families
{
  "hist_gradient_boosting": 0.556908,
  "logistic_regression": 0.547467,
  "random_forest": 0.55598
}
Model-family means are descriptive and include no hyperparameter search.

## E. Core vs Context
{
  "final_holdout": {
    "delta_roc_auc": 0.001961,
    "delta_pr_auc": 0.001185,
    "delta_brier_score": -4.2e-05,
    "delta_mean_forward_return": 0.0
  },
  "wf_2022": {
    "delta_roc_auc": -0.010068,
    "delta_pr_auc": -0.007986,
    "delta_brier_score": 0.002747,
    "delta_mean_forward_return": 0.0
  },
  "wf_2023": {
    "delta_roc_auc": -0.00521,
    "delta_pr_auc": -0.003457,
    "delta_brier_score": 0.000526,
    "delta_mean_forward_return": 0.0
  },
  "wf_2024": {
    "delta_roc_auc": -0.002511,
    "delta_pr_auc": -0.00353,
    "delta_brier_score": 0.001519,
    "delta_mean_forward_return": 0.0
  }
}
Deltas are paired on identical asset/timeframe/horizon/model/split keys. Context remains proxy-only.

## F. Holdout Degradation

Holdout rows analyzed: 31. The holdout was never used in temporal categorization, selection, or optimization. Candidate-level validation-to-holdout ROC-AUC diagnostics are in `phase4_deep_diagnostic.csv`; aggregate summary: {'count': 31.0, 'mean': -0.023737, 'std': 0.006121, 'min': -0.036375, '25%': -0.02781, '50%': -0.023194, '75%': -0.020121, 'max': -0.009786}.

## G. Weak Candidate

The Phase 4.1 WEAK candidate count is 1. Its exact validation and holdout metrics are included in the machine-readable JSON under `weak_candidate`; no new decision rule was applied to it.
{
  "count": 1,
  "rows": [
    {
      "target": "ETH/USDT",
      "timeframe": "4h",
      "horizon": "7d",
      "feature_set_id": "crypto_core_v1",
      "model_id": "logistic_regression",
      "classification": "WEAK",
      "validation": [
        {
          "split_id": "wf_2022",
          "metric_roc_auc": 0.4895890578249984,
          "metric_pr_auc": 0.4589281530794581,
          "metric_balanced_accuracy": 0.5094899350715434,
          "metric_log_loss": 0.6958900864494565,
          "metric_brier_score": 0.2513224308038608,
          "metric_accuracy": 0.5278538812785388,
          "metric_mean_forward_return": -0.0128352471548274,
          "metric_sample_count": 2190.0,
          "metric_positive_rate": 0.4356164383561643
        },
        {
          "split_id": "wf_2023",
          "metric_roc_auc": 0.4758705882352941,
          "metric_pr_auc": 0.5356670874796948,
          "metric_balanced_accuracy": 0.5092100840336135,
          "metric_log_loss": 0.7178534774177022,
          "metric_brier_score": 0.2620732283001206,
          "metric_accuracy": 0.4780821917808219,
          "metric_mean_forward_return": 0.0137038070862001,
          "metric_sample_count": 2190.0,
          "metric_positive_rate": 0.54337899543379
        },
        {
          "split_id": "wf_2024",
          "metric_roc_auc": 0.4805064781606862,
          "metric_pr_auc": 0.5438254863560329,
          "metric_balanced_accuracy": 0.5092409729423182,
          "metric_log_loss": 0.6997218863843815,
          "metric_brier_score": 0.2532677783234237,
          "metric_accuracy": 0.4986338797814207,
          "metric_mean_forward_return": 0.0127267706852982,
          "metric_sample_count": 2196.0,
          "metric_positive_rate": 0.5250455373406193
        }
      ],
      "holdout": [
        {
          "split_id": "final_holdout",
          "metric_roc_auc": 0.5032023018781493,
          "metric_pr_auc": 0.5027050568992062,
          "metric_balanced_accuracy": 0.4999249885478699,
          "metric_log_loss": 0.6971395652294858,
          "metric_brier_score": 0.2519761265035843,
          "metric_accuracy": 0.49812734082397,
          "metric_mean_forward_return": 0.0008880085032477,
          "metric_sample_count": 3738.0,
          "metric_positive_rate": 0.5050829320492242
        }
      ]
    }
  ]
}

## H. Probability / Calibration Data Gap

Stored artifacts contain aggregate probabilities-derived metrics (ROC-AUC, PR-AUC, Brier, Log Loss) and threshold coverage/returns, but no per-prediction probability, timestamp, realized label, or confidence-bin counts. A future instrumented run would need timestamp, asset, timeframe, horizon, model_id, feature_set_id, probability, prediction, realized label, and realized future return per prediction. This phase did not modify the pipeline.

## I. GPU

Phase 4.2 used no GPU and performed no training. A later GPU phase could be justified only by a separately pre-registered question about a model family that cannot be evaluated by the frozen sklearn baselines; this diagnosis alone does not justify TCN or GPU boosting. CPU-only deployment remains required.

## J. Next Research Questions

1. Can per-prediction probability telemetry reproduce the observed Brier/Log-Loss behavior and resolve calibration-bin uncertainty in a separately frozen experiment?
2. Do the observed temporal categories persist under a newly pre-registered chronological sample, without changing the current Phase 4 design or using the current holdout for selection?
3. Does a separately specified proxy-information test explain the unstable Core-vs-Context deltas without promoting the proxy?

## Methodological Limits

- The final holdout is a single out-of-sample period and remains frozen.
- Aggregate artifacts cannot support per-prediction confidence histograms.
- These results describe information value, not a trading strategy.
- No best-model, best-asset, best-horizon, or promotion statement is made.

## Reproducibility
{
  "input_hashes": {
    "data\\research\\phase4\\phase4_ml_baseline_results.csv": "de1677bafce258817032c924a76b1149af86c2c10fcf61cee6fd31387871c0ab",
    "data\\research\\phase4\\phase4_ml_baseline_results.json": "f6d07163c2c4cc7f8d885263f2ff61f73d4884aba8229f3bb76af9ef693c1c8b",
    "data\\research\\phase4\\phase4_ml_baseline_manifest.json": "24192cfac0eea54cbb20aa9cda9464b0ff602a79458ae36abeae5592fcda0d85",
    "data\\research\\phase4\\phase4_analysis.csv": "b4ebc5d5c2d92bd3352bc4c3684fc7abd5b800f41ae0367c1c3a3e85f716fad4",
    "data\\research\\phase4\\phase4_analysis.json": "b8ed44be7c9a9ba99415e1c0f1377e21bf80059be6178366f861a463e3f8711a",
    "data\\research\\phase4\\phase4_analysis_manifest.json": "ebb0ed1ad6ca5418bc8068cb67900137555957659a1f69443fe6515b3e8b1e12",
    "research\\phase4_ml_baseline_analysis.md": "f8860875c67de65322ec65e9e1c3cc51a57b529f46fc8922deef568d9779b7f6",
    "configs\\ml_research_v1.yaml": "73026dd08db19c83059c57c75b65bcebf077280cee941bdcbc1155cb400f6517",
    "data\\research\\phase3\\ml_research_manifest.json": "04008f1be8bf80ba5b9849f3c76cd5924e46649c58653b1f9e0dcf698873f8f5"
  },
  "plots": [
    "data\\research\\phase4\\plots\\deep_diagnostic\\candidate_delta_brier.png",
    "data\\research\\phase4\\plots\\deep_diagnostic\\candidate_delta_roc_auc.png",
    "data\\research\\phase4\\plots\\deep_diagnostic\\candidate_performance_by_asset.png",
    "data\\research\\phase4\\plots\\deep_diagnostic\\candidate_performance_by_horizon.png",
    "data\\research\\phase4\\plots\\deep_diagnostic\\candidate_performance_by_model.png",
    "data\\research\\phase4\\plots\\deep_diagnostic\\candidate_validation_vs_holdout_pr_auc.png",
    "data\\research\\phase4\\plots\\deep_diagnostic\\candidate_validation_vs_holdout_roc_auc.png",
    "data\\research\\phase4\\plots\\deep_diagnostic\\candidate_wf_roc_auc.png"
  ],
  "checks_passed": true,
  "seed": 42
}
