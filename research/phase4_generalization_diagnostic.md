# Phase 4.3 Generalization & Signal-Value Diagnostic

Read-only analysis of frozen Phase 4, 4.1, and 4.2 aggregate artifacts. No model was trained, tuned, selected, promoted, or changed.

## Decision

Evidence classification: **SIGNAL_WEAK**.
This classification describes the evidence base, not an asset, horizon, model, or candidate ranking.

## A. Validation to Holdout

All 144 experiment combinations have three validation folds and one final holdout. ROC-AUC gap summary: {'count': 144.0, 'mean': -0.015711, 'std': 0.037732, 'min': -0.159973, '25%': -0.028349, '50%': -0.019407, '75%': -0.004786, 'max': 0.130338}.
Absolute gap is `holdout - validation`; relative gap is divided by the absolute validation value where meaningful. Negative values indicate degradation for higher-is-better metrics. For loss metrics, a positive gap is deterioration.

## B. WF1/WF2/WF3 vs Holdout

{
  "PATTERN_3_WF_VARIABILITY": 64,
  "PATTERN_4_AUC_WITHOUT_RETURN_COVERAGE_CHANGE": 43,
  "MIXED_DESCRIPTIVE_PATTERN": 18,
  "PATTERN_2_WITHIN_WF_WEAKENING": 16,
  "PATTERN_1_STABLE_WF_HOLDOUT_DROP": 2,
  "PATTERN_5_NEAR_CHANCE_AUC": 1
}
Pattern labels are descriptive only: stable validation with a holdout drop, within-validation weakening, WF variability, AUC without return/coverage movement, and near-chance AUC.

## C. Classification Value vs Return Value

{
  "classification_advantage_rows": 139,
  "return_effect_rows": 144
}
ROC-AUC above 0.5 or PR-AUC above the observed positive-rate baseline does not by itself establish a tradeable return effect. The stored artifacts contain no trade-level PnL, fees, slippage, leverage, or execution data.

## D. Simple Baselines

Majority accuracy, chance ROC-AUC=0.5, positive-rate PR-AUC, positive-rate Brier, and constant positive-rate log loss are computed from stored fold positive rates only. No model was retrained and no backtest was added.
{
  "roc_auc_above_chance_rows": 128,
  "pr_auc_above_positive_rate_rows": 137,
  "accuracy_above_majority_rows": 78,
  "brier_better_than_positive_rate_rows": 42,
  "log_loss_better_than_positive_rate_rows": 42,
  "note": "Comparisons use only stored fold aggregates and are descriptive; no significance test is available."
}
[
  {
    "feature_set_id": "crypto_core_v1",
    "model_id": "hist_gradient_boosting"
  },
  {
    "feature_set_id": "crypto_core_v1",
    "model_id": "logistic_regression"
  },
  {
    "feature_set_id": "crypto_core_v1",
    "model_id": "random_forest"
  },
  {
    "feature_set_id": "crypto_core_v1+crypto_context_proxy_v1",
    "model_id": "hist_gradient_boosting"
  },
  {
    "feature_set_id": "crypto_core_v1+crypto_context_proxy_v1",
    "model_id": "logistic_regression"
  },
  {
    "feature_set_id": "crypto_core_v1+crypto_context_proxy_v1",
    "model_id": "random_forest"
  }
]

## E. Assets, Horizons, and Models

[
  {
    "target": "BTC/USDT"
  },
  {
    "target": "ETH/USDT"
  }
]

[
  {
    "horizon_bucket": "1-2h"
  },
  {
    "horizon_bucket": "1-3w"
  },
  {
    "horizon_bucket": "12-24h"
  },
  {
    "horizon_bucket": "3-7d"
  },
  {
    "horizon_bucket": "4-8h"
  },
  {
    "horizon_bucket": "longer frozen horizons"
  }
]

[
  {
    "model_id": "hist_gradient_boosting"
  },
  {
    "model_id": "logistic_regression"
  },
  {
    "model_id": "random_forest"
  }
]
No group is called best or preferred; differences are descriptive.

## F. Core vs Context Proxy

{
  "holdout": {
    "roc_auc": {
      "mean": 0.008927026731400322,
      "median": 0.0038703291171645615,
      "std": 0.03290917822971705,
      "min": -0.0817869406444946,
      "max": 0.11592862756655853,
      "iqr": 0.01611673829093506,
      "positive_fraction": 0.6805555555555556
    },
    "pr_auc": {
      "mean": 0.007325295287313912,
      "median": 0.0023252387705416377,
      "std": 0.027439440495008797,
      "min": -0.052250592271556484,
      "max": 0.11850987005891922,
      "iqr": 0.015474557065372832,
      "positive_fraction": 0.6388888888888888
    },
    "brier_score": {
      "mean": 0.019782095667796464,
      "median": 0.0014876220400439344,
      "std": 0.039145921087680065,
      "min": -0.007508694426814622,
      "max": 0.17502911090211348,
      "iqr": 0.016341646746744147,
      "positive_fraction": 0.6944444444444444
    },
    "log_loss": {
      "mean": 0.06057221075343059,
      "median": 0.0031887232855513648,
      "std": 0.13189177120647524,
      "min": -0.015834524546186057,
      "max": 0.6372739253446369,
      "iqr": 0.03726700876812894,
      "positive_fraction": 0.6944444444444444
    },
    "mean_forward_return": {
      "mean": 0.0,
      "median": 0.0,
      "std": 0.0,
      "min": 0.0,
      "max": 0.0,
      "iqr": 0.0,
      "positive_fraction": 0.0
    },
    "signal_coverage_up_0_6": {
      "mean": 0.13623703810174692,
      "median": 0.039066944069388404,
      "std": 0.21744489526598226,
      "min": -0.1996753246753247,
      "max": 0.7674418604651163,
      "iqr": 0.14908390700124288,
      "positive_fraction": 0.7777777777777778
    }
  },
  "validation": {
    "roc_auc": {
      "mean": 0.011303910704164252,
      "median": -0.0031045663088580255,
      "std": 0.060954896614145375,
      "min": -0.17132950224680266,
      "max": 0.2674845440494591,
      "iqr": 0.03658965581446,
      "positive_fraction": 0.37962962962962965
    },
    "pr_auc": {
      "mean": 0.008531573066224329,
      "median": -0.003412106046410457,
      "std": 0.05474844334082782,
      "min": -0.11620011852640821,
      "max": 0.2414190111823129,
      "iqr": 0.03129115968931255,
      "positive_fraction": 0.37962962962962965
    },
    "brier_score": {
      "mean": 0.029124744069374802,
      "median": 0.008199413621416893,
      "std": 0.04750501555371851,
      "min": -0.004585367180102495,
      "max": 0.3036075188526046,
      "iqr": 0.02887710497761964,
      "positive_fraction": 0.9444444444444444
    },
    "log_loss": {
      "mean": 0.0943610409519356,
      "median": 0.018427485640265695,
      "std": 0.18878391563877145,
      "min": -0.011498775603385858,
      "max": 1.3833976654186868,
      "iqr": 0.07887462908519793,
      "positive_fraction": 0.9444444444444444
    },
    "mean_forward_return": {
      "mean": 0.0,
      "median": 0.0,
      "std": 0.0,
      "min": 0.0,
      "max": 0.0,
      "iqr": 0.0,
      "positive_fraction": 0.0
    },
    "signal_coverage_up_0_6": {
      "mean": 0.04481230728609703,
      "median": -0.0006849706055767507,
      "std": 0.17375191812798865,
      "min": -0.3287795992714025,
      "max": 0.6045662100456621,
      "iqr": 0.072322616368232,
      "positive_fraction": 0.4861111111111111
    }
  }
}
Deltas are paired on identical target/timeframe/horizon/model/split keys. Positive Brier or Log Loss delta is worse; positive ROC-AUC/PR-AUC/return delta is higher. The proxy is not promoted.

## G. Missing Data

Available: fold-level aggregate classification metrics, threshold coverage, thresholded forward-return aggregates, split IDs, date ranges, sample counts, model IDs, feature-set IDs, and horizons.

Missing: individual probabilities, prediction timestamps, individual labels, individual future returns, calibration buckets, reliability-curve data, and trade-level execution outcomes. Therefore confidence intervals, calibration curves, and artificial significance tests are **NOT AVAILABLE FROM STORED AGGREGATES**.

## H. Holdout and GPU Status

The holdout was used only for final OOS measurement and validation-to-holdout diagnostics. It was not used for feature, model, threshold, horizon, candidate, or strategy selection. No GPU/TCN training was performed. **NO JUSTIFIED GPU EXPERIMENT YET**: the current weakness is unresolved generalization and missing per-prediction telemetry, not a demonstrated capacity limit of sklearn baselines.

## I. Next Research Questions

1. Can a separately frozen, instrumented run capture per-prediction probability/label/return telemetry for proper calibration and signal-value analysis?
2. Does a new chronological validation design reproduce the observed WF-to-holdout behavior without reusing this holdout for selection?
3. Can the Core-vs-Context instability be explained descriptively across time and assets before any new feature-set experiment is considered?

## Reproducibility
{
  "input_hashes": {
    "data\\research\\phase4\\phase4_ml_baseline_results.csv": "de1677bafce258817032c924a76b1149af86c2c10fcf61cee6fd31387871c0ab",
    "data\\research\\phase4\\phase4_ml_baseline_results.json": "f6d07163c2c4cc7f8d885263f2ff61f73d4884aba8229f3bb76af9ef693c1c8b",
    "data\\research\\phase4\\phase4_ml_baseline_manifest.json": "24192cfac0eea54cbb20aa9cda9464b0ff602a79458ae36abeae5592fcda0d85",
    "data\\research\\phase4\\phase4_analysis.csv": "b4ebc5d5c2d92bd3352bc4c3684fc7abd5b800f41ae0367c1c3a3e85f716fad4",
    "data\\research\\phase4\\phase4_analysis.json": "b8ed44be7c9a9ba99415e1c0f1377e21bf80059be6178366f861a463e3f8711a",
    "data\\research\\phase4\\phase4_analysis_manifest.json": "ebb0ed1ad6ca5418bc8068cb67900137555957659a1f69443fe6515b3e8b1e12",
    "data\\research\\phase4\\phase4_deep_diagnostic.csv": "add708a590da53c02c381be17c85f03ba5fbd1bd32aaed99b09491bb8179ad9d",
    "data\\research\\phase4\\phase4_deep_diagnostic.json": "b59d4b6d4d844a0aa3ec3a1f9ee01ab2b6cf1a2a196428152de09d679d332ee1",
    "configs\\ml_research_v1.yaml": "73026dd08db19c83059c57c75b65bcebf077280cee941bdcbc1155cb400f6517",
    "data\\research\\phase3\\ml_research_manifest.json": "04008f1be8bf80ba5b9849f3c76cd5924e46649c58653b1f9e0dcf698873f8f5"
  },
  "plots": [
    "data\\research\\phase4\\plots\\generalization\\wf_validation_holdout_roc_auc.png",
    "data\\research\\phase4\\plots\\generalization\\validation_vs_holdout_roc_auc.png",
    "data\\research\\phase4\\plots\\generalization\\validation_vs_holdout_pr_auc.png",
    "data\\research\\phase4\\plots\\generalization\\generalization_gap_by_horizon.png",
    "data\\research\\phase4\\plots\\generalization\\generalization_gap_by_asset.png",
    "data\\research\\phase4\\plots\\generalization\\generalization_gap_by_model.png",
    "data\\research\\phase4\\plots\\generalization\\core_vs_context_generalization.png",
    "data\\research\\phase4\\plots\\generalization\\forward_return_vs_roc_auc.png"
  ],
  "seed": 42,
  "holdout_used_for_selection": false
}

