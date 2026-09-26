# Phase 4.1 Frozen ML Baseline Analysis

Read-only analysis of existing Phase 4 artifacts. No model was retrained, selected, promoted, or changed.

## 1. Executive Summary

Analyzed 576 records: 432 validation and 144 final-holdout records. Artifact checks passed: 15/15.
The classification is diagnostic only. It is not a model ranking and does not authorize trading deployment.

## 2. Research Design

Frozen matrix: 24 BTC/ETH dataset-horizon tasks × 2 feature sets × 3 sklearn baselines × 3 validation folds plus one final holdout evaluation.
Validation-only classification uses all three walk-forward folds. Holdout values are reported after, but never enter, classification or selection.

## 3. Databases, Models, and Feature Sets

Assets: BTC/USDT, ETH/USDT. Input timeframes: 1h and 4h. Horizons: 1h, 2h, 4h, 8h, 12h, 24h, 3d, 7d, 14d, 21d, 28d.
Models: LogisticRegression, HistGradientBoostingClassifier, RandomForestClassifier.
Feature sets: crypto_core_v1 and crypto_core_v1+crypto_context_proxy_v1. The context set remains an optional proxy context, not an automatically valuable input.

## 4. Validation Results

The complete machine-readable per-combination summary is in `data/research/phase4/phase4_analysis.csv`; the table below is a compact view.
  target timeframe horizon                         feature_set_id               model_id   classification  validation_roc_auc_mean  validation_roc_auc_std  validation_pr_auc_mean  validation_brier_score_mean  validation_mean_forward_return_mean
BTC/USDT        1h     12h                         crypto_core_v1 hist_gradient_boosting            MIXED                 0.531525                0.009769                0.533719                     0.252136                             0.000508
BTC/USDT        1h     12h                         crypto_core_v1    logistic_regression ROBUST CANDIDATE                 0.544988                0.009552                0.547024                     0.248832                             0.000508
BTC/USDT        1h     12h                         crypto_core_v1          random_forest            MIXED                 0.544841                0.011150                0.542182                     0.249799                             0.000508
BTC/USDT        1h     12h crypto_core_v1+crypto_context_proxy_v1 hist_gradient_boosting            MIXED                 0.531280                0.008075                0.528389                     0.262540                             0.000508
BTC/USDT        1h     12h crypto_core_v1+crypto_context_proxy_v1    logistic_regression            MIXED                 0.532643                0.010053                0.541072                     0.253012                             0.000508
BTC/USDT        1h     12h crypto_core_v1+crypto_context_proxy_v1          random_forest            MIXED                 0.541998                0.012942                0.538073                     0.255991                             0.000508
BTC/USDT        1h      1h                         crypto_core_v1 hist_gradient_boosting ROBUST CANDIDATE                 0.566858                0.006398                0.564707                     0.246666                             0.000043
BTC/USDT        1h      1h                         crypto_core_v1    logistic_regression ROBUST CANDIDATE                 0.554986                0.007741                0.553116                     0.247829                             0.000043
BTC/USDT        1h      1h                         crypto_core_v1          random_forest ROBUST CANDIDATE                 0.564314                0.005365                0.561773                     0.247138                             0.000043
BTC/USDT        1h      1h crypto_core_v1+crypto_context_proxy_v1 hist_gradient_boosting ROBUST CANDIDATE                 0.559413                0.003988                0.557565                     0.247579                             0.000043
BTC/USDT        1h      1h crypto_core_v1+crypto_context_proxy_v1    logistic_regression ROBUST CANDIDATE                 0.550986                0.005028                0.549939                     0.248579                             0.000043
BTC/USDT        1h      1h crypto_core_v1+crypto_context_proxy_v1          random_forest ROBUST CANDIDATE                 0.563206                0.004478                0.560176                     0.247447                             0.000043
BTC/USDT        1h     24h                         crypto_core_v1 hist_gradient_boosting            MIXED                 0.522768                0.012865                0.528671                     0.253606                             0.001026
BTC/USDT        1h     24h                         crypto_core_v1    logistic_regression ROBUST CANDIDATE                 0.537535                0.006472                0.544637                     0.249378                             0.001026
BTC/USDT        1h     24h                         crypto_core_v1          random_forest            MIXED                 0.535562                0.016272                0.538419                     0.250674                             0.001026
BTC/USDT        1h     24h crypto_core_v1+crypto_context_proxy_v1 hist_gradient_boosting            MIXED                 0.545389                0.007804                0.542830                     0.277247                             0.001026
BTC/USDT        1h     24h crypto_core_v1+crypto_context_proxy_v1    logistic_regression            MIXED                 0.529381                0.004422                0.548492                     0.255881                             0.001026
BTC/USDT        1h     24h crypto_core_v1+crypto_context_proxy_v1          random_forest            MIXED                 0.539988                0.017705                0.540005                     0.266283                             0.001026
BTC/USDT        1h      2h                         crypto_core_v1 hist_gradient_boosting ROBUST CANDIDATE                 0.561157                0.006106                0.559109                     0.247617                             0.000086
BTC/USDT        1h      2h                         crypto_core_v1    logistic_regression ROBUST CANDIDATE                 0.551757                0.008595                0.553094                     0.248155                             0.000086
BTC/USDT        1h      2h                         crypto_core_v1          random_forest ROBUST CANDIDATE                 0.563572                0.006019                0.561096                     0.247296                             0.000086
BTC/USDT        1h      2h crypto_core_v1+crypto_context_proxy_v1 hist_gradient_boosting            MIXED                 0.553561                0.012589                0.550012                     0.249669                             0.000086
BTC/USDT        1h      2h crypto_core_v1+crypto_context_proxy_v1    logistic_regression            MIXED                 0.548400                0.008121                0.550202                     0.249427                             0.000086
BTC/USDT        1h      2h crypto_core_v1+crypto_context_proxy_v1          random_forest ROBUST CANDIDATE                 0.560669                0.011116                0.559736                     0.247986                             0.000086
BTC/USDT        1h      3d                         crypto_core_v1 hist_gradient_boosting            MIXED                 0.500501                0.035352                0.518027                     0.258579                             0.003183
BTC/USDT        1h      3d                         crypto_core_v1    logistic_regression            MIXED                 0.500082                0.032093                0.531527                     0.250861                             0.003183
BTC/USDT        1h      3d                         crypto_core_v1          random_forest            MIXED                 0.502914                0.035817                0.518938                     0.256005                             0.003183
BTC/USDT        1h      3d crypto_core_v1+crypto_context_proxy_v1 hist_gradient_boosting            MIXED                 0.511300                0.020061                0.532934                     0.314193                             0.003183
BTC/USDT        1h      3d crypto_core_v1+crypto_context_proxy_v1    logistic_regression            MIXED                 0.515153                0.025143                0.547128                     0.264117                             0.003183
BTC/USDT        1h      3d crypto_core_v1+crypto_context_proxy_v1          random_forest            MIXED                 0.532523                0.027900                0.556828                     0.286077                             0.003183

## 5. Walk-Forward Stability

For each combination, the analysis exports mean/std/min/max for ROC-AUC, PR-AUC, balanced accuracy, Brier, log loss, forward return, thresholded returns, and coverage. The neutral labels are defined before interpreting results:
- ROBUST CANDIDATE: all three AUC values >= 0.50, mean AUC >= 0.505, AUC std <= 0.02, at least two PR-AUC values >= 0.50, max Brier <= 0.25, and 0.60-up coverage range <= 0.50.
- MIXED: at least one validation AUC >= 0.50 or mean AUC >= 0.50, but robust criteria are not met.
- WEAK: no validation AUC >= 0.50 and mean AUC < 0.50.
- FAILED: fewer than three validation folds or missing core metrics.
No single strong walk-forward period is sufficient for ROBUST CANDIDATE.

## 6. Multi-Horizon Analysis

Mean validation ROC-AUC by input timeframe and horizon:
timeframe horizon  metric_roc_auc
       1h     12h        0.534500
       1h      1h        0.556796
       1h     24h        0.530487
       1h      2h        0.551865
       1h      3d        0.513457
       1h      4h        0.548381
       1h      7d        0.511787
       1h      8h        0.541867
       4h     14d        0.512427
       4h     21d        0.540496
       4h     28d        0.540079
       4h      7d        0.513680
Longer horizons, shorter horizons, and input-timeframe effects are descriptive patterns only; no horizon is selected or optimized here.

## 7. BTC vs ETH

  target  metric_roc_auc  metric_pr_auc  metric_brier_score  metric_mean_forward_return
BTC/USDT        0.540525       0.559801            0.270695                    0.008062
ETH/USDT        0.525445       0.527536            0.267168                    0.004513
Differences between BTC and ETH are reported as asset-specific behavior, not as a basis for deployment selection.

## 8. Core vs Context Proxy

Paired context deltas use identical asset, timeframe, horizon, model, and walk-forward keys. Positive ROC-AUC/PR-AUC/return deltas and negative Brier deltas are directionally favorable, but consistency matters more than isolated values.
  target timeframe horizon               model_id split_id  delta_roc_auc  delta_pr_auc  delta_brier_score  delta_mean_forward_return
BTC/USDT        1h      1h    logistic_regression  wf_2022      -0.007566     -0.004825           0.001257                        0.0
BTC/USDT        1h      1h hist_gradient_boosting  wf_2022      -0.015060     -0.012944           0.001806                        0.0
BTC/USDT        1h      1h          random_forest  wf_2022      -0.007389     -0.007041           0.000879                        0.0
BTC/USDT        1h      1h    logistic_regression  wf_2023      -0.005285     -0.003761           0.000466                        0.0
BTC/USDT        1h      1h hist_gradient_boosting  wf_2023      -0.005061     -0.008314           0.000802                        0.0
BTC/USDT        1h      1h          random_forest  wf_2023       0.003348      0.003643          -0.000236                        0.0
BTC/USDT        1h      1h    logistic_regression  wf_2024       0.000850     -0.000944           0.000525                        0.0
BTC/USDT        1h      1h hist_gradient_boosting  wf_2024      -0.002214     -0.000166           0.000133                        0.0
BTC/USDT        1h      1h          random_forest  wf_2024       0.000717     -0.001394           0.000284                        0.0
BTC/USDT        1h      2h    logistic_regression  wf_2022      -0.004560     -0.002246           0.002588                        0.0
BTC/USDT        1h      2h hist_gradient_boosting  wf_2022      -0.017295     -0.018818           0.004574                        0.0
BTC/USDT        1h      2h          random_forest  wf_2022      -0.008152     -0.008944           0.001011                        0.0
BTC/USDT        1h      2h    logistic_regression  wf_2023      -0.004077     -0.003782           0.000186                        0.0
BTC/USDT        1h      2h hist_gradient_boosting  wf_2023      -0.005064     -0.005547           0.000785                        0.0
BTC/USDT        1h      2h          random_forest  wf_2023       0.004115      0.007036          -0.000335                        0.0
BTC/USDT        1h      2h    logistic_regression  wf_2024      -0.001433     -0.002650           0.001042                        0.0
BTC/USDT        1h      2h hist_gradient_boosting  wf_2024      -0.000430     -0.002924           0.000795                        0.0
BTC/USDT        1h      2h          random_forest  wf_2024      -0.004672     -0.002174           0.001394                        0.0
BTC/USDT        1h      4h    logistic_regression  wf_2022      -0.005073     -0.002798           0.005228                        0.0
BTC/USDT        1h      4h hist_gradient_boosting  wf_2022      -0.014203     -0.013895           0.006004                        0.0
BTC/USDT        1h      4h          random_forest  wf_2022      -0.012113     -0.010607           0.002361                        0.0
BTC/USDT        1h      4h    logistic_regression  wf_2023      -0.005926     -0.006846           0.000431                        0.0
BTC/USDT        1h      4h hist_gradient_boosting  wf_2023      -0.004461      0.000259           0.001045                        0.0
BTC/USDT        1h      4h          random_forest  wf_2023      -0.002649      0.002437           0.000901                        0.0
BTC/USDT        1h      4h    logistic_regression  wf_2024      -0.002441     -0.005564           0.002069                        0.0
BTC/USDT        1h      4h hist_gradient_boosting  wf_2024       0.002498     -0.000072           0.002512                        0.0
BTC/USDT        1h      4h          random_forest  wf_2024      -0.005477     -0.004578           0.002588                        0.0
BTC/USDT        1h      8h    logistic_regression  wf_2022      -0.004546     -0.002589           0.006490                        0.0
BTC/USDT        1h      8h hist_gradient_boosting  wf_2022      -0.002368     -0.012423           0.008101                        0.0
BTC/USDT        1h      8h          random_forest  wf_2022      -0.001376     -0.010450           0.004334                        0.0
Aggregate delta diagnostics:
       delta_roc_auc  delta_pr_auc  delta_brier_score  delta_mean_forward_return
count     216.000000    216.000000         216.000000                      216.0
mean        0.011304      0.008532           0.029125                        0.0
std         0.061096      0.054876           0.047615                        0.0
min        -0.171330     -0.116200          -0.004585                        0.0
25%        -0.012594     -0.012507           0.001584                        0.0
50%        -0.003105     -0.003412           0.008199                        0.0
75%         0.023996      0.018784           0.030461                        0.0
max         0.267485      0.241419           0.303608                        0.0
No automatic proxy promotion is made.

## 9. Classification Counts

{
  "MIXED": 112,
  "ROBUST CANDIDATE": 31,
  "WEAK": 1
}

## 10. Calibration and Classification vs Market Relevance

Brier score and log loss are analyzed as aggregate calibration diagnostics. The artifacts do not contain per-prediction probabilities, confidence bins, or actual-positive-rate-by-confidence-bin records, so the requested 0.50-0.55/.../>0.70 calibration table cannot be reconstructed without new inference output. Thresholded coverage and forward-return columns are retained as descriptive diagnostics only.
ROC-AUC/PR-AUC measure classification information. Mean forward return, thresholded forward return, and signal coverage measure a separate market-information dimension. A high AUC therefore is not interpreted as a trading signal recommendation.

## 11. Validation to Holdout

Holdout summary is deliberately separate. The following describes degradation/stability after the frozen validation analysis; it did not influence classification:
       holdout_minus_validation_roc_auc  holdout_minus_validation_pr_auc  holdout_minus_validation_brier_score  holdout_minus_validation_mean_forward_return
count                        144.000000                       144.000000                            144.000000                                    144.000000
mean                          -0.015711                        -0.025002                             -0.005119                                     -0.005746
std                            0.037732                         0.036305                              0.020329                                      0.009359
min                           -0.159973                        -0.168123                             -0.074977                                     -0.038145
25%                           -0.028349                        -0.029642                             -0.007135                                     -0.008109
50%                           -0.019407                        -0.020518                             -0.000362                                     -0.000959
75%                           -0.004786                        -0.009042                              0.001880                                     -0.000085
max                            0.130338                         0.087435                              0.088681                                     -0.000009

## 12. Hardware

Phase 4 training backend: CPU-only sklearn. The RTX 3070 was present, but CUDA was not active and GPU training was not used. GPU availability is not interpreted as a result. CPU-only inference remains the architectural requirement.

## 13. Failure Modes and Limitations

- Results are aggregate fold diagnostics, not a trading backtest or promotion test.
- Proxy context is observational and remains proxy-only.
- No per-prediction confidence distribution is stored, limiting calibration analysis.
- Holdout is one chronological period and cannot establish future persistence.
- Threshold metrics are descriptive and were not optimized.

## 14. Research Conclusions

The artifacts support a neutral assessment of temporal generalization, asset differences, horizon behavior, and proxy deltas. They do not support automatic model promotion, strategy changes, threshold changes, leverage changes, or live/paper-trading changes.

## 15. Explicit Next-Step Candidates

- Review the machine-readable paired deltas and stability summaries with human sign-off.
- If calibration is required, design a separate pre-registered data capture step for per-prediction probabilities; do not retrofit thresholds from this report.
- Keep any future experiment separate from the frozen Phase 4 artifacts and do not modify the trading runtime as part of this analysis.

## Reproducibility

{
  "seed": 42,
  "input_hashes": {
    "phase4_ml_baseline_results.csv": "de1677bafce258817032c924a76b1149af86c2c10fcf61cee6fd31387871c0ab",
    "phase4_ml_baseline_results.json": "f6d07163c2c4cc7f8d885263f2ff61f73d4884aba8229f3bb76af9ef693c1c8b",
    "phase4_ml_baseline_manifest.json": "24192cfac0eea54cbb20aa9cda9464b0ff602a79458ae36abeae5592fcda0d85",
    "phase4_ml_baseline_report.md": "210c32ad13d24cf65d9dc17228ffa8dab297c46dd5a2ebd8401d6cc8a7392b02"
  },
  "plots": [
    "data\\research\\phase4\\plots\\validation_pr_auc_by_horizon.png",
    "data\\research\\phase4\\plots\\validation_roc_auc_by_horizon.png",
    "data\\research\\phase4\\plots\\validation_signal_coverage_by_horizon.png",
    "data\\research\\phase4\\plots\\validation_vs_holdout_roc_auc.png"
  ]
}
