# Phase 4.4 Label-Value Diagnostic

## 1. Executive Summary

No model was trained. Existing local OHLCV was used only to reconstruct frozen future outcomes. Evidence classification: **LABEL_VALUE_UNCLEAR**. Final conclusion: **NO LABEL-BASED ML ADVANTAGE DEMONSTRATED**.

## 2. Starting Point

Phase 4.3 ended with SIGNAL_WEAK, validation-to-holdout degradation, no stable model-family effect, and no stable Context Proxy advantage. This phase tests whether the target definition itself could explain part of that weakness.

## 3. Label Inventory

{
  "future_return": {
    "definition": "close[t+h] / close[t] - 1",
    "kind": "continuous regression",
    "status": "COMPUTED_READ_ONLY"
  },
  "future_log_return": {
    "definition": "log(close[t+h] / close[t])",
    "kind": "continuous regression",
    "status": "COMPUTED_READ_ONLY"
  },
  "direction": {
    "definition": "1 if future_return > 0 else 0",
    "kind": "binary classification",
    "status": "COMPUTED_READ_ONLY"
  },
  "cost_aware_direction": {
    "definition": "1 if future_return > 0.002 else 0",
    "kind": "binary classification",
    "status": "COMPUTED_READ_ONLY"
  },
  "mae": {
    "definition": "min(low[t+1:t+h]) / close[t] - 1",
    "kind": "continuous outcome",
    "status": "COMPUTED_READ_ONLY"
  },
  "mfe": {
    "definition": "max(high[t+1:t+h]) / close[t] - 1",
    "kind": "continuous outcome",
    "status": "COMPUTED_READ_ONLY"
  },
  "stop_hit_probability": {
    "definition": "no frozen implementation or stop specification",
    "kind": "classification/probability",
    "status": "NOT_AVAILABLE"
  },
  "holding_time": {
    "definition": "no frozen Phase 3 implementation; triple-barrier time_to_barrier is a separate label",
    "kind": "duration",
    "status": "NOT_AVAILABLE"
  }
}

`future_return`, `future_log_return`, `direction`, `cost_aware_direction`, MAE, and MFE were reconstructed read-only. `stop_hit_probability` and `holding_time` are not sufficiently implemented or persisted and remain unavailable.

## 4. Direction Diagnostic

`direction` is `future_return > 0`, so every positive move counts equally, including moves below the registered 20 bps cost-aware threshold. See the CSV for per-task and per-fold distributions.

## 5. Cost-Aware Direction

`cost_aware_direction` uses the frozen registered 20 bps diagnostic threshold. It is descriptive only; no threshold was optimized and no label was selected.

## 6. Continuous Outcomes

Future return/log-return, MAE, and MFE distributions and correlations are included in the JSON/CSV. These describe outcome value, not predictive model value.

## 7. WF and Holdout

WF1/WF2/WF3 and final holdout are reported separately. The holdout was never used for label selection, threshold selection, or optimization.

## 8. Signal to Return Alignment

Prediction-quantile to return alignment is **NOT AVAILABLE** because no alternative-label model predictions are stored and no new training was performed. Classification quality cannot be inferred from label prevalence alone.

## 9. Hypotheses

- Hypothesis A: remains plausible because Phase 4.3 showed weak and unstable generalization.
- Hypothesis B: structurally plausible because raw direction includes sub-cost moves, but not demonstrated as predictive value.
- Hypothesis C: remains plausible because classification metrics did not consistently align with return aggregates.
- Hypothesis D: remains plausible where label distributions vary by asset/horizon/period; this audit does not select among them.

## 10. Limitations

- No per-prediction probabilities, predictions, labels, or trade-level outcomes are persisted.
- No confidence intervals or artificial significance tests are computed.
- MAE/MFE are descriptive outcomes, not strategy rules.
- Stop-hit probability and holding time lack a frozen operational definition in the stored Phase 3 pipeline.

## 11. Final Research Conclusion

The existing evidence does not demonstrate that changing the label would produce a stable ML advantage. The label structure supports investigating cost-aware targets in a separately frozen future experiment, but it does not justify TCN, GPU, feature expansion, or Phase 5 now.

## 12. Next Research Questions

1. If ML research is resumed, can one small pre-registered diagnostic compare raw direction, cost-aware direction, and continuous return using the same frozen features/splits and no holdout selection?
2. Can per-prediction probabilities and realized returns be persisted to evaluate calibration and prediction-quantile return alignment?

## Reproducibility
{
  "input_hashes": {
    "configs\\ml_research_v1.yaml": "73026dd08db19c83059c57c75b65bcebf077280cee941bdcbc1155cb400f6517",
    "data\\research\\phase3\\ml_research_manifest.json": "04008f1be8bf80ba5b9849f3c76cd5924e46649c58653b1f9e0dcf698873f8f5",
    "data\\research\\phase4\\phase4_ml_baseline_results.csv": "de1677bafce258817032c924a76b1149af86c2c10fcf61cee6fd31387871c0ab",
    "data\\research\\phase4\\phase4_analysis.csv": "b4ebc5d5c2d92bd3352bc4c3684fc7abd5b800f41ae0367c1c3a3e85f716fad4",
    "data\\research\\phase4\\phase4_deep_diagnostic.json": "b59d4b6d4d844a0aa3ec3a1f9ee01ab2b6cf1a2a196428152de09d679d332ee1",
    "data\\research\\phase4\\phase4_generalization_diagnostic.json": "00e3a207854c6e5705b81b89ef0846435dc97af43347bb0532a1e9b79a210418",
    "data\\market_crypto_2017\\BTC_USDT\\1h\\ccxt.parquet": "5a25e901504ef9f5409d2af31d685df522f8f61d0bbc8b9e0a8642c4fca88f21",
    "data\\market_crypto_2017\\BTC_USDT\\4h\\ccxt.parquet": "9b1e523a096d45ca024a29c300ea37cb64cd64b2f9a389729e7424346e66cf98",
    "data\\market_crypto_2017\\ETH_USDT\\1h\\ccxt.parquet": "2e0b9b02b2bd9bb53fb0c96c95a163cbe82bb9ad47080fa5c64b964467ce202f",
    "data\\market_crypto_2017\\ETH_USDT\\4h\\ccxt.parquet": "a8b138a456e060450014a7e685e42abc902c8c87e479792b8107a05c8dd3946b"
  },
  "plots": [
    "data\\research\\phase4\\plots\\label_value\\direction_vs_cost_aware.png",
    "data\\research\\phase4\\plots\\label_value\\future_return_by_asset.png",
    "data\\research\\phase4\\plots\\label_value\\validation_label_stability.png"
  ],
  "training_performed": false,
  "holdout_used_for_selection": false
}
