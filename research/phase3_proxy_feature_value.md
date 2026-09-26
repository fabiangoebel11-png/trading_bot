# Phase 3.2 Proxy Feature Value

## Scope

This is a controlled research diagnostic for the existing BTC/ETH causal
pipeline. It does not select a trading strategy, promote a model, or modify
live, paper, risk, or execution code.

The run evaluates four fixed target/timeframe/horizon combinations:

- BTC/USDT 1h -> 4h and 24h
- ETH/USDT 1h -> 7d
- ETH/USDT 4h -> 14d

For each combination, six feature groups are evaluated with Logistic
Regression and HistGradientBoostingClassifier. The proxy source is the local
`ml_macro_ESF_NQF_VIX_URTH_2600d.csv` cache, aligned by the existing causal
adapter. The groups are crypto-only, all proxies, and one group each for ES,
NQ, VIX, and URTH.

## Protocol

- Train: data through 2021, shortened by one horizon purge and one horizon
  embargo before the validation boundary.
- Validation: 2022-01-01 through 2023-12-31.
- Holdout: 2024-01-01 through 2026-09-01.
- Scaling: `StandardScaler` fitted on train only.
- Model fit: each model is fitted on train only; the same fitted model is
  evaluated on validation and holdout.
- Holdout selection: prohibited and recorded as `false` in every result.

The run produced 96 records: 4 datasets x 6 feature groups x 2 models x 2
evaluation splits.

## Results

Mean ROC-AUC across all four datasets and both models:

| Feature group | Validation | Holdout |
|---|---:|---:|
| crypto-only | 0.5133 | 0.5188 |
| all proxies | 0.5269 | 0.5195 |
| ES | 0.5157 | 0.5173 |
| NQ | 0.5136 | 0.5146 |
| VIX | 0.5143 | 0.5278 |
| URTH | 0.5174 | 0.5179 |

These averages do not establish reproducible incremental value. The all-proxy
group has a small validation advantage but does not preserve a clear advantage
on holdout. VIX has the strongest average holdout ROC-AUC, but this is not
stable evidence across every dataset/model pair. Several proxy groups also
increase holdout log loss and reduce the diagnostic signed-return statistic.

## Conclusion

Current status: `NO_EVIDENCE` for robust, reproducible proxy value. The results
are period- and model-dependent and remain research diagnostics only. No proxy
group should be promoted into a live, paper, or strategy-selection path from
this run.

Artifacts:

- `data/research/phase3/proxy_feature_value.csv`
- `data/research/phase3/proxy_feature_value.json`
- `data/research/phase3/proxy_feature_value_manifest.json`