# ML Output Contract

The current classifier emits probabilities in the order `SHORT`, `NO_TRADE`,
`LONG`. Historical labels are triple-barrier outcomes: the first configured
take-profit or stop barrier wins, and a time-barrier outcome is `NO_TRADE`.

`core.ml.scoring.score_trade_quality` converts probabilities and measured or
predicted favorable/adverse movement into a deterministic score from 0 to 100.
The score is a reporting layer, not a separately trained target. Threshold
metrics are available through `threshold_metrics` for gates such as 80 and 90.

Model sidecars contain the feature list, sequence length, architecture,
normalization parameters, label configuration, random seed, and training
cutoff. A final model trained on historical data must not be used to score
earlier historical bars; use the saved out-of-fold confidence series for that.