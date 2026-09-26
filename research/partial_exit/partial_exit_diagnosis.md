# Partial Exit Post-Research Diagnosis

## 1. Executive Summary

**Decision: NO ROBUST PARTIAL EXIT ADVANTAGE FOUND**

Partial exits generally monetize early movement but give up too much of the long trend; variants with nearly unchanged results often did not execute meaningful TP events.

The diagnosis reran the study after correcting partial-exit accounting. The previous report was not treated as final evidence because realized TP/stop/signal PnL had not fully entered the bar-return series and entry fees were absent from trade net returns.

## 2. Technical Analysis

The original research tested 19 variants. The diagnosis adds 4 explicitly labeled diagnostic policies: two training-only MFE-quantile policies and two simple scale-outs.

Entries and direction remain fixed at `breakout_atr_60_1.0`; only stateful management changes. Entries fill next open, stops precede TP on the same bar, and MFE-derived targets use training trades only.

### Original Policy Families

| Policy | Allocation | TP method | TP values | Stop | Final exit |
|---|---|---|---|---|---|
| `runner_25_25_50` | (0.25, 0.25, 0.5) | atr_multiple | (1.0, 2.0) | unchanged | signal |
| `runner_33_33_34` | (0.33, 0.33, 0.34) | atr_multiple | (1.0, 2.0) | unchanged | signal |
| `runner_50_50` | (0.5, 0.5) | atr_multiple | (1.5,) | unchanged | signal |
| `runner_20_30_50` | (0.2, 0.3, 0.5) | atr_multiple | (1.0, 2.0) | unchanged | signal |
| `partial_breakeven` | (0.25, 0.25, 0.5) | atr_multiple | (1.0, 2.0) | break_even | signal |
| `partial_atr_trailing` | (0.25, 0.25, 0.5) | atr_multiple | (1.0, 2.0) | atr_trailing | atr_trailing |
| `partial_chandelier` | (0.25, 0.25, 0.5) | atr_multiple | (1.0, 2.0) | chandelier | chandelier |
| `partial_donchian_runner` | (0.25, 0.25, 0.5) | atr_multiple | (1.0, 2.0) | donchian | donchian |
| `partial_ema_runner` | (0.25, 0.25, 0.5) | atr_multiple | (1.0, 2.0) | ema | ema |
| `atr_grid_0.5_1.0` | (0.25, 0.25, 0.5) | atr_multiple | (0.5, 1.0) | unchanged | signal |
| `atr_grid_1.0_2.0` | (0.25, 0.25, 0.5) | atr_multiple | (1.0, 2.0) | unchanged | signal |
| `atr_grid_1.5_3.0` | (0.25, 0.25, 0.5) | atr_multiple | (1.5, 3.0) | unchanged | signal |
| `atr_grid_2.0_4.0` | (0.25, 0.25, 0.5) | atr_multiple | (2.0, 4.0) | unchanged | signal |
| `fixed_pct_0.005_0.010` | (0.25, 0.25, 0.5) | fixed_pct | (0.005, 0.01) | unchanged | signal |
| `fixed_pct_0.010_0.020` | (0.25, 0.25, 0.5) | fixed_pct | (0.01, 0.02) | unchanged | signal |
| `vol_normalized_0.50_0.75` | (0.25, 0.25, 0.5) | vol_normalized | (0.5, 0.75) | unchanged | signal |
| `vol_normalized_0.75_1.00` | (0.25, 0.25, 0.5) | vol_normalized | (0.75, 1.0) | unchanged | signal |
| `atr_dynamic_0.50_1.00` | (0.25, 0.25, 0.5) | atr_dynamic | (0.5, 1.0) | unchanged | signal |
| `atr_dynamic_0.75_1.00` | (0.25, 0.25, 0.5) | atr_dynamic | (0.75, 1.0) | unchanged | signal |

## 3. Baseline vs Partial Exits

All rows below are diagnostic comparisons; none is promoted.

| Symbol | Policy | OOS return | OOS Sharpe | OOS Sortino | OOS max DD | OOS PF | Exp. | Win rate | Median trade | Worst trade | TP any | WF Sharpe | Stress Sharpe |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BTC/USDT | `runner_25_25_50` | 0.11% | 0.073 | 0.040 | -16.60% | 1.04 | 0.021% | 46.8% | -0.095% | -3.202% | 68.3% | 0.434 | 0.293 |
| BTC/USDT | `runner_33_33_34` | 0.80% | 0.093 | 0.050 | -15.20% | 1.04 | 0.022% | 51.6% | 0.051% | -3.202% | 68.3% | 0.443 | 0.340 |
| BTC/USDT | `runner_50_50` | 1.58% | 0.124 | 0.066 | -15.21% | 1.06 | 0.033% | 53.2% | 0.059% | -3.202% | 60.3% | 0.424 | 0.415 |
| BTC/USDT | `runner_20_30_50` | 0.91% | 0.101 | 0.056 | -16.34% | 1.05 | 0.028% | 46.0% | -0.119% | -3.202% | 68.3% | 0.446 | 0.319 |
| BTC/USDT | `partial_breakeven` | -2.46% | -0.035 | -0.017 | -18.64% | 0.99 | -0.003% | 62.7% | 0.075% | -3.202% | 68.3% | 0.460 | 0.298 |
| BTC/USDT | `partial_atr_trailing` | 3.41% | 0.190 | 0.096 | -17.51% | 1.08 | 0.045% | 49.2% | -0.080% | -3.202% | 68.3% | 0.627 | 0.426 |
| BTC/USDT | `partial_chandelier` | -2.94% | -0.044 | -0.023 | -18.50% | 0.99 | -0.005% | 45.9% | -0.181% | -3.202% | 63.7% | 0.491 | 0.310 |
| BTC/USDT | `partial_donchian_runner` | -0.91% | 0.037 | 0.020 | -16.90% | 1.02 | 0.013% | 46.9% | -0.095% | -3.202% | 67.2% | 0.415 | 0.284 |
| BTC/USDT | `partial_ema_runner` | 1.10% | 0.106 | 0.055 | -18.38% | 1.05 | 0.025% | 42.3% | -0.261% | -3.202% | 60.6% | 0.613 | 0.326 |
| BTC/USDT | `atr_grid_0.5_1.0` | -8.06% | -0.254 | -0.136 | -19.36% | 0.91 | -0.049% | 45.2% | -0.100% | -3.202% | 78.6% | 0.106 | 0.055 |
| BTC/USDT | `atr_grid_1.0_2.0` | 0.11% | 0.073 | 0.040 | -16.60% | 1.04 | 0.021% | 46.8% | -0.095% | -3.202% | 68.3% | 0.434 | 0.293 |
| BTC/USDT | `atr_grid_1.5_3.0` | 5.77% | 0.258 | 0.144 | -14.50% | 1.11 | 0.067% | 44.4% | -0.208% | -3.202% | 60.3% | 0.607 | 0.495 |
| BTC/USDT | `atr_grid_2.0_4.0` | 4.61% | 0.217 | 0.121 | -17.75% | 1.09 | 0.062% | 42.9% | -0.338% | -3.202% | 50.8% | 0.666 | 0.456 |
| BTC/USDT | `fixed_pct_0.005_0.010` | -7.48% | -0.224 | -0.121 | -19.33% | 0.92 | -0.043% | 46.0% | -0.084% | -3.202% | 70.6% | 0.087 | -0.063 |
| BTC/USDT | `fixed_pct_0.010_0.020` | 1.34% | 0.116 | 0.065 | -17.12% | 1.05 | 0.032% | 46.0% | -0.241% | -3.202% | 55.6% | 0.456 | 0.290 |
| BTC/USDT | `vol_normalized_0.50_0.75` | -2.74% | 0.026 | 0.015 | -22.33% | 1.03 | 0.020% | 35.7% | -0.540% | -3.202% | 0.0% | 0.405 | 0.273 |
| BTC/USDT | `vol_normalized_0.75_1.00` | -2.74% | 0.026 | 0.015 | -22.33% | 1.03 | 0.020% | 35.7% | -0.540% | -3.202% | 0.0% | 0.405 | 0.273 |
| BTC/USDT | `atr_dynamic_0.50_1.00` | -3.61% | -0.066 | -0.035 | -16.76% | 0.98 | -0.010% | 45.2% | -0.119% | -3.202% | 76.2% | 0.344 | 0.178 |
| BTC/USDT | `atr_dynamic_0.75_1.00` | -2.40% | -0.015 | -0.008 | -15.79% | 1.00 | 0.001% | 46.0% | -0.078% | -3.202% | 69.8% | 0.361 | 0.219 |
| BTC/USDT | `mfe_quantile_q75_50_100` | 3.36% | 0.180 | 0.104 | -17.68% | 1.07 | 0.054% | 38.1% | -0.492% | -3.202% | 31.0% | 0.577 | 0.647 |
| BTC/USDT | `mfe_quantile_q75_75_100` | 2.87% | 0.166 | 0.096 | -18.34% | 1.07 | 0.053% | 36.5% | -0.540% | -3.202% | 17.5% | 0.522 | 0.576 |
| BTC/USDT | `simple_scaleout_25_25_50` | -6.76% | -0.219 | -0.098 | -21.44% | 0.92 | -0.038% | 62.7% | 0.106% | -3.202% | 65.7% | 0.240 | 0.152 |
| BTC/USDT | `simple_scaleout_50_50` | -3.49% | -0.068 | -0.032 | -17.35% | 0.98 | -0.010% | 60.3% | 0.285% | -3.202% | 60.3% | 0.338 | 0.216 |
| ETH/USDT | `runner_25_25_50` | 53.94% | 1.160 | 0.620 | -21.77% | 1.51 | 0.463% | 48.1% | -0.224% | -8.407% | 68.5% | 1.070 | 1.088 |
| ETH/USDT | `runner_33_33_34` | 36.42% | 0.955 | 0.491 | -20.37% | 1.39 | 0.333% | 52.8% | 0.208% | -8.407% | 68.5% | 1.046 | 0.983 |
| ETH/USDT | `runner_50_50` | 47.83% | 1.060 | 0.562 | -20.73% | 1.45 | 0.425% | 50.9% | 0.167% | -8.407% | 56.5% | 1.083 | 1.185 |
| ETH/USDT | `runner_20_30_50` | 56.14% | 1.185 | 0.635 | -21.80% | 1.52 | 0.477% | 48.1% | -0.195% | -8.407% | 68.5% | 1.089 | 1.093 |
| ETH/USDT | `partial_breakeven` | 26.88% | 0.752 | 0.336 | -20.73% | 1.36 | 0.273% | 67.6% | 0.184% | -8.407% | 68.5% | 0.963 | 0.807 |
| ETH/USDT | `partial_atr_trailing` | 19.88% | 0.570 | 0.268 | -19.50% | 1.24 | 0.214% | 50.0% | -0.038% | -8.407% | 68.5% | 0.940 | 0.841 |
| ETH/USDT | `partial_chandelier` | 28.87% | 0.763 | 0.373 | -19.51% | 1.32 | 0.269% | 50.0% | 0.004% | -8.407% | 64.9% | 0.825 | 0.915 |
| ETH/USDT | `partial_donchian_runner` | 52.85% | 1.143 | 0.611 | -21.77% | 1.50 | 0.456% | 48.1% | -0.224% | -8.407% | 68.5% | 1.025 | 1.029 |
| ETH/USDT | `partial_ema_runner` | 25.87% | 0.695 | 0.337 | -21.29% | 1.32 | 0.258% | 46.2% | -0.218% | -5.192% | 63.2% | 1.070 | 0.865 |
| ETH/USDT | `atr_grid_0.5_1.0` | 35.15% | 0.876 | 0.458 | -20.55% | 1.41 | 0.339% | 45.4% | -0.183% | -8.407% | 77.8% | 0.771 | 0.861 |
| ETH/USDT | `atr_grid_1.0_2.0` | 53.94% | 1.160 | 0.620 | -21.77% | 1.51 | 0.463% | 48.1% | -0.224% | -8.407% | 68.5% | 1.070 | 1.088 |
| ETH/USDT | `atr_grid_1.5_3.0` | 56.16% | 1.136 | 0.613 | -21.74% | 1.48 | 0.483% | 46.3% | -0.532% | -8.407% | 56.5% | 1.096 | 1.147 |
| ETH/USDT | `atr_grid_2.0_4.0` | 63.33% | 1.174 | 0.623 | -20.90% | 1.51 | 0.531% | 45.4% | -0.546% | -8.407% | 53.7% | 1.176 | 1.245 |
| ETH/USDT | `fixed_pct_0.005_0.010` | 36.28% | 0.918 | 0.498 | -16.83% | 1.44 | 0.346% | 45.4% | -0.289% | -3.874% | 75.9% | 0.695 | 0.820 |
| ETH/USDT | `fixed_pct_0.010_0.020` | 47.03% | 1.070 | 0.576 | -18.94% | 1.47 | 0.420% | 49.1% | -0.243% | -6.078% | 63.9% | 0.994 | 1.091 |
| ETH/USDT | `vol_normalized_0.50_0.75` | 119.45% | 1.451 | 0.789 | -26.62% | 1.78 | 0.864% | 39.8% | -0.866% | -8.407% | 0.9% | 1.112 | 1.118 |
| ETH/USDT | `vol_normalized_0.75_1.00` | 120.02% | 1.453 | 0.789 | -26.62% | 1.78 | 0.868% | 39.8% | -0.866% | -8.407% | 0.0% | 1.112 | 1.121 |
| ETH/USDT | `atr_dynamic_0.50_1.00` | 50.38% | 1.077 | 0.574 | -21.77% | 1.49 | 0.445% | 49.1% | -0.235% | -8.407% | 70.4% | 1.044 | 1.108 |
| ETH/USDT | `atr_dynamic_0.75_1.00` | 46.72% | 0.984 | 0.519 | -21.64% | 1.44 | 0.426% | 48.1% | -0.149% | -8.407% | 58.3% | 1.011 | 1.152 |
| ETH/USDT | `mfe_quantile_q75_50_100` | 77.27% | 1.303 | 0.715 | -21.41% | 1.57 | 0.613% | 41.7% | -0.824% | -8.407% | 34.3% | 1.259 | 1.410 |
| ETH/USDT | `mfe_quantile_q75_75_100` | 86.99% | 1.369 | 0.765 | -22.15% | 1.61 | 0.667% | 40.7% | -0.866% | -8.407% | 25.9% | 1.278 | 1.346 |
| ETH/USDT | `simple_scaleout_25_25_50` | -3.62% | -0.024 | -0.010 | -24.36% | 1.01 | 0.007% | 66.7% | 0.193% | -8.407% | 67.5% | 0.588 | 0.507 |
| ETH/USDT | `simple_scaleout_50_50` | 32.36% | 0.826 | 0.412 | -19.17% | 1.34 | 0.318% | 56.5% | 0.525% | -8.407% | 56.5% | 1.057 | 1.030 |
| SOL/USDT | `runner_25_25_50` | 10.42% | 0.350 | 0.187 | -21.85% | 1.15 | 0.157% | 45.8% | -0.383% | -4.743% | 65.6% | -0.515 | -0.573 |
| SOL/USDT | `runner_33_33_34` | 5.17% | 0.236 | 0.124 | -21.12% | 1.09 | 0.093% | 49.0% | -0.196% | -4.743% | 65.6% | -0.463 | -0.656 |
| SOL/USDT | `runner_50_50` | 19.64% | 0.546 | 0.288 | -19.52% | 1.24 | 0.238% | 54.2% | 0.281% | -5.110% | 59.4% | -0.424 | -0.478 |
| SOL/USDT | `runner_20_30_50` | 10.55% | 0.352 | 0.187 | -22.26% | 1.15 | 0.159% | 45.8% | -0.470% | -4.743% | 65.6% | -0.519 | -0.578 |
| SOL/USDT | `partial_breakeven` | 25.46% | 0.685 | 0.341 | -18.95% | 1.33 | 0.283% | 65.6% | 0.225% | -4.743% | 65.6% | -0.330 | -0.346 |
| SOL/USDT | `partial_atr_trailing` | 3.06% | 0.182 | 0.091 | -23.72% | 1.07 | 0.074% | 44.8% | -0.348% | -4.743% | 65.6% | -0.541 | -0.552 |
| SOL/USDT | `partial_chandelier` | 6.86% | 0.274 | 0.140 | -24.78% | 1.12 | 0.112% | 43.8% | -0.496% | -4.743% | 61.9% | -0.447 | -0.453 |
| SOL/USDT | `partial_donchian_runner` | 7.36% | 0.283 | 0.147 | -22.11% | 1.12 | 0.121% | 45.9% | -0.443% | -4.743% | 64.3% | -0.454 | -0.580 |
| SOL/USDT | `partial_ema_runner` | 1.90% | 0.154 | 0.076 | -25.96% | 1.07 | 0.062% | 36.6% | -0.547% | -3.709% | 57.1% | -0.500 | -0.672 |
| SOL/USDT | `atr_grid_0.5_1.0` | 4.95% | 0.229 | 0.122 | -22.75% | 1.11 | 0.097% | 47.9% | -0.027% | -4.743% | 76.0% | -0.635 | -0.721 |
| SOL/USDT | `atr_grid_1.0_2.0` | 10.42% | 0.350 | 0.187 | -21.85% | 1.15 | 0.157% | 45.8% | -0.383% | -4.743% | 65.6% | -0.515 | -0.573 |
| SOL/USDT | `atr_grid_1.5_3.0` | 23.81% | 0.604 | 0.325 | -20.20% | 1.26 | 0.281% | 43.8% | -0.359% | -5.110% | 59.4% | -0.458 | -0.419 |
| SOL/USDT | `atr_grid_2.0_4.0` | 25.64% | 0.611 | 0.331 | -23.63% | 1.25 | 0.299% | 42.7% | -0.549% | -5.110% | 47.9% | -0.449 | -0.388 |
| SOL/USDT | `fixed_pct_0.005_0.010` | 2.84% | 0.177 | 0.093 | -22.29% | 1.08 | 0.074% | 45.8% | -0.212% | -4.743% | 79.2% | -0.810 | -0.767 |
| SOL/USDT | `fixed_pct_0.010_0.020` | 6.07% | 0.256 | 0.137 | -22.87% | 1.11 | 0.109% | 47.9% | -0.138% | -4.743% | 69.8% | -0.694 | -0.714 |
| SOL/USDT | `vol_normalized_0.50_0.75` | 25.62% | 0.537 | 0.275 | -29.52% | 1.28 | 0.356% | 37.5% | -0.927% | -5.110% | 0.0% | -0.445 | -0.372 |
| SOL/USDT | `vol_normalized_0.75_1.00` | 25.62% | 0.537 | 0.275 | -29.52% | 1.28 | 0.356% | 37.5% | -0.927% | -5.110% | 0.0% | -0.445 | -0.372 |
| SOL/USDT | `atr_dynamic_0.50_1.00` | 18.30% | 0.500 | 0.262 | -20.50% | 1.22 | 0.230% | 46.9% | -0.162% | -5.110% | 64.6% | -0.283 | -0.357 |
| SOL/USDT | `atr_dynamic_0.75_1.00` | 10.80% | 0.348 | 0.177 | -22.27% | 1.15 | 0.171% | 45.8% | -0.447% | -5.110% | 52.1% | -0.380 | -0.398 |
| SOL/USDT | `mfe_quantile_q75_50_100` | 27.03% | 0.627 | 0.346 | -24.16% | 1.26 | 0.315% | 38.5% | -0.837% | -5.110% | 34.4% | -0.514 | -0.422 |
| SOL/USDT | `mfe_quantile_q75_75_100` | 27.18% | 0.614 | 0.338 | -26.00% | 1.26 | 0.322% | 38.5% | -0.869% | -5.110% | 20.8% | -0.510 | -0.357 |
| SOL/USDT | `simple_scaleout_25_25_50` | 14.31% | 0.459 | 0.219 | -21.34% | 1.20 | 0.173% | 65.0% | 0.225% | -4.743% | 66.0% | -0.384 | -0.434 |
| SOL/USDT | `simple_scaleout_50_50` | 30.16% | 0.758 | 0.382 | -19.16% | 1.33 | 0.325% | 59.4% | 0.594% | -5.110% | 59.4% | -0.393 | -0.472 |

Relative deltas, fees, slippage, exposure, holding time, MAE and MFE are available for every validation/OOS row in the JSON and CSV outputs.

## 4. ETH `breakout_atr_60_1.0`

Baseline OOS: return 118.52%, Sharpe 1.441, Sortino 0.782, max DD -26.67%, expectancy 0.862%.

The ETH result is not a case where every partial exit is slightly worse. Runner, break-even, trailing, and fixed/ATR variants show different trade-offs; the near-baseline vol-normalized rows have negligible TP activation and therefore do not demonstrate a meaningful scale-out.

## 5. Regime Analysis

Regimes are the existing phase-2 flags: trend, range, high volatility, low volatility, crash, and recovery. Results are descriptive OOS slices only; no regime was selected after seeing its final result.

### BTC/USDT

- Baseline trend: 15.92% return, Sharpe 0.705, max DD -17.00%, trades 64
- Baseline range: -16.71% return, Sharpe -2.061, max DD -19.23%, trades 62
- Baseline high_volatility: 15.02% return, Sharpe 1.330, max DD -13.08%, trades 28
- Baseline low_volatility: -1.60% return, Sharpe -0.508, max DD -4.04%, trades 18
- Baseline crash: 22.38% return, Sharpe 11.828, max DD -6.16%, trades 2
- Baseline recovery: -9.15% return, Sharpe -3.496, max DD -10.41%, trades 10
- No regime-specific tactic is promoted; any attractive slice requires a pre-registered rule and fresh walk-forward/OOS validation.
### ETH/USDT

- Baseline trend: 131.08% return, Sharpe 2.015, max DD -28.16%, trades 76
- Baseline range: -5.43% return, Sharpe -0.883, max DD -10.07%, trades 32
- Baseline high_volatility: 66.19% return, Sharpe 2.821, max DD -23.46%, trades 25
- Baseline low_volatility: 6.97% return, Sharpe 1.745, max DD -5.15%, trades 10
- Baseline crash: 54.84% return, Sharpe 17.214, max DD -5.69%, trades 6
- Baseline recovery: -8.15% return, Sharpe -1.116, max DD -10.92%, trades 15
- No regime-specific tactic is promoted; any attractive slice requires a pre-registered rule and fresh walk-forward/OOS validation.
### SOL/USDT

- Baseline trend: 26.75% return, Sharpe 0.688, max DD -28.79%, trades 77
- Baseline range: -1.35% return, Sharpe -0.318, max DD -4.19%, trades 19
- Baseline high_volatility: -1.53% return, Sharpe 0.274, max DD -29.55%, trades 19
- Baseline low_volatility: -0.76% return, Sharpe -0.125, max DD -5.37%, trades 8
- Baseline crash: 61.48% return, Sharpe 18.396, max DD -7.82%, trades 5
- Baseline recovery: -12.70% return, Sharpe -1.119, max DD -25.62%, trades 19
- No regime-specific tactic is promoted; any attractive slice requires a pre-registered rule and fresh walk-forward/OOS validation.

## 6. MAE/MFE Analysis

MFE/MAE distributions use raw baseline trade paths; OOS quantiles are not used to tune targets. TP reach rates are expressed relative to each trade's entry ATR.

- BTC/USDT: OOS trades 126, MFE quantiles {'p25': 0.517197496171673, 'p50': 1.4180827217132874, 'p60': 2.1416780510018274, 'p70': 2.974050554307539, 'p75': 3.4194705953909166, 'p80': 3.8831295218622164, 'p90': 5.0680324044261535}, MAE absolute quantiles {'p25': 0.5269542015035739, 'p50': 0.9114330390815772, 'p60': 1.1185593842111374, 'p70': 1.3986141552170706, 'p75': 1.5460590265111485, 'p80': 1.8125535250825164, 'p90': 2.3406940854601768}, TP reach {'0.5x_atr_pct': 87.3015873015873, '1x_atr_pct': 76.98412698412699, '1.5x_atr_pct': 65.07936507936508, '2x_atr_pct': 57.936507936507944, '3x_atr_pct': 42.06349206349206}
- ETH/USDT: OOS trades 108, MFE quantiles {'p25': 0.9428335898529039, 'p50': 2.50209588710808, 'p60': 3.067960911613543, 'p70': 4.844392447290373, 'p75': 5.75611216489601, 'p80': 6.8270031544177, 'p90': 12.182093384440432}, MAE absolute quantiles {'p25': 0.5331010236173744, 'p50': 1.4460675089064567, 'p60': 1.5838821544946848, 'p70': 2.048675086183204, 'p75': 2.4692513596505123, 'p80': 2.876107318519918, 'p90': 3.649548594042825}, TP reach {'0.5x_atr_pct': 85.18518518518519, '1x_atr_pct': 73.14814814814815, '1.5x_atr_pct': 63.888888888888886, '2x_atr_pct': 58.333333333333336, '3x_atr_pct': 50.0}
- SOL/USDT: OOS trades 96, MFE quantiles {'p25': 1.0481639124104047, 'p50': 2.613284974769814, 'p60': 3.7227068012085973, 'p70': 4.74760065779114, 'p75': 5.48014650996895, 'p80': 7.512104465202718, 'p90': 12.197100733157113}, MAE absolute quantiles {'p25': 0.9351206862370454, 'p50': 1.669223569566769, 'p60': 2.058358888579126, 'p70': 2.410362826399759, 'p75': 2.593540555661844, 'p80': 3.196012314909835, 'p90': 3.808204035724322}, TP reach {'0.5x_atr_pct': 85.41666666666666, '1x_atr_pct': 70.83333333333334, '1.5x_atr_pct': 60.416666666666664, '2x_atr_pct': 51.041666666666664, '3x_atr_pct': 42.70833333333333}

## 7. Parameter Sensitivity

Sensitivity is summarized by family range and sign persistence. Narrow winners without neighboring support are treated as unstable.

- `atr_dynamic`: ['atr_dynamic_0.50_1.00', 'atr_dynamic_0.75_1.00']; OOS Sharpe range 0.984 to 1.077; validation-positive 100.0%; OOS-positive 100.0%.
- `atr_grid`: ['atr_grid_0.5_1.0', 'atr_grid_1.0_2.0', 'atr_grid_1.5_3.0', 'atr_grid_2.0_4.0']; OOS Sharpe range 0.876 to 1.174; validation-positive 100.0%; OOS-positive 100.0%.
- `fixed_pct`: ['fixed_pct_0.005_0.010', 'fixed_pct_0.010_0.020']; OOS Sharpe range 0.918 to 1.070; validation-positive 100.0%; OOS-positive 100.0%.
- `mfe_quantile`: ['mfe_quantile_q75_50_100', 'mfe_quantile_q75_75_100']; OOS Sharpe range 1.303 to 1.369; validation-positive 100.0%; OOS-positive 100.0%.
- `partial`: ['partial_breakeven', 'partial_atr_trailing', 'partial_chandelier', 'partial_donchian_runner', 'partial_ema_runner']; OOS Sharpe range 0.570 to 1.143; validation-positive 100.0%; OOS-positive 100.0%.
- `runner`: ['runner_25_25_50', 'runner_33_33_34', 'runner_50_50', 'runner_20_30_50']; OOS Sharpe range 0.955 to 1.185; validation-positive 100.0%; OOS-positive 100.0%.
- `simple_scaleout`: ['simple_scaleout_25_25_50', 'simple_scaleout_50_50']; OOS Sharpe range -0.024 to 0.826; validation-positive 100.0%; OOS-positive 50.0%.
- `vol_normalized`: ['vol_normalized_0.50_0.75', 'vol_normalized_0.75_1.00']; OOS Sharpe range 1.451 to 1.453; validation-positive 100.0%; OOS-positive 100.0%.

## 8. Overfitting Analysis

**OVERFIT RISK**: allocations, target methods/distances, stop modes, final exits, volatility scaling, MFE quantiles, and asset/regime selection create many interacting degrees of freedom. The diagnosis deliberately does not select a winner from final OOS or stress results.

## 9. Event-Level Analysis

### BTC/USDT versus `runner_25_25_50`
- strong_trend at 2019-11-21 09:00:00+00:00: baseline 8.567% gross, MFE 14.452%, exit `signal_exit`; variant matched=True, net 4.681970246827506%, events [{'stage': 1, 'fraction': 0.25, 'price': 7891.405561519201, 'reason': 'tp1', 'timestamp': Timestamp('2019-11-21 11:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.25, 'price': 7824.884402804914, 'reason': 'tp2', 'timestamp': Timestamp('2019-11-21 14:00:00+0000', tz='UTC')}, {'stage': 3, 'fraction': 0.5, 'price': 7258.541418, 'reason': 'signal_exit', 'timestamp': Timestamp('2019-11-23 15:00:00+0000', tz='UTC')}]
- weak_breakout at 2019-11-29 14:00:00+00:00: baseline -0.439% gross, MFE 1.381%, exit `signal_exit`; variant matched=True, net -0.1684264521461803%, events [{'stage': 1, 'fraction': 0.25, 'price': 7825.408929420857, 'reason': 'tp1', 'timestamp': Timestamp('2019-11-29 15:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.75, 'price': 7707.498192, 'reason': 'signal_exit', 'timestamp': Timestamp('2019-11-30 11:00:00+0000', tz='UTC')}]
- whipsaw_or_stop at 2019-11-29 14:00:00+00:00: baseline -0.439% gross, MFE 1.381%, exit `signal_exit`; variant matched=True, net -0.1684264521461803%, events [{'stage': 1, 'fraction': 0.25, 'price': 7825.408929420857, 'reason': 'tp1', 'timestamp': Timestamp('2019-11-29 15:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.75, 'price': 7707.498192, 'reason': 'signal_exit', 'timestamp': Timestamp('2019-11-30 11:00:00+0000', tz='UTC')}]
- reversal_after_favorable_move at 2020-04-02 17:00:00+00:00: baseline -2.901% gross, MFE 4.068%, exit `atr_stop`; variant matched=True, net -1.832958528762226%, events [{'stage': 1, 'fraction': 0.25, 'price': 7036.176519478457, 'reason': 'tp1', 'timestamp': Timestamp('2020-04-02 18:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.75, 'price': 6715.983785121313, 'reason': 'stop', 'timestamp': Timestamp('2020-04-02 20:00:00+0000', tz='UTC')}]
### ETH/USDT versus `runner_25_25_50`
- strong_trend at 2019-11-21 12:00:00+00:00: baseline 9.933% gross, MFE 17.969%, exit `signal_exit`; variant matched=True, net 5.743719388454414%, events [{'stage': 1, 'fraction': 0.25, 'price': 166.57323341365714, 'reason': 'tp1', 'timestamp': Timestamp('2019-11-21 14:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.25, 'price': 163.9977184136571, 'reason': 'tp2', 'timestamp': Timestamp('2019-11-21 15:00:00+0000', tz='UTC')}, {'stage': 3, 'fraction': 0.5, 'price': 151.550304, 'reason': 'signal_exit', 'timestamp': Timestamp('2019-11-23 04:00:00+0000', tz='UTC')}]
- weak_breakout at 2019-12-04 01:00:00+00:00: baseline -0.976% gross, MFE 0.934%, exit `signal_exit`; variant matched=True, net -0.6493413388559877%, events [{'stage': 1, 'fraction': 0.25, 'price': 143.41477722, 'reason': 'tp1', 'timestamp': Timestamp('2019-12-04 06:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.75, 'price': 145.939182, 'reason': 'signal_exit', 'timestamp': Timestamp('2019-12-04 11:00:00+0000', tz='UTC')}]
- whipsaw_or_stop at 2019-12-04 01:00:00+00:00: baseline -0.976% gross, MFE 0.934%, exit `signal_exit`; variant matched=True, net -0.6493413388559877%, events [{'stage': 1, 'fraction': 0.25, 'price': 143.41477722, 'reason': 'tp1', 'timestamp': Timestamp('2019-12-04 06:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.75, 'price': 145.939182, 'reason': 'signal_exit', 'timestamp': Timestamp('2019-12-04 11:00:00+0000', tz='UTC')}]
- reversal_after_favorable_move at 2020-01-14 18:00:00+00:00: baseline -0.927% gross, MFE 4.911%, exit `signal_exit`; variant matched=True, net -0.08092976453988814%, events [{'stage': 1, 'fraction': 0.25, 'price': 168.6469070142286, 'reason': 'tp1', 'timestamp': Timestamp('2020-01-15 00:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.75, 'price': 162.377518, 'reason': 'signal_exit', 'timestamp': Timestamp('2020-01-15 16:00:00+0000', tz='UTC')}]
### SOL/USDT versus `runner_25_25_50`
- strong_trend at 2020-09-05 10:00:00+00:00: baseline 8.152% gross, MFE 30.711%, exit `signal_exit`; variant matched=True, net 8.470192704677814%, events [{'stage': 1, 'fraction': 0.25, 'price': 2.8486227334817142, 'reason': 'tp1', 'timestamp': Timestamp('2020-09-05 11:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.25, 'price': 2.648968524910286, 'reason': 'tp2', 'timestamp': Timestamp('2020-09-05 12:00:00+0000', tz='UTC')}, {'stage': 3, 'fraction': 0.5, 'price': 2.7744547799999997, 'reason': 'signal_exit', 'timestamp': Timestamp('2020-09-06 01:00:00+0000', tz='UTC')}]
- weak_breakout at 2020-08-18 07:00:00+00:00: baseline -5.564% gross, MFE 1.208%, exit `atr_stop`; variant matched=True, net -5.656186132030347%, events [{'stage': 1, 'fraction': 1.0, 'price': 3.0753215870434283, 'reason': 'stop', 'timestamp': Timestamp('2020-08-18 09:00:00+0000', tz='UTC')}]
- whipsaw_or_stop at 2020-08-18 07:00:00+00:00: baseline -5.564% gross, MFE 1.208%, exit `atr_stop`; variant matched=True, net -5.656186132030347%, events [{'stage': 1, 'fraction': 1.0, 'price': 3.0753215870434283, 'reason': 'stop', 'timestamp': Timestamp('2020-08-18 09:00:00+0000', tz='UTC')}]
- reversal_after_favorable_move at 2020-08-30 09:00:00+00:00: baseline -3.887% gross, MFE 4.523%, exit `signal_exit`; variant matched=True, net -2.1896720596618144%, events [{'stage': 1, 'fraction': 0.25, 'price': 4.657327051116571, 'reason': 'tp1', 'timestamp': Timestamp('2020-08-30 12:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.75, 'price': 4.3317334800000005, 'reason': 'signal_exit', 'timestamp': Timestamp('2020-08-31 04:00:00+0000', tz='UTC')}]

## 10. Risk-Management Analysis

RISK MANAGEMENT BENEFIT CANDIDATES REQUIRE FRESH HOLDOUT
. Similar return alone is not called a risk benefit; the candidate must also improve Sharpe, drawdown, worst trade, and have meaningful TP activity.

## 11. Simple Scale-Out Test

**PROMISING BUT NOT ROBUST**. The two simple policies were tested without aggressive tuning and remain research candidates only.

- `simple_scaleout_25_25_50` on BTC/USDT: OOS return -6.76%, Sharpe -0.219, max DD -21.44%, TP activation 65.7%.
- `simple_scaleout_50_50` on BTC/USDT: OOS return -3.49%, Sharpe -0.068, max DD -17.35%, TP activation 60.3%.
- `simple_scaleout_25_25_50` on ETH/USDT: OOS return -3.62%, Sharpe -0.024, max DD -24.36%, TP activation 67.5%.
- `simple_scaleout_50_50` on ETH/USDT: OOS return 32.36%, Sharpe 0.826, max DD -19.17%, TP activation 56.5%.
- `simple_scaleout_25_25_50` on SOL/USDT: OOS return 14.31%, Sharpe 0.459, max DD -21.34%, TP activation 66.0%.
- `simple_scaleout_50_50` on SOL/USDT: OOS return 30.16%, Sharpe 0.758, max DD -19.16%, TP activation 59.4%.

## 12. Conclusion

Early TP realization reduces participation in the persistent breakout trend, while extra fills add fees/slippage. The few near-baseline variants often have very low real TP activation and therefore are not meaningful scale-outs.

Do not promote; retain Full-Position Management as the current control and only continue with a pre-registered, low-dimensional holdout study.

## 13. Recommended Next Research Step

Run one ETH-only, pre-registered holdout experiment with exactly two simple policies plus the full-exit control, fixed costs, funding explicitly sourced or marked unavailable, and a minimum TP-activation threshold.

## 14. What Is Not Robust

- all 19 original partial-exit variants as promotion candidates
- vol-normalized variants as meaningful partial exits when TP activation is negligible
- dynamic leverage as evidence of management quality
- regime-specific selection based on final OOS outcomes
- funding-adjusted conclusions while funding is NOT TESTED

## 15. What Remains To Test

- Funding-inclusive ETH holdout with an actual funding series.
- One pre-registered simple scale-out family only, with a minimum TP activation rate and no asset/regime cherry-picking.
- Independent future data after the research cutoff.
- If no simple rule passes, keep full-position management and stop the partial-exit branch.
