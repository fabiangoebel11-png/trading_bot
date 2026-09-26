# Partial Exit Research

Decision: **NO ROBUST PARTIAL EXIT ADVANTAGE FOUND**
Base strategy: `breakout_atr_60_1.0`
Policies tested: 19
Funding: `NOT TESTED: no local funding cache`
Missing data: `none`

## Baseline
- BTC/USDT: validation Sharpe -0.276, OOS Sharpe 0.008, OOS return -3.45%, OOS max DD -22.89%
- ETH/USDT: validation Sharpe 0.949, OOS Sharpe 1.441, OOS return 118.52%, OOS max DD -26.67%
- SOL/USDT: validation Sharpe -0.719, OOS Sharpe 0.529, OOS return 25.04%, OOS max DD -29.59%

## Variant Results

| Symbol | Policy | Class | Validation Sharpe | OOS Sharpe | OOS Return | OOS Max DD | Stress Sharpe | Stress profitable |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BTC/USDT | `runner_25_25_50` | NOT ROBUST | -0.471 | 0.251 | 5.29% | -15.12% | 0.446 | 65.5% |
| BTC/USDT | `runner_33_33_34` | NOT ROBUST | -0.616 | 0.290 | 6.01% | -13.69% | 0.510 | 69.5% |
| BTC/USDT | `runner_50_50` | NOT ROBUST | -0.421 | 0.303 | 6.83% | -13.70% | 0.562 | 66.5% |
| BTC/USDT | `runner_20_30_50` | NOT ROBUST | -0.445 | 0.277 | 6.12% | -14.85% | 0.459 | 65.5% |
| BTC/USDT | `partial_breakeven` | POTENTIALLY ROBUST | -0.179 | 0.162 | 2.58% | -16.13% | 0.460 | 68.5% |
| BTC/USDT | `partial_atr_trailing` | NOT ROBUST | -0.592 | 0.375 | 8.75% | -15.58% | 0.548 | 74.0% |
| BTC/USDT | `partial_chandelier` | NOT ROBUST | -0.451 | 0.155 | 2.44% | -16.51% | 0.485 | 60.5% |
| BTC/USDT | `partial_donchian_runner` | NOT ROBUST | -0.442 | 0.217 | 4.30% | -15.39% | 0.441 | 64.5% |
| BTC/USDT | `partial_ema_runner` | NOT ROBUST | -0.355 | 0.315 | 7.01% | -15.49% | 0.514 | 76.0% |
| BTC/USDT | `atr_grid_0.5_1.0` | NOT ROBUST | -0.817 | -0.063 | -3.31% | -16.50% | 0.232 | 52.0% |
| BTC/USDT | `atr_grid_1.0_2.0` | NOT ROBUST | -0.471 | 0.251 | 5.29% | -15.12% | 0.446 | 65.5% |
| BTC/USDT | `atr_grid_1.5_3.0` | NOT ROBUST | -0.400 | 0.424 | 11.24% | -13.16% | 0.612 | 74.0% |
| BTC/USDT | `atr_grid_2.0_4.0` | NOT ROBUST | -0.363 | 0.372 | 10.01% | -16.53% | 0.585 | 73.0% |
| BTC/USDT | `fixed_pct_0.005_0.010` | NOT ROBUST | -0.850 | -0.036 | -2.70% | -17.48% | 0.114 | 51.5% |
| BTC/USDT | `fixed_pct_0.010_0.020` | NOT ROBUST | -0.510 | 0.291 | 6.57% | -15.65% | 0.426 | 65.5% |
| BTC/USDT | `vol_normalized_0.50_0.75` | POTENTIALLY ROBUST | -0.145 | 0.153 | 2.29% | -20.00% | 0.379 | 64.0% |
| BTC/USDT | `vol_normalized_0.75_1.00` | POTENTIALLY ROBUST | -0.145 | 0.153 | 2.29% | -20.00% | 0.379 | 64.0% |
| BTC/USDT | `atr_dynamic_0.50_1.00` | NOT ROBUST | -0.578 | 0.116 | 1.37% | -15.00% | 0.296 | 61.5% |
| BTC/USDT | `atr_dynamic_0.75_1.00` | NOT ROBUST | -0.547 | 0.161 | 2.65% | -13.41% | 0.330 | 62.0% |
| ETH/USDT | `runner_25_25_50` | POTENTIALLY ROBUST | 1.016 | 1.262 | 60.50% | -21.51% | 1.193 | 100.0% |
| ETH/USDT | `runner_33_33_34` | NOT ROBUST | 0.978 | 1.070 | 42.24% | -20.11% | 1.091 | 97.0% |
| ETH/USDT | `runner_50_50` | POTENTIALLY ROBUST | 1.116 | 1.162 | 54.13% | -20.47% | 1.278 | 100.0% |
| ETH/USDT | `runner_20_30_50` | POTENTIALLY ROBUST | 1.017 | 1.285 | 62.80% | -21.54% | 1.204 | 100.0% |
| ETH/USDT | `partial_breakeven` | NOT ROBUST | 1.225 | 0.868 | 32.29% | -20.41% | 0.911 | 89.5% |
| ETH/USDT | `partial_atr_trailing` | NOT ROBUST | 1.047 | 0.680 | 24.99% | -19.23% | 0.946 | 93.5% |
| ETH/USDT | `partial_chandelier` | NOT ROBUST | 0.762 | 0.879 | 34.68% | -19.18% | 1.004 | 93.5% |
| ETH/USDT | `partial_donchian_runner` | POTENTIALLY ROBUST | 1.032 | 1.245 | 59.36% | -21.51% | 1.133 | 99.5% |
| ETH/USDT | `partial_ema_runner` | NOT ROBUST | 0.991 | 0.813 | 31.72% | -20.51% | 0.979 | 95.5% |
| ETH/USDT | `atr_grid_0.5_1.0` | NOT ROBUST | 0.712 | 0.984 | 40.91% | -20.29% | 0.970 | 93.0% |
| ETH/USDT | `atr_grid_1.0_2.0` | POTENTIALLY ROBUST | 1.016 | 1.262 | 60.50% | -21.51% | 1.193 | 100.0% |
| ETH/USDT | `atr_grid_1.5_3.0` | POTENTIALLY ROBUST | 1.039 | 1.232 | 62.82% | -21.48% | 1.233 | 100.0% |
| ETH/USDT | `atr_grid_2.0_4.0` | POTENTIALLY ROBUST | 1.134 | 1.264 | 70.30% | -20.64% | 1.317 | 100.0% |
| ETH/USDT | `fixed_pct_0.005_0.010` | NOT ROBUST | 0.675 | 1.029 | 42.09% | -16.55% | 0.941 | 92.0% |
| ETH/USDT | `fixed_pct_0.010_0.020` | NOT ROBUST | 0.867 | 1.174 | 53.30% | -18.67% | 1.190 | 99.0% |
| ETH/USDT | `vol_normalized_0.50_0.75` | POTENTIALLY ROBUST | 1.010 | 1.520 | 128.80% | -26.33% | 1.185 | 99.5% |
| ETH/USDT | `vol_normalized_0.75_1.00` | POTENTIALLY ROBUST | 1.010 | 1.522 | 129.40% | -26.33% | 1.188 | 99.5% |
| ETH/USDT | `atr_dynamic_0.50_1.00` | POTENTIALLY ROBUST | 1.062 | 1.176 | 56.79% | -21.51% | 1.206 | 99.5% |
| ETH/USDT | `atr_dynamic_0.75_1.00` | POTENTIALLY ROBUST | 1.121 | 1.079 | 52.98% | -21.38% | 1.229 | 100.0% |
| SOL/USDT | `runner_25_25_50` | NOT ROBUST | -1.023 | 0.446 | 14.92% | -21.14% | -0.503 | 26.5% |
| SOL/USDT | `runner_33_33_34` | NOT ROBUST | -1.174 | 0.345 | 9.46% | -19.36% | -0.570 | 27.5% |
| SOL/USDT | `runner_50_50` | NOT ROBUST | -1.043 | 0.642 | 24.51% | -18.70% | -0.375 | 36.0% |
| SOL/USDT | `runner_20_30_50` | NOT ROBUST | -1.024 | 0.446 | 15.05% | -21.66% | -0.503 | 28.0% |
| SOL/USDT | `partial_breakeven` | NOT ROBUST | -0.897 | 0.786 | 30.57% | -18.33% | -0.261 | 39.5% |
| SOL/USDT | `partial_atr_trailing` | NOT ROBUST | -0.977 | 0.287 | 7.26% | -22.57% | -0.496 | 29.5% |
| SOL/USDT | `partial_chandelier` | NOT ROBUST | -1.026 | 0.384 | 11.64% | -23.22% | -0.345 | 35.0% |
| SOL/USDT | `partial_donchian_runner` | NOT ROBUST | -1.107 | 0.380 | 11.85% | -20.66% | -0.495 | 25.5% |
| SOL/USDT | `partial_ema_runner` | NOT ROBUST | -1.141 | 0.273 | 6.78% | -24.21% | -0.561 | 27.0% |
| SOL/USDT | `atr_grid_0.5_1.0` | NOT ROBUST | -1.135 | 0.332 | 9.23% | -20.96% | -0.621 | 19.0% |
| SOL/USDT | `atr_grid_1.0_2.0` | NOT ROBUST | -1.023 | 0.446 | 14.92% | -21.14% | -0.503 | 26.5% |
| SOL/USDT | `atr_grid_1.5_3.0` | NOT ROBUST | -0.951 | 0.694 | 28.85% | -19.58% | -0.337 | 32.5% |
| SOL/USDT | `atr_grid_2.0_4.0` | NOT ROBUST | -0.848 | 0.695 | 30.76% | -23.04% | -0.310 | 30.0% |
| SOL/USDT | `fixed_pct_0.005_0.010` | NOT ROBUST | -1.140 | 0.281 | 7.04% | -20.49% | -0.668 | 17.5% |
| SOL/USDT | `fixed_pct_0.010_0.020` | NOT ROBUST | -1.111 | 0.355 | 10.40% | -21.08% | -0.629 | 19.5% |
| SOL/USDT | `vol_normalized_0.50_0.75` | POTENTIALLY ROBUST | -0.652 | 0.602 | 30.74% | -28.97% | -0.322 | 23.5% |
| SOL/USDT | `vol_normalized_0.75_1.00` | POTENTIALLY ROBUST | -0.652 | 0.602 | 30.74% | -28.97% | -0.322 | 23.5% |
| SOL/USDT | `atr_dynamic_0.50_1.00` | NOT ROBUST | -0.921 | 0.591 | 23.12% | -19.89% | -0.288 | 30.0% |
| SOL/USDT | `atr_dynamic_0.75_1.00` | NOT ROBUST | -0.831 | 0.434 | 15.32% | -21.67% | -0.325 | 26.5% |

## Guardrails

- Strategy selection is separate from risk/product leverage.
- OOS was not used for tactic parameter tuning.
- Same-bar stop precedence is conservative and deterministic.
- No live or paper order path was changed by this research runner.
- Structure/swing targets were not tested because no separate causal structure contract was available.
