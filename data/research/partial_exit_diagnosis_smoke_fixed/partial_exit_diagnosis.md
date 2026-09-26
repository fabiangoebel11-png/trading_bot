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
| ETH/USDT | `runner_25_25_50` | 53.94% | 1.160 | 0.620 | -21.77% | 1.51 | 0.463% | 48.1% | -0.224% | -8.407% | 68.5% | 0.985 | 1.137 |
| ETH/USDT | `runner_33_33_34` | 36.42% | 0.955 | 0.491 | -20.37% | 1.39 | 0.333% | 52.8% | 0.208% | -8.407% | 68.5% | 0.883 | 0.993 |
| ETH/USDT | `runner_50_50` | 47.83% | 1.060 | 0.562 | -20.73% | 1.45 | 0.425% | 50.9% | 0.167% | -8.407% | 56.5% | 0.970 | 1.295 |
| ETH/USDT | `runner_20_30_50` | 56.14% | 1.185 | 0.635 | -21.80% | 1.52 | 0.477% | 48.1% | -0.195% | -8.407% | 68.5% | 1.015 | 1.146 |
| ETH/USDT | `partial_breakeven` | 26.88% | 0.752 | 0.336 | -20.73% | 1.36 | 0.273% | 67.6% | 0.184% | -8.407% | 68.5% | 0.774 | 0.170 |
| ETH/USDT | `partial_atr_trailing` | 19.88% | 0.570 | 0.268 | -19.50% | 1.24 | 0.214% | 50.0% | -0.038% | -8.407% | 68.5% | 0.861 | -0.066 |
| ETH/USDT | `partial_chandelier` | 28.87% | 0.763 | 0.373 | -19.51% | 1.32 | 0.269% | 50.0% | 0.004% | -8.407% | 64.9% | 0.803 | 0.347 |
| ETH/USDT | `partial_donchian_runner` | 52.85% | 1.143 | 0.611 | -21.77% | 1.50 | 0.456% | 48.1% | -0.224% | -8.407% | 68.5% | 0.977 | 0.908 |
| ETH/USDT | `partial_ema_runner` | 25.87% | 0.695 | 0.337 | -21.29% | 1.32 | 0.258% | 46.2% | -0.218% | -5.192% | 63.2% | 0.839 | 0.445 |
| ETH/USDT | `atr_grid_0.5_1.0` | 35.15% | 0.876 | 0.458 | -20.55% | 1.41 | 0.339% | 45.4% | -0.183% | -8.407% | 77.8% | 0.640 | 1.223 |
| ETH/USDT | `atr_grid_1.0_2.0` | 53.94% | 1.160 | 0.620 | -21.77% | 1.51 | 0.463% | 48.1% | -0.224% | -8.407% | 68.5% | 0.985 | 1.137 |
| ETH/USDT | `atr_grid_1.5_3.0` | 56.16% | 1.136 | 0.613 | -21.74% | 1.48 | 0.483% | 46.3% | -0.532% | -8.407% | 56.5% | 1.020 | 1.278 |
| ETH/USDT | `atr_grid_2.0_4.0` | 63.33% | 1.174 | 0.623 | -20.90% | 1.51 | 0.531% | 45.4% | -0.546% | -8.407% | 53.7% | 1.112 | 1.504 |
| ETH/USDT | `fixed_pct_0.005_0.010` | 36.28% | 0.918 | 0.498 | -16.83% | 1.44 | 0.346% | 45.4% | -0.289% | -3.874% | 75.9% | 0.616 | 1.156 |
| ETH/USDT | `fixed_pct_0.010_0.020` | 47.03% | 1.070 | 0.576 | -18.94% | 1.47 | 0.420% | 49.1% | -0.243% | -6.078% | 63.9% | 0.853 | 1.333 |
| ETH/USDT | `vol_normalized_0.50_0.75` | 119.45% | 1.451 | 0.789 | -26.62% | 1.78 | 0.864% | 39.8% | -0.866% | -8.407% | 0.9% | 1.094 | 1.341 |
| ETH/USDT | `vol_normalized_0.75_1.00` | 120.02% | 1.453 | 0.789 | -26.62% | 1.78 | 0.868% | 39.8% | -0.866% | -8.407% | 0.0% | 1.095 | 1.341 |
| ETH/USDT | `atr_dynamic_0.50_1.00` | 50.38% | 1.077 | 0.574 | -21.77% | 1.49 | 0.445% | 49.1% | -0.235% | -8.407% | 70.4% | 0.978 | 1.182 |
| ETH/USDT | `atr_dynamic_0.75_1.00` | 46.72% | 0.984 | 0.519 | -21.64% | 1.44 | 0.426% | 48.1% | -0.149% | -8.407% | 58.3% | 1.022 | 1.283 |
| ETH/USDT | `mfe_quantile_q75_50_100` | 77.27% | 1.303 | 0.715 | -21.41% | 1.57 | 0.613% | 41.7% | -0.824% | -8.407% | 34.3% | 1.204 | 1.672 |
| ETH/USDT | `mfe_quantile_q75_75_100` | 86.99% | 1.369 | 0.765 | -22.15% | 1.61 | 0.667% | 40.7% | -0.866% | -8.407% | 25.9% | 1.247 | 1.499 |
| ETH/USDT | `simple_scaleout_25_25_50` | -3.62% | -0.024 | -0.010 | -24.36% | 1.01 | 0.007% | 66.7% | 0.193% | -8.407% | 67.5% | 0.374 | 0.126 |
| ETH/USDT | `simple_scaleout_50_50` | 32.36% | 0.826 | 0.412 | -19.17% | 1.34 | 0.318% | 56.5% | 0.525% | -8.407% | 56.5% | 0.861 | 0.912 |

Relative deltas, fees, slippage, exposure, holding time, MAE and MFE are available for every validation/OOS row in the JSON and CSV outputs.

## 4. ETH `breakout_atr_60_1.0`

Baseline OOS: return 118.52%, Sharpe 1.441, Sortino 0.782, max DD -26.67%, expectancy 0.862%.

The ETH result is not a case where every partial exit is slightly worse. Runner, break-even, trailing, and fixed/ATR variants show different trade-offs; the near-baseline vol-normalized rows have negligible TP activation and therefore do not demonstrate a meaningful scale-out.

## 5. Regime Analysis

Regimes are the existing phase-2 flags: trend, range, high volatility, low volatility, crash, and recovery. Results are descriptive OOS slices only; no regime was selected after seeing its final result.

### ETH/USDT

- Baseline trend: 131.08% return, Sharpe 2.015, max DD -28.16%, trades 76
- Baseline range: -5.43% return, Sharpe -0.883, max DD -10.07%, trades 32
- Baseline high_volatility: 66.19% return, Sharpe 2.821, max DD -23.46%, trades 25
- Baseline low_volatility: 6.97% return, Sharpe 1.745, max DD -5.15%, trades 10
- Baseline crash: 54.84% return, Sharpe 17.214, max DD -5.69%, trades 6
- Baseline recovery: -8.15% return, Sharpe -1.116, max DD -10.92%, trades 15
- No regime-specific tactic is promoted; any attractive slice requires a pre-registered rule and fresh walk-forward/OOS validation.

## 6. MAE/MFE Analysis

MFE/MAE distributions use raw baseline trade paths; OOS quantiles are not used to tune targets. TP reach rates are expressed relative to each trade's entry ATR.

- ETH/USDT: OOS trades 108, MFE quantiles {'p25': 0.9428335898529039, 'p50': 2.50209588710808, 'p60': 3.067960911613543, 'p70': 4.844392447290373, 'p75': 5.75611216489601, 'p80': 6.8270031544177, 'p90': 12.182093384440432}, MAE absolute quantiles {'p25': 0.5331010236173744, 'p50': 1.4460675089064567, 'p60': 1.5838821544946848, 'p70': 2.048675086183204, 'p75': 2.4692513596505123, 'p80': 2.876107318519918, 'p90': 3.649548594042825}, TP reach {'0.5x_atr_pct': 85.18518518518519, '1x_atr_pct': 73.14814814814815, '1.5x_atr_pct': 63.888888888888886, '2x_atr_pct': 58.333333333333336, '3x_atr_pct': 50.0}

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

### ETH/USDT versus `runner_25_25_50`
- strong_trend at 2019-11-21 12:00:00+00:00: baseline 9.933% gross, MFE 17.969%, exit `signal_exit`; variant matched=True, net 5.743719388454414%, events [{'stage': 1, 'fraction': 0.25, 'price': 166.57323341365714, 'reason': 'tp1', 'timestamp': Timestamp('2019-11-21 14:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.25, 'price': 163.9977184136571, 'reason': 'tp2', 'timestamp': Timestamp('2019-11-21 15:00:00+0000', tz='UTC')}, {'stage': 3, 'fraction': 0.5, 'price': 151.550304, 'reason': 'signal_exit', 'timestamp': Timestamp('2019-11-23 04:00:00+0000', tz='UTC')}]
- weak_breakout at 2019-12-04 01:00:00+00:00: baseline -0.976% gross, MFE 0.934%, exit `signal_exit`; variant matched=True, net -0.6493413388559877%, events [{'stage': 1, 'fraction': 0.25, 'price': 143.41477722, 'reason': 'tp1', 'timestamp': Timestamp('2019-12-04 06:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.75, 'price': 145.939182, 'reason': 'signal_exit', 'timestamp': Timestamp('2019-12-04 11:00:00+0000', tz='UTC')}]
- whipsaw_or_stop at 2019-12-04 01:00:00+00:00: baseline -0.976% gross, MFE 0.934%, exit `signal_exit`; variant matched=True, net -0.6493413388559877%, events [{'stage': 1, 'fraction': 0.25, 'price': 143.41477722, 'reason': 'tp1', 'timestamp': Timestamp('2019-12-04 06:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.75, 'price': 145.939182, 'reason': 'signal_exit', 'timestamp': Timestamp('2019-12-04 11:00:00+0000', tz='UTC')}]
- reversal_after_favorable_move at 2020-01-14 18:00:00+00:00: baseline -0.927% gross, MFE 4.911%, exit `signal_exit`; variant matched=True, net -0.08092976453988814%, events [{'stage': 1, 'fraction': 0.25, 'price': 168.6469070142286, 'reason': 'tp1', 'timestamp': Timestamp('2020-01-15 00:00:00+0000', tz='UTC')}, {'stage': 2, 'fraction': 0.75, 'price': 162.377518, 'reason': 'signal_exit', 'timestamp': Timestamp('2020-01-15 16:00:00+0000', tz='UTC')}]

## 10. Risk-Management Analysis

RISK MANAGEMENT BENEFIT NOT ESTABLISHED
. Similar return alone is not called a risk benefit; the candidate must also improve Sharpe, drawdown, worst trade, and have meaningful TP activity.

## 11. Simple Scale-Out Test

**PROMISING BUT NOT ROBUST**. The two simple policies were tested without aggressive tuning and remain research candidates only.

- `simple_scaleout_25_25_50` on ETH/USDT: OOS return -3.62%, Sharpe -0.024, max DD -24.36%, TP activation 67.5%.
- `simple_scaleout_50_50` on ETH/USDT: OOS return 32.36%, Sharpe 0.826, max DD -19.17%, TP activation 56.5%.

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
