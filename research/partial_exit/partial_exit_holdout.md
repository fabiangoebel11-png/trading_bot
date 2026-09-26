# Partial Exit Holdout Test

**Final classification: NO ROBUST PARTIAL EXIT ADVANTAGE FOUND**

Research-only result. No GUI, tactics engine, live order, paper order, or database integration was changed.

## 1. Registered Policies

| Policy | Allocation | TP levels | Stop / trailing |
|---|---|---|---|
| `control_full_position` | [1.0] | - | unchanged full-position ATR stop / none |
| `scaleout_a_25_25_50` | [0.25, 0.25, 0.5] | [1.0, 2.0] | break-even plus fee/slippage buffer, then ATR trailing runner / existing causal ATR trailing logic |
| `scaleout_b_50_50` | [0.5, 0.5] | [1.5] | break-even plus fee/slippage buffer, then ATR trailing runner / existing causal ATR trailing logic |

## 2. Holdout Period

Chronological final 30% after the pre-existing 50% train and 20% validation split. Exact periods are stored in the manifest.

## 3. Data Integrity

Status: **NOT_PRISTINE_PRIOR_DIAGNOSTIC_READ**. The previous descriptive diagnosis evaluated the same historical OOS bars. No tuning was performed here, but no future untouched data is available to certify a pristine holdout.

Funding: `NOT TESTED`.

## 4. ETH Primary Test
Control: return 120.02%, Sharpe 1.453, Sortino 0.789, max DD -26.62%, trades 108.
- `scaleout_a_25_25_50`: return 1.34%, Sharpe 0.121, Sortino 0.048, max DD -22.58%, TP1 68.5%, TP2 35.2%, classification `RISK MANAGEMENT BENEFIT – NOT RETURN ADVANTAGE`; return advantage -118.69pp, risk advantage 4.04pp.
- `scaleout_b_50_50`: return 10.84%, Sharpe 0.369, Sortino 0.168, max DD -21.87%, TP1 56.9%, TP2 0.0%, classification `RISK MANAGEMENT BENEFIT – NOT RETURN ADVANTAGE`; return advantage -109.18pp, risk advantage 4.75pp.

## 5. BTC/SOL Secondary Tests
### BTC/USDT
Control: return -2.74%, Sharpe 0.026, Sortino 0.015, max DD -22.33%, trades 126.
- `scaleout_a_25_25_50`: return -0.76%, Sharpe 0.029, Sortino 0.013, max DD -18.91%, TP1 68.3%, TP2 33.3%, classification `PROMISING BUT NOT ROBUST`; return advantage 1.97pp, risk advantage 3.42pp.
- `scaleout_b_50_50`: return 0.07%, Sharpe 0.067, Sortino 0.030, max DD -16.38%, TP1 59.7%, TP2 0.0%, classification `PROMISING BUT NOT ROBUST`; return advantage 2.81pp, risk advantage 5.95pp.
### SOL/USDT
Control: return 25.62%, Sharpe 0.537, Sortino 0.275, max DD -29.52%, trades 96.
- `scaleout_a_25_25_50`: return 14.84%, Sharpe 0.480, Sortino 0.229, max DD -22.54%, TP1 65.6%, TP2 35.4%, classification `RISK MANAGEMENT BENEFIT – NOT RETURN ADVANTAGE`; return advantage -10.78pp, risk advantage 6.98pp.
- `scaleout_b_50_50`: return 16.17%, Sharpe 0.498, Sortino 0.236, max DD -22.73%, TP1 58.8%, TP2 0.0%, classification `RISK MANAGEMENT BENEFIT – NOT RETURN ADVANTAGE`; return advantage -9.45pp, risk advantage 6.79pp.

## 6. Control vs Scale-Out A

The full comparison table is in the CSV; the registered A row is shown below.
- BTC/USDT: return -0.76% vs control -2.74%, Sharpe 0.029, max DD -18.91%, class `PROMISING BUT NOT ROBUST`.
- ETH/USDT: return 1.34% vs control 120.02%, Sharpe 0.121, max DD -22.58%, class `RISK MANAGEMENT BENEFIT – NOT RETURN ADVANTAGE`.
- SOL/USDT: return 14.84% vs control 25.62%, Sharpe 0.480, max DD -22.54%, class `RISK MANAGEMENT BENEFIT – NOT RETURN ADVANTAGE`.

## 7. Control vs Scale-Out B

- BTC/USDT: return 0.07% vs control -2.74%, Sharpe 0.067, max DD -16.38%, class `PROMISING BUT NOT ROBUST`.
- ETH/USDT: return 10.84% vs control 120.02%, Sharpe 0.369, max DD -21.87%, class `RISK MANAGEMENT BENEFIT – NOT RETURN ADVANTAGE`.
- SOL/USDT: return 16.17% vs control 25.62%, Sharpe 0.498, max DD -22.73%, class `RISK MANAGEMENT BENEFIT – NOT RETURN ADVANTAGE`.

## 8. TP Activation Rates

TP1, TP2, partial-event rate, average closed percentage, fees, slippage, MAE, MFE, and holding time are reported per row in the CSV/JSON.

## 9. Risk Metrics

Worst day/week/month, consecutive losses, tail loss, and drawdown duration are included for every policy comparison.

## 10. Stress Test

Each policy used the same 200 seeded contiguous 365-day windows after doubled fees/slippage, adverse fills, one-bar execution delay, and random fill deviations.
- ETH/USDT `control_full_position`: profitable 97.5%, median return 23.23%, worst return -1.97%, stress Sharpe 0.841, stress max DD -37.44%.
- BTC/USDT `control_full_position`: profitable 0.0%, median return -20.34%, worst return -28.12%, stress Sharpe -1.125, stress max DD -29.22%.
- SOL/USDT `control_full_position`: profitable 18.0%, median return -7.53%, worst return -25.48%, stress Sharpe -0.074, stress max DD -40.65%.
- BTC/USDT `scaleout_a_25_25_50`: profitable 0.0%, median return -19.47%, worst return -25.71%, stress Sharpe -1.923, stress max DD -26.71%.
- BTC/USDT `scaleout_b_50_50`: profitable 0.0%, median return -14.67%, worst return -19.92%, stress Sharpe -1.258, stress max DD -20.71%.
- ETH/USDT `scaleout_a_25_25_50`: profitable 5.0%, median return -12.72%, worst return -21.80%, stress Sharpe -0.737, stress max DD -30.86%.
- ETH/USDT `scaleout_b_50_50`: profitable 0.0%, median return -16.04%, worst return -23.77%, stress Sharpe -0.836, stress max DD -29.13%.
- SOL/USDT `scaleout_a_25_25_50`: profitable 16.5%, median return -8.26%, worst return -18.67%, stress Sharpe -0.375, stress max DD -28.70%.
- SOL/USDT `scaleout_b_50_50`: profitable 3.0%, median return -12.28%, worst return -23.92%, stress Sharpe -0.578, stress max DD -30.24%.

## 11. Statistical Stability

Trade-level paired bootstrap uses 2,000 resamples. Rows below the pre-registered trade/pair gates are classified as `INSUFFICIENT DATA`; no result is selected by return alone.

## 12. Look-Ahead Validation

Runtime probe: **PASSED**. ATR uses prior completed bars, TP uses current bar high/low only after the entry exists, same-bar stop has priority over TP, future exit-bar high/low cannot alter earlier fills.

## 13. Fees / Slippage

Every actual entry, partial fill, and final exit receives the registered fee and slippage assumptions; funding is explicitly not tested.

## 14. Final Classification

**NO ROBUST PARTIAL EXIT ADVANTAGE FOUND**

The historical candidate `SOL/USDT partial_breakeven` is documented separately as `PROMISING BUT NOT ROBUST` and is not promoted.

## 15. Limitations

The same historical OOS bars were already read by the prior descriptive diagnosis, so this repository cannot honestly label them a pristine untouched holdout. No policy or parameter was changed after the manifest, and no holdout result was used for selection.

## 16. Next Research Step

Collect a genuinely future, unseen period and rerun this exact immutable manifest, including actual funding data. Until then, full-position management remains the control and partial exits remain research-only.
