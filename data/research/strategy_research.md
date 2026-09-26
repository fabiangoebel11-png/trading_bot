# Strategy Research

Decision: **NO ROBUST STRATEGY FOUND**

Data: 2019-11-17T22:00:00+00:00 to 2026-09-21T22:00:00+00:00 (59967 union bars), crypto only.
SPY/QQQ: `INSUFFICIENT_DATA` because no local OHLCV files were available.

## Candidates

### conservative
Longer breakout confirmation, wider stop and half portfolio risk budget.

Rules: name=conservative, fast_ma_window=30, slow_ma_window=150, donchian_entry_window=80, donchian_exit_window=30, atr_window=20, min_atr_pct=0.0025, atr_stop_multiplier=3.5, cooldown_candles=12, max_portfolio_risk_pct=0.005, min_leverage=1.0, max_leverage=2.0

| Sample | Sharpe | Return % | Max DD % | Win rate % |
|---|---:|---:|---:|---:|
| train | 0.769 | 237.68 | -56.76 | 49.50 |
| validation | 0.073 | -12.70 | -46.80 | 46.76 |
| oos | 0.687 | 31.82 | -33.53 | 46.55 |
| stress median | 0.633 | 20.70 | n/a | 64.00 profitable windows |

Robust gate: **FAIL**

### balanced
Reference EMA/Donchian trend-following specification.

Rules: name=balanced, fast_ma_window=20, slow_ma_window=100, donchian_entry_window=55, donchian_exit_window=20, atr_window=14, min_atr_pct=0.0015, atr_stop_multiplier=3.0, cooldown_candles=6, max_portfolio_risk_pct=0.01, min_leverage=2.0, max_leverage=4.0

| Sample | Sharpe | Return % | Max DD % | Win rate % |
|---|---:|---:|---:|---:|
| train | 0.686 | -45.83 | -91.71 | 48.42 |
| validation | 0.352 | -30.11 | -79.17 | 45.69 |
| oos | 0.631 | 26.11 | -58.87 | 48.83 |
| stress median | 0.604 | -12.25 | n/a | 40.00 profitable windows |

Robust gate: **FAIL**

### aggressive
Faster breakout response and shorter cooldown within the same execution model.

Rules: name=aggressive, fast_ma_window=10, slow_ma_window=50, donchian_entry_window=30, donchian_exit_window=12, atr_window=10, min_atr_pct=0.001, atr_stop_multiplier=2.5, cooldown_candles=3, max_portfolio_risk_pct=0.01, min_leverage=2.0, max_leverage=4.0

| Sample | Sharpe | Return % | Max DD % | Win rate % |
|---|---:|---:|---:|---:|
| train | 0.281 | -96.72 | -99.92 | 48.80 |
| validation | -0.904 | -91.87 | -94.64 | 46.43 |
| oos | 0.140 | -34.89 | -67.02 | 48.36 |
| stress median | -0.020 | -59.74 | n/a | 16.00 profitable windows |

Robust gate: **FAIL**
