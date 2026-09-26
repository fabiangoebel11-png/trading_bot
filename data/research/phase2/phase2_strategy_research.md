# Phase 2 Strategy Research

Decision: **ROBUST STRATEGY FOUND**

Families: donchian_trend, ema_momentum, breakout_atr, volatility_breakout, mean_reversion, momentum_trend_filter, trend_momentum_combo
Candidates per asset: 58
SPY/QQQ: `INSUFFICIENT_DATA`; no local OHLCV series was used.
Funding: `NOT TESTED`; no local funding cache was available and this runner does not download data.

## Robust Candidates

### ETH/USDT / breakout_atr_20_0.5
Family: breakout_atr
Entry: close clears prior 20-bar range by 0.5 ATR. Stop: 2.5x ATR from entry. Exit: exit on close across fast EMA. Flat during NaN/warmup; after an ATR stop, re-entry waits for a flat signal.
Position risk: 0.500%; parameter stability: 1.00
Validation: Sharpe 0.613, return 22.49%, max DD -34.86%
Walk-forward median Sharpe: 0.528
OOS: Sharpe 1.005, return 92.89%, max DD -30.89%, trades 238
MAE/MFE train quantiles: MAE q50=-0.015422510904013365, q75=-0.007761694986364587, q90=-0.0026068260562961587; MFE q50=0.02501138952164017, q75=0.0555835505992156, q90=0.11615512863841675
Stress: median Sharpe 0.519, profitable windows 65.0%
Leverage: evaluated only after this strategy passed the underlying gate.

### ETH/USDT / breakout_atr_20_1.0
Family: breakout_atr
Entry: close clears prior 20-bar range by 1.0 ATR. Stop: 2.5x ATR from entry. Exit: exit on close across fast EMA. Flat during NaN/warmup; after an ATR stop, re-entry waits for a flat signal.
Position risk: 0.500%; parameter stability: 1.00
Validation: Sharpe 0.613, return 20.37%, max DD -24.56%
Walk-forward median Sharpe: 0.682
OOS: Sharpe 1.142, return 93.51%, max DD -39.20%, trades 154
MAE/MFE train quantiles: MAE q50=-0.014896884977133484, q75=-0.006307531981566361, q90=-0.0021421671587120933; MFE q50=0.03199247486823542, q75=0.07137523081288971, q90=0.1234416339655395
Stress: median Sharpe 0.739, profitable windows 86.5%
Leverage: evaluated only after this strategy passed the underlying gate.

### ETH/USDT / breakout_atr_40_0.5
Family: breakout_atr
Entry: close clears prior 40-bar range by 0.5 ATR. Stop: 2.5x ATR from entry. Exit: exit on close across fast EMA. Flat during NaN/warmup; after an ATR stop, re-entry waits for a flat signal.
Position risk: 0.500%; parameter stability: 1.00
Validation: Sharpe 0.932, return 39.51%, max DD -25.72%
Walk-forward median Sharpe: 0.678
OOS: Sharpe 1.195, return 116.06%, max DD -27.46%, trades 197
MAE/MFE train quantiles: MAE q50=-0.01641520964877119, q75=-0.007081686362325285, q90=-0.0026509543293202764; MFE q50=0.02836436873477094, q75=0.06006741497465079, q90=0.11895695130538603
Stress: median Sharpe 0.644, profitable windows 82.5%
Leverage: evaluated only after this strategy passed the underlying gate.

### ETH/USDT / breakout_atr_40_1.0
Family: breakout_atr
Entry: close clears prior 40-bar range by 1.0 ATR. Stop: 2.5x ATR from entry. Exit: exit on close across fast EMA. Flat during NaN/warmup; after an ATR stop, re-entry waits for a flat signal.
Position risk: 0.500%; parameter stability: 1.00
Validation: Sharpe 0.966, return 36.83%, max DD -18.29%
Walk-forward median Sharpe: 0.966
OOS: Sharpe 1.212, return 94.40%, max DD -35.69%, trades 123
MAE/MFE train quantiles: MAE q50=-0.014366905157814747, q75=-0.005736509540604617, q90=-0.002045127022580307; MFE q50=0.037228204238993645, q75=0.07475262747303868, q90=0.122362444295625
Stress: median Sharpe 0.979, profitable windows 98.0%
Leverage: evaluated only after this strategy passed the underlying gate.

### ETH/USDT / breakout_atr_60_0.5
Family: breakout_atr
Entry: close clears prior 60-bar range by 0.5 ATR. Stop: 2.5x ATR from entry. Exit: exit on close across fast EMA. Flat during NaN/warmup; after an ATR stop, re-entry waits for a flat signal.
Position risk: 0.500%; parameter stability: 1.00
Validation: Sharpe 0.814, return 31.61%, max DD -25.77%
Walk-forward median Sharpe: 0.779
OOS: Sharpe 1.488, return 164.23%, max DD -29.61%, trades 176
MAE/MFE train quantiles: MAE q50=-0.016730854220594837, q75=-0.007700629349100263, q90=-0.003167540782192146; MFE q50=0.028595635879020254, q75=0.06264144655149684, q90=0.1218723209901775
Stress: median Sharpe 0.663, profitable windows 92.0%
Leverage: evaluated only after this strategy passed the underlying gate.

### ETH/USDT / breakout_atr_60_1.0
Family: breakout_atr
Entry: close clears prior 60-bar range by 1.0 ATR. Stop: 2.5x ATR from entry. Exit: exit on close across fast EMA. Flat during NaN/warmup; after an ATR stop, re-entry waits for a flat signal.
Position risk: 0.500%; parameter stability: 1.00
Validation: Sharpe 0.949, return 35.12%, max DD -16.30%
Walk-forward median Sharpe: 1.117
OOS: Sharpe 1.441, return 118.52%, max DD -26.67%, trades 108
MAE/MFE train quantiles: MAE q50=-0.014616461475573606, q75=-0.005971168501286039, q90=-0.002045127022580307; MFE q50=0.040078540730708656, q75=0.07475262747303868, q90=0.12493283278739814
Stress: median Sharpe 1.130, profitable windows 99.5%
Leverage: evaluated only after this strategy passed the underlying gate.
