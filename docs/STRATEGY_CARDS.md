# Strategy Cards

## TREND_BREAKOUT v1.0.0

- Entry: prior-range breakout with aligned trend.
- Stop: the wider of ATR floor and stored adverse excursion.
- Trailing: ratchet by the initial stop distance; never loosen.
- Take-profit: measured MFE, with a minimum 0.5R objective.
- Exit: stop, target, opposite regime, stale data, or breakout failure.
- Typical hold: hours to several days.
- Risk: informational range 0.25%-2.0%; actual risk comes from the user/account calculation.
- Avoid: weak score, stale data, insufficient warmup, or range-bound conditions.

## MOMENTUM v1.0.0

- Entry: directional momentum agrees with the active regime.
- Stop/trailing/exit: ATR and stored excursion rules, never widened after entry.
- Typical hold: minutes to hours.
- Avoid: opposing regime, weak momentum, or extreme volatility.

## MEAN_REVERSION v1.0.0

- Entry: price extension from a reference range while the regime is non-trending.
- Stop: outside the invalidation band.
- Take-profit: reference mean or conservative measured target.
- Exit: regime transition, stop, or reference mean.
- Typical hold: hours to several days.
- Status: informational only; Phase 2 did not establish robust promotion evidence.

## REGIME_ADAPTIVE v1.0.0

- Entry: select a family only after the regime is known.
- Stops, trailing, targets, and exits: inherited from the selected family.
- Typical hold: hours to two weeks.
- Avoid: unknown regime or insufficient data.

The cards describe decisions. They do not place orders. Leverage is derived after entry, stop, risk budget, and notional are known.