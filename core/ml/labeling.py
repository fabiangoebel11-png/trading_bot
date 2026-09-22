"""Triple-barrier labeling (López de Prado style) for the trend-continuation model.

Why not a simple fixed-horizon return label: on noisy intraday data, "return
after N bars" mislabels a bar as a trend-continuation loss even when price
touched a large profit and gave it back within the window, and it ignores
volatility -- a 1% move is a breakout in a quiet regime and noise in a violent
one. The triple barrier instead asks "which happens first: price travels
+k*ATR (up), -k*ATR (down), or neither within N bars (flat/time-out)", which
is both regime-adaptive (barrier width scales with ATR) and directly matches
how a stop/take-profit-managed position would actually resolve.

Every label also carries ``t1`` -- the timestamp at which its outcome became
known. This is required by ``core/ml/dataset.py``'s purged cross-validation:
a label's information window is ``[t, t1]``, and any training sample whose
window overlaps the test fold leaks future information into training unless
it is purged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def triple_barrier_labels(
    close: pd.Series, atr: pd.Series, horizon: int, atr_multiple: float
) -> pd.DataFrame:
    """Return a DataFrame (aligned to ``close.index``) with columns:
      - ``label``: +1 (upper barrier touched first), -1 (lower touched first),
        0 (neither touched within ``horizon`` bars -- time barrier).
      - ``t1``: timestamp of the touch (or of the time barrier if neither hit).

    Rows in the last ``horizon`` bars can't observe a full future window and
    are set to NaN/NaT (drop before training -- there is no look-ahead trick
    that fixes this, the future data simply doesn't exist yet).

    Vectorized across the ``horizon`` dimension (one pass per step-ahead
    offset) rather than per-bar, since a plain Python double loop over tens of
    thousands of bars * 48-bar horizon would be far too slow.
    """
    n = len(close)
    prices = close.to_numpy(dtype=np.float64)
    atr_arr = atr.to_numpy(dtype=np.float64)

    upper = prices + atr_multiple * atr_arr
    lower = prices - atr_multiple * atr_arr

    label = np.zeros(n, dtype=np.float64)
    touch_offset = np.full(n, horizon, dtype=np.int64)  # default: time barrier at +horizon
    touched = np.zeros(n, dtype=bool)

    for step in range(1, horizon + 1):
        idx = np.arange(0, n - step)
        future_price = prices[idx + step]
        still_open = ~touched[idx]

        hit_upper = still_open & (future_price >= upper[idx])
        hit_lower = still_open & (future_price <= lower[idx])

        up_idx = idx[hit_upper]
        down_idx = idx[hit_lower]
        label[up_idx] = 1.0
        touch_offset[up_idx] = step
        touched[up_idx] = True

        label[down_idx] = -1.0
        touch_offset[down_idx] = step
        touched[down_idx] = True

    incomplete = np.arange(n) >= (n - horizon)
    label[incomplete] = np.nan

    t1_pos = np.clip(np.arange(n) + touch_offset, 0, n - 1)
    t1 = close.index.to_numpy()[t1_pos]
    t1 = pd.Series(t1, index=close.index)
    t1[incomplete] = pd.NaT

    return pd.DataFrame({"label": label, "t1": t1}, index=close.index)
