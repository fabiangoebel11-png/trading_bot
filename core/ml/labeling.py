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
  close: pd.Series,
  atr: pd.Series,
  horizon: int,
  atr_multiple: float,
  *,
  high: pd.Series | None = None,
  low: pd.Series | None = None,
  stop_atr_multiple: float | None = None,
  take_profit_atr_multiple: float | None = None,
) -> pd.DataFrame:
    """Return a DataFrame (aligned to ``close.index``) with columns:
      - ``label``: +1 (upper barrier touched first), -1 (lower touched first),
        0 (neither touched within ``horizon`` bars -- time barrier).
      - ``t1``: timestamp of the touch (or of the time barrier if neither hit).
      - ``mfe`` / ``mae``: maximum favorable/adverse excursion as a return.
      - ``barrier_return`` / ``time_to_barrier``: resolved return and offset.

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

    take_profit_multiple = atr_multiple if take_profit_atr_multiple is None else take_profit_atr_multiple
    stop_multiple = atr_multiple if stop_atr_multiple is None else stop_atr_multiple
    upper = prices + take_profit_multiple * atr_arr
    lower = prices - stop_multiple * atr_arr
    high_arr = prices if high is None else high.to_numpy(dtype=np.float64)
    low_arr = prices if low is None else low.to_numpy(dtype=np.float64)

    label = np.zeros(n, dtype=np.float64)
    touch_offset = np.full(n, horizon, dtype=np.int64)  # default: time barrier at +horizon
    touched = np.zeros(n, dtype=bool)

    for step in range(1, horizon + 1):
        idx = np.arange(0, n - step)
        future_high = high_arr[idx + step]
        future_low = low_arr[idx + step]
        still_open = ~touched[idx]

        hit_upper = still_open & (future_high >= upper[idx])
        hit_lower = still_open & (future_low <= lower[idx])

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

    mfe = np.full(n, np.nan, dtype=np.float64)
    mae = np.full(n, np.nan, dtype=np.float64)
    barrier_return = np.full(n, np.nan, dtype=np.float64)
    for start in range(max(0, n - horizon)):
      end = min(n, start + horizon + 1)
      future_high = high_arr[start + 1 : end]
      future_low = low_arr[start + 1 : end]
      if len(future_high) == 0 or not np.isfinite(prices[start]):
        continue
      mfe[start] = np.max(future_high) / prices[start] - 1.0
      mae[start] = np.min(future_low) / prices[start] - 1.0
      resolved = int(touch_offset[start])
      barrier_return[start] = prices[min(start + resolved, n - 1)] / prices[start] - 1.0

    incomplete = np.arange(n) >= (n - horizon)
    mfe[incomplete] = np.nan
    mae[incomplete] = np.nan
    barrier_return[incomplete] = np.nan
    time_to_barrier = touch_offset.astype(np.float64)
    time_to_barrier[incomplete] = np.nan
    return pd.DataFrame(
      {
        "label": label,
        "t1": t1,
        "mfe": mfe,
        "mae": mae,
        "barrier_return": barrier_return,
        "time_to_barrier": time_to_barrier,
      },
      index=close.index,
    )


def qualified_trade_labels(
    close: pd.Series,
    atr: pd.Series,
    horizon: int,
    atr_multiple: float,
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    stop_atr_multiple: float | None = None,
    take_profit_atr_multiple: float | None = None,
    min_quality_ratio: float = 1.5,
    min_atr_move: float = 0.5,
) -> pd.DataFrame:
    """Re-label the raw triple-barrier outcome into a trade-quality-aware target.

    Directional signals are preserved only when the future move is both large
    enough and asymmetrically favorable relative to the adverse excursion. Weak or
    noise-driven windows are deliberately reclassified as NO_TRADE so the model
    learns actual opportunities instead of being forced to label every bar.
    """
    raw = triple_barrier_labels(
        close,
        atr,
        horizon,
        atr_multiple,
        high=high,
        low=low,
        stop_atr_multiple=stop_atr_multiple,
        take_profit_atr_multiple=take_profit_atr_multiple,
    )
    tradeable = np.zeros(len(raw), dtype=bool)
    close_arr = close.to_numpy(dtype=np.float64)
    atr_arr = atr.to_numpy(dtype=np.float64)
    mfe = np.nan_to_num(raw["mfe"].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    mae = np.nan_to_num(raw["mae"].to_numpy(dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    favorable = np.maximum(mfe, 0.0)
    adverse = np.maximum(-mae, 0.0)
    min_move = np.maximum(min_atr_move * atr_arr / np.maximum(np.abs(close_arr), 1e-9), 1e-9)
    tradeable = (favorable >= min_move) & (favorable >= min_quality_ratio * np.maximum(adverse, 1e-9))
    tradeable |= (adverse >= min_move) & (adverse >= min_quality_ratio * np.maximum(favorable, 1e-9))

    qualified = raw.copy()
    qualified["label"] = 0.0
    qualified.loc[raw["label"] == 1.0, "label"] = np.where(tradeable[raw["label"] == 1.0], 1.0, 0.0)
    qualified.loc[raw["label"] == -1.0, "label"] = np.where(tradeable[raw["label"] == -1.0], -1.0, 0.0)
    qualified["tradeable"] = tradeable.astype(np.float64)
    qualified["quality_ratio"] = np.divide(
        favorable,
        np.maximum(adverse, 1e-9),
        out=np.zeros_like(favorable, dtype=np.float64),
        where=np.maximum(adverse, 1e-9) > 0.0,
    )
    return qualified


def build_horizon_label_map(
    close: pd.Series,
    atr: pd.Series,
    horizons: int | list[int] | tuple[int, ...],
    atr_multiple: float,
    *,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    stop_atr_multiple: float | None = None,
    take_profit_atr_multiple: float | None = None,
    min_quality_ratio: float = 1.5,
    min_atr_move: float = 0.5,
) -> dict[int, pd.DataFrame]:
    """Compute trade-quality labels for each configured forecast horizon."""
    normalized = tuple(int(h) for h in ([horizons] if isinstance(horizons, int) else horizons))
    labels: dict[int, pd.DataFrame] = {}
    for horizon in normalized:
        labels[horizon] = qualified_trade_labels(
            close,
            atr,
            horizon,
            atr_multiple,
            high=high,
            low=low,
            stop_atr_multiple=stop_atr_multiple,
            take_profit_atr_multiple=take_profit_atr_multiple,
            min_quality_ratio=min_quality_ratio,
            min_atr_move=min_atr_move,
        )
    return labels
