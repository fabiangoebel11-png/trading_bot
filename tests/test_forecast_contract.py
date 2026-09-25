import pandas as pd

from core.ml.forecast_contract import context_feature_frame


def test_context_alignment_is_backward_and_reports_missing_series() -> None:
    target = pd.date_range(pd.Timestamp.now(tz="UTC").floor("h"), periods=3, freq="1h", tz="UTC")
    context = {
        "VIX": pd.DataFrame({"close": [20.0, 21.0]}, index=pd.DatetimeIndex([target[0] - pd.Timedelta(hours=2), target[1]])),
    }
    _, snapshot = context_feature_frame(context, target, max_age=pd.Timedelta(days=3))
    assert snapshot.status == "MISSING"
    vix = next(item for item in snapshot.series if item.asset == "VIX")
    assert vix.status == "OK"
    assert any(item.asset == "GOLD" and item.status == "MISSING" for item in snapshot.series)
