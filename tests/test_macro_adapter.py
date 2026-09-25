import pandas as pd

from core.ml.macro_adapter import align_macro_proxies, load_macro_csv, requested_asset_statuses


def _macro(tmp_path):
    path = tmp_path / "macro_ESF_VIX_URTH_3600d.csv"
    pd.DataFrame({
        "Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "equity": [100.0, 101.0, 102.0],
        "vix": [20.0, 19.0, 18.0],
        "world": [50.0, 51.0, 52.0],
    }).to_csv(path, index=False)
    return path


def test_macro_alignment_requires_availability_lag(tmp_path):
    path = _macro(tmp_path)
    # Register the temporary file under the production filename contract.
    from core.ml import macro_adapter
    macro_adapter.MACRO_FILES[str(path)] = macro_adapter.MACRO_FILES["data/macro_ESF_VIX_URTH_3600d.csv"]
    target = pd.date_range("2024-01-01", periods=5, freq="12h", tz="UTC")
    result = align_macro_proxies(path, target, availability_lag=pd.Timedelta(days=1), max_staleness=pd.Timedelta(days=3))
    assert result.features.iloc[0].isna().all()
    assert result.features.loc["2024-01-02 12:00Z", "macro_es_f_proxy_value"] == 100.0
    assert result.provenance["proxy_or_real"].eq("PROXY").all()


def test_future_macro_mutation_does_not_change_earlier_features(tmp_path):
    path = _macro(tmp_path)
    from core.ml import macro_adapter
    macro_adapter.MACRO_FILES[str(path)] = macro_adapter.MACRO_FILES["data/macro_ESF_VIX_URTH_3600d.csv"]
    target = pd.date_range("2024-01-01", periods=7, freq="12h", tz="UTC")
    before = align_macro_proxies(path, target, availability_lag=pd.Timedelta(days=1), max_staleness=pd.Timedelta(days=3)).features
    raw, _ = load_macro_csv(path)
    raw.loc[pd.Timestamp("2024-01-03", tz="UTC"), "equity"] = 999.0
    raw.reset_index(names="Date").to_csv(path, index=False)
    after = align_macro_proxies(path, target, availability_lag=pd.Timedelta(days=1), max_staleness=pd.Timedelta(days=3)).features
    pd.testing.assert_frame_equal(before.iloc[:4], after.iloc[:4])


def test_stale_macro_values_are_invalid(tmp_path):
    path = _macro(tmp_path)
    from core.ml import macro_adapter
    macro_adapter.MACRO_FILES[str(path)] = macro_adapter.MACRO_FILES["data/macro_ESF_VIX_URTH_3600d.csv"]
    target = pd.DatetimeIndex(["2024-01-01 00:00Z", "2024-01-10 00:00Z"])
    result = align_macro_proxies(path, target, availability_lag=pd.Timedelta(days=1), max_staleness=pd.Timedelta(days=3))
    assert result.features.iloc[1].isna().all()


def test_later_availability_removes_value_at_target_time(tmp_path):
    path = _macro(tmp_path)
    from core.ml import macro_adapter
    macro_adapter.MACRO_FILES[str(path)] = macro_adapter.MACRO_FILES["data/macro_ESF_VIX_URTH_3600d.csv"]
    target = pd.DatetimeIndex(["2024-01-02 12:00Z"])
    early = align_macro_proxies(path, target, availability_lag=pd.Timedelta(days=1), max_staleness=pd.Timedelta(days=3))
    late = align_macro_proxies(path, target, availability_lag=pd.Timedelta(days=2), max_staleness=pd.Timedelta(days=3))
    assert early.features.iloc[0]["macro_es_f_proxy_value"] == 100.0
    assert late.features.iloc[0].isna().all()


def test_weekend_target_does_not_create_new_macro_observation(tmp_path):
    path = _macro(tmp_path)
    from core.ml import macro_adapter
    macro_adapter.MACRO_FILES[str(path)] = macro_adapter.MACRO_FILES["data/macro_ESF_VIX_URTH_3600d.csv"]
    target = pd.DatetimeIndex(["2024-01-05 12:00Z", "2024-01-06 12:00Z"])
    result = align_macro_proxies(path, target, availability_lag=pd.Timedelta(days=1), max_staleness=pd.Timedelta(days=3))
    assert result.features.iloc[1].equals(result.features.iloc[0])


def test_requested_statuses_do_not_claim_missing_real_assets():
    statuses = {status.requested_asset: status.status for status in requested_asset_statuses()}
    assert statuses["SPY"] == "AVAILABLE_PROXY"
    assert statuses["QQQ"] == "AVAILABLE_PROXY"
    assert statuses["SOL/USDT"] == "INSUFFICIENT_DATA"