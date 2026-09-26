import pandas as pd

from research.phase3_proxy_evaluation import proxy_feature_groups, split_masks


def test_proxy_groups_are_explicit_and_proxy_named() -> None:
    columns = [
        "btc_return_1",
        "macro_es_f_proxy_value",
        "macro_es_f_proxy_return_1",
        "macro_nq_f_proxy_value",
        "macro_vix_proxy_value",
        "macro_urth_proxy_return_1",
    ]
    groups = proxy_feature_groups(columns)
    assert set(groups) == {
        "crypto_only", "crypto_plus_all_proxy", "crypto_plus_es",
        "crypto_plus_nq", "crypto_plus_vix", "crypto_plus_urth",
    }
    assert "macro_es_f_proxy_value" in groups["crypto_plus_es"]
    assert "macro_nq_f_proxy_value" not in groups["crypto_plus_es"]
    assert all("spy" not in name.lower() and "qqq" not in name.lower() for name in columns)


def test_split_masks_leave_purged_train_before_validation_and_frozen_holdout() -> None:
    index = pd.date_range("2019-01-01", "2026-08-31", freq="h", tz="UTC")
    train, validation, holdout, spec = split_masks(index, pd.Timedelta(hours=24))
    assert train.any() and validation.any() and holdout.any()
    assert index[train].max() < pd.Timestamp("2021-12-31 23:59:59", tz="UTC")
    assert index[validation].min() >= pd.Timestamp("2022-01-01", tz="UTC")
    assert index[holdout].min() >= pd.Timestamp("2024-01-01", tz="UTC")
    assert spec.purge_hours == 24
    assert spec.embargo_hours == 24