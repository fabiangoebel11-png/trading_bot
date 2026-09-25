import numpy as np
import pandas as pd
import pytest
import yaml

from core.ml.research_design import (
    assert_selection_split,
    label_end_times,
    purged_train_mask,
    validate_feature_availability,
    validate_target_context_isolation,
)


def test_target_context_isolation_is_strict() -> None:
    validate_target_context_isolation("BTC/USDT", ["ES_F_PROXY", "VIX"])
    with pytest.raises(ValueError, match="target_asset"):
        validate_target_context_isolation("BTC/USDT", ["BTC/USDT"])
    with pytest.raises(ValueError, match="not allowed"):
        validate_target_context_isolation("ES_F_PROXY", [])


def test_holdout_cannot_enter_selection() -> None:
    assert_selection_split("validation")
    with pytest.raises(ValueError, match="evaluation-only"):
        assert_selection_split("final_holdout")


def test_feature_availability_requires_causal_contract() -> None:
    contract = {
        "feature_name": "macro_vix_proxy_return",
        "source_asset": "VIX",
        "source_timestamp": "2024-01-02T00:00:00Z",
        "availability_timestamp": "2024-01-04T00:00:00Z",
        "alignment_timestamp": "2024-01-04T00:00:00Z",
        "timeframe": "1d",
        "transformation": "return_1",
        "staleness_limit": "5d",
        "proxy_or_real": "PROXY",
        "causal_status": "CAUSAL",
    }
    validate_feature_availability(contract, pd.Timestamp("2024-01-04", tz="UTC"))
    contract["availability_timestamp"] = "2024-01-05T00:00:00Z"
    with pytest.raises(ValueError, match="after prediction"):
        validate_feature_availability(contract, pd.Timestamp("2024-01-04", tz="UTC"))


@pytest.mark.parametrize("horizon", ["4h", "24h", "7d", "14d"])
def test_purge_and_embargo_for_overlapping_labels(horizon: str) -> None:
    delta = pd.Timedelta(horizon)
    samples = pd.date_range("2024-01-01", periods=20, freq="h", tz="UTC")
    ends = label_end_times(samples, delta)
    mask = purged_train_mask(samples, ends, pd.Timestamp("2024-01-01 12:00", tz="UTC"), delta)
    assert mask.dtype == np.bool_
    assert not mask[-1]
    assert (samples[mask] < pd.Timestamp("2024-01-01 12:00", tz="UTC") - delta).all()
    assert (ends[mask] < pd.Timestamp("2024-01-01 12:00", tz="UTC")).all()


def test_label_end_is_strictly_future() -> None:
    samples = pd.date_range("2024-01-01", periods=3, freq="4h", tz="UTC")
    ends = label_end_times(samples, pd.Timedelta("14d"))
    assert (ends > samples).all()


def test_frozen_manifest_is_reproducible_and_proxy_safe() -> None:
    config_path = "configs/ml_research_v1.yaml"
    manifest_path = "data/research/phase3/ml_research_manifest.json"
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = __import__("json").load(handle)
    assert config["feature_set_version"] == "crypto_core_v1"
    assert config["proxy_data"]["promoted"] is False
    assert set(manifest["target_assets"]) == {"BTC/USDT", "ETH/USDT"}
    assert set(manifest["context_assets"]) == {"ES_F_PROXY", "NQ_F_PROXY", "VIX", "URTH"}
    assert len(manifest["research_matrix"]) == 24
    assert all(row["status"] == "READY" for row in manifest["research_matrix"])
    assert all(row["target"] in manifest["target_assets"] for row in manifest["research_matrix"])
    assert all("SPY" not in row["target"] and "QQQ" not in row["target"] for row in manifest["research_matrix"])