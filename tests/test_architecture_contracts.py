from datetime import timedelta

import pandas as pd
import pytest

from core.architecture import (
    AssetRole,
    BacktestRequest,
    DecisionAction,
    DecisionRecord,
    ExperimentManifest,
    FeatureSetSpec,
    HorizonClass,
    HorizonSpec,
    LabelSpec,
    StrategyConfig,
    StrategyStatus,
    align_asof,
    assert_no_future_features,
    content_hash,
    from_payload,
    load_data_universe,
    to_payload,
)


def _strategy() -> StrategyConfig:
    return StrategyConfig(
        strategy_id="btc-breakout",
        version="1.0.0",
        name="BTC breakout research",
        asset="BTC/USDT",
        direction="BOTH",
        asset_class="crypto",
        horizon=HorizonClass.H1,
        observation_timeframe="1h",
        strategy_family="breakout_atr",
        entry_logic={"lookback": 20},
        filter_logic={"min_atr_pct": 0.001},
        model_id=None,
        feature_set_id="crypto-price-v1",
        regime_filter={"enabled": True},
        stop_logic={"kind": "atr"},
        initial_stop={"atr_multiple": 2.5},
        trailing_stop={"enabled": True},
        exit_logic={"kind": "ema"},
        partial_exit_config={"enabled": False},
        risk_profile="normal",
        leverage_policy={"max": 2.0},
    )


def test_data_universe_separates_targets_and_features() -> None:
    universe = load_data_universe("configs/data_universe.yaml")
    assert {"BTC/USDT", "ETH/USDT", "SOL/USDT"} <= set(universe.targets())
    assert "VIX" in universe.feature_assets()
    assert AssetRole.TARGET in universe.asset("BTC/USDT").roles
    assert AssetRole.TARGET not in universe.asset("VIX").roles


def test_schema_roundtrip_is_canonical() -> None:
    feature_set = FeatureSetSpec(
        "crypto-price", "1.0.0", ("BTC/USDT", "ETH/USDT"), ("1h",),
        ("price", "trend", "volatility"), {"returns": 48},
        {"direction": "backward", "max_staleness_minutes": 120},
        {"method": "train_only_zscore"}, "labels-v1", "2026-09-23T00:00:00+00:00",
    )
    label = LabelSpec("labels-v1", (HorizonSpec("4h", 4, HorizonClass.H1),))
    experiment = ExperimentManifest(
        "exp-1", "git:test", "universe-v1", feature_set.feature_set_id, label.label_version,
        "btc-breakout", "1.0.0", {"type": "baseline"}, {}, {"start": "2020", "end": "2024"},
        {"start": "2024", "end": "2025"}, {"folds": 3}, {"start": "2025", "end": "2026"},
        None, {"runs": 10}, {"fee": 0.0005}, {"bps": 2}, {"status": "NOT_TESTED"}, {},
    )
    decision = DecisionRecord(
        "decision-1", "2026-09-23T12:00:00+00:00", "BTC/USDT", "LONG", "btc-breakout", "1.0.0",
        HorizonClass.H1, "1h", "trend", None, "crypto-price", 0.8, 0.01, -0.02, 0.04,
        100.0, 95.0, 10.0, 0.02, 1.5, 0.1, 0.0, DecisionAction.PAPER_ENTRY, ("SCORE_PASS",),
    )
    for value, cls in ((feature_set, FeatureSetSpec), (label, LabelSpec), (experiment, ExperimentManifest), (decision, DecisionRecord), (_strategy(), StrategyConfig)):
        restored = from_payload(cls, to_payload(value))
        assert to_payload(restored) == to_payload(value)
        assert content_hash(restored) == content_hash(value)


def test_horizon_boundaries_and_timeframe_are_distinct() -> None:
    assert HorizonSpec("1h", 1, HorizonClass.H1).horizon_class == HorizonClass.H1
    assert HorizonSpec("7d", 168, HorizonClass.H3).hours == 168
    with pytest.raises(ValueError):
        HorizonSpec("30d", 720, HorizonClass.H5)
    strategy = _strategy()
    assert strategy.observation_timeframe == "1h"
    assert strategy.horizon == HorizonClass.H1


def test_asof_alignment_respects_lag_and_staleness() -> None:
    source = pd.DataFrame(
        {"value": [10.0, 20.0]},
        index=pd.DatetimeIndex(["2026-01-01 10:00Z", "2026-01-01 12:00Z"]),
    )
    target = pd.DatetimeIndex(["2026-01-01 10:30Z", "2026-01-01 11:30Z", "2026-01-01 12:30Z", "2026-01-01 13:30Z"])
    aligned = align_asof(source, target, availability_lag=timedelta(hours=1), max_staleness=timedelta(hours=2))
    assert pd.isna(aligned["value"].iloc[0])
    assert aligned["value"].iloc[1] == 10.0
    assert aligned["value"].iloc[2] == 10.0
    assert aligned["value"].iloc[3] == 20.0


def test_future_feature_observation_is_rejected() -> None:
    target = pd.DatetimeIndex(["2026-01-01 10:00Z"])
    features = pd.DataFrame({"observed_at": [pd.Timestamp("2026-01-01 10:01Z")]}, index=target)
    with pytest.raises(ValueError, match="after prediction"):
        assert_no_future_features(features, target)


def test_research_strategy_cannot_become_implicit_paper_state() -> None:
    assert _strategy().status == StrategyStatus.RESEARCH
    assert BacktestRequest(_strategy(), "dataset-1", {}, {}, {}).strategy.status == StrategyStatus.RESEARCH