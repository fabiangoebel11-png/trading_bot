from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.ml.providers import (
    AssetSpec,
    TwelveDataHistoricalProvider,
    asset_specs_from_config,
    canonical_cache_path,
    prepare_asset_specs,
)
from train import load_config


def _payload(timestamp: str, close: float) -> bytes:
    return json.dumps({"values": [{"datetime": timestamp, "open": close - 1, "high": close + 1, "low": close - 2, "close": close, "volume": 10}]}).encode()


def test_twelve_data_parses_chunked_responses_and_deduplicates(monkeypatch) -> None:
    calls = []

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return self.body

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return Response(json.dumps({"values": [
            {"datetime": "2026-01-01 00:00:00", "open": "99", "high": "101", "low": "98", "close": "100", "volume": "10"},
            {"datetime": "2026-01-01 00:00:00", "open": "99", "high": "101", "low": "98", "close": "100", "volume": "10"},
        ]}).encode())

    monkeypatch.setattr("core.ml.providers.urlopen", fake_urlopen)
    provider = TwelveDataHistoricalProvider({"max_requests_per_day": 10, "max_requests_per_minute": 100}, api_key="test-key")
    frame = provider.load(AssetSpec("QQQ", "QQQ", "twelve_data", "5m", 365, proxy_of="QQQ"))
    assert len(calls) >= 2
    assert len(frame) == 1
    assert frame.index.tz is not None
    assert provider.last_report["source_symbol"] == "QQQ"


def test_twelve_data_missing_key_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("TWELVE_DATA_API_KEY", raising=False)
    provider = TwelveDataHistoricalProvider({"api_key_env": "TWELVE_DATA_API_KEY"})
    with pytest.raises(RuntimeError, match="TWELVE_DATA_API_KEY"):
        provider.load(AssetSpec("QQQ", "QQQ", "twelve_data", "5m", 1))


def test_twelve_data_quota_stops_before_next_request(monkeypatch) -> None:
    calls = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps({"values": []}).encode()

    monkeypatch.setattr("core.ml.providers.urlopen", lambda request, timeout: (calls.append(request.full_url) or Response()))
    provider = TwelveDataHistoricalProvider({"max_requests_per_day": 1, "max_requests_per_minute": 100}, api_key="secret")
    provider.load(AssetSpec("QQQ", "QQQ", "twelve_data", "5m", 365))
    assert len(calls) == 1
    assert provider.requests_used == 1


def test_twelve_data_five_minute_history_is_capped_at_one_year(monkeypatch) -> None:
    requests = []

    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps({"values": []}).encode()

    monkeypatch.setattr("core.ml.providers.urlopen", lambda request, timeout: (requests.append(request.full_url) or Response()))
    provider = TwelveDataHistoricalProvider({"max_requests_per_day": 1, "max_requests_per_minute": 100}, api_key="secret")
    provider.load(AssetSpec("QQQ", "QQQ", "twelve_data", "5m", 1095))
    assert provider.last_report["expected_candles"] == 365 * 288
    assert "start_date=" in requests[0]
    assert TwelveDataHistoricalProvider.estimate_requests(1095, "5m") == TwelveDataHistoricalProvider.estimate_requests(365, "5m")


def test_equity_profile_uses_one_year_intraday_and_long_daily_history() -> None:
    config = load_config("configs/training_equity_intraday.yaml")
    specs = asset_specs_from_config(config)
    for asset in ("NASDAQ100_PROXY", "SP500_PROXY"):
        by_timeframe = {spec.timeframe: spec for spec in specs if spec.name == asset}
        assert all(by_timeframe[timeframe].history_days == 365 for timeframe in ("5m", "15m", "1h", "4h"))
        assert by_timeframe["1d"].history_days == 14600


def test_config_uses_explicit_proxy_symbols_and_local_derived_frames() -> None:
    config = load_config("configs/training.yaml")
    specs = asset_specs_from_config(config)
    qqq = [spec for spec in specs if spec.name == "NASDAQ100_PROXY"]
    spy = [spec for spec in specs if spec.name == "SP500_PROXY"]
    assert qqq[0].symbol == "QQQ" and qqq[0].proxy_of == "QQQ"
    assert spy[0].symbol == "SPY" and spy[0].proxy_of == "SPY"
    assert {spec.provider for spec in qqq if spec.timeframe in {"15m", "1h", "4h"}} == {"local_derived"}
    assert "DAX" not in config.ml.assets


def test_local_resampling_produces_expected_15m_1h_4h_caches(tmp_path: Path) -> None:
    config = load_config("configs/training.yaml")
    config.ml.assets = ["NASDAQ100_PROXY"]
    config.ml.optional_assets = []
    config.ml.context_assets = {}
    config.ml.timeframes = ["5m", "15m", "1h", "4h"]
    config.ml.timeframe_history_days = {"5m": 1, "15m": 1, "1h": 1, "4h": 1}
    config.ml.market_cache_dir = str(tmp_path / "market")
    base_index = pd.date_range("2026-01-01", periods=48, freq="5min", tz="UTC")
    base = pd.DataFrame({"open": np.arange(48) + 100, "high": np.arange(48) + 101, "low": np.arange(48) + 99, "close": np.arange(48) + 100.5, "volume": 1.0}, index=base_index)

    class FakeTwelve:
        last_report = {"status": "success", "requested_start": "2026-01-01T00:00:00+00:00", "requested_end": "2026-01-02T00:00:00+00:00"}
        def load(self, asset): return base

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("core.ml.providers.provider_for", lambda *args, **kwargs: FakeTwelve())
    try:
        _, reports = prepare_asset_specs(asset_specs_from_config(config), cache_root=config.ml.market_cache_dir, cache_dir=str(tmp_path), twelve_data_settings=config.data.twelve_data)
    finally:
        monkeypatch.undo()
    assert len(reports) == 4
    assert canonical_cache_path(config.ml.market_cache_dir, AssetSpec("NASDAQ100_PROXY", "QQQ", "local_derived", "15m", 1)).exists()
    assert len(pd.read_parquet(Path(config.ml.market_cache_dir) / "NASDAQ100_PROXY" / "1h" / "local_derived.parquet")) == 5
