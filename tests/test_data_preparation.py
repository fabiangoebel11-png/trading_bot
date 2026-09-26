from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core import data_loader
from core.ml.data import quality_report
from core.ml.providers import AssetSpec, EODHDHistoricalProvider, canonical_cache_path


class _FakeExchange:
    rateLimit = 0

    def __init__(self, config=None):
        self.config = config

    def parse_timeframe(self, timeframe: str) -> int:
        return {"1h": 3600}[timeframe]

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int):
        if since >= 1_700_010_800_000:
            return []
        return [
            [1_700_000_000_000, 100, 101, 99, 100.5, 1],
            [1_700_003_600_000, 100.5, 102, 100, 101, 2],
            [1_700_003_600_000, 100.5, 102, 100, 101, 2],
        ]


def test_ccxt_pagination_deduplicates_and_reports_partial(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(data_loader.ccxt, "fake", _FakeExchange, raising=False)
    frame, report = data_loader.fetch_ohlcv_history(
        "BTC/USDT", "1h", 3650, cache_dir=str(tmp_path), exchange_id="fake", use_cache=False, return_report=True
    )
    assert len(frame) == 2
    assert frame.index.is_monotonic_increasing
    assert report["status"] == "partial"
    assert report["coverage_ratio"] < 1.0
    assert not (tmp_path / "ohlcv_BTC-USDT_3650d.csv").exists()


def test_ccxt_complete_stale_seed_requests_a_nonzero_tail_page(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class TailExchange(_FakeExchange):
        def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int):
            calls.append((since, limit))
            return []

    monkeypatch.setattr(data_loader.ccxt, "fake", TailExchange, raising=False)
    now = pd.Timestamp.utcnow().tz_convert("UTC").floor("h")
    seed = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1.0]},
        index=pd.DatetimeIndex([now - pd.Timedelta(days=3)]),
    )
    data_loader.fetch_ohlcv_history(
        "BTC/USDT", "1h", 3, cache_dir=str(tmp_path), exchange_id="fake", use_cache=False, seed_frame=seed,
    )

    assert calls
    assert calls[0][1] >= 1


def test_ccxt_substantial_late_seed_continues_from_its_tail(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class TailExchange(_FakeExchange):
        def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int):
            calls.append((since, limit))
            return []

    monkeypatch.setattr(data_loader.ccxt, "fake", TailExchange, raising=False)
    now = pd.Timestamp.utcnow().tz_convert("UTC").floor("h")
    index = pd.date_range(now - pd.Timedelta(hours=61), periods=58, freq="h")
    seed = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=index)
    data_loader.fetch_ohlcv_history(
        "BTC/USDT", "1h", 3, cache_dir=str(tmp_path), exchange_id="fake", use_cache=False, seed_frame=seed,
    )

    assert calls
    assert calls[0][0] == int((index.max() + pd.Timedelta(hours=1)).timestamp() * 1000)


def test_ccxt_oversized_seed_pages_by_requested_window_not_total_cache(monkeypatch, tmp_path: Path, capsys) -> None:
    now = pd.Timestamp.utcnow().tz_convert("UTC").floor("h")
    seed_index = pd.date_range(end=now - pd.Timedelta(days=120), periods=19_927, freq="1h")
    seed = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0},
        index=seed_index,
    )
    calls = []

    class RecoveringExchange(_FakeExchange):
        def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int):
            calls.append((since, limit))
            end_ms = int(now.timestamp() * 1000)
            return [
                [timestamp, 100.0, 101.0, 99.0, 100.0, 1.0]
                for timestamp in range(since, end_ms + 1, 3_600_000)
            ][:limit]

    monkeypatch.setattr(data_loader.ccxt, "fake", RecoveringExchange, raising=False)
    frame = data_loader.fetch_ohlcv_history(
        "BTC/USDT", "1h", 21, cache_dir=str(tmp_path), exchange_id="fake", use_cache=False,
        seed_frame=seed,
    )

    assert len(calls) <= 2
    assert calls[0][1] > 1
    assert calls[0][0] >= int((now - pd.Timedelta(days=21)).timestamp() * 1000)
    assert len(frame) >= len(seed)
    assert frame.index.max() > seed_index[-1].tz_localize(None) + pd.Timedelta(days=100)
    assert "angefragte Kerzen geladen" in capsys.readouterr().out


def test_quality_report_distinguishes_session_volume_and_coverage() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC")
    frame = pd.DataFrame(
        {"open": [1, 1, 1], "high": [1, 1, 1], "low": [1, 1, 1], "close": [1, 1, 1], "volume": [0, 0, 0]},
        index=index,
    )
    report = quality_report(
        frame,
        asset="VIX",
        timeframe="1d",
        provider="yfinance",
        cache_path="data/market/VIX/1d/yfinance.parquet",
        market_type="equity_index",
        session_timezone="America/New_York",
        requested_start="2024-01-01T00:00:00+00:00",
        requested_end="2024-01-03T00:00:00+00:00",
    )
    assert report["volume_semantics"] == "unavailable"
    assert report["missing_ratio"] == 0.0
    assert report["coverage_ratio"] == 1.0


def test_eodhd_chunking_uses_multiple_requests_without_network(monkeypatch) -> None:
    calls = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps([
                {"timestamp": 1700000000, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}
            ]).encode()

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return _Response()

    monkeypatch.setenv("EODHD_API_KEY", "test-key")
    monkeypatch.setattr("core.ml.providers.urlopen", fake_urlopen)
    provider = EODHDHistoricalProvider()
    frame = provider.load(AssetSpec("NASDAQ-100", "NDX.INDX", "eodhd", "5m", 61, market_type="equity_index"))
    assert len(calls) == 3
    assert len(frame) == 1
    assert "NDX.INDX" in calls[0]


def test_cache_path_contains_asset_timeframe_and_provider() -> None:
    path = canonical_cache_path("data/market", AssetSpec("BTC/USDT", "BTC/USDT", "ccxt", "5m", 30))
    assert path.as_posix().endswith("BTC_USDT/5m/ccxt.parquet")
