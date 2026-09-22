"""Tests for discrete leverage-tier discovery (execution/exchange_client.py:
CCXTExchangeClient.fetch_leverage_brackets) -- must come from real market
metadata, never a hardcoded guess, with a graceful fallback."""
from __future__ import annotations

from execution.exchange_client import CCXTExchangeClient


def _client(dry_run: bool) -> CCXTExchangeClient:
    return CCXTExchangeClient(dry_run=dry_run, testnet=True)


def test_dry_run_returns_fallback_without_touching_exchange() -> None:
    client = _client(dry_run=True)
    steps = client.fetch_leverage_brackets("BTC/USDT", fallback_steps=[1, 2, 5, 10])
    assert steps == [1, 2, 5, 10]


def test_uses_fetch_leverage_tiers_when_available(monkeypatch) -> None:
    client = _client(dry_run=False)
    monkeypatch.setattr(client, "_ensure_markets", lambda: None)
    client.exchange.has = {"fetchLeverageTiers": True}
    client.exchange.fetch_leverage_tiers = lambda symbols: {
        "BTC/USDT": [{"maxLeverage": 5}, {"maxLeverage": 10}, {"maxLeverage": 20}]
    }
    steps = client.fetch_leverage_brackets("BTC/USDT", fallback_steps=[1, 2])
    assert steps == [5.0, 10.0, 20.0]


def test_falls_back_to_market_leverage_ceiling_when_no_tiers(monkeypatch) -> None:
    client = _client(dry_run=False)
    monkeypatch.setattr(client, "_ensure_markets", lambda: None)
    client.exchange.has = {"fetchLeverageTiers": False}
    client.exchange.markets = {"BTC/USDT": {"limits": {"leverage": {"max": 5}}}}
    steps = client.fetch_leverage_brackets("BTC/USDT", fallback_steps=[1, 2])
    assert steps == [1, 2, 3, 4, 5]


def test_falls_back_on_exception(monkeypatch) -> None:
    client = _client(dry_run=False)
    monkeypatch.setattr(client, "_ensure_markets", lambda: None)
    client.exchange.has = {"fetchLeverageTiers": True}

    def _raise(symbols):
        raise RuntimeError("network down")

    client.exchange.fetch_leverage_tiers = _raise
    steps = client.fetch_leverage_brackets("BTC/USDT", fallback_steps=[1, 2, 3])
    assert steps == [1, 2, 3]


def test_falls_back_when_no_metadata_available(monkeypatch) -> None:
    client = _client(dry_run=False)
    monkeypatch.setattr(client, "_ensure_markets", lambda: None)
    client.exchange.has = {"fetchLeverageTiers": False}
    client.exchange.markets = {}
    steps = client.fetch_leverage_brackets("BTC/USDT", fallback_steps=[1, 2, 3])
    assert steps == [1, 2, 3]
