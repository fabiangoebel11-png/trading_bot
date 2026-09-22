"""Read-only testnet-readiness check: prints the exact market metadata a
Bybit/Binance-style exchange exposes for every traded symbol (leverage tiers,
lot size / minimum order amount, tick size) plus the position-sizing config
that would be in effect -- so all of it can be eyeballed against real
exchange docs BEFORE the first testnet order.

Never places an order, never touches DRY_RUN/testnet env vars, never mutates
any config file. Those must stay explicit operator decisions (see .env.example);
this script only prints what WOULD apply, using a local, ephemeral config
instance built here for reporting purposes only.

Run with: python -m uv run research/testnet_readiness_check.py
"""
from __future__ import annotations

from core.config import TradingBotConfig
from execution.exchange_client import CCXTExchangeClient


def _testnet_readiness_config() -> TradingBotConfig:
    """A conservative, testnet-appropriate config for this readiness report
    only -- NOT applied anywhere automatically. Position-selection (top-1),
    low notional caps, ML strictly shadow-only (production_enabled stays the
    dataclass default, False)."""
    config = TradingBotConfig()
    config.selection.enabled = True
    config.selection.max_active_positions = 1
    config.capital.max_absolute_position_size_usdt = 50.0
    config.capital.max_total_notional_usdt = 50.0
    config.ml.enabled = True
    assert config.ml.production_enabled is False, "must stay manual-only, see TrendMLConfig docstring"
    return config


def main() -> None:
    config = _testnet_readiness_config()
    client = CCXTExchangeClient(
        exchange_id=config.data.exchange_id, market_type=config.data.market_type, dry_run=True, testnet=True
    )
    client._ensure_markets()  # public call, safe without API keys

    print("=== Testnet-Readiness-Check (rein lesend, keine Order, kein DRY_RUN-Wechsel) ===")
    print(f"Exchange: {config.data.exchange_id} ({config.data.market_type})")
    print(
        f"Position-Sizing-Config fuer diesen Check: max_active_positions="
        f"{config.selection.max_active_positions}, max_absolute_position_size_usdt="
        f"{config.capital.max_absolute_position_size_usdt}, max_total_notional_usdt="
        f"{config.capital.max_total_notional_usdt}"
    )
    print(
        f"ML-Status: enabled={config.ml.enabled}, production_enabled={config.ml.production_enabled} "
        "(muss False bleiben -- ML bleibt Shadow-only bis ein Mensch die Holdout-Promotion manuell freigibt)"
    )
    print(f"Konfigurierter Margin-Modus (wird bei Start via configure_account gesetzt): {config.execution.exchange_margin_mode}")

    for symbol in config.data.symbols:
        print(f"\n--- {symbol} ---")
        market = (client.exchange.markets or {}).get(symbol)
        if market is None:
            print("  Keine Marktmetadaten gefunden (Symbol evtl. auf diesem Testnet nicht gelistet).")
            continue

        leverage_steps = client.fetch_leverage_brackets(symbol, config.execution.fallback_leverage_steps)
        print(f"  Erlaubte Leverage-Stufen: {leverage_steps}")

        limits = market.get("limits", {}) or {}
        amount_limits = limits.get("amount", {}) or {}
        cost_limits = limits.get("cost", {}) or {}
        print(f"  Lot-Size / Min-Amount: min={amount_limits.get('min')}, max={amount_limits.get('max')}")
        print(f"  Min-Notional (minOrderAmt-Aequivalent): {cost_limits.get('min')}")

        precision = market.get("precision", {}) or {}
        print(f"  Tick-Size (Preis-Precision): {precision.get('price')}")
        print(f"  Amount-Precision: {precision.get('amount')}")

    print(
        "\nHinweis: DRY_RUN/Testnet-Modus wird ausschliesslich ueber Umgebungsvariablen (.env) "
        "gesteuert, nicht durch dieses oder ein anderes Skript. Vor dem ersten echten Testnet-"
        "Order-Versuch: die obigen Werte gegen die Bybit/Binance-Dokumentation gegenpruefen."
    )


if __name__ == "__main__":
    main()
