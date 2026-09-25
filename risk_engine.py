"""Risk-Engine: harter, deterministischer Algorithmus (kein ML).

Nimmt die rohen ML-Outputs (Score, erwartete MAE/MFE, ATR, Kapital) entgegen
und berechnet daraus konkrete, ausführbare Trade-Parameter: Hebel (gecappt),
Stop-Loss, Take-Profit, Knock-Out-Barriere und Positionsgröße.

Bewusst zustandslos/reine Funktionen -- kein DB-/Netzwerkzugriff hier, damit
dieses Modul unabhängig unit-testbar bleibt und von ``paper_broker.py`` (Paper
Trading) sowie später einer echten Order-Ausführung identisch wiederverwendet
werden kann.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

# --- Instrument-specific leverage ceilings -----------------------------------
# These are technical ceilings, never a target leverage.  A plain ETF price
# series is an underlying, not a CFD, future, certificate, or margin product.
# It therefore has no synthetic liquidation/KO level and is capped at 1x.
MAX_LEVERAGE_CRYPTO = 10.0
MAX_LEVERAGE_EQUITY_UNDERLYING = 1.0

# Fixed-Fractional-Risk-Sizing: wie viel Prozent des GESAMTEN Paper-Kapitals
# darf ein einzelner Trade verlieren, wenn der Stop exakt getroffen wird.
DEFAULT_RISK_PER_TRADE_PCT = 0.02
# Harte Obergrenze, wie viel Prozent des Kapitals ÜBERHAUPT in einem einzelnen
# Trade gebunden sein darf (Margin), unabhängig vom Risiko-Betrag -- verhindert,
# dass ein sehr enger Stop (kleines stop_distance_pct) rechnerisch einen
# absurd hohen Hebel/eine absurd hohe Positionsgröße erzeugt.
DEFAULT_MAX_POSITION_FRACTION = 0.35
# ATR-Puffer: der Stop darf nie enger sein als ATR * dieser Faktor, selbst wenn
# das Modell eine sehr kleine erwartete MAE vorhersagt (Modell-Fehleinschätzung
# nicht 1:1 vertrauen, ATR ist die "brutale" Marktrealität).
DEFAULT_ATR_BUFFER_MULTIPLE = 1.5
# Sicherheitsabstand zwischen Stop-Loss und Knock-Out-Barriere: die KO-Schwelle
# liegt IMMER etwas jenseits des Stops, damit der Stop zuerst greift (ein KO
# ist ein Total-/Fastverlust der Position, der Stop soll das verhindern).
DEFAULT_KO_SAFETY_BUFFER_PCT = 0.15
# Minimales Chance/Risiko-Verhältnis (Take-Profit-Distanz / Stop-Distanz) --
# wenn die vom Modell erwartete MFE kleiner ist als das, wird der Trade nicht
# aufgewertet (kein künstliches Aufblasen des TP), aber die Kennzahl wird
# transparent < 1.0 ausgewiesen (Caller kann dann z.B. das Sizing skalieren).


@dataclass(frozen=True)
class StrategyProfile:
    """Deterministic risk-policy adjustments for an always-visible what-if quote."""

    name: str
    atr_buffer_multiple: float
    risk_per_trade_pct: float
    max_position_fraction: float


AGGRESSIVE_PROFILE = StrategyProfile(
    name="aggressive",
    atr_buffer_multiple=1.0,
    risk_per_trade_pct=0.025,
    max_position_fraction=0.45,
)
CONSERVATIVE_PROFILE = StrategyProfile(
    name="conservative",
    atr_buffer_multiple=2.25,
    risk_per_trade_pct=0.01,
    max_position_fraction=0.25,
)


@dataclass(frozen=True)
class TradeSetup:
    """Rohe, bereits vom ML gelieferte Inputs für einen möglichen Trade."""

    asset: str
    asset_class: str            # "crypto" | "equity"
    direction: str               # "LONG" | "SHORT"
    entry_price: float
    atr: float                   # absolute ATR in Preis-Einheiten
    expected_mae: float          # ML: erwartete negative Exkursion, als Bruchteil (z.B. -0.02)
    expected_mfe: float          # ML: erwartete positive Exkursion, als Bruchteil (z.B. 0.05)
    score: float                 # 0-100 Setup-Score
    capital_eur: float           # aktuelles GESAMTES Paper-Wallet-Kapital
    open_risk_eur: float = 0.0   # bereits durch andere offene Trades gebundenes Risiko-Budget
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT
    max_position_fraction: float = DEFAULT_MAX_POSITION_FRACTION
    atr_buffer_multiple: float = DEFAULT_ATR_BUFFER_MULTIPLE
    ko_safety_buffer_pct: float = DEFAULT_KO_SAFETY_BUFFER_PCT
    max_portfolio_risk_pct: float = 0.06   # Summe aller offenen Risiko-Budgets darf das nie überschreiten
    profile: str = "base"
    # ``equity_underlying`` is the safe default for QQQ/SPY price data.  A
    # leveraged equity product must state its instrument and exchange ceiling
    # explicitly; asset class alone is not enough to infer leverage.
    instrument_type: str | None = None
    exchange_max_leverage: float | None = None
    available_margin_eur: float | None = None
    margin_in_use_eur: float = 0.0
    confidence: float = 1.0
    volatility_multiplier: float = 1.0
    correlation_multiplier: float = 1.0
    drawdown_pct: float = 0.0
    max_drawdown_pct: float = 0.10
    max_notional_eur: float | None = None
    asset_risk_multiplier: float = 1.0


@dataclass(frozen=True)
class RiskParameters:
    asset: str
    direction: str
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    knockout_barrier_price: float
    leverage: float
    position_size_eur: float     # Notional
    margin_eur: float            # tatsächlich gebundenes Kapital (Notional / Leverage)
    quantity: float               # Stück/Coins
    risk_amount_eur: float       # max. Verlust in EUR, wenn Stop exakt trifft
    risk_reward_ratio: float
    stop_distance_pct: float
    take_profit_distance_pct: float
    rejected: bool = False
    rejection_reason: str = ""
    profile: str = "base"
    trade_quality: float = 0.0
    allowed_risk_fraction: float = 0.0
    risk_based_notional_eur: float = 0.0
    max_allowed_leverage: float = 0.0
    instrument_type: str = ""
    protection_model: str = ""
    liquidation_price: float | None = None
    safety_barrier_price: float | None = None


def instrument_type_for(asset_class: str, instrument_type: str | None = None) -> str:
    """Resolve the product being modeled without guessing equity leverage."""
    if instrument_type:
        return instrument_type
    if asset_class == "crypto":
        return "crypto_perpetual"
    if asset_class == "equity":
        return "equity_underlying"
    raise ValueError(f"Unknown asset_class {asset_class!r}, expected 'crypto' or 'equity'")


def max_leverage_for(
    asset_class: str,
    instrument_type: str | None = None,
    exchange_max_leverage: float | None = None,
) -> float:
    """Return a hard product/exchange ceiling, never a sizing target.

    For CFD/future/margin products the caller must provide the actual exchange
    or broker limit.  Normal ETF data is deliberately treated as 1x cash
    exposure until such a product is explicitly configured.
    """
    resolved = instrument_type_for(asset_class, instrument_type)
    if resolved == "crypto_perpetual":
        ceiling = MAX_LEVERAGE_CRYPTO
    elif resolved == "equity_underlying":
        ceiling = MAX_LEVERAGE_EQUITY_UNDERLYING
    elif resolved in {"equity_cfd", "equity_future", "equity_margin", "crypto_future"}:
        if exchange_max_leverage is None or not np.isfinite(exchange_max_leverage):
            raise ValueError(f"actual exchange_max_leverage required for {resolved}")
        ceiling = float(exchange_max_leverage)
    else:
        raise ValueError(f"unsupported instrument_type {resolved!r}")
    if exchange_max_leverage is not None:
        ceiling = min(ceiling, float(exchange_max_leverage))
    if not np.isfinite(ceiling) or ceiling <= 0:
        raise ValueError("instrument leverage ceiling must be positive and finite")
    return float(ceiling)


def allowed_risk_fraction(
    *,
    base_risk_fraction: float,
    score: float,
    confidence: float = 1.0,
    drawdown_pct: float = 0.0,
    max_drawdown_pct: float = 0.10,
    volatility_multiplier: float = 1.0,
    correlation_multiplier: float = 1.0,
) -> float:
    """Calculate risk budget from quality and current portfolio conditions.

    Drawdown scaling is deliberately continuous and parameter-light: risk is
    unchanged at zero drawdown and reaches zero at the configured account
    drawdown stop.  Volatility/correlation inputs are multipliers produced by
    the portfolio layer, not hidden leverage rules.
    """
    values = (base_risk_fraction, score, confidence, drawdown_pct, max_drawdown_pct, volatility_multiplier, correlation_multiplier)
    if not all(np.isfinite(value) for value in values) or base_risk_fraction <= 0 or max_drawdown_pct <= 0:
        return 0.0
    quality = float(np.clip(score / 100.0, 0.0, 1.0) * np.clip(confidence, 0.0, 1.0))
    drawdown_scale = float(np.clip(1.0 - max(drawdown_pct, 0.0) / max_drawdown_pct, 0.0, 1.0))
    return float(max(0.0, base_risk_fraction * quality * drawdown_scale * np.clip(volatility_multiplier, 0.0, 1.0) * np.clip(correlation_multiplier, 0.0, 1.0)))


def select_effective_leverage(
    risk_based_notional: float,
    max_margin: float,
    max_allowed_leverage: float,
) -> float:
    """Select the smallest leverage needed to fit notional in margin.

    The risk engine chooses notional first.  Leverage only determines how much
    margin is needed for that already-capped exposure, and can never exceed the
    instrument/exchange ceiling.
    """
    if not all(np.isfinite(value) for value in (risk_based_notional, max_margin, max_allowed_leverage)):
        return 0.0
    if risk_based_notional <= 0 or max_margin <= 0 or max_allowed_leverage <= 0:
        return 0.0
    required = risk_based_notional / max_margin
    return float(np.clip(max(1.0, required), 1.0, max_allowed_leverage))


def compute_stop_distance_pct(entry_price: float, atr: float, expected_mae: float, atr_buffer_multiple: float) -> float:
    """Stop-Distanz = max(|ML-MAE|, ATR-Puffer), niemals enger als beides."""
    if entry_price <= 0:
        raise ValueError("entry_price must be positive")
    atr_pct = abs(atr) / entry_price * atr_buffer_multiple if np.isfinite(atr) else 0.0
    mae_pct = abs(expected_mae) if np.isfinite(expected_mae) else 0.0
    return float(max(atr_pct, mae_pct, 1e-6))


def compute_take_profit_distance_pct(expected_mfe: float, stop_distance_pct: float, min_reward_risk: float = 0.5) -> float:
    """Take-Profit-Distanz direkt aus der ML-MFE-Schätzung, mit einer weichen
    Untergrenze (min_reward_risk * stop_distance_pct) gegen ein wirtschaftlich
    sinnlos enges Ziel -- die MFE-Schätzung selbst wird nie künstlich
    aufgebläht, nur nach unten sauber begrenzt."""
    mfe_pct = abs(expected_mfe) if np.isfinite(expected_mfe) else 0.0
    return float(max(mfe_pct, min_reward_risk * stop_distance_pct))


def fixed_fractional_position_fraction(capital_eur: float, open_risk_eur: float, risk_amount_eur: float, max_portfolio_risk_pct: float) -> float:
    """Skaliert das Risiko-Budget herunter, falls bereits offene Trades einen
    Teil des Portfolio-weiten Risiko-Budgets belegen (geteiltes 500-EUR-Wallet,
    nicht mehrere unabhängige Konten)."""
    if capital_eur <= 0:
        return 0.0
    budget_eur = max(capital_eur * max_portfolio_risk_pct - open_risk_eur, 0.0)
    return float(min(risk_amount_eur, budget_eur))


def kelly_fraction(win_rate: float, reward_risk_ratio: float, cap: float = 0.5) -> float:
    """Half-Kelly (konservativ) als OPTIONALE Sizing-Alternative, nur nutzbar
    wenn belastbare historische win_rate/reward_risk-Statistiken vorliegen
    (z.B. aus dem OOS-Report). Liefert 0.0 bei nicht-edge-positiven Eingaben,
    niemals negativ. Nicht der Standardpfad (siehe ``compute_risk_parameters``,
    default bleibt Fixed-Fractional) -- Kelly reagiert sehr empfindlich auf
    Schätzfehler in win_rate/reward_risk, gerade bei wenig Live-Historie."""
    if not (0.0 < win_rate < 1.0) or reward_risk_ratio <= 0:
        return 0.0
    full_kelly = win_rate - (1.0 - win_rate) / reward_risk_ratio
    return float(np.clip(full_kelly * 0.5, 0.0, cap))


def compute_knockout_barrier(entry_price: float, direction: str, leverage: float, stop_distance_pct: float, ko_safety_buffer_pct: float) -> float:
    """KO-Barriere liegt hinter dem Stop-Loss (Sicherheitsabstand), aber
    innerhalb dessen, was der Hebel selbst als Liquidations-/KO-Schwelle
    implizieren würde -- der schärfere (näher am Einstieg liegende) der beiden
    Werte gewinnt, damit die Barriere nie "hinter" dem theoretischen
    Totalverlust-Punkt des Produkts liegt."""
    if leverage <= 0:
        raise ValueError("leverage must be positive")
    leverage_implied_pct = 1.0 / leverage
    ko_distance_pct = stop_distance_pct * (1.0 + ko_safety_buffer_pct)
    # Never place the KO barrier beyond the product's own leverage-implied
    # knock-out distance (that would be a fantasy price the issuer would
    # never actually honor).
    effective_pct = min(ko_distance_pct, leverage_implied_pct) if leverage_implied_pct > 0 else ko_distance_pct
    if direction == "LONG":
        return float(entry_price * (1.0 - effective_pct))
    if direction == "SHORT":
        return float(entry_price * (1.0 + effective_pct))
    raise ValueError(f"Unknown direction {direction!r}")


def compute_risk_parameters(setup: TradeSetup) -> RiskParameters:
    """Zentrale Risk-Engine-Funktion: aus einem ``TradeSetup`` werden
    Hebel/Stop/Take-Profit/KO/Positionsgröße berechnet. Fail-closed: bei
    ungültigen Eingaben (nicht-finite Werte, Kapital <= 0) wird ein
    ``rejected=True``-Ergebnis mit Größe 0 zurückgegeben statt eine
    kaputte Order zu produzieren."""
    if setup.direction not in {"LONG", "SHORT"}:
        return _rejected(setup, f"invalid direction {setup.direction!r}")
    if setup.capital_eur <= 0 or setup.entry_price <= 0:
        return _rejected(setup, "non-positive capital or entry price")
    if not all(np.isfinite(v) for v in (setup.entry_price, setup.atr, setup.expected_mae, setup.expected_mfe, setup.score, setup.capital_eur)):
        return _rejected(setup, "non-finite input")

    try:
        instrument_type = instrument_type_for(setup.asset_class, setup.instrument_type)
        max_leverage = max_leverage_for(setup.asset_class, instrument_type, setup.exchange_max_leverage)
    except ValueError as exc:
        return _rejected(setup, str(exc))

    stop_distance_pct = compute_stop_distance_pct(setup.entry_price, setup.atr, setup.expected_mae, setup.atr_buffer_multiple)
    take_profit_distance_pct = compute_take_profit_distance_pct(setup.expected_mfe, stop_distance_pct)
    risk_reward_ratio = take_profit_distance_pct / stop_distance_pct if stop_distance_pct > 0 else 0.0

    allowed_fraction = allowed_risk_fraction(
        base_risk_fraction=setup.risk_per_trade_pct * setup.asset_risk_multiplier,
        score=setup.score,
        confidence=setup.confidence,
        drawdown_pct=setup.drawdown_pct,
        max_drawdown_pct=setup.max_drawdown_pct,
        volatility_multiplier=setup.volatility_multiplier,
        correlation_multiplier=setup.correlation_multiplier,
    )
    raw_risk_amount_eur = setup.capital_eur * allowed_fraction
    risk_amount_eur = fixed_fractional_position_fraction(setup.capital_eur, setup.open_risk_eur, raw_risk_amount_eur, setup.max_portfolio_risk_pct)
    if risk_amount_eur <= 0:
        return _rejected(setup, "portfolio risk budget exhausted by already-open trades")

    risk_based_notional = risk_amount_eur / stop_distance_pct
    available_margin_eur = setup.available_margin_eur if hasattr(setup, "available_margin_eur") else None
    if available_margin_eur is None:
        available_margin_eur = setup.capital_eur - max(setup.margin_in_use_eur, 0.0)
    max_margin_eur = max(min(float(available_margin_eur), setup.capital_eur) * setup.max_position_fraction, 0.0)
    max_notional_eur = max_margin_eur * max_leverage
    if setup.max_notional_eur is not None:
        if not np.isfinite(setup.max_notional_eur) or setup.max_notional_eur <= 0:
            return _rejected(setup, "invalid max_notional_eur")
        max_notional_eur = min(max_notional_eur, setup.max_notional_eur)
    position_size_eur = float(min(risk_based_notional, max_notional_eur))
    leverage = select_effective_leverage(position_size_eur, max_margin_eur, max_leverage)
    if leverage <= 0:
        return _rejected(setup, "no available margin for risk-based notional")
    # Required margin is derived only after the final notional and leverage.
    margin_eur = max(position_size_eur / leverage, 1e-9)
    risk_amount_eur = position_size_eur * stop_distance_pct

    quantity = position_size_eur / setup.entry_price
    stop_loss_price = setup.entry_price * (1.0 - stop_distance_pct) if setup.direction == "LONG" else setup.entry_price * (1.0 + stop_distance_pct)
    take_profit_price = setup.entry_price * (1.0 + take_profit_distance_pct) if setup.direction == "LONG" else setup.entry_price * (1.0 - take_profit_distance_pct)
    has_exchange_liquidation = instrument_type in {"crypto_perpetual", "crypto_future", "equity_cfd", "equity_future", "equity_margin"}
    knockout_barrier_price = compute_knockout_barrier(setup.entry_price, setup.direction, leverage, stop_distance_pct, setup.ko_safety_buffer_pct) if has_exchange_liquidation else 0.0
    liquidation_distance_pct = 1.0 / leverage if has_exchange_liquidation else None
    liquidation_price = None
    if liquidation_distance_pct is not None:
        liquidation_price = float(
            setup.entry_price * (1.0 - liquidation_distance_pct)
            if setup.direction == "LONG"
            else setup.entry_price * (1.0 + liquidation_distance_pct)
        )

    return RiskParameters(
        asset=setup.asset,
        direction=setup.direction,
        entry_price=setup.entry_price,
        stop_loss_price=stop_loss_price,
        take_profit_price=take_profit_price,
        knockout_barrier_price=knockout_barrier_price,
        leverage=leverage,
        position_size_eur=position_size_eur,
        margin_eur=margin_eur,
        quantity=quantity,
        risk_amount_eur=risk_amount_eur,
        risk_reward_ratio=risk_reward_ratio,
        stop_distance_pct=stop_distance_pct,
        take_profit_distance_pct=take_profit_distance_pct,
        profile=getattr(setup, "profile", "base"),
        trade_quality=float(np.clip(setup.score / 100.0, 0.0, 1.0) * np.clip(setup.confidence, 0.0, 1.0)),
        allowed_risk_fraction=allowed_fraction,
        risk_based_notional_eur=risk_based_notional,
        max_allowed_leverage=max_leverage,
        instrument_type=instrument_type,
        protection_model="exchange_liquidation_estimate" if has_exchange_liquidation else "underlying_no_liquidation",
        liquidation_price=liquidation_price,
        safety_barrier_price=knockout_barrier_price if has_exchange_liquidation else None,
    )


def _rejected(setup: TradeSetup, reason: str) -> RiskParameters:
    return RiskParameters(
        asset=setup.asset, direction=setup.direction, entry_price=setup.entry_price,
        stop_loss_price=0.0, take_profit_price=0.0, knockout_barrier_price=0.0,
        leverage=0.0, position_size_eur=0.0, margin_eur=0.0, quantity=0.0,
        risk_amount_eur=0.0, risk_reward_ratio=0.0, stop_distance_pct=0.0,
        take_profit_distance_pct=0.0, rejected=True, rejection_reason=reason,
        profile=getattr(setup, "profile", "base"),
        instrument_type=instrument_type_for(setup.asset_class, setup.instrument_type) if setup.asset_class in {"crypto", "equity"} else "",
    )


def compute_profile_parameters(setup: TradeSetup, profile: StrategyProfile) -> RiskParameters:
    """Compute an independent, actionable what-if setup for ``profile``.

    This function deliberately does not inspect the ML score to decide whether
    to run. A low score still produces a transparent hypothetical setup; the
    caller decides whether it may be traded. Profile leverage is bounded by
        the asset-class hard ceiling and never increases with holding duration.
        The profile changes risk budget and stop policy only; effective leverage is still derived downstream.
    """
    if profile.atr_buffer_multiple <= 0 or profile.risk_per_trade_pct < 0:
        return _rejected(replace(setup, profile=profile.name), "invalid strategy profile")
    adjusted = replace(
        setup,
        profile=profile.name,
        atr_buffer_multiple=profile.atr_buffer_multiple,
        risk_per_trade_pct=profile.risk_per_trade_pct,
        max_position_fraction=profile.max_position_fraction,
    )
    return compute_risk_parameters(adjusted)


def compute_strategy_profiles(setup: TradeSetup) -> dict[str, RiskParameters]:
    """Return both profiles on every cycle, regardless of signal quality."""
    return {
        AGGRESSIVE_PROFILE.name: compute_profile_parameters(setup, AGGRESSIVE_PROFILE),
        CONSERVATIVE_PROFILE.name: compute_profile_parameters(setup, CONSERVATIVE_PROFILE),
    }


def ratchet_trailing_stop(direction: str, current_stop: float, current_price: float, stop_distance_pct: float) -> float:
    """ATR-Trailing-Stop-Ratchet, identisch im Prinzip zu ``core.risk.
    apply_dynamic_stop`` (Trend-Strategie), hier aber pro offenem Paper-Trade
    inkrementell pro Poll aufgerufen statt vektorisiert über eine ganze Serie:
    der Stop bewegt sich nur in Richtung des Trades, nie zurück."""
    candidate = current_price * (1.0 - stop_distance_pct) if direction == "LONG" else current_price * (1.0 + stop_distance_pct)
    if direction == "LONG":
        return float(max(current_stop, candidate))
    return float(min(current_stop, candidate))
