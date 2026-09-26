import importlib
import json
import sys
from datetime import datetime, timedelta, timezone

import app
import pandas as pd
import pytest


def test_app_import_does_not_load_legacy_inference_path() -> None:
    sys.modules.pop("app", None)
    module = importlib.import_module("app")
    assert not hasattr(module, "load_real_analysis")
    assert not hasattr(module, "_strategy_setup_summary")
    assert not hasattr(module, "infer_model_comparison")
    assert not hasattr(module, "load_cached_ohlcv")
    assert hasattr(module, "_render_realtime_strategy")
    assert module._trade_direction_excursions("SHORT", 0.05, -0.03) == (0.03, -0.05)
    assert module._trade_direction_excursions("LONG", 0.05, -0.03) == (0.05, -0.03)
    assert module._trade_direction_excursions("SHORT", -0.02, -0.05) == (0.05, 0.0)
    assert module._trade_direction_excursions("LONG", 0.05, 0.02) == (0.05, 0.0)


def test_time_stop_is_twice_the_selected_horizon(monkeypatch) -> None:
    messages = []
    monkeypatch.setattr(app.st, "info", lambda message: messages.append(message))

    app._render_time_stop("4h")

    assert messages == [
        "Time-Stop: 8 Stunden (Trade zum Marktpreis schließen, falls TP1 nicht erreicht wurde)"
    ]


def test_fx_context_retains_rate_but_marks_stale_quote(monkeypatch) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    market = {
        "asset": app.FX_ASSET, "last_price": 1.08, "timestamp": timestamp,
        "freshness_ok": 1,
    }
    monkeypatch.setattr(app, "_market_snapshot", lambda asset: market if asset == app.FX_ASSET else None)

    fresh = app._usd_per_eur_context()
    assert fresh is not None and fresh["fresh"]
    assert fresh["usd_per_eur"] == pytest.approx(1.08)

    market["freshness_ok"] = 0
    stale = app._usd_per_eur_context()
    assert stale is not None and not stale["fresh"]
    assert stale["usd_per_eur"] == pytest.approx(1.08)


def test_trade_planner_leverage_only_changes_margin() -> None:
    import app

    low = app._build_planner_summary("BTC/USDT", 10_000.0, 0.01, "LONG", 100.0, 1.0, 2.0)
    high = app._build_planner_summary("BTC/USDT", 10_000.0, 0.01, "LONG", 100.0, 1.0, 4.0)

    assert not low["rejected"] and not high["rejected"]
    assert low["risk_amount_eur"] == high["risk_amount_eur"]
    assert low["position_size_eur"] == high["position_size_eur"]
    assert low["margin_eur"] == high["margin_eur"] * 2.0


def test_realtime_selector_auto_ranks_horizons_and_leaves_leverage_to_risk_engine(monkeypatch) -> None:
    calls = []

    def fake_setup(*, asset, horizon, profile, capital, risk_pct, selected_leverage, available_margin_eur):
        calls.append((horizon, selected_leverage, capital, risk_pct))
        votes = {
            "model_1": {"label": "TCN", "score": 70.0 if horizon == "4h" else 80.0},
            "model_2": {"label": "Chronos-2", "score": 70.0},
        }
        payload = {
            "votes": votes,
            "summary": {
                "rejected": False, "margin_exceeded": False, "risk_budget_shortfall": False,
                "stop_distance_pct": 0.01 if horizon == "4h" else 0.02,
            },
            "direction": "LONG" if horizon == "4h" else "SHORT",
            "market": {"htf_trend": "UP"},
            "forecast": {"expected_return": 0.03 if horizon == "4h" else -0.04},
        }
        return payload, ""

    monkeypatch.setattr(app, "_database_setup_summary", fake_setup)

    payload, error = app._select_realtime_setup("BTC/USDT", "recommended", 2_500.0)

    assert error == ""
    assert payload is not None
    assert payload["horizon"] == "4h"
    assert payload["strategy_label"] == "TCN + Chronos Consensus"
    assert payload["htf_trend"] == "UP"
    assert payload["expected_reward_r"] == pytest.approx(3.0)
    assert {horizon for horizon, _, _, _ in calls} == {"4h", "8h"}
    assert all(leverage is None for _, leverage, _, _ in calls)
    assert all(capital == 2_500.0 for _, _, capital, _ in calls)


def test_realtime_selector_blocks_candidates_below_minimum_expected_r(monkeypatch) -> None:
    def fake_setup(*, asset, horizon, profile, capital, risk_pct, selected_leverage, available_margin_eur):
        return {
            "votes": {
                "model_1": {"label": "TCN", "score": 80.0},
                "model_2": {"label": "Chronos-2", "score": 70.0},
            },
            "summary": {
                "rejected": False, "margin_exceeded": False, "risk_budget_shortfall": False,
                "stop_distance_pct": 0.01,
            },
            "direction": "LONG",
            "market": {},
            "forecast": {"expected_return": 0.005},
        }, ""

    monkeypatch.setattr(app, "_database_setup_summary", fake_setup)

    payload, error = app._select_realtime_setup("BTC/USDT", "recommended", 2_500.0)

    assert payload is None
    assert "0.50R" in error
    assert "1.0R" in error


def test_realtime_planner_calculates_leverage_automatically() -> None:
    summary = app._build_planner_summary(
        "BTC/USDT", 10_000.0, 0.018, "LONG", 100.0, 2.0, None,
        profile="recommended", available_margin_eur=10_000.0,
    )

    assert not summary["rejected"]
    assert 1.0 < summary["selected_leverage"] <= summary["max_allowed_leverage"]
    assert summary["risk_amount_eur"] == summary["requested_risk_eur"]


def test_manual_strategy_assistant_plans_existing_short_without_forecast(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    market = {
        "asset": "BTC/USDT", "last_price": 990.0, "atr": 5.0,
        "market_open": 1, "session_type": "24_7", "data_age_seconds": 0.0,
        "feed_healthy": 1, "freshness_ok": 1, "warmup_ready": 1,
        "updated_at": now.isoformat(),
    }
    monkeypatch.setattr(
        app, "load_table",
        lambda table: pd.DataFrame([{"status": "READY"}]) if table == "runtime_status" else pd.DataFrame(),
    )
    monkeypatch.setattr(app, "_market_snapshot", lambda _asset: market)
    monkeypatch.setattr(
        app,
        "_usd_per_eur_context",
        lambda: {"usd_per_eur": 1.08, "age_seconds": 0.0, "fresh": True, "timestamp": now.isoformat()},
    )
    monkeypatch.setattr(app, "_load_assistant_forecast_row", lambda _asset, _horizon: None)

    payload, error = app._build_position_management_summary(
        asset="BTC/USDT", capital=10_000.0,
        direction="SHORT", entry_price=25.0,
        allocated_margin_eur=2_500.0, selected_leverage=5.0,
    )

    assert error == ""
    assert payload is not None
    summary = payload["summary"]
    expected_stop_pct = (5.0 / 990.0) * app.POSITION_STOP_ATR_MULTIPLE
    assert payload["horizon"] == "1h"
    assert payload["product_stop_pct"] == pytest.approx(expected_stop_pct * 100.0)
    assert summary["stop_loss_price"] == pytest.approx(25.0 * (1.0 + expected_stop_pct))
    assert summary["take_profit_1_price"] == pytest.approx(25.0 * (1.0 - app.POSITION_TP1_R * expected_stop_pct))
    assert summary["take_profit_2_price"] == pytest.approx(25.0 * (1.0 - app.POSITION_TP2_R * expected_stop_pct))
    assert summary["trailing_stop_activation_price"] == summary["take_profit_1_price"]
    assert summary["trailing_stop_price"] == pytest.approx(25.0 * (1.0 - 0.5 * expected_stop_pct))
    assert payload["model_direction"] is None
    assert summary["tp1_close_pct"] == 50.0
    assert summary["tp2_close_pct"] == 25.0
    assert summary["trailing_close_pct"] == 25.0
    assert payload["actual_stop_risk_eur"] == pytest.approx(2_500.0 * 5.0 * expected_stop_pct)
    assert summary["quantity"] == pytest.approx(summary["position_size_eur"] * 1.08 / 25.0)
    assert payload["fx_context"]["usd_per_eur"] == pytest.approx(1.08)

    higher_leverage, higher_error = app._build_position_management_summary(
        asset="BTC/USDT", capital=10_000.0,
        direction="SHORT", entry_price=25.0,
        allocated_margin_eur=2_500.0, selected_leverage=10.0,
    )
    assert higher_error == ""
    assert higher_leverage is not None
    assert higher_leverage["actual_stop_risk_eur"] == pytest.approx(payload["actual_stop_risk_eur"] * 2.0)


def test_strategy_assistant_automatically_selects_best_fresh_horizon(monkeypatch) -> None:
    now = datetime.now(timezone.utc).isoformat()

    def make_forecast(horizon: str, tcn_score: float, chronos_score: float) -> dict:
        return {
            "asset": "BTC/USDT", "asset_class": "crypto", "horizon": horizon,
            "direction": "LONG", "forecast_status": "MODEL_AGREEMENT", "usable_for_decision": 1,
            "freshness": "FRESH", "forecast_timestamp": now,
            "individual_forecasts_json": json.dumps([
                {
                    "forecast_source": "DIRECT_MODEL", "model": "tcn", "direction": "LONG",
                    "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
                    "forecast_timestamp": now, "opportunity_score": tcn_score,
                },
                {
                    "forecast_source": "CHRONOS_2", "model": "chronos2", "direction": "LONG",
                    "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
                    "forecast_timestamp": now, "opportunity_score": chronos_score,
                    "p10_price": 99.0, "p50_price": 101.0, "p90_price": 103.0,
                },
            ]),
        }

    rows = {"4h": make_forecast("4h", 80.0, 60.0), "8h": make_forecast("8h", 72.0, 70.0)}
    monkeypatch.setattr(app, "_load_assistant_forecast_row", lambda _asset, horizon: rows.get(horizon))
    monkeypatch.setattr(app, "_market_snapshot", lambda _asset: pd.Series({"htf_trend": "UP"}))

    horizon, forecast, score, reason = app._select_position_model_context("BTC/USDT")

    assert horizon == "8h"
    assert forecast is rows["8h"]
    assert score == 70.0
    assert "Fresh model agreement" in reason


def test_global_capital_and_risk_profiles_scale_the_sizing_basis() -> None:
    assert app._risk_profile_risk_fraction("conservative") == 0.01
    assert app._risk_profile_risk_fraction("recommended") == 0.018
    assert app._risk_profile_risk_fraction("aggressive") > 0.20

    low_capital = app._build_planner_summary(
        "BTC/USDT", 10_000.0, 0.018, "LONG", 100.0, 2.0, 5.0,
        profile="recommended", available_margin_eur=10_000.0,
    )
    high_capital = app._build_planner_summary(
        "BTC/USDT", 20_000.0, 0.018, "LONG", 100.0, 2.0, 5.0,
        profile="recommended", available_margin_eur=20_000.0,
    )

    assert low_capital["risk_amount_eur"] == low_capital["requested_risk_eur"]
    assert high_capital["risk_amount_eur"] == high_capital["requested_risk_eur"]
    assert high_capital["risk_amount_eur"] == low_capital["risk_amount_eur"] * 2.0
    assert high_capital["position_size_eur"] == low_capital["position_size_eur"] * 2.0


def test_crypto_and_equity_leverage_caps_are_hard_limited() -> None:
    assert app._max_leverage_for_asset("BTC/USDT") == 10.0
    assert app._max_leverage_for_asset("QQQ") == 100.0

    crypto = app._build_planner_summary(
        "BTC/USDT", 10_000.0, 0.01, "LONG", 100.0, 2.0, 10.0,
        profile="recommended", available_margin_eur=10_000.0,
    )
    crypto_over_cap = app._build_planner_summary(
        "BTC/USDT", 10_000.0, 0.01, "LONG", 100.0, 2.0, 11.0,
        profile="recommended", available_margin_eur=10_000.0,
    )
    equity = app._build_planner_summary(
        "QQQ", 10_000.0, 0.018, "LONG", 100.0, 0.1, 100.0,
        profile="recommended", available_margin_eur=10_000.0,
    )

    assert not crypto["rejected"] and crypto["selected_leverage"] == 10.0
    assert crypto_over_cap["rejected"]
    assert not equity["rejected"] and equity["selected_leverage"] == 100.0


def test_radar_model_mapping_matches_crypto_and_equity_payloads() -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    crypto = pd.Series({
        "asset": "BTC/USDT",
        "asset_class": "crypto",
        "individual_forecasts_json": json.dumps([
            {
                "forecast_source": "DIRECT_MODEL", "model": "tcn", "direction": "LONG",
                "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
                "forecast_timestamp": timestamp, "opportunity_score": 80.0,
            },
            {
                "forecast_source": "CHRONOS_2", "model": "chronos2", "direction": "LONG",
                "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
                "forecast_timestamp": timestamp,
                "opportunity_score": 70.0,
                "p10_price": 99.0, "p50_price": 101.0, "p90_price": 103.0,
            },
        ]),
    })
    equity = pd.Series({
        "asset": "QQQ",
        "asset_class": "equity",
        "individual_forecasts_json": json.dumps([
            {"model": "RULE_STRATEGY", "direction": "SHORT", "score": 80.0},
            {
                "forecast_source": "CHRONOS_2", "model": "chronos2", "direction": "SHORT",
                "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
                "forecast_timestamp": timestamp,
                "opportunity_score": 70.0,
                "p10_price": 97.0, "p50_price": 99.0, "p90_price": 102.0,
            },
        ]),
    })

    crypto_votes = app._forecast_votes(crypto)
    equity_votes = app._forecast_votes(equity)

    assert crypto_votes["model_1"]["label"] == "TCN"
    assert crypto_votes["model_2"]["label"] == "Chronos-2"
    assert app._agreement_label(crypto_votes) == "Ja"
    assert equity_votes["model_1"]["label"] == "Chronos-2"
    assert equity_votes["model_2"]["label"] == "Rule-Strategy"
    assert app._agreement_label(equity_votes) == "Ja"


def test_radar_htf_mapping_shows_na_when_backend_has_no_htf_fields() -> None:
    market = pd.Series({"asset": "BTC/USDT", "atr": 2.0, "atr_pct": 0.02})

    assert app._htf_trend(market, None) == "N/A"
    assert app._htf_trend(pd.Series({"htf_trend": "UP"}), None) == "UP"


def test_chronos_neutral_score_cannot_form_radar_agreement() -> None:
    forecast = pd.Series({
        "asset": "BTC/USDT",
        "asset_class": "crypto",
        "individual_forecasts_json": json.dumps([
            {
                "forecast_source": "DIRECT_MODEL", "model": "tcn", "direction": "LONG",
                "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
                "forecast_timestamp": datetime.now(timezone.utc).isoformat(),
            },
            {
                "forecast_source": "CHRONOS_2", "model": "chronos2", "direction": "LONG",
                "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
                "forecast_timestamp": datetime.now(timezone.utc).isoformat(),
                "opportunity_score": 50.1,
                "p10_price": 99.0, "p50_price": 100.01, "p90_price": 101.0,
            },
        ]),
    })

    votes = app._forecast_votes(forecast)

    assert votes["model_1"]["direction"] == "LONG"
    assert votes["model_2"]["direction"] == "-"
    assert app._agreement_label(votes) == "Neutral"


def test_trade_planner_flags_margin_and_risk_budget_shortfall() -> None:
    summary = app._build_planner_summary(
        "BTC/USDT", 10_000.0, 0.05, "LONG", 100.0, 1.0, 9.0,
        profile="aggressive", available_margin_eur=1_000.0,
    )

    assert summary["rejected"]
    assert "available margin" in summary["reason"].lower()


def test_empty_or_partial_database_forecasts_fail_closed(monkeypatch) -> None:
    tables = {
        "runtime_status": pd.DataFrame([{"status": "READY"}]),
        "assistant_forecasts": pd.DataFrame(columns=["asset"]),
    }
    monkeypatch.setattr(app, "load_table", lambda table: tables.get(table, pd.DataFrame()))

    payload, error = app._database_setup_summary(
        "BTC/USDT", "4h", "conservative", 500.0, 0.01, 5.0, 500.0,
    )

    assert payload is None
    assert "warming up" in error.lower()


def test_strategy_setup_blocks_missing_chronos_quantiles_and_stale_forecasts(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    common = {
        "asset": "BTC/USDT",
        "asset_class": "crypto",
        "horizon": "4h",
        "direction": "LONG",
        "forecast_status": "MODEL_AGREEMENT",
        "usable_for_decision": 1,
        "freshness": "FRESH",
        "forecast_timestamp": now.isoformat(),
        "individual_forecasts_json": json.dumps([
            {
                "forecast_source": "DIRECT_MODEL", "model": "tcn", "direction": "LONG",
                "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
                "forecast_timestamp": now.isoformat(),
            },
            {
                "forecast_source": "CHRONOS_2", "model": "chronos2", "direction": "LONG",
                "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
                "forecast_timestamp": now.isoformat(),
            },
        ]),
    }
    monkeypatch.setattr(app, "_load_assistant_forecast_row", lambda _asset, _horizon: common)
    monkeypatch.setattr(
        app,
        "load_table",
        lambda table: pd.DataFrame([{"status": "READY"}]) if table == "runtime_status" else pd.DataFrame(),
    )
    market = {
        "asset": "BTC/USDT", "last_price": 100.0, "atr": 2.0,
        "market_open": 1, "session_type": "24_7", "data_age_seconds": 0.0,
        "feed_healthy": 1, "freshness_ok": 1, "warmup_ready": 1,
        "updated_at": now.isoformat(),
    }
    monkeypatch.setattr(app, "_market_snapshot", lambda _asset: market)
    monkeypatch.setattr(
        app,
        "_usd_per_eur_context",
        lambda: {"usd_per_eur": 1.08, "age_seconds": 0.0, "fresh": True, "timestamp": now.isoformat()},
    )
    missing_quantiles, error = app._database_setup_summary(
        "BTC/USDT", "4h", "conservative", 500.0, 0.01, 5.0, 500.0,
    )
    assert missing_quantiles is None
    assert "quantile" in error.lower()

    common["individual_forecasts_json"] = json.dumps([
        {
            "forecast_source": "DIRECT_MODEL", "model": "tcn", "direction": "LONG",
            "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
            "forecast_timestamp": now.isoformat(),
        },
        {
            "forecast_source": "CHRONOS_2", "model": "chronos2", "direction": "LONG",
            "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
            "forecast_timestamp": now.isoformat(),
                "opportunity_score": 70.0,
            "p10_price": 99.0, "p50_price": 101.0, "p90_price": 103.0,
        },
    ])
    common["forecast_timestamp"] = (now - timedelta(hours=2)).isoformat()
    stale, stale_error = app._database_setup_summary(
        "BTC/USDT", "4h", "conservative", 500.0, 0.01, 5.0, 500.0,
    )
    assert stale is None
    assert "stale" in stale_error.lower()

    common["forecast_timestamp"] = now.isoformat()
    payload, error = app._database_setup_summary(
        "BTC/USDT", "4h", "conservative", 500.0, 0.01, 5.0, 500.0,
    )
    assert payload is not None
    assert error == ""
    assert payload["summary"]["risk_amount_eur"] == payload["summary"]["requested_risk_eur"]
    assert payload["horizon_volatility_scale"] == pytest.approx(2.0)
    assert payload["sizing_atr"] == pytest.approx(4.0)
    assert payload["summary"]["stop_distance_pct"] == pytest.approx(0.09)
    assert payload["summary"]["quantity"] == pytest.approx(
        payload["summary"]["position_size_eur"] * 1.08 / payload["entry_price"]
    )

    market["updated_at"] = (now - timedelta(minutes=5)).isoformat()
    stale_market, market_error = app._database_setup_summary(
        "BTC/USDT", "4h", "conservative", 500.0, 0.01, 5.0, 500.0,
    )
    assert stale_market is None
    assert "market snapshot is stale" in market_error.lower()

    market["updated_at"] = now.isoformat()
    common["individual_forecasts_json"] = json.dumps([
        {
            "forecast_source": "DIRECT_MODEL", "model": "tcn", "direction": "LONG",
            "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
            "forecast_timestamp": now.isoformat(),
        },
        {
            "forecast_source": "CHRONOS_2", "model": "chronos2", "direction": "LONG",
            "forecast_status": "MODEL_UNAVAILABLE", "freshness": "UNKNOWN",
            "forecast_timestamp": None, "p10_price": 99.0, "p50_price": 101.0, "p90_price": 103.0,
        },
    ])
    invalid_component, component_error = app._database_setup_summary(
        "BTC/USDT", "4h", "conservative", 500.0, 0.01, 5.0, 500.0,
    )
    assert invalid_component is None
    assert "do not agree" in component_error.lower()

    market["updated_at"] = now.isoformat()
    monkeypatch.setattr(
        app,
        "load_table",
        lambda table: pd.DataFrame([{"status": "DEGRADED"}]) if table == "runtime_status" else pd.DataFrame(),
    )
    degraded, runtime_error = app._database_setup_summary(
        "BTC/USDT", "4h", "conservative", 500.0, 0.01, 5.0, 500.0,
    )
    assert degraded is None
    assert "runtime status is degraded" in runtime_error.lower()


def test_closed_or_stale_equity_market_never_builds_a_setup(monkeypatch) -> None:
    now = datetime.now(timezone.utc)
    forecast = {
        "asset": "SPY", "asset_class": "equity", "horizon": "4h",
        "direction": "LONG", "forecast_status": "MODEL_AGREEMENT",
        "usable_for_decision": 1, "freshness": "FRESH",
        "forecast_timestamp": now.isoformat(),
        "individual_forecasts_json": json.dumps([
            {"model": "RULE_STRATEGY", "direction": "LONG", "score": 80.0},
            {
                "forecast_source": "CHRONOS_2", "model": "chronos2", "direction": "LONG",
                "forecast_status": "PARTIALLY_VALIDATED", "freshness": "FRESH",
                "forecast_timestamp": now.isoformat(), "opportunity_score": 70.0,
                "p10_price": 99.0, "p50_price": 101.0, "p90_price": 103.0,
            },
        ]),
    }
    market = {
        "asset": "SPY", "last_price": 100.0, "atr": 2.0,
        "market_open": 0, "session_type": "CLOSED", "data_age_seconds": 3600.0,
        "feed_healthy": 0, "freshness_ok": 0, "warmup_ready": 1,
        "updated_at": now.isoformat(),
    }
    monkeypatch.setattr(app, "_load_assistant_forecast_row", lambda _asset, _horizon: forecast)
    monkeypatch.setattr(app, "_market_snapshot", lambda _asset: market)
    monkeypatch.setattr(
        app, "load_table",
        lambda table: pd.DataFrame([{"status": "READY"}]) if table == "runtime_status" else pd.DataFrame(),
    )

    payload, closed_error = app._database_setup_summary("SPY", "4h", "recommended", 500.0, 0.018, 2.0, 500.0)
    assert payload is None
    assert "MARKET_CLOSED" in closed_error

    market.update(market_open=1, session_type="REGULAR", feed_healthy=0, freshness_ok=0)
    payload, stale_error = app._database_setup_summary("SPY", "4h", "recommended", 500.0, 0.018, 2.0, 500.0)
    assert payload is None
    assert "DATA_STALE" in stale_error


