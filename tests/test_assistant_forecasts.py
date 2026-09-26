import json
from datetime import datetime, timezone

from core.ml.assistant_forecasts import AssistantForecast, PUBLIC_HORIZONS, apply_oos_quality_gate, combine_model_forecasts, combine_rule_chronos_forecast, direct_swing_forecasts, direct_tcn_forecasts
import state_db


def _tcn_output(direction="LONG"):
    return {
        "probabilities": (0.2, 0.1, 0.7), "expected_return": 0.02,
        "expected_mfe": 0.03, "expected_mae": -0.01, "expected_duration": 4.0,
        "opportunity_score": 71.0, "direction": direction,
    }


def test_crypto_registry_exposes_exact_heads_and_never_silently_falls_back() -> None:
    forecasts = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1", timestamp="2026-01-01T00:00:00+00:00",
        outputs={1: _tcn_output(), 4: _tcn_output(), 24: _tcn_output()},
    )
    by_horizon = {item.horizon: item for item in forecasts}
    assert tuple(by_horizon) == PUBLIC_HORIZONS
    assert by_horizon["1h"].forecast_source == "DIRECT_MODEL"
    assert by_horizon["1d"].forecast_source == "DIRECT_MODEL"
    assert by_horizon["8h"].forecast_status == "UNAVAILABLE"
    assert by_horizon["20d"].forecast_status == "UNAVAILABLE"


def test_malformed_tcn_head_is_unavailable_instead_of_raising() -> None:
    forecasts = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(),
        outputs={4: {"probabilities": (1.1, -0.1), "expected_return": "bad"}},
    )

    assert forecasts[1].forecast_status == "UNAVAILABLE"


def test_equity_swing_registry_uses_only_exact_trading_day_heads() -> None:
    outputs = {day: {"expected_return": 0.01 * day, "expected_mfe": 0.04, "expected_mae": -0.02, "expected_duration": float(day), "opportunity_score": 70.0} for day in (1, 3, 5, 10, 20)}
    forecasts = direct_swing_forecasts(asset="SPY", model="swing", model_version="v1", timestamp="2026-01-01T00:00:00+00:00", outputs=outputs)
    by_horizon = {item.horizon: item for item in forecasts}
    assert by_horizon["5d"].forecast_source == "DIRECT_MODEL"
    assert by_horizon["4h"].forecast_status == "UNAVAILABLE"
    assert by_horizon["1d"].expected_duration == 1.0


def _chronos_equity_forecast(direction: str = "LONG", freshness: str = "FRESH") -> AssistantForecast:
    return AssistantForecast(
        asset="QQQ", asset_class="equity", timeframe="1h", horizon="4h", forecast_source="CHRONOS_2",
        model="chronos2", model_version="v2", direction=direction, probability_short=None,
        probability_neutral=None, probability_long=None, expected_return=0.01, expected_mfe=None,
        expected_mae=None, expected_duration=None, opportunity_score=70.0, confidence="MEDIUM",
        forecast_timestamp=datetime.now(timezone.utc).isoformat(), target_timestamp=None,
        data_quality="AVAILABLE", forecast_status="PARTIALLY_VALIDATED", freshness=freshness,
        forecast_age_minutes=0.0, p10_price=99.0, p50_price=101.0, p90_price=103.0,
    )


def _chronos_crypto_forecast(direction: str = "LONG") -> AssistantForecast:
    return AssistantForecast(
        asset="BTC/USDT", asset_class="crypto", timeframe="1h", horizon="4h", forecast_source="CHRONOS_2",
        model="chronos2", model_version="v2", direction=direction, probability_short=None,
        probability_neutral=None, probability_long=None, expected_return=0.01 if direction == "LONG" else -0.01,
        expected_mfe=None, expected_mae=None, expected_duration=None, opportunity_score=70.0, confidence="MEDIUM",
        forecast_timestamp=datetime.now(timezone.utc).isoformat(), target_timestamp=None,
        data_quality="AVAILABLE", forecast_status="PARTIALLY_VALIDATED", freshness="FRESH",
        forecast_age_minutes=0.0, p10_price=99.0, p50_price=101.0, p90_price=103.0,
    )


def test_chronos_end_point_quantiles_are_not_public_mfe_or_mae() -> None:
    class Output:
        status = "AVAILABLE"
        expected_return = 0.01
        expected_favorable_move = 0.08
        expected_adverse_move = -0.05
        input_timeframe = "1h"
        model_id = "chronos2"
        model_version = "v2"
        direction = "LONG"
        model_score = 70.0
        confidence = "MEDIUM"
        data_timestamp = "2026-01-01T00:00:00+00:00"
        data_quality = "AVAILABLE"
        reason = "test"
        forecast_low = 99.0
        forecast_median = 101.0
        forecast_high = 103.0

    from core.ml.assistant_forecasts import chronos2_forecast

    forecast = chronos2_forecast(asset="QQQ", asset_class="equity", horizon="4h", output=Output())

    assert forecast.expected_return == 0.01
    assert forecast.expected_mfe is None
    assert forecast.expected_mae is None
    assert (forecast.p10_price, forecast.p50_price, forecast.p90_price) == (99.0, 101.0, 103.0)


def test_chronos_without_quantiles_is_unavailable() -> None:
    class Output:
        status = "AVAILABLE"
        expected_return = 0.01
        input_timeframe = "1h"
        model_id = "chronos2"
        model_version = "v2"
        direction = "LONG"
        data_timestamp = datetime.now(timezone.utc).isoformat()

    from core.ml.assistant_forecasts import chronos2_forecast

    forecast = chronos2_forecast(asset="BTC/USDT", asset_class="crypto", horizon="4h", output=Output())

    assert forecast.forecast_status == "UNAVAILABLE"
    assert "quantile" in forecast.reason.lower()


def test_rule_chronos_gate_requires_fresh_matching_directions() -> None:
    matched = combine_rule_chronos_forecast(
        asset="QQQ", horizon="4h", rule_direction="LONG", rule_score=80.0,
        chronos=_chronos_equity_forecast(),
    )
    conflict = combine_rule_chronos_forecast(
        asset="QQQ", horizon="4h", rule_direction="SHORT", rule_score=80.0,
        chronos=_chronos_equity_forecast(),
    )
    stale = combine_rule_chronos_forecast(
        asset="QQQ", horizon="4h", rule_direction="LONG", rule_score=80.0,
        chronos=_chronos_equity_forecast(freshness="STALE"),
    )

    assert matched.forecast_status == "MODEL_AGREEMENT" and matched.usable_for_decision
    assert matched.expected_mfe is None and matched.expected_mae is None
    assert conflict.forecast_status == "MODEL_DISAGREEMENT" and not conflict.usable_for_decision
    assert stale.forecast_status == "UNAVAILABLE" and not stale.usable_for_decision


def test_equity_intraday_tcn_route_is_disabled() -> None:
    forecasts = direct_tcn_forecasts(
        asset="QQQ",
        asset_class="equity",
        model="equity_tcn",
        model_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(),
        outputs={1: _tcn_output(), 4: _tcn_output(), 8: _tcn_output(), 12: _tcn_output()},
    )

    by_horizon = {item.horizon: item for item in forecasts}
    assert all(by_horizon[horizon].forecast_status == "UNAVAILABLE" for horizon in ("1h", "4h", "8h", "12h"))
    assert all("TCN is disabled" in by_horizon[horizon].reason for horizon in ("1h", "4h", "8h", "12h"))


def test_forecast_matrix_replaces_the_complete_asset_snapshot(tmp_path) -> None:
    database = tmp_path / "state.db"
    state_db.init_db(database)
    first = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1", timestamp="2026-01-01T00:00:00+00:00",
        outputs={1: _tcn_output()},
    )
    second = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v2", timestamp="2026-01-01T01:00:00+00:00",
        outputs={4: _tcn_output("SHORT")},
    )
    with state_db.connect(database) as connection:
        state_db.replace_assistant_forecasts(connection, "BTC/USDT", [item.as_dict() for item in first])
    with state_db.connect(database) as connection:
        state_db.replace_assistant_forecasts(connection, "BTC/USDT", [item.as_dict() for item in second])
        rows = connection.execute("SELECT horizon, forecast_status, model_version FROM assistant_forecasts WHERE asset = ?", ("BTC/USDT",)).fetchall()

    assert len(rows) == len(PUBLIC_HORIZONS)
    assert {row["horizon"] for row in rows} == set(PUBLIC_HORIZONS)
    assert next(row for row in rows if row["horizon"] == "1h")["forecast_status"] == "UNAVAILABLE"
    assert next(row for row in rows if row["horizon"] == "4h")["model_version"] == "v2"


def test_combination_uses_median_only_for_directional_agreement() -> None:
    first = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(), outputs={4: _tcn_output()},
    )[1]
    second = _chronos_crypto_forecast()
    combined = combine_model_forecasts([first, second])
    assert combined.forecast_status == "MODEL_AGREEMENT"
    assert combined.forecast_source == "ENSEMBLE"
    assert combined.direction == "LONG"
    assert combined.combination_method == "DIRECTION_AGREEMENT_MEDIAN"
    assert combined.usable_for_decision
    assert len(combined.individual_forecasts) == 2


def test_raw_tcn_forecast_is_never_decision_usable_without_ensemble() -> None:
    forecasts = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(), outputs={4: _tcn_output()},
    )

    assert not forecasts[1].usable_for_decision


def test_combination_never_aggregates_conflicting_directions() -> None:
    first = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(), outputs={4: _tcn_output("LONG")},
    )[1]
    second = _chronos_crypto_forecast("SHORT")
    combined = combine_model_forecasts([first, second])
    assert combined.forecast_status == "MODEL_DISAGREEMENT"
    assert combined.direction is None
    assert combined.expected_return is None


def test_single_available_model_never_becomes_decision_usable() -> None:
    first = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(), outputs={4: _tcn_output()},
    )[1]
    missing = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(), outputs={},
    )[1]

    combined = combine_model_forecasts([first, missing])

    assert combined.forecast_status == "MODEL_PARTIAL_AGREEMENT"
    assert not combined.usable_for_decision


def test_oos_quality_gate_keeps_forecasts_visible_without_promoting_to_validated() -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    forecasts = direct_tcn_forecasts(
        asset="BTC/USDT",
        asset_class="crypto",
        model="tcn",
        model_version="prod-v1",
        timestamp=timestamp,
        outputs={1: _tcn_output(), 4: _tcn_output(), 8: _tcn_output(), 12: _tcn_output(), 24: _tcn_output()},
    )
    by_horizon = {item.horizon: apply_oos_quality_gate(item) for item in forecasts}

    assert by_horizon["1h"].quality_status == "DEGRADED"
    assert by_horizon["1h"].forecast_status == "DEGRADED"
    assert by_horizon["1h"].usable_for_decision is False
    assert by_horizon["1h"].direction == "LONG"

    assert by_horizon["4h"].quality_status == "PARTIALLY_VALIDATED"
    assert by_horizon["4h"].forecast_status == "PARTIALLY_VALIDATED"
    assert by_horizon["4h"].usable_for_decision is False
    assert by_horizon["8h"].usable_for_decision is False

    assert by_horizon["12h"].quality_status == "PARTIALLY_VALIDATED"
    assert by_horizon["12h"].usable_for_decision is False
    assert by_horizon["1d"].usable_for_decision is False

    assert by_horizon["4h"].forecast_status != "VALIDATED"
    assert by_horizon["8h"].forecast_status != "VALIDATED"


def test_chronos_quantiles_round_trip_through_forecast_json(tmp_path) -> None:
    database = tmp_path / "state.db"
    state_db.init_db(database)
    tcn = direct_tcn_forecasts(
        asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1",
        timestamp=datetime.now(timezone.utc).isoformat(), outputs={4: _tcn_output()},
    )[1]
    combined = combine_model_forecasts([tcn, _chronos_crypto_forecast()])

    with state_db.connect(database) as connection:
        state_db.replace_assistant_forecasts(connection, "BTC/USDT", [combined.as_dict()])
        row = connection.execute(
            "SELECT individual_forecasts_json FROM assistant_forecasts WHERE asset=? AND horizon=?",
            ("BTC/USDT", "4h"),
        ).fetchone()

    details = json.loads(row["individual_forecasts_json"])
    chronos = next(item for item in details if item["forecast_source"] == "CHRONOS_2")
    assert (chronos["p10_price"], chronos["p50_price"], chronos["p90_price"]) == (99.0, 101.0, 103.0)