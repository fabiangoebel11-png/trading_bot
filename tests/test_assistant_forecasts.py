from core.ml.assistant_forecasts import PUBLIC_HORIZONS, apply_oos_quality_gate, combine_model_forecasts, direct_swing_forecasts, direct_tcn_forecasts
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


def test_equity_swing_registry_uses_only_exact_trading_day_heads() -> None:
    outputs = {day: {"expected_return": 0.01 * day, "expected_mfe": 0.04, "expected_mae": -0.02, "expected_duration": float(day), "opportunity_score": 70.0} for day in (1, 3, 5, 10, 20)}
    forecasts = direct_swing_forecasts(asset="SPY", model="swing", model_version="v1", timestamp="2026-01-01T00:00:00+00:00", outputs=outputs)
    by_horizon = {item.horizon: item for item in forecasts}
    assert by_horizon["5d"].forecast_source == "DIRECT_MODEL"
    assert by_horizon["4h"].forecast_status == "UNAVAILABLE"
    assert by_horizon["1d"].expected_duration == 1.0


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
    first, second = direct_tcn_forecasts(asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1", timestamp="2026-01-01T00:00:00+00:00", outputs={4: _tcn_output()})[1], direct_tcn_forecasts(asset="BTC/USDT", asset_class="crypto", model="chronos2", model_version="v1", timestamp="2026-01-01T00:00:00+00:00", outputs={4: _tcn_output()})[1]
    combined = combine_model_forecasts([first, second])
    assert combined.forecast_status == "MODEL_AGREEMENT"
    assert combined.forecast_source == "ENSEMBLE"
    assert combined.direction == "LONG"
    assert combined.combination_method == "DIRECTION_AGREEMENT_MEDIAN"
    assert len(combined.individual_forecasts) == 2


def test_combination_never_aggregates_conflicting_directions() -> None:
    first, second = direct_tcn_forecasts(asset="BTC/USDT", asset_class="crypto", model="tcn", model_version="v1", timestamp="2026-01-01T00:00:00+00:00", outputs={4: _tcn_output("LONG")})[1], direct_tcn_forecasts(asset="BTC/USDT", asset_class="crypto", model="chronos2", model_version="v1", timestamp="2026-01-01T00:00:00+00:00", outputs={4: _tcn_output("SHORT")})[1]
    combined = combine_model_forecasts([first, second])
    assert combined.forecast_status == "MODEL_DISAGREEMENT"
    assert combined.direction is None
    assert combined.expected_return is None


def test_oos_quality_gate_keeps_forecasts_visible_without_promoting_to_validated() -> None:
    forecasts = direct_tcn_forecasts(
        asset="BTC/USDT",
        asset_class="crypto",
        model="tcn",
        model_version="prod-v1",
        timestamp="2026-01-01T00:00:00+00:00",
        outputs={1: _tcn_output(), 4: _tcn_output(), 8: _tcn_output(), 12: _tcn_output(), 24: _tcn_output()},
    )
    by_horizon = {item.horizon: apply_oos_quality_gate(item) for item in forecasts}

    assert by_horizon["1h"].quality_status == "DEGRADED"
    assert by_horizon["1h"].forecast_status == "DEGRADED"
    assert by_horizon["1h"].usable_for_decision is False
    assert by_horizon["1h"].direction == "LONG"

    assert by_horizon["4h"].quality_status == "PARTIALLY_VALIDATED"
    assert by_horizon["4h"].forecast_status == "PARTIALLY_VALIDATED"
    assert by_horizon["4h"].usable_for_decision is True
    assert by_horizon["8h"].usable_for_decision is True

    assert by_horizon["12h"].quality_status == "PARTIALLY_VALIDATED"
    assert by_horizon["12h"].usable_for_decision is False
    assert by_horizon["1d"].usable_for_decision is False

    assert by_horizon["4h"].forecast_status != "VALIDATED"
    assert by_horizon["8h"].forecast_status != "VALIDATED"