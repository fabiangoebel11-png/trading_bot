import pytest

from decision_engine import (
    MarketSnapshot,
    SignalSnapshot,
    analyze_position,
    build_trade_plan,
    calculate_position,
    scenario_analysis,
)
import state_db
from decision_pipeline import analyse_ohlcv, load_cached_ohlcv, model_quality


def _market() -> MarketSnapshot:
    return MarketSnapshot("BTC/USDT", "crypto", 100_000.0, atr=1_000.0, atr_pct=0.01, trend="UP", volatility="NORMAL", data_status="LIVE")


def _signal() -> SignalSnapshot:
    return SignalSnapshot("BTC/USDT", "LONG", 78.0, expected_return=0.02, expected_mfe=0.03, expected_mae=-0.01, expected_duration_bars=8.0)


def test_trade_plan_is_deterministic_and_order_free() -> None:
    plan = build_trade_plan(_market(), _signal(), account=500.0, risk_pct=0.005)
    assert plan is not None
    assert plan.strategy_id == "trend_breakout"
    assert plan.initial_stop < plan.entry_reference < plan.tp1 < plan.tp2
    assert plan.risk_amount == pytest.approx(2.5)
    assert 0 <= plan.signal_quality <= 100


def test_position_math_separates_notional_margin_and_risk() -> None:
    result = calculate_position(account=500.0, entry=100.0, current=102.0, stop=98.0, take_profit=108.0, leverage=3.0, direction="LONG")
    assert result.notional == pytest.approx(525.0)
    assert result.margin == pytest.approx(175.0)
    assert result.risk_amount == pytest.approx(10.5)
    assert result.profit_at_tp > 0
    assert result.liquidation_estimate == pytest.approx(100.0 * (1 - 1 / 3))


def test_position_analysis_and_scenarios_follow_state_machine() -> None:
    analysis = analyze_position(position_id="p1", asset="BTC/USDT", direction="LONG", entry=100.0, current=106.0, quantity=5.0, margin=100.0, stop=95.0, take_profit=110.0)
    assert analysis.state == "HOLD"
    scenarios = scenario_analysis(entry=100.0, current=106.0, quantity=5.0, margin=100.0, stop=95.0, take_profit=110.0, direction="LONG", changes=(-0.2, 0.05))
    assert scenarios[0].state == "INVALIDATED"
    assert scenarios[1].state == "TAKE PROFIT"


def test_stale_or_uncertain_data_does_not_create_plan() -> None:
    market = MarketSnapshot("BTC/USDT", "crypto", 100.0, atr=1.0, data_status="STALE")
    assert build_trade_plan(market, _signal(), account=500.0) is None


def test_assistant_position_and_scenario_persist_without_paper_trade(tmp_path) -> None:
    db_path = tmp_path / "assistant.db"
    state_db.init_db(db_path)
    state_db.save_assistant_position(db_path, {
        "position_id": "manual-1", "asset": "BTC/USDT", "direction": "LONG", "entry_price": 100.0,
        "quantity": 1.0, "margin": 100.0, "stop_price": 95.0, "take_profit_price": 110.0,
    })
    state_db.append_assistant_scenario(db_path, "scenario-1", "manual-1", {"state": "PROFIT"})
    with state_db.connect(db_path) as conn:
        position = conn.execute("SELECT * FROM assistant_positions WHERE position_id = 'manual-1'").fetchone()
        scenario = conn.execute("SELECT * FROM assistant_scenarios WHERE scenario_id = 'scenario-1'").fetchone()
        paper_trade = conn.execute("SELECT COUNT(*) AS count FROM open_trades").fetchone()["count"]
    assert position["status"] == "OPEN"
    assert scenario["subject_id"] == "manual-1"
    assert paper_trade == 0


def test_real_btc_cache_produces_selection_and_plan() -> None:
    frame = load_cached_ohlcv("BTC/USDT", "1h", "data")
    if frame is None:
        pytest.skip("BTC cache unavailable in this checkout")
    result = analyse_ohlcv("BTC/USDT", "1h", frame, capital=500.0, asset_class="crypto")
    assert result.indicators is not None
    assert result.regime in {"TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOLATILITY", "LOW_VOLATILITY", "UNCLEAR"}
    assert result.best_candidate is not None
    assert result.plan is not None
    assert result.plan.notional > 0
    assert result.plan.maximum_loss <= 500.0 * 0.005 + 1e-8


def test_missing_equity_cache_is_explicit() -> None:
    result = analyse_ohlcv("QQQ", "1h", None, asset_class="equity")
    assert result.data_status == "INSUFFICIENT_DATA"
    assert result.plan is None
    assert result.model.status == "MODEL_NOT_AVAILABLE"


def test_model_score_requires_matching_artifact_and_keeps_scores_separate() -> None:
    signal = {"model_type": "crypto_sniper", "score": 81.0, "direction": "LONG", "expected_mfe": 0.02, "expected_mae": -0.01}
    quality = model_quality(signal, asset="BTC/USDT", asset_class="crypto")
    assert quality.status == "AVAILABLE"
    assert quality.score == pytest.approx(81.0)
    assert quality.model_version != "unknown"

    unavailable = model_quality(signal, asset="BTC/USDT", asset_class="crypto", model_root="does-not-exist")
    assert unavailable.status == "MODEL_NOT_AVAILABLE"
    assert unavailable.score is None


def test_equity_provider_cache_is_read_without_fabricated_data() -> None:
    frame = load_cached_ohlcv("QQQ", "1h", "data")
    if frame is None:
        pytest.skip("QQQ provider cache unavailable in this checkout")
    assert len(frame) >= 120
    assert frame.index.is_monotonic_increasing


def test_position_snapshot_and_idea_are_immutable(tmp_path) -> None:
    db_path = tmp_path / "assistant.db"
    state_db.init_db(db_path)
    idea = {"idea_id": "idea-1", "asset": "BTC/USDT", "strategy_id": "momentum", "strategy_version": "1.0.0", "status": "VALID_SETUP", "plan": {"entry": 100.0}, "input_snapshot": {"close": 100.0}}
    state_db.save_assistant_trade_idea(db_path, idea)
    state_db.save_assistant_trade_idea(db_path, idea)
    with pytest.raises(ValueError):
        state_db.save_assistant_trade_idea(db_path, {**idea, "plan": {"entry": 101.0}})
    state_db.append_assistant_position_snapshot(db_path, "snapshot-1", "manual-1", {"state": "HOLD"}, "1.0.0")
    with pytest.raises(ValueError):
        state_db.append_assistant_position_snapshot(db_path, "snapshot-1", "manual-1", {"state": "EXIT"}, "1.0.0")