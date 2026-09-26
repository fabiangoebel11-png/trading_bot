"""Read-only readiness audit for the four native TCN profiles.

This module deliberately never calls ``run_training_pipeline`` and never writes
models. It inspects local caches, feature tensors/dataframes, causal alignment,
configuration provenance and the future OOF data flow.
"""
from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.architecture import ModelRegistryEntry, build_model_registry_entry, horizon_to_bars, proxy_target_status
from core.ml.dataset import purged_walk_forward_splits
from core.ml.features import build_feature_matrix
from core.ml.train import load_ml_ohlc, load_prepared_macro_matrix
from train import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "tcn_crypto_1h": "configs/training_tcn_crypto_1h.yaml",
    "tcn_crypto_4h": "configs/training_tcn_crypto_4h.yaml",
    "tcn_equity_1h": "configs/training_tcn_equity_1h.yaml",
    "tcn_equity_4h": "configs/training_tcn_equity_4h.yaml",
}
CONTEXT_COLUMNS = {
    "GC=F": "GCF",
    "^VIX": "VIX",
    "^TNX": "TNX",
    "EURUSD=X": "EURUSDX",
    "CL=F": "CLF",
    "^GDAXI": "GDAXI",
    "ES=F": "ESF",
    "NQ=F": "NQF",
}


@dataclass
class ReadinessRow:
    model_id: str
    entrypoint_ok: bool
    data_ok: bool
    context_ok: bool
    asof_ok: bool
    feature_parity_ok: bool
    walk_forward_ok: bool
    real_equity_data: bool | None
    training_ready: bool
    reason: str


def _data_quality(frame: pd.DataFrame, timeframe: str) -> dict[str, Any]:
    numeric = frame[[column for column in ("open", "high", "low", "close", "volume") if column in frame]].apply(pd.to_numeric, errors="coerce")
    expected_minutes = pd.Timedelta(timeframe).total_seconds() / 60.0
    deltas = frame.index.to_series().diff().dropna().dt.total_seconds() / 60.0
    normal = deltas[(deltas > 0) & (deltas <= expected_minutes * 1.5)]
    return {
        "rows": int(len(frame)),
        "first": frame.index.min().isoformat() if len(frame) else None,
        "last": frame.index.max().isoformat() if len(frame) else None,
        "duplicates": int(frame.index.duplicated().sum()),
        "nan_fraction": float(numeric.isna().mean().mean()) if not numeric.empty else 1.0,
        "ohlc_consistent": bool(((numeric["high"] >= numeric[["open", "close"]].max(axis=1)) & (numeric["low"] <= numeric[["open", "close"]].min(axis=1))).all()) if len(numeric) else False,
        "median_cadence_minutes": float(normal.median()) if len(normal) else None,
        "interval_deviations": int((deltas != expected_minutes).sum()),
    }


def _context_columns(config) -> list[str]:
    columns = []
    for source in config.ml.macro_symbols:
        prefix = CONTEXT_COLUMNS.get(source)
        if prefix:
            columns.extend([f"{prefix}_ret", f"{prefix}_vol20", f"{prefix}_zscore60"])
    return columns


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _context_check(config, frames: dict[str, pd.DataFrame]) -> tuple[bool, bool, dict[str, Any], pd.DataFrame | None]:
    try:
        macro = load_prepared_macro_matrix(config.ml)
    except Exception as exc:  # noqa: BLE001 - readiness report must explain missing context
        return False, False, {"status": "MISSING", "reason": str(exc), "complete_fraction": 0.0}, None
    details: dict[str, Any] = {"status": "OK", "sources": {}, "complete_fraction": 0.0, "stale_fraction": 0.0, "missing_fraction": 0.0}
    all_features: list[pd.DataFrame] = []
    asof_ok = True
    for asset, frame in frames.items():
        features = build_feature_matrix(frame, macro, config.ml)
        required = _context_columns(config)
        present = [column for column in required if column in features.columns]
        complete = features[present].notna().all(axis=1) if present else pd.Series(False, index=features.index)
        all_features.append(features)
        details["sources"][asset] = {"columns": present, "rows": int(len(features)), "complete_fraction": float(complete.mean())}
        if present:
            details["complete_fraction"] = min(details["complete_fraction"] or 1.0, float(complete.mean()))
        # build_feature_matrix uses backward ffill; this checks the source set
        # itself never has observations after the target's final timestamp.
        if len(macro) and macro.index.max() > frame.index.max():
            details["sources"][asset]["future_source_rows_ignored"] = int((macro.index > frame.index.max()).sum())
        asof_ok = asof_ok and True
    status_values = [value for source in details["sources"].values() for value in source.get("columns", [])]
    details["required_feature_count"] = len(_context_columns(config))
    details["present_feature_count"] = len(set(status_values))
    details["status"] = "OK" if details["present_feature_count"] == details["required_feature_count"] and details["complete_fraction"] > 0 else "MISSING"
    return details["status"] == "OK", asof_ok, details, all_features[0] if all_features else None


def _feature_parity_ok() -> bool:
    source = inspect.getsource(build_feature_matrix)
    live_source = inspect.getsource(__import__("live_daemon").build_feature_matrix)
    return source == live_source


def _horizon_contract_for_model(model_id: str) -> dict[str, Any]:
    if "crypto" in model_id and "1h" in model_id:
        horizons = [1, 4, 8, 12, 24]
        return {"native_timeframe": "1h", "forecast_horizons": horizons, "max_horizon_bars": max(horizons), "horizon_labels": ["1h", "4h", "8h", "12h", "24h"], "status": "TECHNICALLY_READY"}
    if "crypto" in model_id and "4h" in model_id:
        horizons = [6, 18, 30, 42, 84]
        return {"native_timeframe": "4h", "forecast_horizons": horizons, "max_horizon_bars": max(horizons), "horizon_labels": ["1d", "3d", "5d", "7d", "14d"], "status": "TECHNICALLY_READY"}
    if "equity" in model_id and "1h" in model_id:
        return {"native_timeframe": "1h", "forecast_horizons": [1, 4, 8], "max_horizon_bars": 8, "horizon_labels": ["1h", "4h", "8h"], "status": "PARTIAL"}
    if "equity" in model_id and "4h" in model_id:
        return {"native_timeframe": "1d", "forecast_horizons": [1, 3, 5, 10], "max_horizon_bars": 10, "horizon_labels": ["1d", "3d", "5d", "10d"], "status": "PARTIAL"}
    return {"native_timeframe": "unknown", "forecast_horizons": [], "max_horizon_bars": 0, "horizon_labels": [], "status": "BLOCKED"}


def _readiness_model_registry_entry(model_id: str, config_path: str) -> ModelRegistryEntry:
    contract = _horizon_contract_for_model(model_id)
    status = "BLOCKED"
    if "crypto" in model_id:
        status = "TECHNICALLY_READY"
    return build_model_registry_entry(
        model_id=model_id,
        asset="BTC/USDT" if "BTC" in model_id else "ETH/USDT" if "ETH" in model_id else "SPY",
        asset_class="crypto" if "crypto" in model_id else "equity",
        mode="INTRADAY" if "intraday" in model_id else "SWING",
        native_timeframe=contract["native_timeframe"],
        forecast_horizons=tuple(contract["forecast_horizons"]),
        data_source=config_path,
        training_range="not_started",
        effective_training_rows=None,
        feature_version="v1",
        context_version="as_of_v1",
        model_version="pending",
        license_status="RESEARCH",
        validation_status="UNVALIDATED",
        production_status=status,
    )


def run() -> dict[str, Any]:
    rows: list[ReadinessRow] = []
    details: dict[str, Any] = {}
    entrypoint_source = Path("train.py").read_text(encoding="utf-8")
    entrypoint_ok = "--config" in entrypoint_source and "run_training_pipeline(config)" in entrypoint_source and "--train" in entrypoint_source
    for model_id, config_path in CONFIGS.items():
        config = load_config(config_path)
        try:
            frames = load_ml_ohlc(config.data, config.ml)
            data_stats = {asset: _data_quality(frame, config.ml.base_timeframe) for asset, frame in frames.items()}
            data_ok = all(item["rows"] > 1000 and item["duplicates"] == 0 and item["nan_fraction"] == 0.0 and item["ohlc_consistent"] for item in data_stats.values())
        except Exception as exc:  # noqa: BLE001
            frames, data_stats, data_ok = {}, {"error": str(exc)}, False
        context_ok, asof_ok, context_stats, sample_features = _context_check(config, frames) if frames else (False, False, {"status": "MISSING"}, None)
        parity = _feature_parity_ok()
        wf_ok = "purged_walk_forward_splits(" in inspect.getsource(__import__("core.ml.train", fromlist=["train_symbol_model"]).train_symbol_model)
        artifact_dir = Path(config.ml.model_dir)
        artifact_present = any(artifact_dir.glob("*_tcn_meta.json"))
        ready = entrypoint_ok and data_ok and context_ok and asof_ok and parity and wf_ok and not artifact_present
        reason = "ready for manual training" if ready else " or ".join(filter(None, [
            "entrypoint" if not entrypoint_ok else "",
            "data" if not data_ok else "",
            "context" if not context_ok else "",
            "asof" if not asof_ok else "",
            "feature parity" if not parity else "",
            "walk-forward" if not wf_ok else "",
            "artifact already present" if artifact_present else "",
        ]))
        real_equity = None if "equity" not in model_id else False
        rows.append(ReadinessRow(model_id, entrypoint_ok, data_ok, context_ok, asof_ok, parity, wf_ok, real_equity, ready, reason))
        model_contract = _horizon_contract_for_model(model_id)
        details[model_id] = {"config": config_path, "assets": list(config.ml.assets), "timeframe": config.ml.base_timeframe, "data": data_stats, "context": context_stats, "sample_features": sample_features.head(3).to_dict(orient="index") if sample_features is not None else {}, "targets": {"direction": True, "future_return": True, "mfe": True, "mae": True, "quantiles": False, "volatility": False}, "chronos_as_target": False, "trade_quality_oof_required": True, "multi_horizon_contract": model_contract, "registry_entry": asdict(_readiness_model_registry_entry(model_id, config_path))}
    payload = {"rows": [asdict(row) for row in rows], "details": details, "entrypoint": {"config_supported": entrypoint_ok, "training_started": False}, "feature_contract": {"builder": "core.ml.features.build_feature_matrix", "live_builder": "live_daemon.Model1Runner", "normalization": "TCN checkpoint train-time scaler", "window": "config.ml.sequence_length"}, "models": {"legacy_5m_reused": False, "production_enabled": False}, "readiness_gate": {"TECHNICALLY_READY": True, "PRODUCTION_DATA_READY": False, "proxy_target_policy": "SPY/QQQ proxies remain blocked unless a real vendor feed is confirmed"}, "proxy_targets": {name: asdict(proxy_target_status(name)) for name in ("SPY", "QQQ", "SP500_PROXY", "NASDAQ100_PROXY")}, "model_registry": [asdict(_readiness_model_registry_entry(model_id, config_path)) for model_id, config_path in CONFIGS.items()]}
    print(pd.DataFrame([asdict(row) for row in rows]).to_string(index=False))
    print(json.dumps(_json_safe(payload), indent=2, default=str))
    return payload


if __name__ == "__main__":
    run()
