# Trading Bot

## Quickstart

1. **Repository klonen:** `git clone <dein-repo-url>` und in das Projektverzeichnis wechseln.
2. **Umgebung installieren:**
	```powershell
	uv venv
	uv pip install -e .
	```
	Auf CPU-only-Laptops zusätzlich installieren: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`.
3. **Konfiguration anlegen:** `Copy-Item .env.example .env` und die lokalen Credentials in `.env` eintragen. Die echte `.env` wird nie versioniert.
4. **Starten:** `.\start.ps1` öffnet Daemon, Paper-Broker und Streamlit-GUI in separaten Fenstern. Die GUI ist danach unter `http://localhost:8501` erreichbar.

`core.ml.device` verwendet automatisch CUDA, wenn verfügbar, und sonst CPU. Der Daemon meldet beim Start `Inference Device: CUDA` oder `Inference Device: CPU`. Das Startskript startet einen beendeten Prozess nach 10 Sekunden erneut. Zum Beenden das jeweilige Prozessfenster schließen.

## Repository-Inhalt und Modelle

Runtime-Daten, Markt-Caches, Datenbanken, Logs und Credentials sind per `.gitignore` ausgeschlossen. Die vorhandenen PyTorch-Checkpoints sind zusammen etwa 14,45 MB groß und bleiben deshalb zunächst als normale Git-Dateien versionierbar. Falls sie wachsen, sollten sie über Git LFS oder als GitHub-Release-Artefakte verteilt werden. Ohne Checkpoints müssen die Modelle vor dem Betrieb neu trainiert oder manuell bereitgestellt werden.

## ML Training Pipeline

The research pipeline loads configured OHLCV histories, normalizes them,
builds causal technical, cross-asset, and higher-timeframe features, and
creates deterministic triple-barrier labels. Features are turned into causal
sequences and evaluated with purged chronological walk-forward folds. No
random train/test split is used.

Historical OHLCV can be persisted as Parquet with `core.ml.data`; the existing
CCXT cache remains available for compatibility. Different instruments may have
different start dates and market sessions; missing observations are retained as
missing rather than forward-filled across closures.

Read-only historical preparation is explicit:

```text
python train.py --prepare-data --config configs/training.yaml
```

Preparation expands the configured primary universe (`NASDAQ100_PROXY` and
`SP500_PROXY`, explicitly `QQQ` and `SPY`, plus BTC/USDT and ETH/USDT) and
keeps DAX optional. The explicit mappings are: **NASDAQ-100 research proxy =
QQQ** and **S&P 500 research proxy = SPY**. It also includes a small contextual set (VIX, US10Y,
EUR/USD, gold, and WTI) across the configured timeframes. Each request records
its provider and source symbol and is cached under
`data/market/<asset>/<timeframe>/<provider>.parquet`. Existing legacy crypto
CSV caches remain readable and are migrated on demand when a canonical Parquet
file is prepared. Equity session closures are reported separately from missing
24/7 candles; prices are normalized to UTC without forward-filling closures.

US proxy 5m data uses the configured Twelve Data provider and
`TWELVE_DATA_API_KEY`; 15m, 1h, and 4h are derived locally from the same 5m
source. Daily proxy history remains separately sourced from yfinance. Twelve
Data request budgets and retries are configured in `training.yaml`; missing
keys, quota exhaustion, or unavailable history are reported explicitly. No
proxy is substituted automatically.

The repository does not automatically train the model. Start real training
only with this explicit command:

```text
python train.py --train --config configs/training.yaml
```

Plain `python train.py` only prints help. Training uses causal 5m/15m/1h/4h/1d
alignment, purged chronological folds, optional logistic and gradient-boosting
baselines, validation-only temperature calibration, and multitask TCN outputs:
`P(SHORT)`, `P(NO_TRADE)`, `P(LONG)`, expected MFE, and expected MAE.

The fixed train, validation, test, and shadow date ranges are configurable in
the YAML file. Test data is not used for calibration, feature selection, or
threshold selection.

The command writes model weights, metadata, scaler values, and leak-free
out-of-fold confidence under `data/models/` (or the configured model path).
The metadata records the feature list, label barriers, training cutoff,
architecture, and random seed. See `docs/ml_output.md` for the output contract.