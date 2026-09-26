# Trading Bot

## Quickstart

Für einen neuen Windows-Rechner:

1. **Repository klonen:** `git clone <dein-repo-url>` und in das Projektverzeichnis wechseln.
2. **Doppelklick auf `setup.bat`:** Die virtuelle Umgebung und alle Abhängigkeiten werden installiert. Falls `.env` fehlt, fragt das Skript die Telegram-Daten ab und erstellt sie.
3. **Doppelklick auf `start.bat`:** Daemon, Paper-Broker und Streamlit-GUI werden in separaten Fenstern gestartet.

Manuelle Installation:

1. **Umgebung installieren:**
	```powershell
	uv venv --python 3.12
	uv sync --python .\.venv\Scripts\python.exe --extra ml --group dev
	```
	Das Projekt verwendet explizit den offiziellen CUDA-124-PyTorch-Index via `tool.uv.sources` fuer `torch`. `torchvision` und `torchaudio` werden im TCN-Setup nicht verwendet und sind nicht Bestandteil des `ml`-Extras.
3. **Konfiguration anlegen:** `Copy-Item .env.example .env` und die lokalen Credentials in `.env` eintragen. Die echte `.env` wird nie versioniert.
4. **Starten:** `./start.ps1` öffnet Daemon, Paper-Broker und Streamlit-GUI in separaten Fenstern. Die GUI ist danach unter `http://localhost:8501` erreichbar.

## Ordnerübersicht

- Im Root liegen die Windows-Startskripte, `app.py`, die Daemon-/Broker-Einstiegspunkte und gemeinsam importierte Runtime-Module.
- `core/` und `execution/` enthalten Engine und Ausführungslogik; `configs/` enthält YAML-Konfigurationen.
- `research/training/` enthält die modellbezogenen Trainingsstarter; `research/diagnostics/` enthält einmalige Diagnoseprogramme.
- `docs/reports/` und `docs/archive/` enthalten Audit-Ausgaben und historische Übergaben.
- `scripts/manual/` enthält manuelle Telegram-/Testnet-Prüfungen. Der Bybit-Test kann eine Testnet-Order platzieren und wieder stornieren.
- `data/` und `training_logs/` sind für lokale Datenbanken, Caches und Ausgaben; diese Dateien gehören nicht in einen Source-Commit.

Modellspezifische Trainingsstarter werden vom Repository-Root aus aufgerufen:

```powershell
.\.venv\Scripts\python.exe research\training\train_crypto_intraday.py
.\.venv\Scripts\python.exe research\training\train_equity_intraday.py
.\.venv\Scripts\python.exe research\training\train_swing.py --train
```

Der Produktionsstart setzt die Daemon-/GUI-Inferenz weiterhin explizit auf CPU; der globale Override `TRADING_BOT_FORCE_CPU` wird nicht mehr gesetzt, damit ein separater Trainingsprozess CUDA verwenden kann. Training verwendet `training_device: cuda` und bricht hart ab, wenn CUDA nicht verfügbar ist. `core.ml.device` meldet beim Daemon-Start `Inference Device: CUDA` oder `Inference Device: CPU`. Das Startskript startet einen beendeten Prozess nach 10 Sekunden erneut. Zum Beenden das jeweilige Prozessfenster schließen.

## Repository-Inhalt und Modelle

Runtime-Daten, Markt-Caches, Datenbanken, Logs, Credentials und trainierte Checkpoints bleiben lokal und sind per `.gitignore` ausgeschlossen. Checkpoints müssen bei Bedarf separat bereitgestellt oder neu trainiert werden; sie gehören nicht zum Source-Commit.

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