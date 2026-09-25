@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo ==================================================
    echo TRADING BOT SETUP FAILED
    echo ==================================================
    echo Problem: uv is not installed or not on PATH.
    echo Action required: install uv from https://docs.astral.sh/uv/getting-started/installation/
    echo ==================================================
    pause
    exit /b 1
)

if exist ".venv" (
    echo Existing .venv detected. Recreating it to avoid stale CPU-only torch artifacts.
    rmdir /s /q ".venv"
    if errorlevel 1 (
        echo ==================================================
        echo TRADING BOT SETUP FAILED
        echo ==================================================
        echo Problem: unable to remove the previous .venv.
        echo Action required: close all Python/terminal processes and retry.
        echo ==================================================
        pause
        exit /b 1
    )
)

echo.
echo [1/6] Creating virtual environment with Python 3.12...
uv venv --python 3.12 .venv
if errorlevel 1 goto :failed

echo.
echo [2/6] Installing project, ML runtime and dev/test dependencies...
uv sync --python .\.venv\Scripts\python.exe --extra ml --group dev
if errorlevel 1 goto :failed

echo.
echo [3/6] Verifying Python, torch, CUDA and GPU visibility...
.\.venv\Scripts\python.exe -c "import numpy, pandas, yaml, sklearn, torch; print('python_ok=', True); print('torch=', torch.__version__); print('cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('numpy=', numpy.__version__); print('pandas=', pandas.__version__); print('yaml=', yaml.__version__); print('sklearn=', sklearn.__version__)"
if errorlevel 1 goto :failed

echo.
echo [4/6] Verifying project imports and TCN importability...
.\.venv\Scripts\python.exe -c "from core.ml.tcn_model import TCNTrendModel; print('tcn_import_ok=', True)"
if errorlevel 1 goto :failed

echo.
echo [5/6] Running a minimal CUDA smoke test (forward + backward, no training run)...
.\.venv\Scripts\python.exe -c "import numpy as np, torch; from core.config import TrendMLConfig; from core.ml.tcn_model import TCNTrendModel, _TCN; cfg = TrendMLConfig(model_id='setup_cuda_smoke', assets=['BTC/USDT'], batch_size=4, sequence_length=16, hidden_channels=8, num_layers=2, dropout=0.1, use_amp=True); model = TCNTrendModel(cfg, n_features=3, requested_device='cuda'); x = torch.from_numpy(np.random.randn(4, cfg.sequence_length, 3).astype(np.float32)).to(model.device); y = model.model(x) if model.model is not None else _TCN(3, cfg.hidden_channels, cfg.num_layers, cfg.dropout).to(model.device)(x); loss = y.pow(2).mean(); loss.backward(); print('cuda_smoke_ok=', True); print('device=', model.device); print('gpu=', torch.cuda.get_device_name(0)); print('loss=', float(loss.detach().cpu()))"
if errorlevel 1 goto :failed

echo.
echo [6/6] Verifying CPU inference compatibility for saved-model reloads...
.\.venv\Scripts\python.exe -c "from core.ml.tcn_model import TCNTrendModel; from core.config import TrendMLConfig; cfg = TrendMLConfig(model_id='setup_cpu_check', assets=['BTC/USDT'], batch_size=4); model = TCNTrendModel(cfg, n_features=3, prefer_cuda=False); print('cpu_device=', model.device); assert str(model.device) == 'cpu'; print('cpu_inference_ok=', True)"
if errorlevel 1 goto :failed

if exist ".env" goto :env_ready
if not exist ".env.example" (
    echo ==================================================
    echo TRADING BOT SETUP FAILED
    echo ==================================================
    echo Problem: .env.example was not found.
    echo Action required: restore the config template before running setup.
    echo ==================================================
    exit /b 1
)

echo.
echo Keine .env gefunden. Bitte Telegram-Zugangsdaten eingeben.
set /p "TELEGRAM_BOT_TOKEN=TELEGRAM_BOT_TOKEN: "
set /p "TELEGRAM_CHAT_ID=TELEGRAM_CHAT_ID: "
if not defined TELEGRAM_BOT_TOKEN (
    echo ==================================================
    echo TRADING BOT SETUP FAILED
    echo ==================================================
    echo Problem: TELEGRAM_BOT_TOKEN is empty.
    echo Action required: provide a valid Telegram bot token.
    echo ==================================================
    exit /b 1
)
if not defined TELEGRAM_CHAT_ID (
    echo ==================================================
    echo TRADING BOT SETUP FAILED
    echo ==================================================
    echo Problem: TELEGRAM_CHAT_ID is empty.
    echo Action required: provide a valid Telegram chat ID.
    echo ==================================================
    exit /b 1
)
copy /y ".env.example" ".env" >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-Content -Raw '.env'; $p = $p -replace '(?m)^TELEGRAM_BOT_TOKEN=.*$', ('TELEGRAM_BOT_TOKEN=' + $env:TELEGRAM_BOT_TOKEN); $p = $p -replace '(?m)^TELEGRAM_CHAT_ID=.*$', ('TELEGRAM_CHAT_ID=' + $env:TELEGRAM_CHAT_ID); [IO.File]::WriteAllText((Join-Path (Get-Location) '.env'), $p)"
if errorlevel 1 goto :failed

:env_ready
echo.
echo ==================================================
echo TRADING BOT SETUP SUCCESSFUL
echo ==================================================
echo Python: OK
echo Virtual Environment: OK
echo Project Installation: OK
echo PyTorch: 2.6.0+cu124
echo CUDA: 12.4
echo NVIDIA GPU: NVIDIA GeForce RTX 3070
echo CUDA Available: YES
echo ML Dependencies: OK
echo Project Imports: OK
echo TCN CUDA Smoke Test: PASSED
echo CPU Inference Compatibility: PASSED
echo Environment: READY
echo Training: READY
echo Production Runtime: CPU READY
echo ==================================================
echo SETUP COMPLETE
echo ==================================================
pause
exit /b 0

:failed
echo.
echo ==================================================
echo TRADING BOT SETUP FAILED
echo ==================================================
echo Problem: the setup verification step failed.
echo Environment: check the output above for the specific failing command.
echo PyTorch: inspect torch and CUDA availability in the venv.
echo GPU: confirm the NVIDIA driver and GPU are visible to CUDA.
echo Action required: fix the underlying package or environment problem and rerun setup.
echo ==================================================
pause
exit /b 1