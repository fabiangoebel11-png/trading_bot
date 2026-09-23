@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where uv >nul 2>&1
if errorlevel 1 (
    echo [FEHLER] uv wurde nicht gefunden.
    echo Bitte uv installieren: https://docs.astral.sh/uv/getting-started/installation/
    pause
    exit /b 1
)

echo Erstelle virtuelle Umgebung...
uv venv
if errorlevel 1 goto :failed

echo Installiere Trading Bot im Editable Mode...
uv pip install -e .
if errorlevel 1 goto :failed

echo Installiere CPU-Version von PyTorch...
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :failed

if exist ".env" goto :env_ready
if not exist ".env.example" (
    echo [FEHLER] .env.example wurde nicht gefunden.
    goto :failed
)

echo.
echo Keine .env gefunden. Bitte Telegram-Zugangsdaten eingeben.
set /p "TELEGRAM_BOT_TOKEN=TELEGRAM_BOT_TOKEN: "
set /p "TELEGRAM_CHAT_ID=TELEGRAM_CHAT_ID: "
if not defined TELEGRAM_BOT_TOKEN (
    echo [FEHLER] TELEGRAM_BOT_TOKEN darf nicht leer sein.
    goto :failed
)
if not defined TELEGRAM_CHAT_ID (
    echo [FEHLER] TELEGRAM_CHAT_ID darf nicht leer sein.
    goto :failed
)

copy /y ".env.example" ".env" >nul
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Get-Content -Raw '.env'; $p = $p -replace '(?m)^TELEGRAM_BOT_TOKEN=.*$', ('TELEGRAM_BOT_TOKEN=' + $env:TELEGRAM_BOT_TOKEN); $p = $p -replace '(?m)^TELEGRAM_CHAT_ID=.*$', ('TELEGRAM_CHAT_ID=' + $env:TELEGRAM_CHAT_ID); [IO.File]::WriteAllText((Join-Path (Get-Location) '.env'), $p)"
if errorlevel 1 goto :failed

:env_ready
echo.
echo Setup abgeschlossen! Du kannst jetzt start.bat ausführen.
echo Setup abgeschlossen! Du kannst jetzt start.bat ausfuehren.
pause
exit /b 0

:failed
echo.
echo Setup fehlgeschlagen. Bitte die Fehlermeldung pruefen.
pause
exit /b 1