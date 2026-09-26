@echo off
setlocal EnableExtensions
cd /d "%~dp0"

if not exist ".env" (
    echo [HINWEIS] Keine .env gefunden. Bitte zuerst setup.bat ausfuehren.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo [HINWEIS] Keine virtuelle Umgebung gefunden. Bitte zuerst setup.bat ausfuehren.
    pause
    exit /b 1
)
if not exist "start.ps1" (
    echo [FEHLER] start.ps1 wurde nicht gefunden.
    pause
    exit /b 1
)

echo Starte Daemon, Paper-Broker und GUI...
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
if errorlevel 1 (
    echo [FEHLER] start.ps1 konnte nicht gestartet werden.
    pause
    exit /b 1
)
exit /b 0