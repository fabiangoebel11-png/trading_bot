$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$env:TRADING_BOT_INFERENCE_DEVICE = "cpu"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python virtual environment not found at $Python. Create it and install the project first."
}

function Start-ResilientWindow {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$CommandLine
    )

    $childScript = @"
`$Host.UI.RawUI.WindowTitle = '$Name'
Set-Location -LiteralPath '$ProjectRoot'
`$env:PYTHONUNBUFFERED = '1'
while (`$true) {
    Write-Host "[$Name] starting..."
    $CommandLine
    `$exitCode = `$LASTEXITCODE
    Write-Host "[$Name] exited with code `$exitCode. Restarting in 10 seconds..."
    Start-Sleep -Seconds 10
}
"@

    Start-Process -FilePath "powershell.exe" `
        -WorkingDirectory $ProjectRoot `
        -ArgumentList @("-NoLogo", "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $childScript)
}

Start-ResilientWindow -Name "Trading Bot - Live Daemon" -CommandLine "& '$Python' 'live_daemon.py'"
Start-ResilientWindow -Name "Trading Bot - Paper Broker" -CommandLine "& '$Python' 'paper_broker.py'"
Start-ResilientWindow -Name "Trading Bot - Streamlit GUI" -CommandLine "& '$Python' '-m' 'streamlit' 'run' 'app.py'"

Write-Host "Started daemon, paper broker, and GUI in separate windows."
Write-Host "Close a child window to stop that component."
