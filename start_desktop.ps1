# 启用严格错误捕获
$ErrorActionPreference = "Stop"

function Pause-Script {
    try {
        Write-Host "Press any key to continue..."
        $null = $host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    } catch {
        Read-Host "Press Enter to exit..." | Out-Null
    }
}

try {
    if ($PSScriptRoot) {
        Set-Location $PSScriptRoot
    }

    Write-Host "Current Directory: $(Get-Location)"
    Write-Host "Checking python environment..."

    function Get-FreePort {
        param([int]$StartPort = 8501, [int]$MaxPort = 8510)
        for ($p = $StartPort; $p -le $MaxPort; $p++) {
            try {
                $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $p)
                $listener.Start()
                $listener.Stop()
                return $p
            } catch {
                continue
            }
        }
        throw "No free port found in range $StartPort-$MaxPort"
    }

    $runner = Join-Path (Split-Path $PSScriptRoot -Parent) "run_py39.ps1"
    $projectVenv = Join-Path $PSScriptRoot ".venv"
    $venvPy = Join-Path $projectVenv "Scripts\python.exe"

    if (Test-Path $runner) {
        Write-Host "Using python via: $runner"
        & $runner --version
        if (-not (Test-Path $venvPy)) {
            Write-Host "Creating local venv at $projectVenv"
            & $runner -m venv $projectVenv
        }
    } else {
        Write-Host "run_py39.ps1 not found. Falling back to python on PATH."
        python --version
        if (-not (Test-Path $venvPy)) {
            Write-Host "Creating local venv at $projectVenv"
            python -m venv $projectVenv
        }
    }

    if (-not (Test-Path $venvPy)) {
        throw "Local venv python not found: $venvPy"
    }

    Write-Host "Checking dependencies (venv)..."
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r requirements.txt

    $port = Get-FreePort
    Write-Host "Starting War Watch Platform on port $port..."

    $job = Start-Job -ArgumentList $port -ScriptBlock {
        param($p)
        for ($i = 0; $i -lt 60; $i++) {
            try {
                $client = New-Object System.Net.Sockets.TcpClient
                $client.Connect("127.0.0.1", [int]$p)
                $client.Close()
                Start-Process "http://localhost:$p" | Out-Null
                break
            } catch {
                Start-Sleep -Seconds 1
            }
        }
    }

    & $venvPy -m streamlit run platform_app.py --server.address 127.0.0.1 --server.port $port --server.headless true --browser.gatherUsageStats false
    try { Remove-Job $job -Force | Out-Null } catch {}

} catch {
    Write-Host "An error occurred:" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
    Write-Host "Stack Trace:" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace -ForegroundColor Red
} finally {
    # 无论成功失败，最后都暂停，防止窗口立刻关闭
    Pause-Script
}
