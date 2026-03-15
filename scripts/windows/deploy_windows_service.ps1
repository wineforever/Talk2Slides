param(
    [string]$ServiceName = "Talk2Slides",
    [string]$ProjectRoot = "",
    [string]$ConfigPath = "",
    [string]$PythonExe = "",
    [switch]$Reinstall
)

$ErrorActionPreference = "Stop"

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Resolve-ExistingPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "$Label not found: $Path"
    }

    return (Resolve-Path -LiteralPath $Path).Path
}

function Resolve-PythonExe {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ProjectRoot,
        [string]$ExplicitPythonExe = ""
    )

    if (-not [string]::IsNullOrWhiteSpace($ExplicitPythonExe)) {
        $command = Get-Command $ExplicitPythonExe -ErrorAction SilentlyContinue
        if ($command) {
            return $command.Source
        }
        if (Test-Path -LiteralPath $ExplicitPythonExe) {
            return (Resolve-Path -LiteralPath $ExplicitPythonExe).Path
        }
        throw "Python executable not found: $ExplicitPythonExe"
    }

    $venvPython = Join-Path $ProjectRoot "backend\venv\Scripts\python.exe"
    if (Test-Path -LiteralPath $venvPython) {
        return (Resolve-Path -LiteralPath $venvPython).Path
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return $python.Source
    }

    throw "Python executable not found. Install Python or pass -PythonExe."
}

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    & $FilePath @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Description failed with exit code $exitCode."
    }
}

function Ensure-PythonModule {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string]$ImportName,
        [Parameter(Mandatory = $true)]
        [string]$PackageName
    )

    & $PythonExe "-c" "import $ImportName" *> $null
    if ($LASTEXITCODE -eq 0) {
        return
    }

    Write-Host "[INFO] Installing missing dependency: $PackageName"
    Invoke-ExternalCommand `
        -FilePath $PythonExe `
        -Arguments @("-m", "pip", "install", $PackageName) `
        -Description "Install Python package $PackageName"
}

if (-not (Test-Admin)) {
    throw "Please run this script from an elevated PowerShell session."
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = Join-Path $PSScriptRoot "..\.."
}
$ProjectRoot = Resolve-ExistingPath -Path $ProjectRoot -Label "Project root"

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $ProjectRoot "talk2slides.ini"
}
$ConfigPath = Resolve-ExistingPath -Path $ConfigPath -Label "Config file"

$PythonExe = Resolve-PythonExe -ProjectRoot $ProjectRoot -ExplicitPythonExe $PythonExe
$ServiceScript = Resolve-ExistingPath -Path (Join-Path $ProjectRoot "scripts\windows\talk2slides_service.py") -Label "Service script"
$LogPath = Join-Path $ProjectRoot "logs\$ServiceName.service.log"

Ensure-PythonModule -PythonExe $PythonExe -ImportName "win32serviceutil" -PackageName "pywin32"

$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing -and -not $Reinstall) {
    throw "Service '$ServiceName' already exists. Use -Reinstall to replace it."
}

if ($existing -and $Reinstall) {
    Write-Host "[INFO] Removing existing service: $ServiceName"
    if ($existing.Status -ne "Stopped") {
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
    }

    Invoke-ExternalCommand `
        -FilePath $PythonExe `
        -Arguments @($ServiceScript, "--service-name", $ServiceName, "remove") `
        -Description "Remove service $ServiceName"

    $deadline = (Get-Date).AddSeconds(15)
    do {
        Start-Sleep -Milliseconds 500
        $existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
    } while ($existing -and (Get-Date) -lt $deadline)

    if ($existing) {
        throw "Timed out waiting for service '$ServiceName' to be deleted."
    }
}

Write-Host "[INFO] Installing Python service host: $ServiceName"
Invoke-ExternalCommand `
    -FilePath $PythonExe `
    -Arguments @(
        $ServiceScript,
        "--service-name", $ServiceName,
        "--display-name", $ServiceName,
        "--description", "Talk2Slides FastAPI Service",
        "--startup", "auto",
        "install"
    ) `
    -Description "Install service $ServiceName"

Write-Host "[INFO] Saving service parameters"
Invoke-ExternalCommand `
    -FilePath $PythonExe `
    -Arguments @(
        $ServiceScript,
        "--service-name", $ServiceName,
        "--config", $ConfigPath,
        "--project-root", $ProjectRoot,
        "--log-path", $LogPath,
        "configure"
    ) `
    -Description "Configure service $ServiceName"

Write-Host "[INFO] Configuring service restart on failure"
sc.exe failure $ServiceName reset= 86400 actions= restart/5000/restart/5000/restart/5000 | Out-Null

Write-Host "[INFO] Starting service: $ServiceName"
try {
    Start-Service -Name $ServiceName
} catch {
    if (Test-Path -LiteralPath $LogPath) {
        Write-Host "[INFO] Recent service log output:"
        Get-Content -Path $LogPath -Tail 40
    }
    throw
}

Write-Host "[DONE] Deployment completed."
Write-Host "Check service status with: Get-Service $ServiceName"
Write-Host "Service log file: $LogPath"
