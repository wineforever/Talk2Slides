param(
    [string]$ServiceName = "Talk2Slides",
    [string]$ProjectRoot = "",
    [string]$ConfigPath = "",
    [switch]$Reinstall
)

$ErrorActionPreference = "Stop"

function Test-Admin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    throw "请使用管理员权限运行此脚本。"
}

if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
} else {
    $ProjectRoot = (Resolve-Path $ProjectRoot).Path
}

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $ProjectRoot "talk2slides.ini"
} else {
    $ConfigPath = (Resolve-Path $ConfigPath).Path
}

$RunBat = Join-Path $ProjectRoot "run_windows.bat"
if (-not (Test-Path $RunBat)) {
    throw "未找到启动脚本: $RunBat"
}
if (-not (Test-Path $ConfigPath)) {
    throw "未找到配置文件: $ConfigPath"
}

$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing -and -not $Reinstall) {
    throw "服务 '$ServiceName' 已存在。若需重装，请加 -Reinstall 参数。"
}

if ($existing -and $Reinstall) {
    Write-Host "[INFO] 停止并删除已有服务: $ServiceName"
    if ($existing.Status -ne "Stopped") {
        Stop-Service -Name $ServiceName -Force -ErrorAction SilentlyContinue
    }
    sc.exe delete $ServiceName | Out-Null
    Start-Sleep -Seconds 1
}

$binPath = "cmd.exe /c `"$RunBat --config `"$ConfigPath`"`""

Write-Host "[INFO] 创建服务: $ServiceName"
New-Service `
    -Name $ServiceName `
    -DisplayName $ServiceName `
    -Description "Talk2Slides FastAPI Service" `
    -BinaryPathName $binPath `
    -StartupType Automatic

Write-Host "[INFO] 配置服务故障自动重启"
sc.exe failure $ServiceName reset= 86400 actions= restart/5000/restart/5000/restart/5000 | Out-Null

Write-Host "[INFO] 启动服务: $ServiceName"
Start-Service -Name $ServiceName

Write-Host "[DONE] 部署完成。"
Write-Host "查看状态: Get-Service $ServiceName"
Write-Host "查看日志建议: 使用 run_windows.bat 的输出重定向或 Windows 事件查看器。"
