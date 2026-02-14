@echo off
setlocal

chcp 65001 >nul
set "ROOT_DIR=%~dp0"
set "CONFIG_PATH=%ROOT_DIR%talk2slides.ini"

if not exist "%CONFIG_PATH%" (
    echo [ERROR] Config file not found: "%CONFIG_PATH%"
    exit /b 1
)

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Please install Python 3.9+ and add it to PATH.
    exit /b 1
)

python "%ROOT_DIR%scripts\run_windows.py" --config "%CONFIG_PATH%" %*
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%
