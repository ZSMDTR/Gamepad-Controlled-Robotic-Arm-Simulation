@echo off
cd /d "%~dp0"
set /p "LEADER_PORT=Leader serial port (for example COM7): "
if not defined LEADER_PORT (
  echo A serial port is required.
  pause >nul
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_b601_windows.ps1" -LeaderPort "%LEADER_PORT%"
if errorlevel 1 (
  echo Startup stopped. Read the message above, then press any key.
  pause >nul
)
