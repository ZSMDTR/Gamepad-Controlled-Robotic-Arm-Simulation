@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_b601_windows.ps1"
if errorlevel 1 (
  echo.
  echo Startup stopped. Read the message above, then press any key.
  pause >nul
)
