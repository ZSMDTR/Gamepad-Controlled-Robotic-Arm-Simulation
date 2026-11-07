@echo off
cd /d "%~dp0"
.venv\Scripts\python.exe scripts\native_viewer.py --port 8002 --robot b601_rs --open-browser
if errorlevel 1 (
  echo.
  echo MuJoCo could not start. Check the message above, then press any key.
  pause >nul
)
