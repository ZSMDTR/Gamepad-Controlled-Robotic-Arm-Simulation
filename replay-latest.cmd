@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run start-mujoco-b601.cmd first to set up the environment.
  pause >nul
  exit /b 1
)
".venv\Scripts\python.exe" scripts\replay_episode.py --latest
if errorlevel 1 (
  echo.
  echo Replay stopped. Read the message above, then press any key.
  pause >nul
)
