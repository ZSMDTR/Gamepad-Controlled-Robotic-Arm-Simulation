@echo off
cd /d "%~dp0"
echo Open http://127.0.0.1:8001 in your browser.
echo Keep this window open while using the simulation. Close it to stop.
".venv\Scripts\python.exe" -m arm_teleop --port 8001
pause
