@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
if not exist logs mkdir logs
echo Running experiment 03 (about 15-25 minutes). Please keep this window open.
.venv-seed\Scripts\python.exe scripts\steer\exp03_panel.py > logs\exp03.log 2>&1
echo finished with exit code %ERRORLEVEL%. Log: logs\exp03.log
pause
