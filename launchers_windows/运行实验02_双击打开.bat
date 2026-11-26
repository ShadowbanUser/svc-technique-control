@echo off
title SVC experiment 02 - three control mechanisms
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
if not exist logs mkdir logs
echo.
echo Experiment 02 on the GPU. About 10 minutes, nothing is printed here. Please wait ...
.venv-seed\Scripts\python.exe scripts\steer\exp02_mechanisms.py > logs\exp02.log 2>&1
echo finished with exit code %ERRORLEVEL%. Log: logs\exp02.log
echo.
echo ALL DONE. You can close this window.
pause
