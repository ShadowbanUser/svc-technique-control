@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
if not exist logs mkdir logs
echo Experiment 06: objective tests against real technique recordings.
echo Two stages, roughly 1.5 to 2 hours in total.
echo Keep this window open, keep the computer awake, and avoid other GPU-heavy programs.
echo If it is interrupted, just double-click this file again: finished parts are kept.
echo.
echo [1/2] Conversions and distances to the real recordings, about 1.5 hours...
echo ===== %date% %time% ===== >> logs\exp06_groundtruth.log
.venv-seed\Scripts\python.exe scripts\steer\exp06_groundtruth.py >> logs\exp06_groundtruth.log 2>&1
if errorlevel 1 goto failed
echo [2/2] Technique classifier, about 20 minutes, CPU only...
echo ===== %date% %time% ===== >> logs\exp06_classifier.log
.venv-seed\Scripts\python.exe scripts\steer\exp06_classifier.py >> logs\exp06_classifier.log 2>&1
if errorlevel 1 goto failed
echo.
echo Both stages finished. Tell Claude it is done.
pause
exit /b 0
:failed
echo.
echo A stage failed, exit code %ERRORLEVEL%. Tell Claude; the logs are in the logs folder.
pause
exit /b 1
