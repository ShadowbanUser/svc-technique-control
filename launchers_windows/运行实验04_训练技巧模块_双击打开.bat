@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
if not exist logs mkdir logs
echo Experiment 04: train the technique adapter. Three stages, roughly 1 to 1.5 hours in total.
echo Keep this window open. Avoid other GPU-heavy programs while it runs.
echo.
echo [1/3] Preparing VocalSet training pairs, about 10-20 minutes...
.venv-seed\Scripts\python.exe scripts\train\prep_technique_pairs.py > logs\exp04_prep.log 2>&1
if errorlevel 1 goto failed
echo [2/3] Training, about 30-60 minutes...
.venv-seed\Scripts\python.exe scripts\train\train_technique_adapter.py > logs\exp04_train.log 2>&1
if errorlevel 1 goto failed
echo [3/3] Generating test outputs, about 10 minutes...
.venv-seed\Scripts\python.exe scripts\steer\exp04_adapter_eval.py > logs\exp04_eval.log 2>&1
if errorlevel 1 goto failed
echo.
echo All three stages finished. Tell Claude it is done.
pause
exit /b 0
:failed
echo.
echo A stage failed, exit code %ERRORLEVEL%. Tell Claude; the logs are in the logs folder.
pause
exit /b 1
