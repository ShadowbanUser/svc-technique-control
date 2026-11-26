@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
if not exist logs mkdir logs
echo Experiment 05: controls and one ablation. Three stages, roughly 1.5 to 2 hours in total.
echo Keep this window open. Avoid other GPU-heavy programs while it runs.
echo.
echo [1/3] Domain check and strength sweep, about 15 minutes...
.venv-seed\Scripts\python.exe scripts\steer\exp05_controls.py > logs\exp05_controls.log 2>&1
if errorlevel 1 goto failed
echo [2/3] Building the ablation inputs, about 10 minutes...
.venv-seed\Scripts\python.exe scripts\train\prep_self_content.py > logs\exp05_prep_self.log 2>&1
if errorlevel 1 goto failed
echo [3/3] Training the ablation, about 80 minutes...
.venv-seed\Scripts\python.exe scripts\train\train_technique_adapter.py --train-cache cache/techpairs_self --out results/exp05/ablation_nodtw > logs\exp05_ablation.log 2>&1
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
