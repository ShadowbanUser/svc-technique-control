@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
if not exist logs mkdir logs
echo Building the listening test package, about 2-5 minutes...
.venv-seed\Scripts\python.exe scripts\listen\build_listening_package.py > logs\listening_package.log 2>&1
if errorlevel 1 goto failed
echo.
echo Done. The zip file is in this folder. Tell Claude.
pause
exit /b 0
:failed
echo.
echo Failed, exit code %ERRORLEVEL%. Tell Claude; the log is logs\listening_package.log
pause
exit /b 1
