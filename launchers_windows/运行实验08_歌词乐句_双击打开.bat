@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
if not exist logs mkdir logs
echo Experiment 08: the breathy control on sung lyric phrases (GTSinger, English).
echo About 75 minutes. Run it only after experiment 07 has finished.
echo Keep this window open, keep the computer awake, and avoid other GPU-heavy programs.
echo If it is interrupted, just double-click this file again: finished parts are kept.
echo.
echo ===== %date% %time% ===== >> logs\exp08_lyrics.log
.venv-seed\Scripts\python.exe scripts\steer\exp08_lyrics.py >> logs\exp08_lyrics.log 2>&1
if errorlevel 1 goto failed
echo.
echo Finished. Tell Claude it is done.
pause
exit /b 0
:failed
echo.
echo It failed, exit code %ERRORLEVEL%. Tell Claude; the log is logs\exp08_lyrics.log
pause
exit /b 1
