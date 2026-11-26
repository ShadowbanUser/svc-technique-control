@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set PYTHONDONTWRITEBYTECODE=1
if not exist logs mkdir logs
echo Experiment 07: conversion between different singers, scored against real recordings,
echo plus speaker similarity and pitch checks on the experiment 06 outputs.
echo About 1 hour. Needs the experiment 06 results, which are already on this computer.
echo Keep this window open, keep the computer awake, and avoid other GPU-heavy programs.
echo If it is interrupted, just double-click this file again: finished parts are kept.
echo.
echo ===== %date% %time% ===== >> logs\exp07_crosssinger.log
.venv-seed\Scripts\python.exe scripts\steer\exp07_crosssinger.py >> logs\exp07_crosssinger.log 2>&1
if errorlevel 1 goto failed
echo.
echo Finished. Tell Claude it is done.
pause
exit /b 0
:failed
echo.
echo It failed, exit code %ERRORLEVEL%. Tell Claude; the log is logs\exp07_crosssinger.log
pause
exit /b 1
