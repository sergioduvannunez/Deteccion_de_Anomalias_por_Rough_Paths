@echo off
chcp 65001 >nul
title Rough Paths Lab - Regenerar datos
cd /d "%~dp0"

set "PY=C:\Users\aipri\anaconda3\envs\ml_env\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo    REGENERAR DATOS  (solo si cambiaste el backend)
echo ============================================================
echo.
echo    [1/2] Contextos simulados (it / ambiental / eeg) ~1.5 min
"%PY%" -m backend.pipeline_anomalias
echo.
echo    [2/2] Ecuaciones diferenciales neuronales ~2 min
"%PY%" -m backend.pipeline_neuralde
echo.
echo ============================================================
echo    Listo. Ahora ejecuta "Iniciar Rough Paths Lab.bat".
echo ============================================================
pause
