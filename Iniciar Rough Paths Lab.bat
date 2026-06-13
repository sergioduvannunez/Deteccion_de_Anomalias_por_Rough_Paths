@echo off
chcp 65001 >nul
title Rough Paths Lab
cd /d "%~dp0"

rem --- localizar el interprete de Python ---
set "PY=C:\Users\aipri\anaconda3\envs\ml_env\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================================
echo    ROUGH PATHS LAB
echo    Deteccion de anomalias por signaturas + Neural ODE/CDE/RDE
echo ============================================================
echo.
echo    Iniciando servidor... el navegador se abrira solo en
echo    http://localhost:8001
echo.
echo    Para DETENER: cierra esta ventana o pulsa Ctrl+C
echo ============================================================
echo.

"%PY%" "%~dp0frontend\servidor.py"

echo.
echo El servidor se ha detenido.
pause
