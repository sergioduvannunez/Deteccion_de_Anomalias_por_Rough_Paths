@echo off
chcp 65001 >nul
title Rough Paths Lab - Regenerar datos
cd /d "%~dp0"

rem --- Buscar Python dinámicamente (igual que "Iniciar Rough Paths Lab.bat") ---
set "PY="

rem Opción 1: conda en el PATH → detectar el entorno ml_env
where conda >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('conda env list ^| findstr /C:"ml_env"') do (
        for %%j in (%%i) do (
            if exist "%%j\python.exe" set "PY=%%j\python.exe"
        )
    )
)

rem Opción 2: python global del sistema
if "%PY%"=="" (
    where python >nul 2>nul
    if %errorlevel% equ 0 set "PY=python"
)

if "%PY%"=="" (
    echo [ERROR] No se encontro Python en el sistema.
    echo Instala Python desde https://www.python.org/downloads/ y vuelve a intentar.
    pause
    exit /b
)

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
