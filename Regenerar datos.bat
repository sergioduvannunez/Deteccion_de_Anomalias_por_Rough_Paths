@echo off
chcp 65001 >nul
title Rough Paths Lab - Regenerar datos
cd /d "%~dp0"

set "PY="

rem ─── 1. Buscar Python — misma lógica que "Iniciar Rough Paths Lab.bat" ───────

where py >nul 2>nul
if %errorlevel% equ 0 set "PY=py"

if "%PY%"=="" (
    where python >nul 2>nul
    if %errorlevel% equ 0 set "PY=python"
)

if "%PY%"=="" (
    where python3 >nul 2>nul
    if %errorlevel% equ 0 set "PY=python3"
)

if "%PY%"=="" (
    where conda >nul 2>nul
    if %errorlevel% equ 0 (
        for /f "delims=" %%i in ('conda env list 2^>nul ^| findstr /C:"ml_env"') do (
            for %%j in (%%i) do (
                if exist "%%j\python.exe" set "PY=%%j\python.exe"
            )
        )
    )
)

if "%PY%"=="" (
    where conda >nul 2>nul
    if %errorlevel% equ 0 (
        for /f "delims=" %%b in ('conda info --base 2^>nul') do (
            if exist "%%b\python.exe" set "PY=%%b\python.exe"
        )
    )
)

if "%PY%"=="" (
    echo.
    echo  [ERROR] No se encontro Python en este computador.
    echo  Instala Python desde https://www.python.org/downloads/
    echo  ^(marca "Add Python to PATH" durante la instalacion^)
    echo.
    pause
    exit /b 1
)

rem ─── 2. Instalar dependencias automáticamente ───────────────────────────────
echo.
echo    Python encontrado. Instalando dependencias del proyecto...
echo    (puede tardar 1-2 minutos la primera vez)
echo.
"%PY%" -m pip install -r "%~dp0requirements.txt"
if %errorlevel% neq 0 (
    echo.
    echo  [ADVERTENCIA] pip reporto un problema. Se intentara continuar de todas formas.
    echo.
)

rem ─── 3. Regenerar datos del backend ─────────────────────────────────────────
echo.
echo ============================================================
echo    REGENERAR DATOS  (solo si cambiaste el backend)
echo ============================================================
echo.
echo    [1/2] Contextos simulados (it / ambiental / eeg) ~1.5 min
"%PY%" -m backend.pipeline_anomalias
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Fallo pipeline_anomalias. Revisa el mensaje de arriba.
    pause
    exit /b 1
)

echo.
echo    [2/2] Ecuaciones diferenciales neuronales ~2 min
"%PY%" -m backend.pipeline_neuralde
if %errorlevel% neq 0 (
    echo.
    echo  [ERROR] Fallo pipeline_neuralde. Revisa el mensaje de arriba.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo    Listo. Ahora ejecuta "Iniciar Rough Paths Lab.bat".
echo ============================================================
pause
