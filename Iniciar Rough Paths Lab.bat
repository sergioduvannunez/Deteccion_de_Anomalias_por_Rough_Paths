@echo off
chcp 65001 >nul
title Rough Paths Lab
cd /d "%~dp0"

rem --- Buscar Python dinámicamente en Anaconda ---
set "PY="

rem Opción 1: Intentar activar el entorno 'ml_env' usando la variable de entorno de Anaconda
where conda >nul 2>nul
if %errorlevel% equ 0 (
    rem Si conda está en el PATH, lo usamos para encontrar la ruta del entorno
    for /f "tokens=*" %%i in ('conda env list ^| findstr /C:"ml_env"') do (
        for %%j in (%%i) do (
            if exist "%%j\python.exe" set "PY=%%j\python.exe"
        )
    )
)

rem Opción 2: Si no se detectó por Conda, intentamos usar el comando 'python' global del sistema
if "%PY%"=="" (
    where python >nul 2>nul
    if %errorlevel% equ 0 set "PY=python"
)

rem Si después de todo no se encuentra Python, avisamos al usuario
if "%PY%"=="" (
    echo [ERROR] No se encontro el entorno 'ml_env' ni Python en el sistema.
    echo Asegurate de tener Anaconda instalado y el entorno creado.
    pause
    exit /b
)

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

rem Ejecutar el servidor con el Python encontrado
"%PY%" "%~dp0frontend\servidor.py"

echo.
echo El servidor se ha detenido.
pause