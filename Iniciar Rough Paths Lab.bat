@echo off
chcp 65001 >nul
title Rough Paths Lab
cd /d "%~dp0"

set "PY="

rem ─── 1. Buscar Python — de más a menos universal ────────────────────────────

rem py.exe: Python Launcher (incluido en todo Python >= 3.3 de python.org)
where py >nul 2>nul
if %errorlevel% equ 0 set "PY=py"

rem python en el PATH (Anaconda, conda, python.org con PATH marcado)
if "%PY%"=="" (
    where python >nul 2>nul
    if %errorlevel% equ 0 set "PY=python"
)

rem python3 (Git for Windows, MSYS2, algunos entornos no estándar)
if "%PY%"=="" (
    where python3 >nul 2>nul
    if %errorlevel% equ 0 set "PY=python3"
)

rem entorno conda ml_env (si existe en este PC)
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

rem entorno base de Anaconda/Miniconda
if "%PY%"=="" (
    where conda >nul 2>nul
    if %errorlevel% equ 0 (
        for /f "delims=" %%b in ('conda info --base 2^>nul') do (
            if exist "%%b\python.exe" set "PY=%%b\python.exe"
        )
    )
)

rem ─── Sin Python: instrucciones de instalación ───────────────────────────────
if "%PY%"=="" (
    echo.
    echo  [ERROR] No se encontro Python en este computador.
    echo.
    echo  Instala Python y vuelve a abrir este archivo:
    echo.
    echo   Opcion A — Python oficial  ^(recomendado^):
    echo     https://www.python.org/downloads/
    echo     IMPORTANTE: marca "Add Python to PATH" durante la instalacion.
    echo.
    echo   Opcion B — Anaconda  ^(entorno cientifico completo^):
    echo     https://www.anaconda.com/download
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
    echo  [ADVERTENCIA] pip reporto un problema. Se intentara arrancar de todas formas.
    echo  Si el lab no carga, abre una terminal y ejecuta:
    echo      pip install -r requirements.txt
    echo.
)

rem ─── 3. Arrancar el servidor ─────────────────────────────────────────────────
echo.
echo ============================================================
echo    ROUGH PATHS LAB
echo    Deteccion de anomalias por signaturas + Neural ODE/CDE/RDE
echo ============================================================
echo.
echo    Iniciando servidor... el navegador se abrira en
echo    http://localhost:8001
echo.
echo    Para DETENER: cierra esta ventana o pulsa Ctrl+C
echo ============================================================
echo.

"%PY%" "%~dp0run.py"

echo.
echo El servidor se ha detenido.
pause
