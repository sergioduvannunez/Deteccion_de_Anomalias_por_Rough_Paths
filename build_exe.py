#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_exe.py — genera RoughPathsLab.exe (100% autónomo, PyInstaller --onefile)

Doble clic en el .exe → arranca el servidor y abre el navegador con el
laboratorio completo. NO requiere Python ni ninguna instalación previa:
empaqueta el intérprete, el servidor (frontend/servidor.py), la SPA
(index.html + css + js, con Plotly y KaTeX offline) y TODOS los datos
pre-calculados de outputs/ (AMI + simulados + neuralde).

Uso:
    python build_exe.py

El .exe queda en la raíz del proyecto (~100-150 MB).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    print(">>", " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd))
    subprocess.check_call(cmd)


def main():
    try:
        import PyInstaller  # noqa: F401
        print("[OK] PyInstaller ya instalado.")
    except ImportError:
        print("[INFO] Instalando PyInstaller...")
        run([sys.executable, "-m", "pip", "install", "pyinstaller"])

    sep = ";" if sys.platform.startswith("win") else ":"
    front = os.path.join(HERE, "frontend")
    outputs = os.path.join(HERE, "outputs")

    # origen_absoluto ; destino_relativo_en_bundle  (todo bajo sys._MEIPASS)
    pares = [
        (os.path.join(front, "index.html"), "."),
        (os.path.join(front, "css"), "css"),
        (os.path.join(front, "js"), "js"),
        (os.path.join(outputs, "tablas_framework"), "outputs/tablas_framework"),
        (os.path.join(outputs, "series"), "outputs/series"),
        (os.path.join(outputs, "simulados"), "outputs/simulados"),
        (os.path.join(outputs, "neuralde"), "outputs/neuralde"),
    ]

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile", "--console", "--noconfirm",
        "--name", "RoughPathsLab",
        "--distpath", HERE,
        "--workpath", os.path.join(HERE, "build", "_work"),
        "--specpath", os.path.join(HERE, "build"),
        "--hidden-import", "pandas._libs.tslibs.np_datetime",
        "--hidden-import", "pandas._libs.tslibs.nattype",
        "--hidden-import", "pandas._libs.tslibs.timedeltas",
    ]
    for origen, destino in pares:
        if not os.path.exists(origen):
            print(f"[WARN] no existe {origen} — se omite")
            continue
        cmd += ["--add-data", f"{origen}{sep}{destino}"]
    cmd.append(os.path.join(front, "servidor.py"))

    print("\n=== Construyendo RoughPathsLab.exe (2-4 min la primera vez) ===\n")
    run(cmd)

    exe = os.path.join(HERE, "RoughPathsLab.exe")
    size = os.path.getsize(exe) // (1024 * 1024) if os.path.exists(exe) else "?"
    print(f"\n=== Listo: RoughPathsLab.exe ({size} MB) ===")
    print("Doble clic en RoughPathsLab.exe para lanzar el laboratorio.\n")


if __name__ == "__main__":
    main()
