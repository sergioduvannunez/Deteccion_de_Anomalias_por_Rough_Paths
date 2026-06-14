#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — lanzador universal de Rough Paths Lab (Windows / macOS / Linux).

Pensado para funcionar en CUALQUIER computador tras clonar el repositorio:

    python run.py            # arranca el servidor y abre el navegador
    python run.py 8080       # puerto personalizado

Qué hace, en orden:
  1. Comprueba la versión de Python (>= 3.9).
  2. Comprueba que estén las dependencias; si falta alguna, muestra el comando
     exacto para instalarlas y termina sin un traceback confuso.
  3. Lanza frontend/servidor.py con el MISMO intérprete que ejecuta este script
     (así usa el entorno activo, sin rutas absolutas a ninguna máquina).

No depende de Anaconda ni de ningún entorno concreto: usa sys.executable.
"""

import importlib
import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))

# Paquete a importar  ->  nombre con el que se instala vía pip
DEPENDENCIAS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "pyarrow": "pyarrow",   # solo necesario para la muestra/parquet AMI
}


def revisar_python():
    if sys.version_info < (3, 9):
        print(f"[ERROR] Se necesita Python 3.9 o superior (tienes {sys.version.split()[0]}).")
        print("        Instálalo desde https://www.python.org/downloads/ y vuelve a intentar.")
        sys.exit(1)


def revisar_dependencias():
    faltan = []
    for modulo, paquete in DEPENDENCIAS.items():
        try:
            importlib.import_module(modulo)
        except ImportError:
            faltan.append(paquete)
    if faltan:
        print("[ERROR] Faltan dependencias de Python: " + ", ".join(faltan))
        print("\n  Instálalas con este comando (copia y pega):\n")
        print(f"    {os.path.basename(sys.executable)} -m pip install " + " ".join(faltan))
        print("\n  O todas de una vez:\n")
        print(f"    {os.path.basename(sys.executable)} -m pip install -r requirements.txt\n")
        sys.exit(1)


def main():
    revisar_python()
    revisar_dependencias()

    puerto = sys.argv[1] if len(sys.argv) > 1 else "8001"
    servidor = os.path.join(AQUI, "frontend", "servidor.py")
    if not os.path.exists(servidor):
        print(f"[ERROR] No encuentro {servidor}. ¿Ejecutas run.py desde la raíz del proyecto?")
        sys.exit(1)

    print("Iniciando Rough Paths Lab...")
    # Mismo intérprete que ejecuta run.py: respeta el entorno activo del usuario.
    try:
        subprocess.run([sys.executable, servidor, puerto], cwd=AQUI)
    except KeyboardInterrupt:
        print("\nServidor detenido.")


if __name__ == "__main__":
    main()
