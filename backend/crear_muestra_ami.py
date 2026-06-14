"""
backend/crear_muestra_ami.py
============================
Crea una MUESTRA ANONIMIZADA y pequeña de los datos crudos AMI.

Motivación
----------
El dataset AMI real pesa ~7 GB (230 .parquet) y la columna `ID` identifica
medidores concretos (dato sensible). Para el repositorio se conserva solo una
muestra: pocos medidores, un mes, con los identificadores reales sustituidos
por códigos genéricos `AMI_0001`, `AMI_0002`, ...

El mapeo medidor_real -> código genérico se construye en memoria y NO se
guarda: tras ejecutar este script y borrar los originales, la muestra es
efectivamente anónima (no hay forma de revertir el código al medidor real).

Estructura de los .parquet AMI (no cambia):
    ID    (str)   identificador del medidor   -> se anonimiza
    Year, Month, Day, Hour (int)              timestamp descompuesto
    Value (float|str)                          consumo horario en Wh

Uso:
    python backend/crear_muestra_ami.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:  # consola Windows en UTF-8 (evita errores con acentos)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "Raw_Processed"

# Parámetros de la muestra (pocos datos, entendibles)
ARCHIVO_ORIGEN = "ACTIVE_Enero_6.parquet"   # uno de los .parquet más pequeños
N_CASAS = 60                                  # medidores a conservar
MES_SALIDA = "Enero"                          # nombre de mes para el archivo de salida
SEED = 42


def main() -> None:
    origen = RAW / ARCHIVO_ORIGEN
    if not origen.exists():
        # plan B: tomar el ACTIVE más pequeño que exista
        candidatos = sorted(RAW.glob("ACTIVE_*.parquet"), key=lambda p: p.stat().st_size)
        if not candidatos:
            print("[ERROR] no hay archivos ACTIVE_*.parquet en Raw_Processed")
            sys.exit(1)
        origen = candidatos[0]
        print(f"[INFO] {ARCHIVO_ORIGEN} no existe; uso {origen.name}")

    print(f"Leyendo {origen.name} ...")
    df = pd.read_parquet(origen, columns=["ID", "Year", "Month", "Day", "Hour", "Value"])

    # Homogenizar tipos (algunos meses guardan Value como texto)
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df["ID"] = df["ID"].astype(str)
    df = df.dropna(subset=["Value"])

    # Elegir los N medidores con MAYOR cobertura (más horas registradas):
    # así la muestra permite armar ventanas semanales completas (168 h).
    cobertura = df.groupby("ID").size().sort_values(ascending=False)
    ids_elegidos = list(cobertura.head(N_CASAS).index)
    df = df[df["ID"].isin(ids_elegidos)].copy()

    # --- ANONIMIZACIÓN -------------------------------------------------------
    # Mapeo aleatorio medidor_real -> AMI_0001..AMI_00NN. No se persiste.
    rng = np.random.default_rng(SEED)
    barajados = list(ids_elegidos)
    rng.shuffle(barajados)
    mapeo = {real: f"AMI_{i + 1:04d}" for i, real in enumerate(barajados)}
    df["ID"] = df["ID"].map(mapeo)
    # ------------------------------------------------------------------------

    df = df.sort_values(["ID", "Year", "Month", "Day", "Hour"]).reset_index(drop=True)

    salida = RAW / f"ACTIVE_{MES_SALIDA}_0.parquet"
    tmp = RAW / f"_muestra_{MES_SALIDA}.parquet"   # nombre temporal mientras existe el original
    df.to_parquet(tmp, index=False)

    print(f"  medidores: {df['ID'].nunique()}  | filas: {len(df):,} | "
          f"tamaño: {tmp.stat().st_size / 1024:.0f} KB")
    print(f"  IDs anonimizados: {sorted(df['ID'].unique())[:3]} ... "
          f"{sorted(df['ID'].unique())[-1]}")
    print(f"[OK] muestra escrita en {tmp.name} (se renombrará a {salida.name} "
          f"tras limpiar los originales)")


if __name__ == "__main__":
    main()
