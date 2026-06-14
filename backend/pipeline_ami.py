"""
backend/pipeline_ami.py
=======================
Análisis de detección de anomalías sobre los datos AMI reales (consumo
eléctrico horario), con la suite de detectores del backend.

IMPORTANTE — procedencia de los datos
-------------------------------------
Los resultados AMI que consume el frontend (outputs/tablas_framework/ y
outputs/series/) se calcularon sobre el DATASET COMPLETO original (~7 GB, un
año de medidores reales). Ese dataset NO se incluye en el repositorio por
tamaño y por privacidad. En su lugar se incluye una MUESTRA ANONIMIZADA
(Raw_Processed/ACTIVE_Enero_0.parquet: 60 medidores, un mes, IDs sustituidos
por códigos AMI_xxxx; ver backend/crear_muestra_ami.py).

Por eso este script tiene dos caminos:

  --demo   : corre sobre la MUESTRA anonimizada incluida. Genera resultados
             de DEMOSTRACIÓN (pocas ventanas) en outputs/ami_demo/, SIN tocar
             los resultados oficiales. Sirve para comprobar que el pipeline
             funciona de extremo a extremo con los datos del repo.

  (sin flag): reprocesa el DATASET COMPLETO (si se dispone de él en
             Raw_Processed) y reescribe los resultados oficiales en
             outputs/tablas_framework/. Tarda horas. Acepta --quick para leer
             solo 2 archivos por mes.

Uso:
    python -m backend.pipeline_ami --demo      # demo con la muestra incluida
    python -m backend.pipeline_ami --quick     # dataset completo, 2 files/mes
    python -m backend.pipeline_ami             # dataset completo entero
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

try:
    from .anomalias import InyectorAnomalias, TIPOS_ANOMALIA
    from .cargador import CargadorAMI
    from .detectores import DETECTORES_AMI, SuiteDetectores
    from .evaluacion import evaluar_pu, jaccard_matrices
except ImportError:  # pragma: no cover
    from anomalias import InyectorAnomalias, TIPOS_ANOMALIA
    from cargador import CargadorAMI
    from detectores import DETECTORES_AMI, SuiteDetectores
    from evaluacion import evaluar_pu, jaccard_matrices

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parent.parent
OUT_OFICIAL = BASE / "outputs" / "tablas_framework"
OUT_DEMO = BASE / "outputs" / "ami_demo"


def evaluar_muestra(df: pd.DataFrame, nombre: str, out_dir: Path,
                    fraccion: float = 0.05, depths=(2, 3, 4)) -> None:
    """
    Pasos para una muestra de series semanales (filas con columnas h0..h167):
      1. Inyectar anomalías sintéticas (InyectorAnomalias).
      2. Ajustar la suite de detectores SOLO con las series originales.
      3. Puntuar originales + sintéticos.
      4. Evaluar con protocolo PU (DR/AR) y matrices de Jaccard -> CSV.
    """
    hcols = [f"h{i}" for i in range(168)]
    iny = InyectorAnomalias(fraccion=fraccion, seed=42)
    df_eval = iny.inyectar(df)                       # añade is_synthetic / anomaly_type

    X_train = df[hcols].values.astype(float)         # solo originales -> entrenamiento
    X_eval = df_eval[hcols].values.astype(float)     # originales + sintéticos -> evaluación
    y_syn = df_eval["is_synthetic"].values
    y_tipo = df_eval["anomaly_type"].values

    suite = SuiteDetectores(detectores=DETECTORES_AMI, depths=depths, seed=42)
    suite.ajustar(X_train)
    scores = suite.puntuar(X_eval)

    evaluar_pu(scores, y_syn, y_tipo, TIPOS_ANOMALIA, nombre, out_dir=out_dir)
    jaccard_matrices(scores, y_syn, y_tipo, TIPOS_ANOMALIA, nombre, out_dir=out_dir)
    print(f"  [{nombre}] OK — {len(scores)} detectores, "
          f"{int(y_syn.sum())} sintéticos sobre {len(df)} originales")


def correr_demo() -> None:
    """Demostración end-to-end sobre la muestra anonimizada incluida."""
    print("=== pipeline_ami --demo (muestra anonimizada) ===")
    OUT_DEMO.mkdir(parents=True, exist_ok=True)
    # min_dias bajo: la muestra cubre ~3 semanas de un mes.
    cargador = CargadorAMI(min_dias=7)

    # weekly_month: 1 semana representativa por mes (aquí, el único mes).
    semanas = cargador.muestreo_weekly_month()
    if semanas:
        df_wm = pd.concat(semanas.values(), ignore_index=True)
        # fracción de inyección alta: con pocas ventanas asegura ~5 por tipo.
        evaluar_muestra(df_wm, "weekly_month", OUT_DEMO, fraccion=0.30)

    # monthly: todas las ventanas semanales del mes.
    mensual = cargador.muestreo_monthly()
    if mensual:
        df_m = pd.concat(mensual.values(), ignore_index=True)
        evaluar_muestra(df_m, "monthly", OUT_DEMO, fraccion=0.30)

    print(f"\nOK demo: resultados en {OUT_DEMO.relative_to(BASE)} "
          f"(los oficiales en tablas_framework/ NO se tocaron)")


def correr_completo(quick: bool) -> None:
    """Reprocesa el dataset COMPLETO y reescribe los resultados oficiales."""
    print("=== pipeline_ami (dataset completo) ===")
    OUT_OFICIAL.mkdir(parents=True, exist_ok=True)
    cargador = CargadorAMI(max_files_per_month=2 if quick else None)

    semanas = cargador.muestreo_weekly_month()
    evaluar_muestra(pd.concat(semanas.values(), ignore_index=True),
                    "weekly_month", OUT_OFICIAL)
    if quick:
        print("\nOK pipeline_ami --quick (solo weekly_month)")
        return

    df_anual = cargador.muestreo_annual()
    evaluar_muestra(df_anual, "annual", OUT_OFICIAL)
    evaluar_muestra(df_anual, "monthly", OUT_OFICIAL)

    res, _ = cargador.seleccionar_casas(n_residencial=1, n_comercial=0)
    if res:
        sem = cargador.muestreo_single_house(res[0])
        filas = []
        for mes, lst in sem.items():
            for ts, arr in lst:
                fila = {"ID": res[0], "mes": mes, "ts_inicio": ts}
                fila.update({f"h{j}": arr[j] for j in range(168)})
                filas.append(fila)
        if filas:
            evaluar_muestra(pd.DataFrame(filas), "single_house", OUT_OFICIAL)
    print("\nOK pipeline_ami: outputs/tablas_framework/ regenerado")


def main():
    if "--demo" in sys.argv:
        correr_demo()
    else:
        correr_completo(quick="--quick" in sys.argv)


if __name__ == "__main__":
    main()
