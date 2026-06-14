"""
backend/pipeline_anomalias.py
=============================
Pipeline de detección de anomalías sobre los TRES contextos simulados
multivariados (it, ambiental, eeg).

Por cada contexto:
  1. Genera el dataset masivo con anomalías etiquetadas (simuladores.py).
  2. Ajusta la suite completa de detectores SOLO con ventanas originales
     (clásicos + signatures multivariadas con aumento temporal — el muestreo
     irregular del contexto 'ambiental' entra por los timestamps reales).
  3. Puntúa originales + sintéticos y evalúa con protocolo PU (τ = p90).
  4. Exporta a outputs/simulados/:
       metricas_pu_{ctx}.csv
       jaccard_general_{ctx}.csv (+ una por tipo)
       contexto_{ctx}.json   (metadatos, canales, muestreo, descripciones)
       muestras_{ctx}.json   (pares normal/anómala multicanal para el front)
       detecciones_{ctx}.csv (scores de cada detector sobre los pares)

Uso:
    python -m backend.pipeline_anomalias            # tamaño completo
    python -m backend.pipeline_anomalias --rapido   # versión reducida
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import simuladores
    from .detectores import SuiteDetectores, catalogo_multivariado
    from .evaluacion import evaluar_pu, jaccard_matrices
except ImportError:  # pragma: no cover
    import simuladores
    from detectores import SuiteDetectores, catalogo_multivariado
    from evaluacion import evaluar_pu, jaccard_matrices

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs" / "simulados"

# Tamaño de cada dataset simulado. Valores deliberadamente PEQUEÑOS: el
# proyecto prioriza "pocos datos pero entendibles". Con ~300 ventanas normales
# + 7 tipos x 20 anomalías el umbral PU (percentil 90) y las tasas DR/AR salen
# estables, y los CSV resultantes pesan pocas decenas de KB.
TAM = {"it": 320, "ambiental": 300, "eeg": 260}
PARES_POR_TIPO = 3          # pares normal/anómala exportados al inspector
PUNTOS_MAX_FRONT = 180      # límite de puntos por canal en el JSON del front


def _submuestrear_display(arr_t, arr_x, max_pts=PUNTOS_MAX_FRONT):
    n = len(arr_t)
    if n <= max_pts:
        return arr_t, arr_x
    paso = int(np.ceil(n / max_pts))
    return arr_t[::paso], arr_x[::paso]


def procesar_contexto(nombre: str, n_ventanas: int, q: float = 0.90) -> None:
    t0 = time.time()
    print(f"\n=== Contexto {nombre} (N={n_ventanas}) ===")
    ds = simuladores.SIMULADORES[nombre](n_ventanas=n_ventanas)
    X, t = ds["X"], ds["t"]
    y_syn, y_tipo, base_idx = ds["y_syn"], ds["y_tipo"], ds["base_idx"]
    tipos, canales = ds["tipos"], ds["canales"]
    print(f"  X={X.shape}  anomalias={int(y_syn.sum())}  muestreo={ds['muestreo']['tipo']}")

    # ── Suite de detectores ───────────────────────────────────────────────────
    suite = SuiteDetectores(detectores=catalogo_multivariado((2, 3)),
                            depths=(2, 3), seed=42)
    mask_o = ~y_syn
    suite.ajustar(X[mask_o], t=t[mask_o])
    scores = suite.puntuar(X, t=t)
    print(f"  Detectores puntuados: {len(scores)}  ({time.time()-t0:.0f}s)")

    # ── Métricas PU + Jaccard ─────────────────────────────────────────────────
    df = evaluar_pu(scores, y_syn, y_tipo, tipos, nombre, q=q, out_dir=OUT)
    jaccard_matrices(scores, y_syn, y_tipo, tipos, nombre, q=q, out_dir=OUT)
    top = (df.drop_duplicates("Detector").nlargest(5, "DR_global")
           [["Detector", "DR_global"]])
    print("  Top-5 DR global:")
    for _, r in top.iterrows():
        print(f"    {r['Detector']:22s} {r['DR_global']:.3f}")

    # ── Metadatos del contexto ────────────────────────────────────────────────
    meta = {k: ds[k] for k in ("nombre", "titulo", "icono", "descripcion",
                               "canales", "unidades", "muestreo", "tipos",
                               "descrip_tipos")}
    meta["n_ventanas"] = int(mask_o.sum())
    meta["n_anomalias"] = int(y_syn.sum())
    meta["longitud"] = int(X.shape[1])
    meta["hiperparametros"] = {k: {a: (round(float(b), 5) if isinstance(b, (int, float)) else str(b))
                                   for a, b in v.items() if a != "seed"}
                               for k, v in suite.hiperparametros_efectivos.items()}
    (OUT / f"contexto_{nombre}.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # ── Pares normal/anómala para el inspector ────────────────────────────────
    pares, detecciones = [], []
    taus = {det: float(np.quantile(np.where(np.isnan(s), -1, s)[mask_o], q))
            for det, s in scores.items()}
    par_idx = 0
    idx_syn = np.where(y_syn)[0]
    for tipo in tipos:
        sel = idx_syn[y_tipo[idx_syn] == tipo][:PARES_POR_TIPO]
        for i_syn in sel:
            i_base = int(base_idx[i_syn])
            tt_a, xx_a = _submuestrear_display(t[i_syn], X[i_syn])
            tt_b, xx_b = _submuestrear_display(t[i_base], X[i_base])
            pares.append({
                "par_idx": par_idx, "tipo": tipo,
                "t": np.round(tt_a, 4).tolist(),
                "t_base": np.round(tt_b, 4).tolist(),
                "canales": {c: np.round(xx_a[:, j], 3).tolist()
                            for j, c in enumerate(canales)},
                "base": {c: np.round(xx_b[:, j], 3).tolist()
                         for j, c in enumerate(canales)},
            })
            for det, s in scores.items():
                sa = float(np.nan_to_num(s[i_syn], nan=-1))
                sb = float(np.nan_to_num(s[i_base], nan=-1))
                detecciones.append({
                    "detector": det, "par_idx": par_idx, "tipo": tipo,
                    "score_norm": round(sa, 4),
                    "score_base": round(sb, 4),
                    "tau_norm": round(taus[det], 4),
                    "detectado": bool(sa > taus[det]),
                })
            par_idx += 1

    (OUT / f"muestras_{nombre}.json").write_text(
        json.dumps({"pares": pares}, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(detecciones).to_csv(OUT / f"detecciones_{nombre}.csv", index=False)
    print(f"  Exportado: {par_idx} pares, {len(detecciones)} detecciones "
          f"({time.time()-t0:.0f}s total)")


def main():
    rapido = "--rapido" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    for nombre in ("it", "ambiental", "eeg"):
        n = TAM[nombre] // 4 if rapido else TAM[nombre]
        procesar_contexto(nombre, n)
    print("\nOK pipeline_anomalias: outputs/simulados/ completo")


if __name__ == "__main__":
    main()
