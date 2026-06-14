"""
backend/evaluacion.py
=====================
Protocolo de evaluación PU-Learning (Positive-Unlabeled) + similitud Jaccard.

Genérico respecto a la lista de tipos de anomalía: sirve igual para los 7
tipos AMI que para los 7 tipos de cada contexto simulado.

Protocolo
---------
Solo conocemos positivos (anomalías inyectadas); los "originales" son
no-etiquetados (pueden contener anomalías reales). Entonces:

  τ_M  = cuantil q (default 0.90) de los scores del detector M sobre originales
  AR_M = P(score_M > τ_M | original)      — tasa de alerta (≈ 1-q por diseño)
  DR_M = P(score_M > τ_M | sintético)     — tasa de detección (global y por tipo)

A igualdad de AR (mismo presupuesto de alertas), un DR mayor = mejor detector.

  J(M,M') = |A_M ∩ A_M'| / |A_M ∪ A_M'| sobre las anomalías sintéticas
  detectadas: mide si dos detectores "ven" las mismas anomalías
  (J alto = redundantes; J bajo con DR alto = complementarios → ensamble útil).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from .detectores import familia_de
except ImportError:  # pragma: no cover
    from detectores import familia_de


def evaluar_pu(
    scores: Dict[str, np.ndarray],
    y_syn: np.ndarray,
    y_tipo: np.ndarray,
    tipos: List[str],
    nombre_muestra: str,
    q: float = 0.90,
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Calcula DR/AR global y por tipo. Exporta metricas_pu_{nombre}.csv."""
    mask_o = ~y_syn          # máscara de series originales (no etiquetadas)
    mask_s = y_syn           # máscara de anomalías sintéticas (positivos)
    registros = []
    for det, sc in scores.items():
        # Los métodos de signatura puntúan solo un subconjunto: el resto es NaN.
        # Se reemplaza por -1 (nunca supera el umbral) para no sesgar el cuantil.
        sc = np.where(np.isnan(sc), -1.0, np.asarray(sc, float))
        # Umbral PU: percentil q de los scores SOLO sobre originales.
        tau = float(np.quantile(sc[mask_o], q))
        alertas = sc > tau
        AR = float(alertas[mask_o].mean())                       # falsos positivos (~1-q)
        DR = float(alertas[mask_s].mean()) if mask_s.any() else 0.0  # detección global
        for tipo in tipos:
            m = (y_tipo == tipo) & mask_s                        # sintéticos de este tipo
            registros.append({
                "Detector": det,
                "Familia": familia_de(det),
                "DR_global": round(DR, 4),
                "AR_global": round(AR, 4),
                "tau": round(tau, 4),
                "TipoAnomalia": tipo,
                "DR_tipo": round(float(alertas[m].mean()), 4) if m.any() else np.nan,
                "n_tipo": int(m.sum()),
                "n_orig": int(mask_o.sum()),
                "muestra": nombre_muestra,
            })
    df = pd.DataFrame(registros)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"metricas_pu_{nombre_muestra}.csv", index=False)
    return df


def jaccard_matrices(
    scores: Dict[str, np.ndarray],
    y_syn: np.ndarray,
    y_tipo: np.ndarray,
    tipos: List[str],
    nombre_muestra: str,
    q: float = 0.90,
    out_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """Matriz Jaccard general + una por tipo. Exporta jaccard_{tipo}_{nombre}.csv."""
    mask_o, mask_s = ~y_syn, y_syn
    nombres = list(scores.keys())
    # Para cada detector, vector booleano de qué anomalías marca como alerta.
    alerta_s, alerta_tipo = {}, {}
    for det, sc in scores.items():
        sc = np.where(np.isnan(sc), -1.0, np.asarray(sc, float))
        tau = float(np.quantile(sc[mask_o], q))     # mismo umbral PU que en evaluar_pu
        alerta_s[det] = sc[mask_s] > tau            # alertas sobre TODAS las anomalías
        alerta_tipo[det] = {t: sc[(y_tipo == t) & mask_s] > tau for t in tipos}

    # Índice de Jaccard entre dos conjuntos de alertas: |A∩B| / |A∪B|.
    def jac(a, b):
        u = float(np.logical_or(a, b).sum())
        return float(np.logical_and(a, b).sum()) / u if u > 0 else 0.0

    matrices: Dict[str, pd.DataFrame] = {}
    gen = pd.DataFrame(
        [[jac(alerta_s[m1], alerta_s[m2]) for m2 in nombres] for m1 in nombres],
        index=nombres, columns=nombres)
    matrices["general"] = gen
    for t in tipos:
        if sum(len(alerta_tipo[m][t]) for m in nombres) == 0:
            continue
        matrices[t] = pd.DataFrame(
            [[jac(alerta_tipo[m1][t], alerta_tipo[m2][t]) for m2 in nombres]
             for m1 in nombres],
            index=nombres, columns=nombres)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for clave, m in matrices.items():
            m.round(4).to_csv(out_dir / f"jaccard_{clave}_{nombre_muestra}.csv")
    return matrices
