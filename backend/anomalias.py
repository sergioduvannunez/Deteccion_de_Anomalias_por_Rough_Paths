"""
framework_ami/anomalias.py
==========================
Inyección de 7 tipos de anomalías sobre series semanales brutas (168 h).
Cada inyección se aplica sobre una COPIA de los datos originales.

Tipos:
  1. PartialBypass    — sub-registro proporcional (magnitud ×α, forma igual)
  2. Smoothing        — aplanamiento del perfil semanal
  3. FlipSchedule     — inversión día/noche (roll 12h en cada día)
  4. SuddenDrop       — caída brusca de consumo (magnitud ×γ, forma igual)
  5. SyntheticNoise   — ruido gaussiano sobre la forma
  6. FlatLine         — tramo de horas consecutivas a valor constante [AMI: avería medidor]
  7. SpikeEvent       — pico abrupto de pocas horas [AMI: carga no regular / evento]

Todos los tipos se aplican sobre el array crudo (168 valores reales en Wh/kWh).
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ── Tipos de anomalía disponibles ─────────────────────────────────────────────
TIPOS_ANOMALIA = [
    "PartialBypass",
    "Smoothing",
    "FlipSchedule",
    "SuddenDrop",
    "SyntheticNoise",
    "FlatLine",
    "SpikeEvent",
]


class InyectorAnomalias:
    """
    Inyecta anomalías sobre un DataFrame de series semanales.

    El DataFrame de entrada debe tener columnas h0..h167 (168 valores por fila).
    Se devuelve un DataFrame adicional con las filas sintéticas, manteniendo
    las columnas originales más 'is_synthetic' y 'anomaly_type'.

    Parámetros
    ----------
    fraccion : float
        Fracción de la muestra original a convertir en sintéticos (default 0.05)
    seed : int
        Semilla aleatoria
    """

    def __init__(self, fraccion: float = 0.05, seed: int = 42):
        self.fraccion = fraccion
        self.rng = np.random.default_rng(seed)
        self._H = [f"h{i}" for i in range(168)]

    # ── API principal ─────────────────────────────────────────────────────────
    def inyectar(
        self,
        df_orig: pd.DataFrame,
        tipos: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Genera anomalías sobre copias de df_orig.

        Devuelve df_orig (con is_synthetic=False, anomaly_type='original')
        concatenado con todas las filas sintéticas.
        """
        if tipos is None:
            tipos = TIPOS_ANOMALIA

        df_orig = df_orig.copy()
        df_orig["is_synthetic"] = False
        df_orig["anomaly_type"] = "original"

        n_total    = max(len(tipos), int(len(df_orig) * self.fraccion))
        n_por_tipo = max(1, n_total // len(tipos))

        sinteticos = []
        for atype in tipos:
            idx_base = self.rng.choice(len(df_orig), size=n_por_tipo, replace=True)
            for j, idx in enumerate(idx_base):
                fila_base = df_orig.iloc[idx].copy()
                arr = fila_base[self._H].values.astype(float)
                arr_mod, params = self._aplicar(atype, arr)

                nueva_fila = fila_base.copy()
                for k, col in enumerate(self._H):
                    nueva_fila[col] = arr_mod[k]
                nueva_fila["is_synthetic"] = True
                nueva_fila["anomaly_type"] = atype
                nueva_fila["_params"]      = str(params)
                if "ID" in nueva_fila.index:
                    nueva_fila["ID"] = f"SYN_{atype}_{j}"
                sinteticos.append(nueva_fila)

        df_syn = pd.DataFrame(sinteticos)
        df_eval = pd.concat([df_orig, df_syn], ignore_index=True)
        df_eval["is_synthetic"] = df_eval["is_synthetic"].astype(bool)
        return df_eval

    def _aplicar(
        self, atype: str, arr: np.ndarray
    ) -> Tuple[np.ndarray, dict]:
        """Despacha al método de anomalía correcto."""
        funcs = {
            "PartialBypass":  self._partial_bypass,
            "Smoothing":      self._smoothing,
            "FlipSchedule":   self._flip_schedule,
            "SuddenDrop":     self._sudden_drop,
            "SyntheticNoise": self._synthetic_noise,
            "FlatLine":       self._flat_line,
            "SpikeEvent":     self._spike_event,
        }
        if atype not in funcs:
            raise ValueError(f"Tipo desconocido: {atype}")
        return funcs[atype](arr)

    # ── Implementaciones ──────────────────────────────────────────────────────

    def _partial_bypass(self, arr: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Sub-registro proporcional: toda la energía registrada se reduce por
        factor α ∈ [0.30, 0.60].  Simula bypass de carga (fraude).
        La FORMA del perfil semanal no cambia; solo la magnitud.
        Detectable únicamente mediante features de magnitud absoluta.
        """
        alpha = float(self.rng.uniform(0.30, 0.60))
        return arr * alpha, {"alpha": round(alpha, 3)}

    def _smoothing(self, arr: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Aplanamiento del perfil: cada valor se mezcla con la media semanal
        ponderada por β ∈ [0.10, 0.35].  Simula manipulación de datos para
        ocultar patrones de consumo inusuales.
        Altera la forma de la serie; detectable en espacio de forma.
        """
        beta = float(self.rng.uniform(0.10, 0.35))
        mu   = float(arr.mean())
        arr_mod = beta * arr + (1.0 - beta) * mu
        return arr_mod, {"beta": round(beta, 3)}

    def _flip_schedule(self, arr: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Inversión día/noche: desplaza cada día 12 horas (roll 12 posiciones).
        Se aplica día a día para preservar la continuidad entre días.
        Simula error de zona horaria en medidor o actividad industrial nocturna.
        Es la anomalía más extrema y fácil de detectar.
        """
        arr_mod = arr.copy()
        for d in range(7):
            ini = d * 24
            fin = ini + 24
            arr_mod[ini:fin] = np.roll(arr[ini:fin], 12)
        return arr_mod, {"shift": 12}

    def _sudden_drop(self, arr: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Caída brusca severa: toda la energía cae a γ ∈ [0.05, 0.15] del nivel
        habitual.  Simula corte de suministro, fuga de corriente o fraude extremo.
        Idéntico a PartialBypass pero con factor γ << α.
        """
        gamma = float(self.rng.uniform(0.05, 0.15))
        return arr * gamma, {"gamma": round(gamma, 3)}

    def _synthetic_noise(self, arr: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Ruido gaussiano sobre la forma: cada hora recibe perturbación
        ε_h ~ N(0, σ·δ) donde σ = std de la serie semanal y δ ∈ [0.30, 0.60].
        Los valores negativos se recortan a 0.
        Simula errores de medición o interferencias electromagnéticas.
        """
        delta = float(self.rng.uniform(0.30, 0.60))
        sigma = float(arr.std()) + 1e-8
        noise = self.rng.normal(0.0, sigma * delta, len(arr))
        arr_mod = np.clip(arr + noise, 0.0, None)
        return arr_mod, {"delta": round(delta, 3)}

    def _flat_line(self, arr: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Línea plana: un tramo continuo de 12–48 horas queda fijo en un
        valor constante cercano a 0 (simula avería del medidor o comunicación).
        Inicio aleatorio, duración aleatoria ∈ [12, 48] horas.
        Contexto AMI: frecuente en medidores con fallo de comunicación donde
        el valor de la última lectura se repite hasta reconectar.
        """
        n          = len(arr)
        duracion   = int(self.rng.integers(12, 49))
        inicio     = int(self.rng.integers(0, n - duracion))
        val_flat   = float(arr.mean()) * float(self.rng.uniform(0.0, 0.10))
        arr_mod    = arr.copy()
        arr_mod[inicio: inicio + duracion] = val_flat
        return arr_mod, {"inicio": inicio, "duracion": duracion, "val": round(val_flat, 2)}

    def _spike_event(self, arr: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Pico abrupto: un bloque de 1–6 horas consecutivas recibe un factor
        de amplificación ×k donde k ∈ [4, 12] del valor promedio del día.
        Inicio aleatorio (preferentemente en un día laboral).
        Contexto AMI: carga masiva repentina (aire acondicionado industrial,
        soldadora, recarga de vehículo eléctrico en vivienda sin infraestructura).
        """
        n_dias   = 7
        dia      = int(self.rng.integers(0, n_dias))
        ini_dia  = dia * 24
        duracion = int(self.rng.integers(1, 7))
        hora     = int(self.rng.integers(0, 24 - duracion))
        factor   = float(self.rng.uniform(4.0, 12.0))
        arr_mod  = arr.copy()
        bloque   = slice(ini_dia + hora, ini_dia + hora + duracion)
        arr_mod[bloque] = arr_mod[bloque] * factor
        return arr_mod, {
            "dia": dia, "hora_inicio": hora,
            "duracion": duracion, "factor": round(factor, 2)
        }

    # ── Utilidades ────────────────────────────────────────────────────────────
    @property
    def H_COLS(self) -> List[str]:
        return self._H

    @staticmethod
    def descripcion_tipos() -> pd.DataFrame:
        """DataFrame con descripción teórica de cada tipo."""
        rows = [
            ("PartialBypass",  "Magnitud",  "α ∈ [0.30,0.60]",
             "Bypass parcial: energía registrada × α. Forma invariante. "
             "Detectable solo con features de magnitud (valor absoluto)."),
            ("Smoothing",      "Forma",     "β ∈ [0.10,0.35]",
             "Mezcla con media: β·x + (1-β)·μ. Reduce picos y valles. "
             "Simula manipulación de datos para ocultar patrones."),
            ("FlipSchedule",   "Forma",     "shift=12h/día",
             "Roll 12h por día: invierte ciclo día/noche. Anomalía extrema, "
             "detectada por casi todos los métodos."),
            ("SuddenDrop",     "Magnitud",  "γ ∈ [0.05,0.15]",
             "Caída brusca: energía × γ ≈ 5-15% del nivel normal. "
             "Más extremo que PartialBypass; solo detectable por magnitud."),
            ("SyntheticNoise", "Forma",     "δ ∈ [0.30,0.60]",
             "Ruido gaussiano N(0, σ·δ) por hora. Perturba la forma semanal. "
             "Detectable por métodos de densidad local (LOF, Signatures)."),
            ("FlatLine",       "Forma+Mag", "dur ∈ [12,48]h, val≈0",
             "Tramo de horas consecutivas fijado en valor casi nulo. "
             "Simula avería o pérdida de comunicación del medidor AMI."),
            ("SpikeEvent",     "Forma+Mag", "factor ∈ [4,12]×, dur ∈ [1,6]h",
             "Pico abrupto de pocas horas. Simula carga no regular: VE, "
             "soldadora, AC industrial. Anomalía detectada por IForest y LOF."),
        ]
        return pd.DataFrame(rows, columns=["Tipo","Espacio","Params","Descripcion"])
