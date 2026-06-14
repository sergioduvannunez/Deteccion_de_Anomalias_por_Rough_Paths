"""
backend/registro.py
===================
Registro extensible de detectores de anomalías (patrón Registry).

Añadir un detector nuevo = definir una clase con `ajustar`/`puntuar` y
decorarla. Queda automáticamente disponible en la SuiteDetectores, en los
pipelines y (vía metadatos) en el frontend.

Ejemplo
-------
    from backend.registro import registrar, DetectorBase

    @registrar(
        nombre="MiDetector",
        familia="C-ML",
        vista="pca",
        descripcion="Distancia coseno al centroide global.",
    )
    class MiDetector(DetectorBase):
        def ajustar(self, X):
            self.mu = X.mean(axis=0)
        def puntuar(self, X):
            num = (X * self.mu).sum(1)
            den = np.linalg.norm(X, axis=1) * np.linalg.norm(self.mu) + 1e-12
            return 1.0 - num / den

Vistas de features disponibles (las construye la SuiteDetectores una vez):
    'shape'   : valores re-escalados (RobustScaler) — forma de la serie
    'aug'     : shape + log-energía + log-pico por canal — forma + magnitud
    'pca'     : proyección PCA-k de shape — espacio compacto para densidad
    'sig2..4' : signaturas (camino con aumento temporal) — geometría del camino
    'logsig2' : log-signatura nivel 2 (incrementos + áreas de Lévy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np


class DetectorBase:
    """Interfaz mínima de un detector no supervisado.

    El ciclo de vida es:  det = Clase(**hiperparametros); det.ajustar(X_train);
    s = det.puntuar(X_eval)  con s creciente en anomalía.
    """

    def __init__(self, **hp):
        self.hp = hp
        self.seed = hp.get("seed", 42)

    def ajustar(self, X: np.ndarray) -> None:  # pragma: no cover - interfaz
        raise NotImplementedError

    def puntuar(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


@dataclass
class EspecDetector:
    nombre: str
    familia: str
    vista: str
    fabrica: Callable[..., DetectorBase]
    descripcion: str = ""
    hiperparametros: Dict = field(default_factory=dict)   # defaults declarados
    auto_hp: Optional[Callable[[np.ndarray], Dict]] = None  # HP automáticos f(X)


REGISTRO: Dict[str, EspecDetector] = {}


def registrar(
    nombre: str,
    familia: str,
    vista: str,
    descripcion: str = "",
    auto_hp: Optional[Callable[[np.ndarray], Dict]] = None,
    **defaults,
):
    """Decorador que inscribe la clase/fábrica en el registro global."""

    def _wrap(cls):
        REGISTRO[nombre] = EspecDetector(
            nombre=nombre,
            familia=familia,
            vista=vista,
            fabrica=cls,
            descripcion=descripcion,
            hiperparametros=dict(defaults),
            auto_hp=auto_hp,
        )
        return cls

    return _wrap


def detectores_registrados(familias: Optional[List[str]] = None) -> List[str]:
    """Nombres registrados, opcionalmente filtrados por familia."""
    if familias is None:
        return list(REGISTRO.keys())
    return [n for n, e in REGISTRO.items() if e.familia in familias]


def crear_detector(nombre: str, X_train: Optional[np.ndarray] = None, **overrides) -> DetectorBase:
    """
    Instancia un detector con la política de hiperparámetros:
      defaults declarados  <  auto_hp(X_train)  <  overrides del usuario.
    (optimalidad automatizada, pero siempre seleccionable a mano)
    """
    espec = REGISTRO[nombre]
    hp = dict(espec.hiperparametros)              # 1) defaults declarados en @registrar
    if espec.auto_hp is not None and X_train is not None:
        hp.update(espec.auto_hp(X_train))         # 2) hiperparámetros automáticos f(datos)
    hp.update(overrides)                          # 3) overrides manuales (máxima prioridad)
    return espec.fabrica(**hp)                    # instancia la clase del detector
