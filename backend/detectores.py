"""
backend/detectores.py
=====================
Suite de detectores de anomalías sobre el registro extensible.

Mejoras respecto a framework_ami/detectores.py:
  * Cada detector es una clase registrada (añadir algoritmos = decorar una clase).
  * Hiperparámetros: automáticos (BIC, codo-kneedle, Scott, sqrt-N, ...) pero
    siempre seleccionables a mano vía `hiperparametros={"Detector": {...}}`.
  * Autoencoder REAL (red 2 capas entrenada con el motor autodiff propio),
    con fallback a proxy PCA si el motor no está disponible.
  * RobustPCA con una pasada IRLS de re-ponderación Huber (antes: PCA plano).
  * Conformal kNN vectorizado con searchsorted (antes: bucle Python).
  * Mahalanobis local de signaturas vectorizado (lotes de álgebra lineal).
  * Soporte de series MULTIVARIADAS (N, n, c) y muestreo irregular (pasando
    los tiempos t — quedan codificados en la signatura vía aumento temporal).

Familias:
  A-Estadistico : RobustZMAD, PCAT2Q, KDE, GMM
  B-Clustering  : KMeans, HDBSCAN, LOF, OPTICS
  C-ML          : IForest, OCSVM, Autoencoder
  D-Alternativo : RobustPCA, Conformal
  Signatures    : SigKernel_d*, SigMaHaKNN_d*_k*, SigConformancia, LogSigMaHa
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

from sklearn.cluster import KMeans as _KMeans, OPTICS as _OPTICS
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

try:  # paquete o script suelto
    from . import signaturas as sigmod
    from .registro import DetectorBase, REGISTRO, crear_detector, registrar
except ImportError:  # pragma: no cover
    import signaturas as sigmod
    from registro import DetectorBase, REGISTRO, crear_detector, registrar

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# HIPERPARÁMETROS AUTOMÁTICOS (optimalidad automatizada y justificada)
# ══════════════════════════════════════════════════════════════════════════════

def _hp_lof(X: np.ndarray) -> Dict:
    """k = max(5, sqrt(N)): estándar kNN; balance sesgo-varianza."""
    return {"k": max(5, int(math.sqrt(len(X))))}


def _hp_kde(X: np.ndarray) -> Dict:
    """Regla de Scott: h = N^{-1/(d+4)} — óptima AMISE para núcleo gaussiano."""
    N, d = X.shape
    return {"bandwidth": float(N ** (-1.0 / (d + 4)))}


def _hp_gmm(X: np.ndarray) -> Dict:
    """Número de componentes por mínimo BIC = -2logL + p·logN."""
    Xs = RobustScaler().fit_transform(X)
    if Xs.shape[1] > 5:
        Xs = PCA(n_components=5, random_state=0).fit_transform(Xs)
    max_k = min(10, max(2, len(X) // 20))
    mejor_k, mejor_bic = 2, np.inf
    for k in range(2, max_k + 1):
        try:
            g = GaussianMixture(n_components=k, covariance_type="full",
                                random_state=0, max_iter=100).fit(Xs)
            b = g.bic(Xs)
            if b < mejor_bic:
                mejor_bic, mejor_k = b, k
        except Exception:
            pass
    return {"n_components": mejor_k}


def _hp_kmeans(X: np.ndarray) -> Dict:
    """Codo por método kneedle: k que maximiza la distancia perpendicular
    de la curva WCSS(k) a la cuerda entre sus extremos."""
    Xs = RobustScaler().fit_transform(X)
    if Xs.shape[1] > 5:
        Xs = PCA(n_components=5, random_state=0).fit_transform(Xs)
    ks = list(range(2, min(15, max(3, len(X) // 10)) + 1))
    wcss = []
    for k in ks:
        km = _KMeans(n_clusters=k, random_state=0, n_init=4, max_iter=100).fit(Xs)
        wcss.append(km.inertia_)
    w = np.asarray(wcss, dtype=float)
    if len(w) < 3 or w.max() - w.min() < 1e-12:
        return {"n_clusters": ks[0]}
    x = (np.arange(len(ks)) - 0) / (len(ks) - 1)
    y = (w - w[-1]) / (w[0] - w[-1])
    dist = np.abs(y - (1 - x)) / math.sqrt(2)
    return {"n_clusters": ks[int(np.argmax(dist))]}


def _hp_hdbscan(X: np.ndarray) -> Dict:
    """min_cluster_size ≈ 2% del dataset (mínimo 5): evita fragmentación."""
    mcs = max(5, len(X) // 50)
    return {"min_cluster_size": mcs, "min_samples": max(3, mcs // 3)}


def _hp_ocsvm(X: np.ndarray) -> Dict:
    """nu = fracción de outliers estimada por Z-MAD global, acotada [0.02, 0.15]."""
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) + 1e-10
    z = np.abs(X - med).max(axis=1) / mad.max()
    return {"nu": float(np.clip((z > 3.5).mean(), 0.02, 0.15))}


def _hp_conformal(X: np.ndarray) -> Dict:
    """k = 2·log2(N): crecimiento logarítmico apropiado en alta dimensión."""
    return {"k": max(5, int(2 * math.log2(max(2, len(X)))))}


def _hp_iforest(X: np.ndarray) -> Dict:
    N = len(X)
    return {"max_samples": 256 if N < 2000 else (512 if N < 10000 else 1024)}


def _hp_knn_sig(X: np.ndarray) -> Dict:
    return {"k": max(5, int(math.sqrt(len(X)) / 2))}


# ══════════════════════════════════════════════════════════════════════════════
# FAMILIA A — ESTADÍSTICOS
# ══════════════════════════════════════════════════════════════════════════════

@registrar("RobustZMAD", "A-Estadistico", "aug",
           descripcion="Z-score robusto por mediana/MAD; alerta por el máximo entre features.")
class RobustZMAD(DetectorBase):
    def ajustar(self, X):
        self.med = np.median(X, axis=0)
        self.mad = np.median(np.abs(X - self.med), axis=0) + 1e-10

    def puntuar(self, X):
        return (np.abs(X - self.med) / self.mad).max(axis=1)


@registrar("PCAT2Q", "A-Estadistico", "pca",
           descripcion="Estadístico T² de Hotelling en el subespacio PCA.")
class PCAT2Q(DetectorBase):
    def ajustar(self, X):
        self.mu = X.mean(axis=0)
        cov = np.cov(X.T) + 1e-8 * np.eye(X.shape[1])
        self.cov_inv = np.linalg.inv(cov)

    def puntuar(self, X):
        diff = X - self.mu
        return np.einsum("ni,ij,nj->n", diff, self.cov_inv, diff)


@registrar("KDE", "A-Estadistico", "pca", auto_hp=_hp_kde,
           descripcion="Verosimilitud negativa bajo densidad kernel gaussiana (ancho de Scott).")
class KDE(DetectorBase):
    def ajustar(self, X):
        self.kde = KernelDensity(kernel="gaussian",
                                 bandwidth=self.hp.get("bandwidth", 0.5)).fit(X)

    def puntuar(self, X):
        return -self.kde.score_samples(X)


@registrar("GMM", "A-Estadistico", "pca", auto_hp=_hp_gmm,
           descripcion="Verosimilitud negativa bajo mezcla gaussiana (k por BIC).")
class GMM(DetectorBase):
    def ajustar(self, X):
        self.gmm = GaussianMixture(
            n_components=self.hp.get("n_components", 3),
            covariance_type="full", random_state=self.seed, max_iter=300,
        ).fit(X)

    def puntuar(self, X):
        return -self.gmm.score_samples(X)


# ══════════════════════════════════════════════════════════════════════════════
# FAMILIA B — CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

@registrar("KMeans", "B-Clustering", "pca", auto_hp=_hp_kmeans,
           descripcion="Distancia al centroide más cercano (k por codo-kneedle).")
class KMeansDet(DetectorBase):
    def ajustar(self, X):
        self.km = _KMeans(n_clusters=self.hp.get("n_clusters", 4),
                          random_state=self.seed, n_init=10).fit(X)

    def puntuar(self, X):
        return self.km.transform(X).min(axis=1)


@registrar("HDBSCAN", "B-Clustering", "pca", auto_hp=_hp_hdbscan,
           descripcion="Distancia media a los vecinos no-ruido del clustering jerárquico por densidad.")
class HDBSCANDet(DetectorBase):
    def ajustar(self, X):
        nucleo = X
        try:
            from sklearn.cluster import HDBSCAN as _HDB
            hdb = _HDB(min_cluster_size=self.hp.get("min_cluster_size", 8),
                       min_samples=self.hp.get("min_samples", 3)).fit(X)
            mask = hdb.labels_ >= 0
            if mask.sum() >= 5:
                nucleo = X[mask]
        except Exception:
            pass
        k = min(5, len(nucleo))
        self.nn = NearestNeighbors(n_neighbors=k).fit(nucleo)

    def puntuar(self, X):
        d, _ = self.nn.kneighbors(X)
        return d.mean(axis=1)


@registrar("LOF", "B-Clustering", "aug", auto_hp=_hp_lof,
           descripcion="Local Outlier Factor: densidad local relativa a los k vecinos.")
class LOFDet(DetectorBase):
    def ajustar(self, X):
        self.lof = LocalOutlierFactor(n_neighbors=min(self.hp.get("k", 20), len(X) - 1),
                                      novelty=True).fit(X)

    def puntuar(self, X):
        return -self.lof.score_samples(X)


@registrar("OPTICS", "B-Clustering", "pca",
           descripcion="Distancia mínima al núcleo OPTICS normalizada por alcanzabilidad máxima.")
class OPTICSDet(DetectorBase):
    def ajustar(self, X):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=min(3000, len(X)), replace=False)
        self.X_sub = X[idx]
        opt = _OPTICS(min_samples=max(3, self.hp.get("min_samples", 5))).fit(self.X_sub)
        reach = opt.reachability_.copy()
        finito = reach[np.isfinite(reach)]
        self.max_reach = float(finito.max()) if len(finito) else 1.0

    def puntuar(self, X):
        d = pairwise_distances(X, self.X_sub)
        return d.min(axis=1) / (self.max_reach + 1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# FAMILIA C — MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════

@registrar("IForest", "C-ML", "aug", auto_hp=_hp_iforest,
           descripcion="Isolation Forest: profundidad media de aislamiento en árboles aleatorios.")
class IForestDet(DetectorBase):
    def ajustar(self, X):
        self.ifo = IsolationForest(
            n_estimators=200, max_samples=self.hp.get("max_samples", 256),
            contamination=0.05, random_state=self.seed, n_jobs=-1,
        ).fit(X)

    def puntuar(self, X):
        return -self.ifo.score_samples(X)


@registrar("OCSVM", "C-ML", "aug", auto_hp=_hp_ocsvm,
           descripcion="One-Class SVM RBF: distancia al hiperplano que separa del origen en el RKHS.")
class OCSVMDet(DetectorBase):
    def ajustar(self, X):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=min(3000, len(X)), replace=False)
        self.oc = OneClassSVM(kernel="rbf", nu=self.hp.get("nu", 0.05),
                              gamma="scale").fit(X[idx])

    def puntuar(self, X):
        return -self.oc.score_samples(X)


@registrar("Autoencoder", "C-ML", "aug",
           descripcion="Autoencoder tanh F→64→8→64→F entrenado con autodiff propio; score = error de reconstrucción.")
class AutoencoderDet(DetectorBase):
    """Autoencoder real (motor autodiff NumPy). Fallback: proxy PCA-8."""

    def ajustar(self, X):
        self.sc = RobustScaler().fit(X)
        Xs = np.clip(self.sc.transform(X), -8, 8)
        try:
            try:
                from . import autodiff as ad
            except ImportError:
                import autodiff as ad
            self._ad = ad
            F = Xs.shape[1]
            oculto, cuello = min(64, F), min(8, F)
            rng = np.random.default_rng(self.seed)
            self.params = ad.inicializar_mlp([F, oculto, cuello, oculto, F], rng)
            epochs = int(self.hp.get("epochs", 150))
            lr = float(self.hp.get("lr", 5e-3))
            ad.entrenar_autoencoder(self.params, Xs, epochs=epochs, lr=lr,
                                    batch=min(256, len(Xs)), seed=self.seed)
            self.modo = "autodiff"
        except Exception:
            self.pca = PCA(n_components=min(8, Xs.shape[1] - 1),
                           random_state=self.seed).fit(Xs)
            self.modo = "pca"

    def puntuar(self, X):
        Xs = np.clip(self.sc.transform(X), -8, 8)
        if self.modo == "autodiff":
            Xr = self._ad.aplicar_mlp(self.params, Xs)
        else:
            Xr = self.pca.inverse_transform(self.pca.transform(Xs))
        return np.mean((Xs - Xr) ** 2, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# FAMILIA D — ALTERNATIVOS
# ══════════════════════════════════════════════════════════════════════════════

@registrar("RobustPCA", "D-Alternativo", "aug",
           descripcion="PCA con re-ponderación IRLS-Huber (resta influencia de outliers); score = residuo.")
class RobustPCADet(DetectorBase):
    def ajustar(self, X):
        k = min(5, X.shape[1] - 1)
        pca0 = PCA(n_components=k, random_state=self.seed).fit(X)
        rec = pca0.inverse_transform(pca0.transform(X))
        res = np.linalg.norm(X - rec, axis=1)
        c = np.median(res) * 2.5 + 1e-10          # umbral Huber
        w = np.where(res <= c, 1.0, c / res)       # pesos IRLS
        Xw = (X - X.mean(0)) * w[:, None] + X.mean(0)
        self.pca = PCA(n_components=k, random_state=self.seed).fit(Xw)

    def puntuar(self, X):
        rec = self.pca.inverse_transform(self.pca.transform(X))
        return np.linalg.norm(X - rec, axis=1)


@registrar("Conformal", "D-Alternativo", "aug", auto_hp=_hp_conformal,
           descripcion="p-valor conformal con no-conformidad = distancia al k-ésimo vecino.")
class ConformalDet(DetectorBase):
    def ajustar(self, X):
        self.sc = RobustScaler().fit(X)
        Xs = self.sc.transform(X)
        k = min(self.hp.get("k", 10), len(Xs) - 1)
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k + 1).fit(Xs)
        d_cal, _ = self.nn.kneighbors(Xs)
        self.alpha_cal = np.sort(d_cal[:, k])      # excluye el propio punto

    def puntuar(self, X):
        Xs = self.sc.transform(X)
        d, _ = self.nn.kneighbors(Xs, n_neighbors=self.k + 1)
        alpha = d[:, self.k - 1]                   # k-ésimo vecino real
        n = len(self.alpha_cal)
        n_mayores = n - np.searchsorted(self.alpha_cal, alpha, side="left")
        pval = (1.0 + n_mayores) / (n + 1.0)
        return 1.0 - pval


# ══════════════════════════════════════════════════════════════════════════════
# FAMILIA SIGNATURES
# ══════════════════════════════════════════════════════════════════════════════

class _SigMaHaKNN(DetectorBase):
    """Mahalanobis local en espacio de signaturas (PCA interna si D>60)."""

    def ajustar(self, X):
        self.red = None
        if X.shape[1] > 60:
            self.red = PCA(n_components=60, random_state=self.seed).fit(X)
            X = self.red.transform(X)
        self.base = X

    def puntuar(self, X):
        if self.red is not None:
            X = self.red.transform(X)
        return sigmod.score_mahalanobis_local(
            self.base, X, k=self.hp.get("k", 10),
            ridge=self.hp.get("ridge", 1e-4))


class _SigKernel(DetectorBase):
    """OCSVM con kernel de signaturas normalizado (precomputado)."""

    def ajustar(self, X):
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(X), size=min(3000, len(X)), replace=False)
        self.base = X[idx]

    def puntuar(self, X):
        return sigmod.score_sigkernel_ocsvm(self.base, X,
                                            nu=self.hp.get("nu", 0.05))


class _SigConformancia(DetectorBase):
    """Distancia media a k vecinos en espacio de signaturas estandarizado."""

    def ajustar(self, X):
        self.base = X

    def puntuar(self, X):
        return sigmod.score_conformancia(self.base, X, k=self.hp.get("k", 12))


def _registrar_detectores_sig(depths=(2, 3, 4)) -> None:
    for d in depths:
        vista = f"sig{d}"
        nombre_k = f"SigKernel_d{d}"
        if nombre_k not in REGISTRO:
            registrar(nombre_k, "Signatures", vista, nu=0.05,
                      descripcion=f"OCSVM con kernel de signaturas (nivel {d}).")(
                type(nombre_k, (_SigKernel,), {}))
        for kname, kval in (("k3", 3), ("k10", 10), ("k20", 20)):
            nom = f"SigMaHaKNN_d{d}_{kname}"
            if nom not in REGISTRO:
                registrar(nom, "Signatures", vista, k=kval,
                          descripcion=f"Mahalanobis local k={kval} sobre signaturas nivel {d}.")(
                    type(nom, (_SigMaHaKNN,), {}))
    if "SigConformancia_d2" not in REGISTRO:
        registrar("SigConformancia_d2", "Signatures", "sig2", auto_hp=_hp_knn_sig,
                  descripcion="Distancia de conformancia (media a k vecinos) sobre signaturas nivel 2.")(
            type("SigConformancia_d2", (_SigConformancia,), {}))
    if "LogSigMaHa_d2" not in REGISTRO:
        registrar("LogSigMaHa_d2", "Signatures", "logsig2", k=10,
                  descripcion="Mahalanobis local sobre log-signatura nivel 2 (incrementos + áreas de Lévy).")(
            type("LogSigMaHa_d2", (_SigMaHaKNN,), {}))


_registrar_detectores_sig()


# ══════════════════════════════════════════════════════════════════════════════
# CATÁLOGOS
# ══════════════════════════════════════════════════════════════════════════════

DETECTORES_CLASICOS = [
    "RobustZMAD", "PCAT2Q", "KDE", "GMM",
    "KMeans", "HDBSCAN", "LOF", "OPTICS",
    "IForest", "OCSVM", "Autoencoder", "RobustPCA", "Conformal",
]

# Catálogo AMI clásico (compatible con los CSV pre-calculados, 25 detectores)
DETECTORES_AMI = DETECTORES_CLASICOS + [
    f"Sig{base}_d{d}{sfx}"
    for d in (2, 3, 4)
    for base, sfx in (("Kernel", ""), ("MaHaKNN", "_k3"), ("MaHaKNN", "_k10"), ("MaHaKNN", "_k20"))
]


def catalogo_multivariado(depths=(2, 3)) -> List[str]:
    """Catálogo para series multivariadas: clásicos + signatures de los
    niveles pedidos + los dos detectores extra (demostración de extensión)."""
    dets = list(DETECTORES_CLASICOS)
    for d in depths:
        dets += [f"SigKernel_d{d}", f"SigMaHaKNN_d{d}_k3",
                 f"SigMaHaKNN_d{d}_k10", f"SigMaHaKNN_d{d}_k20"]
    dets += ["SigConformancia_d2", "LogSigMaHa_d2"]
    return dets


def familia_de(nombre: str) -> str:
    if nombre in REGISTRO:
        return REGISTRO[nombre].familia
    return "Signatures" if "Sig" in nombre else "Desconocido"


# ══════════════════════════════════════════════════════════════════════════════
# SUITE
# ══════════════════════════════════════════════════════════════════════════════

class SuiteDetectores:
    """
    Orquesta vistas de features + detectores registrados.

    Parámetros
    ----------
    detectores      : lista de nombres (default: catálogo según dimensión)
    hiperparametros : {"Detector": {hp: valor}} — overrides manuales
    pca_components  : k del subespacio PCA de la vista 'pca'
    depths          : niveles de signatura a construir
    seed            : reproducibilidad

    Uso
    ---
        suite = SuiteDetectores(depths=(2,3))
        suite.ajustar(X_train, t=t_train)        # X (N,n) ó (N,n,c)
        scores = suite.puntuar(X_eval, t=t_eval) # {nombre: scores en [0,1]}
    """

    def __init__(
        self,
        detectores: Optional[List[str]] = None,
        hiperparametros: Optional[Dict[str, Dict]] = None,
        pca_components: int = 10,
        depths: Tuple[int, ...] = (2, 3, 4),
        seed: int = 42,
    ):
        self.nombres = detectores
        self.overrides = hiperparametros or {}
        self.pca_k = pca_components
        self.depths = depths
        self.seed = seed
        self._dets: Dict[str, DetectorBase] = {}
        self._fit = {}

    # ── Vistas de features ────────────────────────────────────────────────────
    def _vistas(self, X: np.ndarray, t: Optional[np.ndarray], fit: bool) -> Dict[str, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        multic = X.ndim == 3
        N = X.shape[0]
        plano = X.reshape(N, -1)

        # magnitud por canal
        if multic:
            energia = np.log1p(np.abs(X).sum(axis=1))          # (N, c)
            pico = np.log1p(np.abs(X).max(axis=1))             # (N, c)
        else:
            energia = np.log1p(np.abs(X).sum(axis=1, keepdims=True))
            pico = np.log1p(np.abs(X).max(axis=1, keepdims=True))
        aug = np.hstack([plano, energia, pico])

        if fit:
            self._fit["sc_shape"] = RobustScaler().fit(plano)
            self._fit["sc_aug"] = RobustScaler().fit(aug)
            k = min(self.pca_k, N - 1, plano.shape[1])
            shape_sc = self._fit["sc_shape"].transform(plano)
            self._fit["pca"] = PCA(n_components=k, random_state=self.seed).fit(shape_sc)
            self._fit["sc_pca"] = RobustScaler().fit(self._fit["pca"].transform(shape_sc))

        shape_sc = self._fit["sc_shape"].transform(plano)
        vistas = {
            "shape": shape_sc,
            "aug": self._fit["sc_aug"].transform(aug),
            "pca": self._fit["sc_pca"].transform(self._fit["pca"].transform(shape_sc)),
        }

        # Signaturas: el aumento temporal codifica el muestreo (t irregular)
        caminos = sigmod.construir_caminos(X, t=t, aumento_tiempo=True,
                                           normalizar="ventana")
        for dpt in self.depths:
            vistas[f"sig{dpt}"] = sigmod.signaturas_lote(caminos, dpt)
        vistas["logsig2"] = sigmod.logsig_nivel2_lote(caminos)
        return vistas

    # ── API ───────────────────────────────────────────────────────────────────
    def ajustar(self, X_train: np.ndarray, t: Optional[np.ndarray] = None,
                verbose: bool = True) -> None:
        if self.nombres is None:
            self.nombres = (DETECTORES_AMI if np.asarray(X_train).ndim == 2
                            else catalogo_multivariado(self.depths))
        vistas = self._vistas(X_train, t, fit=True)
        self._dets = {}
        for nombre in self.nombres:
            espec = REGISTRO.get(nombre)
            if espec is None or espec.vista not in vistas:
                if verbose:
                    print(f"  [skip] {nombre} (vista no disponible)")
                continue
            Xv = vistas[espec.vista]
            det = crear_detector(nombre, Xv, seed=self.seed,
                                 **self.overrides.get(nombre, {}))
            det.ajustar(Xv)
            self._dets[nombre] = det
        if verbose:
            print(f"  Suite ajustada: {len(self._dets)} detectores")

    def puntuar(self, X_eval: np.ndarray, t: Optional[np.ndarray] = None,
                normalizar: bool = True) -> Dict[str, np.ndarray]:
        vistas = self._vistas(X_eval, t, fit=False)
        scores: Dict[str, np.ndarray] = {}
        for nombre, det in self._dets.items():
            espec = REGISTRO[nombre]
            s = np.asarray(det.puntuar(vistas[espec.vista]), dtype=float)
            if normalizar:
                lo, hi = np.nanpercentile(s, 1), np.nanpercentile(s, 99)
                s = np.clip((s - lo) / (hi - lo + 1e-10), 0.0, 1.0)
            scores[nombre] = s
        return scores

    @property
    def hiperparametros_efectivos(self) -> Dict[str, Dict]:
        return {n: dict(d.hp) for n, d in self._dets.items()}
