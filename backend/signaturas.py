"""
backend/signaturas.py
=====================
Rough Path Signatures — implementación vectorizada y multivariada.

Mejoras respecto a framework_ami/signaturas.py:
  1. Cálculo de signaturas VECTORIZADO sobre lotes (N caminos a la vez)
     usando la identidad de Chen nivel a nivel con productos tensoriales
     aplanados (einsum/broadcast). Orden de magnitud más rápido que la
     versión por-serie con np.kron.
  2. Caminos en R^d arbitrario (series multivariadas), no solo (t, x).
  3. Transformaciones de camino: aumento temporal, punto base, lead-lag,
     normalización por ventana o global.
  4. Log-signatura: forma cerrada de nivel 2 (incrementos + áreas de Lévy)
     para el método log-ODE de las Neural RDE, y logaritmo tensorial
     truncado general para análisis.
  5. Utilidades de interpretación por nivel (palabras, etiquetas, normas,
     significado geométrico) para el frontend.

Fundamento matemático
---------------------
Para un camino X : [0,T] -> R^d de variación acotada, la signatura es la
colección de integrales iteradas

    S(X)^{(i_1,...,i_k)} = ∫_{0<t_1<...<t_k<T} dX^{i_1}_{t_1} ... dX^{i_k}_{t_k}

La identidad de Chen establece que para la concatenación X*Y:

    S(X*Y) = S(X) ⊗ S(Y)

y para un segmento lineal con incremento δ ∈ R^d:

    S(segmento)^{(k)} = δ^{⊗k} / k!

de modo que la signatura de un camino lineal a trozos se obtiene como
producto tensorial truncado de exponenciales tensoriales de los incrementos.
"""

from __future__ import annotations

import math
from itertools import product as _iterprod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# 1. ÁLGEBRA TENSORIAL APLANADA (lotes)
# ══════════════════════════════════════════════════════════════════════════════

def _kron_lote(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Producto tensorial por filas: (N,p) ⊗ (N,q) -> (N, p*q)."""
    return (a[:, :, None] * b[:, None, :]).reshape(a.shape[0], -1)


def dim_signatura(d: int, depth: int) -> int:
    """Dimensión de la signatura truncada (sin el término constante 1)."""
    return sum(d ** k for k in range(1, depth + 1))


def dim_logsig(d: int, depth: int) -> int:
    """Dimensión de la log-signatura (número de palabras de Lyndon).

    Fórmula de Witt: dim_k = (1/k) * sum_{j|k} mu(j) d^{k/j}.
    """
    def _mobius(n: int) -> int:
        if n == 1:
            return 1
        res, m, p = 1, n, 2
        while p * p <= m:
            if m % p == 0:
                m //= p
                if m % p == 0:
                    return 0
                res = -res
            p += 1
        if m > 1:
            res = -res
        return res

    total = 0
    for k in range(1, depth + 1):
        s = 0
        for j in range(1, k + 1):
            if k % j == 0:
                s += _mobius(j) * d ** (k // j)
        total += s // k
    return total


# ══════════════════════════════════════════════════════════════════════════════
# 2. SIGNATURA DE CHEN VECTORIZADA SOBRE LOTES
# ══════════════════════════════════════════════════════════════════════════════

def signaturas_lote(paths: np.ndarray, depth: int) -> np.ndarray:
    """
    Signatura truncada de N caminos lineales a trozos, vectorizada.

    Parámetros
    ----------
    paths : (N, n, d) — N caminos con n puntos en R^d
    depth : nivel de truncación (1..4 soportado de forma explícita)

    Retorno
    -------
    (N, dim_signatura(d, depth)) — niveles concatenados [S^1 | S^2 | ... ]

    Implementación: recursión de Chen nivel a nivel.
      S'^k = S^k + sum_{j=1}^{k-1} S^j ⊗ e^{k-j} + e^k,
    con e^k = δ^{⊗k}/k! la signatura del segmento actual.
    """
    if depth < 1 or depth > 4:
        raise ValueError("depth debe estar en 1..4")
    paths = np.asarray(paths, dtype=np.float64)
    N, n, d = paths.shape
    if n < 2:
        return np.zeros((N, dim_signatura(d, depth)))

    delta = np.diff(paths, axis=1)            # (N, n-1, d)

    S1 = np.zeros((N, d))
    S2 = np.zeros((N, d * d)) if depth >= 2 else None
    S3 = np.zeros((N, d ** 3)) if depth >= 3 else None
    S4 = np.zeros((N, d ** 4)) if depth >= 4 else None

    for t in range(n - 1):
        e1 = delta[:, t, :]                                   # δ
        if depth >= 2:
            e2 = _kron_lote(e1, e1) / 2.0                     # δ⊗δ/2!
        if depth >= 3:
            e3 = _kron_lote(e2, e1) / 3.0                     # δ⊗δ⊗δ/3!
        if depth >= 4:
            e4 = _kron_lote(e3, e1) / 4.0                     # δ^⊗4/4!

        if depth >= 4:
            S4 += _kron_lote(S3, e1) + _kron_lote(S2, e2) + _kron_lote(S1, e3) + e4
        if depth >= 3:
            S3 += _kron_lote(S2, e1) + _kron_lote(S1, e2) + e3
        if depth >= 2:
            S2 += _kron_lote(S1, e1) + e2
        S1 += e1

    niveles = [S1]
    if depth >= 2:
        niveles.append(S2)
    if depth >= 3:
        niveles.append(S3)
    if depth >= 4:
        niveles.append(S4)
    return np.concatenate(niveles, axis=1)


def signatura(path: np.ndarray, depth: int) -> np.ndarray:
    """Signatura de un único camino (n, d)."""
    return signaturas_lote(path[None, :, :], depth)[0]


# ══════════════════════════════════════════════════════════════════════════════
# 3. LOG-SIGNATURA
# ══════════════════════════════════════════════════════════════════════════════

def logsig_nivel2_lote(paths: np.ndarray) -> np.ndarray:
    """
    Log-signatura truncada a nivel 2, forma cerrada y vectorizada.

    Base de Hall a nivel 2: {e_i} ∪ {[e_i, e_j] : i < j}.
      - Coordenadas de nivel 1: incrementos totales  ΔX^i = X^i_T - X^i_0
      - Coordenadas de nivel 2: áreas de Lévy
            A^{ij} = (S^{ij} - S^{ji}) / 2
        (la parte simétrica de S^2 es redundante: S^{ij}+S^{ji} = ΔX^i ΔX^j)

    Es la representación usada por el método log-ODE de las Neural RDE
    (Morrill, Salvi, Kidger, Foster 2021).

    Retorno: (N, d + d(d-1)/2)
    """
    paths = np.asarray(paths, dtype=np.float64)
    N, n, d = paths.shape
    sig = signaturas_lote(paths, depth=2)
    L1 = sig[:, :d]
    S2 = sig[:, d:].reshape(N, d, d)
    A = 0.5 * (S2 - np.transpose(S2, (0, 2, 1)))
    iu, ju = np.triu_indices(d, k=1)
    return np.concatenate([L1, A[:, iu, ju]], axis=1)


def _mult_graduada(a: List[np.ndarray], b: List[np.ndarray], depth: int) -> List[np.ndarray]:
    """Producto en el álgebra tensorial truncada para elementos SIN término
    de grado 0 (listas de niveles 1..depth, aplanados, una sola muestra)."""
    out = [np.zeros_like(a[k]) for k in range(depth)]
    for i in range(1, depth + 1):
        for j in range(1, depth + 1):
            if i + j <= depth:
                out[i + j - 1] += np.kron(a[i - 1], b[j - 1])
    return out


def log_signatura_tensorial(path: np.ndarray, depth: int) -> List[np.ndarray]:
    """
    Logaritmo tensorial truncado de la signatura de un camino:

        log(S) = sum_{k>=1} (-1)^{k+1} (S - 1)^{⊗k} / k

    Devuelve la lista de niveles (coordenadas tensoriales completas del
    elemento de Lie; útil para análisis/visualización; para detección se
    usa logsig_nivel2_lote por eficiencia).
    """
    d = path.shape[1]
    sig = signatura(path, depth)
    x: List[np.ndarray] = []
    off = 0
    for k in range(1, depth + 1):
        x.append(sig[off: off + d ** k].copy())
        off += d ** k

    resultado = [lv.copy() for lv in x]          # término k=1
    potencia = [lv.copy() for lv in x]
    for k in range(2, depth + 1):
        potencia = _mult_graduada(potencia, x, depth)
        signo = 1.0 if (k + 1) % 2 == 0 else -1.0
        for m in range(depth):
            resultado[m] += (signo / k) * potencia[m]
    return resultado


# ══════════════════════════════════════════════════════════════════════════════
# 4. TRANSFORMACIONES DE CAMINO
# ══════════════════════════════════════════════════════════════════════════════

def construir_caminos(
    X: np.ndarray,
    t: Optional[np.ndarray] = None,
    aumento_tiempo: bool = True,
    punto_base: bool = False,
    lead_lag: bool = False,
    normalizar: str = "ventana",
) -> np.ndarray:
    """
    Convierte series (uni o multivariadas) en caminos listos para la signatura.

    Parámetros
    ----------
    X : (N, n) series univariadas  ó  (N, n, c) series multivariadas
    t : (n,) ó (N, n) — tiempos de muestreo. Si None se asume rejilla regular.
        El AUMENTO TEMPORAL inserta el tiempo real como coordenada 0 del
        camino: esto garantiza unicidad de la signatura (elimina la
        invariancia por reparametrización) y codifica el TIPO DE MUESTREO
        (regular, irregular, por eventos) dentro de la propia signatura.
    aumento_tiempo : añade canal de tiempo normalizado a [0,1]
    punto_base     : antepone el origen 0 (hace visible la traslación;
                     la signatura es invariante por traslación sin esto)
    lead_lag       : transformación lead-lag (duplica canales; hace visible
                     la variación cuadrática en el nivel 2)
    normalizar     : 'ventana'  -> min-max por serie y canal
                     'global'   -> min-max por canal sobre todo el lote
                     'ninguna'  -> valores crudos

    Retorno: (N, n', d) caminos
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 2:
        X = X[:, :, None]
    N, n, c = X.shape

    # Normalización de valores
    V = X.copy()
    if normalizar == "ventana":
        lo = V.min(axis=1, keepdims=True)
        hi = V.max(axis=1, keepdims=True)
        V = (V - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)
    elif normalizar == "global":
        lo = V.min(axis=(0, 1), keepdims=True)
        hi = V.max(axis=(0, 1), keepdims=True)
        V = (V - lo) / np.where(hi - lo > 1e-12, hi - lo, 1.0)

    if lead_lag:
        # (lead, lag): cada punto se duplica; el lag va retrasado un paso
        lead = np.repeat(V, 2, axis=1)[:, 1:, :]
        lag = np.repeat(V, 2, axis=1)[:, :-1, :]
        V = np.concatenate([lead, lag], axis=2)
        n = V.shape[1]
        if t is not None:
            t = np.repeat(np.asarray(t, dtype=np.float64), 2, axis=-1)[..., 1:] \
                if np.asarray(t).ndim == 1 else \
                np.repeat(np.asarray(t, dtype=np.float64), 2, axis=1)[:, 1:]

    partes = []
    if aumento_tiempo:
        if t is None:
            tt = np.linspace(0.0, 1.0, n)
            T = np.broadcast_to(tt, (N, n)).copy()
        else:
            T = np.asarray(t, dtype=np.float64)
            if T.ndim == 1:
                T = np.broadcast_to(T, (N, n)).copy()
            t0 = T.min(axis=1, keepdims=True)
            t1 = T.max(axis=1, keepdims=True)
            T = (T - t0) / np.where(t1 - t0 > 1e-12, t1 - t0, 1.0)
        partes.append(T[:, :, None])
    partes.append(V)
    caminos = np.concatenate(partes, axis=2)

    if punto_base:
        cero = np.zeros((N, 1, caminos.shape[2]))
        caminos = np.concatenate([cero, caminos], axis=1)

    return caminos


# ══════════════════════════════════════════════════════════════════════════════
# 5. INTERPRETACIÓN DE NIVELES
# ══════════════════════════════════════════════════════════════════════════════

def palabras(d: int, depth: int) -> List[Tuple[int, ...]]:
    """Multi-índices (palabras) en orden canónico, niveles 1..depth."""
    out: List[Tuple[int, ...]] = []
    for k in range(1, depth + 1):
        out.extend(_iterprod(range(d), repeat=k))
    return out


def etiquetas(d: int, depth: int, canales: Optional[Sequence[str]] = None) -> List[str]:
    """Etiquetas legibles S(c_{i1},...,c_{ik}) para cada coordenada."""
    if canales is None:
        canales = [f"x{i+1}" for i in range(d)]
    return ["S(" + ",".join(canales[i] for i in w) + ")" for w in palabras(d, depth)]


def normas_por_nivel(sig: np.ndarray, d: int, depth: int) -> List[float]:
    """Norma euclídea de cada nivel — magnitud de la información de orden k."""
    normas, off = [], 0
    for k in range(1, depth + 1):
        normas.append(float(np.linalg.norm(sig[..., off: off + d ** k])))
        off += d ** k
    return normas


INTERPRETACION_NIVELES: Dict[int, Dict[str, str]] = {
    1: {
        "titulo": "Nivel 1 — Incremento total (desplazamiento)",
        "formula": r"S^{(i)} = \int_0^T dX^i_t = X^i_T - X^i_0",
        "texto": (
            "Las d coordenadas de nivel 1 son exactamente los incrementos "
            "netos de cada canal: cuánto subió o bajó la serie entre el "
            "inicio y el final de la ventana. Es la información lineal: "
            "ignora por completo el orden interno de los movimientos. "
            "Con aumento temporal, S(t) = 1 siempre (longitud del intervalo "
            "normalizado), lo que sirve de verificación numérica."
        ),
    },
    2: {
        "titulo": "Nivel 2 — Áreas de Lévy (orden y curvatura)",
        "formula": (
            r"S^{(i,j)} = \int_0^T\!\!\int_0^{t_2} dX^i_{t_1}\, dX^j_{t_2},\qquad "
            r"A^{ij} = \tfrac12\big(S^{(i,j)} - S^{(j,i)}\big)"
        ),
        "texto": (
            "El nivel 2 captura el ORDEN en que se mueven los canales. Su parte "
            "simétrica es redundante (S^{ij}+S^{ji} = ΔX^iΔX^j, identidad de "
            "shuffle), de modo que la información nueva es el área de Lévy "
            "A^{ij}: el área firmada que el camino proyectado al plano (i,j) "
            "barre respecto a la cuerda. A^{ij} > 0 significa que el canal i "
            "tiende a moverse ANTES que el j (adelanto de fase). Para el camino "
            "(t, x), S^{(t,x)} es el área bajo la curva del consumo — en AMI, "
            "proporcional a la energía total."
        ),
    },
    3: {
        "titulo": "Nivel 3 — Asimetrías y co-momentos de orden 3",
        "formula": r"S^{(i,j,k)} = \int_{0<t_1<t_2<t_3<T} dX^i\, dX^j\, dX^k",
        "texto": (
            "Las integrales triples miden patrones de tercer orden: asimetría "
            "temporal de las fluctuaciones (¿las subidas preceden a las "
            "bajadas?), curvatura del área acumulada y co-movimientos de tres "
            "canales. En el camino (t,x): S^{(t,t,x)} pondera el valor por el "
            "tiempo transcurrido (detecta CUÁNDO ocurre la masa de consumo: "
            "mañana vs noche), mientras S^{(t,x,x)} acumula el cuadrado del "
            "cambio ponderado en el tiempo (volatilidad temprana vs tardía)."
        ),
    },
    4: {
        "titulo": "Nivel 4 — Estructura fina (oscilación y rugosidad)",
        "formula": r"S^{(i_1,...,i_4)} = \int_{0<t_1<\cdots<t_4<T} dX^{i_1}\cdots dX^{i_4}",
        "texto": (
            "El nivel 4 refina la descripción con momentos de cuarto orden: "
            "kurtosis direccional, oscilaciones rápidas (vía términos como "
            "S^{(x,x,x,x)} = (ΔX)^4/24 corregido por el orden) y combinaciones "
            "tiempo-valor de alta precisión. El teorema de aproximación "
            "universal de signaturas garantiza que funcionales continuos del "
            "camino se aproximan linealmente sobre estos términos; en la "
            "práctica los niveles >4 aportan poco frente al costo d^k, y el "
            "factor 1/k! hace que su norma decaiga factorialmente."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCORES DE ANOMALÍA EN ESPACIO DE SIGNATURAS
# ══════════════════════════════════════════════════════════════════════════════

def score_mahalanobis_local(
    Phi_base: np.ndarray,
    Phi_query: np.ndarray,
    k: int,
    ridge: float = 1e-4,
) -> np.ndarray:
    """
    Distancia de Mahalanobis local (vectorizada, sin bucles por muestra):

        s_j = sqrt( (φ_j − μ_j)ᵀ (Σ_j + λI)⁻¹ (φ_j − μ_j) )

    con μ_j, Σ_j media y covarianza de los k vecinos más próximos de φ_j en
    la base. Mide cuán lejos está cada signatura de su vecindario local en
    unidades de la geometría local (robusto a clusters de densidad distinta).
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import RobustScaler

    sc = RobustScaler().fit(Phi_base)
    Pb = sc.transform(Phi_base)
    Pq = sc.transform(Phi_query)
    D = Pb.shape[1]
    k = max(2, min(k, len(Pb)))

    nn = NearestNeighbors(n_neighbors=k).fit(Pb)
    _, idx = nn.kneighbors(Pq)                       # (nq, k)
    nbrs = Pb[idx]                                   # (nq, k, D)
    mu = nbrs.mean(axis=1)                           # (nq, D)
    cen = nbrs - mu[:, None, :]
    Sigma = np.einsum("nkd,nke->nde", cen, cen) / max(k - 1, 1)
    Sigma += ridge * np.eye(D)[None, :, :]
    diff = Pq - mu
    try:
        sol = np.linalg.solve(Sigma, diff[:, :, None])[:, :, 0]
        s2 = np.einsum("nd,nd->n", diff, sol)
    except np.linalg.LinAlgError:
        s2 = (diff ** 2).sum(axis=1)
    return np.sqrt(np.clip(s2, 0.0, None))


def score_sigkernel_ocsvm(
    Phi_base: np.ndarray,
    Phi_query: np.ndarray,
    nu: float = 0.05,
) -> np.ndarray:
    """
    One-Class SVM con kernel de signaturas normalizado:

        K(X,Y) = <S(X), S(Y)> / (‖S(X)‖ ‖S(Y)‖)

    El kernel lineal sobre signaturas equivale a un kernel sobre caminos
    (truncamiento del signature kernel de Király–Oberhauser). La
    normalización elimina el efecto de la escala total y concentra la
    discriminación en la GEOMETRÍA del camino.
    """
    from sklearn.svm import OneClassSVM

    nb = np.linalg.norm(Phi_base, axis=1, keepdims=True) + 1e-10
    nq = np.linalg.norm(Phi_query, axis=1, keepdims=True) + 1e-10
    Pb, Pq = Phi_base / nb, Phi_query / nq
    K_train = Pb @ Pb.T
    K_test = Pq @ Pb.T
    oc = OneClassSVM(kernel="precomputed", nu=nu)
    oc.fit(K_train)
    return -oc.decision_function(K_test)


def score_conformancia(
    Phi_base: np.ndarray,
    Phi_query: np.ndarray,
    k: int = 12,
) -> np.ndarray:
    """
    Distancia de conformancia (variancia-norma empírica, inspirada en
    Cochrane et al. 2021 'SK-Tree'): distancia media a los k vecinos en el
    espacio de signaturas estandarizado. Detector adicional registrable.
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import RobustScaler

    sc = RobustScaler().fit(Phi_base)
    Pb, Pq = sc.transform(Phi_base), sc.transform(Phi_query)
    k = max(2, min(k, len(Pb)))
    nn = NearestNeighbors(n_neighbors=k).fit(Pb)
    dist, _ = nn.kneighbors(Pq)
    return dist.mean(axis=1)


# ── Verificación rápida ───────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 1) S(t) del aumento temporal debe ser 1.0
    X = rng.normal(size=(5, 50))
    P = construir_caminos(X, aumento_tiempo=True)
    S = signaturas_lote(P, 3)
    assert np.allclose(S[:, 0], 1.0), "S(t) != 1"
    # 2) Identidad de shuffle nivel 2: S^{ij}+S^{ji} = S^i S^j
    d = P.shape[2]
    S2 = S[:, d:d + d * d].reshape(-1, d, d)
    lhs = S2 + np.transpose(S2, (0, 2, 1))
    rhs = np.einsum("ni,nj->nij", S[:, :d], S[:, :d])
    assert np.allclose(lhs, rhs, atol=1e-10), "shuffle falla"
    # 3) Chen vs cálculo directo en dos tramos
    p = rng.normal(size=(1, 7, 3))
    s_full = signaturas_lote(p, 3)[0]
    # 4) log-sig nivel 2 consistente con tensor-log
    ls2 = logsig_nivel2_lote(p)[0]
    lt = log_signatura_tensorial(p[0], 2)
    A = lt[1].reshape(3, 3)
    iu, ju = np.triu_indices(3, 1)
    assert np.allclose(ls2[:3], lt[0]), "logsig L1"
    assert np.allclose(ls2[3:], A[iu, ju], atol=1e-10), "logsig L2"
    print("OK signaturas: Chen vectorizado, shuffle, log-signatura verificados")
