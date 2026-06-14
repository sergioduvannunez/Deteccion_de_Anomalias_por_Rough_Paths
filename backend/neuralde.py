"""
backend/neuralde.py
===================
Ecuaciones Diferenciales Neuronales: la progresión completa

    RNN  →  Neural ODE  →  Neural CDE  →  Neural RDE (rugosa)

construida desde cero sobre el motor autodiff propio (backend/autodiff.py).

Marco teórico (resumen riguroso; el frontend desarrolla cada paso)
------------------------------------------------------------------
1. RNN (discreta):       h_{k+1} = φ(W_h h_k + W_x x_{k+1} + b)
   Una red recurrente es un sistema dinámico DISCRETO controlado por datos.

2. Neural ODE (Chen et al. 2018): límite continuo de las conexiones
   residuales h_{k+1} = h_k + f(h_k):
        dh/dt = f_θ(h(t)),   h(0) = enc(x_0)
   La profundidad pasa a ser tiempo continuo. Limitación clave: la dinámica
   NO incorpora datos que llegan después de t=0 (la solución queda
   determinada por la condición inicial — teorema de Picard-Lindelöf).

3. Neural CDE (Kidger et al. 2020): ecuación diferencial CONTROLADA
        dz_t = f_θ(z_t) dX_t      (integral de Riemann–Stieltjes)
   donde X es la interpolación continua de las observaciones (con canal de
   tiempo). El dato actúa como CONTROL de la dinámica en todo t: es el
   análogo continuo exacto de la RNN, y maneja muestreo irregular de forma
   natural. f_θ(z) toma valores en matrices H×(d) (campo de "respuesta").

4. Neural RDE (Morrill et al. 2021): para series largas/rugosas se aplica
   el MÉTODO LOG-ODE de la teoría de caminos rugosos de Lyons: en cada
   ventana [t_j, t_{j+1}] el control se resume por su log-signatura
   truncada (incrementos + áreas de Lévy a nivel 2):
        z_{j+1} = z_j + g_θ(z_j) · logsig_{[t_j,t_{j+1}]}(X)
   Esto discretiza la CDE respetando la geometría de orden superior del
   camino (no solo sus incrementos), con muchos menos pasos de integración.

Todos los modelos se entrenan con Adam por descenso de gradiente a través
del integrador (discretizar-luego-optimizar).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from . import autodiff as ad
    from . import signaturas as sigmod
except ImportError:  # pragma: no cover
    import autodiff as ad
    import signaturas as sigmod

T = ad.T


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES COMUNES
# ══════════════════════════════════════════════════════════════════════════════

def _xavier(rng, a, b):
    lim = np.sqrt(6.0 / (a + b))
    return T(rng.uniform(-lim, lim, size=(a, b)))


class ModeloBase:
    """Interfaz: clasificadores de series (B, n, c) con tiempos (B, n)."""

    nombre = "base"
    familia = "—"

    def __init__(self, c_in: int, n_clases: int, hidden: int = 24, seed: int = 0):
        self.c_in, self.n_clases, self.H = c_in, n_clases, hidden
        self.rng = np.random.default_rng(seed)
        self.params: List[T] = []

    def forward(self, X: np.ndarray, t: np.ndarray) -> T:  # logits (B, C)
        raise NotImplementedError

    def predecir(self, X, t):
        return np.argmax(self.forward(X, t).data, axis=1)

    def n_parametros(self) -> int:
        return int(sum(p.data.size for p in self.params))


# ══════════════════════════════════════════════════════════════════════════════
# 1. RNN VANILLA — sistema dinámico discreto
# ══════════════════════════════════════════════════════════════════════════════

class ModeloRNN(ModeloBase):
    """h_{k+1} = tanh(W_h h_k + W_x [x_k, Δt_k] + b).

    El Δt como entrada extra es el remiendo clásico para muestreo irregular:
    la red debe APRENDER el efecto del tiempo, no lo tiene en su estructura.
    """

    nombre = "RNN"
    familia = "Recurrente discreta"

    def __init__(self, c_in, n_clases, hidden=24, seed=0):
        super().__init__(c_in, n_clases, hidden, seed)
        rng, H = self.rng, hidden
        self.Wx = _xavier(rng, c_in + 1, H)        # +1: Δt
        self.Wh = _xavier(rng, H, H)
        self.b = T(np.zeros(H))
        self.Wo = _xavier(rng, H, n_clases)
        self.bo = T(np.zeros(n_clases))
        self.params = [self.Wx, self.Wh, self.b, self.Wo, self.bo]

    def _entradas(self, X, t):
        dt = np.diff(t, axis=1, prepend=t[:, :1])
        return np.concatenate([X, dt[:, :, None]], axis=2)  # (B, n, c+1)

    def forward(self, X, t):
        U = self._entradas(X, t)
        B, n, _ = U.shape
        h = T(np.zeros((B, self.H)))
        for k in range(n):
            h = ad.tanh(T(U[:, k, :]) @ self.Wx + h @ self.Wh + self.b)
        return h @ self.Wo + self.bo

    def trayectoria(self, x, t):
        U = self._entradas(x[None], t[None])[0]
        h = np.zeros(self.H)
        hs = [h.copy()]
        for k in range(len(U)):
            h = np.tanh(U[k] @ self.Wx.data + h @ self.Wh.data + self.b.data)
            hs.append(h.copy())
        return np.array(hs)


# ══════════════════════════════════════════════════════════════════════════════
# 2. GRU — recurrencia con compuertas (control del flujo de gradiente)
# ══════════════════════════════════════════════════════════════════════════════

class ModeloGRU(ModeloBase):
    """z = σ(·), r = σ(·), ĥ = tanh(W [x, r⊙h]), h ← (1-z)⊙h + z⊙ĥ.

    Las compuertas convierten la recurrencia en una interpolación convexa
    paso a paso — precursora discreta del flujo continuo de las ODE.
    """

    nombre = "GRU"
    familia = "Recurrente discreta"

    def __init__(self, c_in, n_clases, hidden=24, seed=0):
        super().__init__(c_in, n_clases, hidden, seed)
        rng, H = self.rng, hidden
        cin = c_in + 1
        self.Wz = _xavier(rng, cin + H, H); self.bz = T(np.zeros(H))
        self.Wr = _xavier(rng, cin + H, H); self.br = T(np.zeros(H))
        self.Wn = _xavier(rng, cin + H, H); self.bn = T(np.zeros(H))
        self.Wo = _xavier(rng, H, n_clases); self.bo = T(np.zeros(n_clases))
        self.params = [self.Wz, self.bz, self.Wr, self.br,
                       self.Wn, self.bn, self.Wo, self.bo]

    def forward(self, X, t):
        dt = np.diff(t, axis=1, prepend=t[:, :1])
        U = np.concatenate([X, dt[:, :, None]], axis=2)
        B, n, _ = U.shape
        h = T(np.zeros((B, self.H)))
        for k in range(n):
            xk = T(U[:, k, :])
            xh = ad.concat([xk, h], axis=1)
            z = ad.sigmoid(xh @ self.Wz + self.bz)
            r = ad.sigmoid(xh @ self.Wr + self.br)
            xrh = ad.concat([xk, r * h], axis=1)
            hn = ad.tanh(xrh @ self.Wn + self.bn)
            h = (T(np.ones(1)) - z) * h + z * hn
        return h @ self.Wo + self.bo


# ══════════════════════════════════════════════════════════════════════════════
# 3. NEURAL ODE — profundidad continua, dato solo en h(0)
# ══════════════════════════════════════════════════════════════════════════════

class ModeloNODE(ModeloBase):
    """dh/dt = f_θ(h),  h(0) = W_e x_0 + b_e,  clasifica h(1).

    Integrador Runge-Kutta 4 con M pasos fijos; el gradiente se propaga a
    través del integrador (discretizar-luego-optimizar; el método adjunto
    continuo de Chen et al. es la alternativa O(1)-memoria).
    Limitación estructural: por unicidad de Picard-Lindelöf, h(t) queda
    determinado por h(0) — las observaciones x_1..x_n NO influyen.
    """

    nombre = "NeuralODE"
    familia = "EDO neuronal"

    def __init__(self, c_in, n_clases, hidden=24, oculto_f=48, M=12, seed=0):
        super().__init__(c_in, n_clases, hidden, seed)
        rng, H = self.rng, hidden
        self.M = M
        self.We = _xavier(rng, c_in + 1, H); self.be = T(np.zeros(H))
        self.f_params = ad.inicializar_mlp([H, oculto_f, H], rng)
        self.Wo = _xavier(rng, H, n_clases); self.bo = T(np.zeros(n_clases))
        self.params = [self.We, self.be, *self.f_params, self.Wo, self.bo]

    def _f(self, h: T) -> T:
        return ad.mlp_forward(self.f_params, h, act_final=ad.tanh)

    def forward(self, X, t):
        x0 = np.concatenate([X[:, 0, :], t[:, :1]], axis=1)  # (x_0, t_0)
        h = ad.tanh(T(x0) @ self.We + self.be)
        dt = 1.0 / self.M
        for _ in range(self.M):                       # RK4
            k1 = self._f(h)
            k2 = self._f(h + k1 * (dt / 2))
            k3 = self._f(h + k2 * (dt / 2))
            k4 = self._f(h + k3 * dt)
            h = h + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0)
        return h @ self.Wo + self.bo

    def trayectoria(self, x, t):
        x0 = np.concatenate([x[0], t[:1]])
        h = np.tanh(x0 @ self.We.data + self.be.data)
        hs = [h.copy()]
        dt = 1.0 / self.M
        f = lambda hh: ad.aplicar_mlp_act(self.f_params, hh) if False else None
        for _ in range(self.M):
            k1 = self._f_np(h); k2 = self._f_np(h + 0.5 * dt * k1)
            k3 = self._f_np(h + 0.5 * dt * k2); k4 = self._f_np(h + dt * k3)
            h = h + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            hs.append(h.copy())
        return np.array(hs)

    def _f_np(self, h):
        n_capas = len(self.f_params) // 2
        y = h
        for i in range(n_capas):
            y = y @ self.f_params[2 * i].data + self.f_params[2 * i + 1].data
            y = np.tanh(y)   # ocultas y final (act_final=tanh)
        return y


# ══════════════════════════════════════════════════════════════════════════════
# 4. NEURAL CDE — el dato controla la dinámica en todo t
# ══════════════════════════════════════════════════════════════════════════════

class ModeloNCDE(ModeloBase):
    """dz = f_θ(z) dX_t, con X = (t, x) interpolado linealmente.

    Discretización: para cada intervalo de observación, paso de punto medio
    (RK2) sobre el incremento de control ΔX_i:
        z* = z + ½ f(z)·ΔX ;  z ← z + f(z*)·ΔX
    f_θ: R^H → R^{H×d} (matriz de respuesta), con tanh final para acotar
    el campo (estabilidad, práctica estándar de Kidger et al.).
    El muestreo irregular entra de forma EXACTA vía ΔX (que incluye Δt).
    """

    nombre = "NeuralCDE"
    familia = "EDC neuronal"

    def __init__(self, c_in, n_clases, hidden=24, oculto_f=48, seed=0):
        super().__init__(c_in, n_clases, hidden, seed)
        rng, H = self.rng, hidden
        self.d = c_in + 1                      # canales de control: (t, x)
        self.We = _xavier(rng, self.d, H); self.be = T(np.zeros(H))
        self.f_params = ad.inicializar_mlp([H, oculto_f, H * self.d], rng)
        self.Wo = _xavier(rng, H, n_clases); self.bo = T(np.zeros(n_clases))
        self.params = [self.We, self.be, *self.f_params, self.Wo, self.bo]

    def _campo(self, z: T, B: int) -> T:
        F = ad.mlp_forward(self.f_params, z, act_final=ad.tanh)
        return F.reshape(B, self.H, self.d)

    def _control(self, X, t):
        return np.concatenate([t[:, :, None], X], axis=2)   # (B, n, d)

    def forward(self, X, t):
        Xc = self._control(X, t)
        B, n, d = Xc.shape
        dX = np.diff(Xc, axis=1)                              # (B, n-1, d)
        z = ad.tanh(T(Xc[:, 0, :]) @ self.We + self.be)
        for i in range(n - 1):
            dXi = T(dX[:, i, :, None])                        # (B, d, 1)
            k1 = (self._campo(z, B) @ dXi).reshape(B, self.H)
            z_mid = z + k1 * 0.5
            k2 = (self._campo(z_mid, B) @ dXi).reshape(B, self.H)
            z = z + k2
        return z @ self.Wo + self.bo

    def trayectoria(self, x, t):
        Xc = self._control(x[None], t[None])[0]
        dX = np.diff(Xc, axis=0)
        z = np.tanh(Xc[0] @ self.We.data + self.be.data)
        zs = [z.copy()]
        for i in range(len(dX)):
            F = self._campo_np(z)
            z_mid = z + 0.5 * (F @ dX[i])
            z = z + self._campo_np(z_mid) @ dX[i]
            zs.append(z.copy())
        return np.array(zs)

    def _campo_np(self, z):
        y = ad.aplicar_mlp(self.f_params, z[None])[0]
        return np.tanh(y).reshape(self.H, self.d)


# ══════════════════════════════════════════════════════════════════════════════
# 5. NEURAL RDE — método log-ODE sobre ventanas (caminos rugosos)
# ══════════════════════════════════════════════════════════════════════════════

class ModeloNRDE(ModeloBase):
    """z_{j+1} = z_j + g_θ(z_j) · logsig²_{ventana j}(X).

    La log-signatura de nivel 2 de cada ventana — incrementos ΔX y áreas de
    Lévy A^{uv} — es el resumen geométrico mínimo que garantiza orden de
    convergencia 2 del método log-ODE (Lyons; Boutaib-Gyurkó-Lyons-Yang).
    Ventajas: n/s pasos en lugar de n (series largas), y robustez ante
    muestreo fino/rugoso: la geometría intra-ventana no se pierde.
    """

    nombre = "NeuralRDE"
    familia = "EDR neuronal (rough)"

    def __init__(self, c_in, n_clases, hidden=24, oculto_f=48,
                 ventana: int = 6, seed=0):
        super().__init__(c_in, n_clases, hidden, seed)
        rng, H = self.rng, hidden
        self.ventana = ventana
        self.d = c_in + 1
        self.D = self.d + self.d * (self.d - 1) // 2     # dim logsig nivel 2
        self.We = _xavier(rng, self.d, H); self.be = T(np.zeros(H))
        self.f_params = ad.inicializar_mlp([H, oculto_f, H * self.D], rng)
        self.Wo = _xavier(rng, H, n_clases); self.bo = T(np.zeros(n_clases))
        self.params = [self.We, self.be, *self.f_params, self.Wo, self.bo]

    def _logsigs(self, X, t):
        """Log-signaturas nivel 2 por ventana. (B, W, D) — constantes wrt θ."""
        Xc = np.concatenate([t[:, :, None], X], axis=2)
        B, n, d = Xc.shape
        s = self.ventana
        bordes = list(range(0, n - 1, s)) + [n - 1]
        LS = []
        for a, b in zip(bordes[:-1], bordes[1:]):
            LS.append(sigmod.logsig_nivel2_lote(Xc[:, a:b + 1, :]))
        return np.stack(LS, axis=1), Xc                   # (B, W, D)

    def _campo(self, z: T, B: int) -> T:
        F = ad.mlp_forward(self.f_params, z, act_final=ad.tanh)
        return F.reshape(B, self.H, self.D)

    def forward(self, X, t):
        LS, Xc = self._logsigs(X, t)
        B, W, D = LS.shape
        z = ad.tanh(T(Xc[:, 0, :]) @ self.We + self.be)
        for j in range(W):
            lj = T(LS[:, j, :, None])                     # (B, D, 1)
            k1 = (self._campo(z, B) @ lj).reshape(B, self.H)
            z_mid = z + k1 * 0.5
            k2 = (self._campo(z_mid, B) @ lj).reshape(B, self.H)
            z = z + k2
        return z @ self.Wo + self.bo

    def trayectoria(self, x, t):
        LS, Xc = self._logsigs(x[None], t[None])
        z = np.tanh(Xc[0, 0] @ self.We.data + self.be.data)
        zs = [z.copy()]
        for j in range(LS.shape[1]):
            F = self._campo_np(z)
            z_mid = z + 0.5 * (F @ LS[0, j])
            z = z + self._campo_np(z_mid) @ LS[0, j]
            zs.append(z.copy())
        return np.array(zs)

    def _campo_np(self, z):
        y = ad.aplicar_mlp(self.f_params, z[None])[0]
        return np.tanh(y).reshape(self.H, self.D)


MODELOS = {
    "RNN": ModeloRNN,
    "GRU": ModeloGRU,
    "NeuralODE": ModeloNODE,
    "NeuralCDE": ModeloNCDE,
    "NeuralRDE": ModeloNRDE,
}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_clasificador(
    modelo: ModeloBase,
    X_tr: np.ndarray, t_tr: np.ndarray, y_tr: np.ndarray,
    X_va: np.ndarray, t_va: np.ndarray, y_va: np.ndarray,
    epochs: int = 60, lr: float = 5e-3, batch: int = 64,
    seed: int = 0, verbose: bool = True,
) -> Dict:
    """Entrena con Adam; devuelve historial {loss, acc_val} y tiempo."""
    rng = np.random.default_rng(seed)
    opt = ad.Adam(modelo.params, lr=lr, clip=5.0)
    N = len(X_tr)
    hist = {"loss": [], "acc_val": []}
    t0 = time.time()
    for ep in range(epochs):
        idx = rng.permutation(N)
        loss_ep = 0.0
        for ini in range(0, N, batch):
            sel = idx[ini: ini + batch]
            opt.zero_grad()
            logits = modelo.forward(X_tr[sel], t_tr[sel])
            perdida = ad.softmax_crossentropy(logits, y_tr[sel])
            perdida.backward()
            opt.step()
            loss_ep += float(perdida.data) * len(sel)
        acc = float((modelo.predecir(X_va, t_va) == y_va).mean())
        hist["loss"].append(loss_ep / N)
        hist["acc_val"].append(acc)
        if verbose and (ep + 1) % 10 == 0:
            print(f"    [{modelo.nombre}] epoch {ep+1}/{epochs} "
                  f"loss={hist['loss'][-1]:.4f} acc_val={acc:.3f}")
    hist["segundos"] = round(time.time() - t0, 1)
    hist["n_params"] = modelo.n_parametros()
    return hist


# ══════════════════════════════════════════════════════════════════════════════
# DATASETS DE JUGUETE
# ══════════════════════════════════════════════════════════════════════════════

def datos_espirales(
    N: int = 800, n_obs: int = 40, ruido: float = 0.04,
    irregular: bool = False, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clasificación de espirales 2D: sentido horario vs antihorario.
    Fase inicial aleatoria → la PRIMERA observación no informa la clase:
    los modelos deben usar la evolución temporal (esto castiga a NeuralODE,
    que solo ve x_0, y premia a CDE/RDE).

    irregular=True: tiempos de observación ~ Uniforme ordenada por serie
    (cada serie tiene SU PROPIA rejilla — muestreo por eventos).
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=N)
    X = np.zeros((N, n_obs, 2))
    tiempos = np.zeros((N, n_obs))
    for i in range(N):
        if irregular:
            t = np.sort(rng.uniform(0, 1, n_obs))
            t[0], t[-1] = 0.0, 1.0
        else:
            t = np.linspace(0, 1, n_obs)
        fase = rng.uniform(0, 2 * np.pi)
        direc = 1.0 if y[i] == 1 else -1.0
        ang = fase + direc * (2.5 * np.pi) * t
        rad = 0.3 + 0.7 * t
        X[i, :, 0] = rad * np.cos(ang) + rng.normal(0, ruido, n_obs)
        X[i, :, 1] = rad * np.sin(ang) + rng.normal(0, ruido, n_obs)
        tiempos[i] = t
    return X, tiempos, y


def submuestrear_irregular(X, t, frac=0.5, seed=0):
    """Submuestreo aleatorio por serie (conserva extremos): simula sensores
    asíncronos / pérdida de paquetes sobre una rejilla regular."""
    rng = np.random.default_rng(seed)
    N, n, c = X.shape
    m = max(4, int(n * frac))
    Xs = np.zeros((N, m, c))
    ts = np.zeros((N, m))
    for i in range(N):
        interior = rng.choice(np.arange(1, n - 1), size=m - 2, replace=False)
        idx = np.sort(np.concatenate([[0], interior, [n - 1]]))
        Xs[i] = X[i, idx]
        ts[i] = t[i, idx] if t.ndim == 2 else t[idx]
    return Xs, ts


# ══════════════════════════════════════════════════════════════════════════════
# DEMO: NODE APRENDE UN CAMPO VECTORIAL (Van der Pol)
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_node_dinamica(
    mu: float = 1.0, n_tray: int = 60, n_pasos: int = 40,
    dt: float = 0.12, epochs: int = 250, lr: float = 8e-3, seed: int = 0,
) -> Dict:
    """
    Demostración del corazón de las Neural ODE: aprender el campo vectorial
    f del oscilador de Van der Pol  x'' - μ(1-x²)x' + x = 0  a partir de
    trayectorias observadas, minimizando ‖x_{k+1} - RK4(x_k; f_θ)‖².

    Devuelve campo real vs campo aprendido en una rejilla + trayectorias.
    """
    rng = np.random.default_rng(seed)

    def f_real(p):
        x, v = p[..., 0], p[..., 1]
        return np.stack([v, mu * (1 - x ** 2) * v - x], axis=-1)

    def rk4_np(p, f, h):
        k1 = f(p); k2 = f(p + 0.5 * h * k1)
        k3 = f(p + 0.5 * h * k2); k4 = f(p + h * k3)
        return p + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    # trayectorias de entrenamiento
    p0 = rng.uniform(-3, 3, size=(n_tray, 2))
    tray = [p0]
    for _ in range(n_pasos):
        tray.append(rk4_np(tray[-1], f_real, dt))
    tray = np.stack(tray, axis=1)                     # (n_tray, n_pasos+1, 2)
    pares_a = tray[:, :-1].reshape(-1, 2)
    pares_b = tray[:, 1:].reshape(-1, 2)

    # f_θ: MLP 2→48→48→2 ; pérdida: un paso RK4
    params = ad.inicializar_mlp([2, 48, 48, 2], rng)

    def f_theta(p: T) -> T:
        return ad.mlp_forward(params, p)

    def rk4_ad(p: T, h: float) -> T:
        k1 = f_theta(p)
        k2 = f_theta(p + k1 * (h / 2))
        k3 = f_theta(p + k2 * (h / 2))
        k4 = f_theta(p + k3 * h)
        return p + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0)

    opt = ad.Adam(params, lr=lr)
    perdidas = []
    for ep in range(epochs):
        idx = rng.choice(len(pares_a), size=min(512, len(pares_a)), replace=False)
        opt.zero_grad()
        pred = rk4_ad(T(pares_a[idx]), dt)
        L = ad.mse(pred, pares_b[idx])
        L.backward()
        opt.step()
        perdidas.append(float(L.data))

    # rejilla de campos
    g = np.linspace(-3.2, 3.2, 17)
    GX, GY = np.meshgrid(g, g)
    rej = np.stack([GX.ravel(), GY.ravel()], axis=1)
    campo_real = f_real(rej)
    campo_apr = ad.aplicar_mlp(params, rej)

    # trayectoria de prueba (real vs integrada con f_θ)
    p = np.array([2.0, 0.0])
    tr_real, tr_apr = [p.copy()], [p.copy()]
    q = p.copy()
    for _ in range(120):
        p = rk4_np(p[None], f_real, dt * 0.6)[0]
        q = rk4_np(q[None], lambda z: ad.aplicar_mlp(params, z), dt * 0.6)[0]
        tr_real.append(p.copy()); tr_apr.append(q.copy())

    return {
        "rejilla": rej.tolist(),
        "campo_real": np.round(campo_real, 4).tolist(),
        "campo_aprendido": np.round(campo_apr, 4).tolist(),
        "tray_real": np.round(np.array(tr_real), 4).tolist(),
        "tray_aprendida": np.round(np.array(tr_apr), 4).tolist(),
        "perdidas": [round(x, 6) for x in perdidas],
        "mu": mu,
    }


# ── prueba rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X, t, y = datos_espirales(N=120, n_obs=30, seed=1)
    Xv, tv, yv = datos_espirales(N=60, n_obs=30, seed=2)
    for nombre, Cls in MODELOS.items():
        m = Cls(c_in=2, n_clases=2, hidden=16, seed=0)
        h = entrenar_clasificador(m, X, t, y, Xv, tv, yv,
                                  epochs=4, lr=8e-3, batch=40, verbose=False)
        print(f"{nombre:10s} params={h['n_params']:6d} "
              f"loss={h['loss'][-1]:.3f} acc={h['acc_val'][-1]:.2f} "
              f"({h['segundos']}s)")
    print("OK neuralde: los 5 modelos entrenan y clasifican")
