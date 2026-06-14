"""
backend/autodiff.py
===================
Motor de diferenciación automática en modo reverso (backpropagation) sobre
NumPy. Sin dependencias externas: es la base de TODOS los modelos neuronales
del proyecto (Autoencoder de la suite de detectores y las cuatro familias
RNN → Neural ODE → Neural CDE → Neural RDE).

Fundamento
----------
Cada operación construye un nodo de un grafo dirigido acíclico. El método
`backward()` recorre el grafo en orden topológico inverso aplicando la regla
de la cadena vectorial:

    ∂L/∂x = Σ_{y hijo de x} (∂y/∂x)ᵀ ∂L/∂y

El soporte de broadcasting requiere "des-difundir" los gradientes
(sumar sobre los ejes expandidos) — véase `_unbroadcast`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# NODO TENSORIAL
# ══════════════════════════════════════════════════════════════════════════════

def _unbroadcast(g: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    """Reduce el gradiente g a `shape` sumando los ejes difundidos."""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g.reshape(shape)


class T:
    """Tensor con gradiente. Envuelve un np.ndarray float64."""

    __slots__ = ("data", "grad", "_bw", "_padres")

    def __init__(self, data, padres: tuple = ()):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: Optional[np.ndarray] = None
        self._bw = None
        self._padres = padres

    # ── infraestructura ───────────────────────────────────────────────────────
    @property
    def shape(self):
        return self.data.shape

    def _acum(self, g: np.ndarray) -> None:
        self.grad = g if self.grad is None else self.grad + g

    def backward(self) -> None:
        orden, visto = [], set()

        def topo(v: "T"):
            if id(v) in visto:
                return
            visto.add(id(v))
            for p in v._padres:
                topo(p)
            orden.append(v)

        topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(orden):
            if v._bw is not None and v.grad is not None:
                v._bw(v.grad)

    # ── operaciones básicas ───────────────────────────────────────────────────
    def __add__(self, otro):
        otro = otro if isinstance(otro, T) else T(otro)
        out = T(self.data + otro.data, (self, otro))

        def bw(g):
            self._acum(_unbroadcast(g, self.data.shape))
            otro._acum(_unbroadcast(g, otro.data.shape))
        out._bw = bw
        return out

    __radd__ = __add__

    def __neg__(self):
        out = T(-self.data, (self,))
        out._bw = lambda g: self._acum(-g)
        return out

    def __sub__(self, otro):
        otro = otro if isinstance(otro, T) else T(otro)
        return self + (-otro)

    def __rsub__(self, otro):
        return T(otro) + (-self)

    def __mul__(self, otro):
        otro = otro if isinstance(otro, T) else T(otro)
        out = T(self.data * otro.data, (self, otro))

        def bw(g):
            self._acum(_unbroadcast(g * otro.data, self.data.shape))
            otro._acum(_unbroadcast(g * self.data, otro.data.shape))
        out._bw = bw
        return out

    __rmul__ = __mul__

    def __truediv__(self, otro):
        otro = otro if isinstance(otro, T) else T(otro)
        out = T(self.data / otro.data, (self, otro))

        def bw(g):
            self._acum(_unbroadcast(g / otro.data, self.data.shape))
            otro._acum(_unbroadcast(-g * self.data / otro.data ** 2,
                                    otro.data.shape))
        out._bw = bw
        return out

    def __matmul__(self, otro):
        otro = otro if isinstance(otro, T) else T(otro)
        out = T(self.data @ otro.data, (self, otro))

        def bw(g):
            self._acum(g @ otro.data.swapaxes(-1, -2))
            otro._acum(self.data.swapaxes(-1, -2) @ g)
        out._bw = bw
        return out

    def __getitem__(self, idx):
        out = T(self.data[idx], (self,))

        def bw(g):
            full = np.zeros_like(self.data)
            np.add.at(full, idx, g)
            self._acum(full)
        out._bw = bw
        return out

    def reshape(self, *shape):
        orig = self.data.shape
        out = T(self.data.reshape(*shape), (self,))
        out._bw = lambda g: self._acum(g.reshape(orig))
        return out

    def sum(self, axis=None, keepdims=False):
        out = T(self.data.sum(axis=axis, keepdims=keepdims), (self,))

        def bw(g):
            if axis is None:
                self._acum(np.broadcast_to(g, self.data.shape).copy())
            else:
                if not keepdims:
                    g = np.expand_dims(g, axis)
                self._acum(np.broadcast_to(g, self.data.shape).copy())
        out._bw = bw
        return out

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / n)


# ── funciones elementales ─────────────────────────────────────────────────────

def tanh(x: T) -> T:
    y = np.tanh(x.data)
    out = T(y, (x,))
    out._bw = lambda g: x._acum(g * (1.0 - y * y))
    return out


def sigmoid(x: T) -> T:
    y = 1.0 / (1.0 + np.exp(-np.clip(x.data, -30, 30)))
    out = T(y, (x,))
    out._bw = lambda g: x._acum(g * y * (1.0 - y))
    return out


def relu(x: T) -> T:
    y = np.maximum(x.data, 0.0)
    out = T(y, (x,))
    out._bw = lambda g: x._acum(g * (x.data > 0))
    return out


def exp(x: T) -> T:
    y = np.exp(np.clip(x.data, -60, 60))
    out = T(y, (x,))
    out._bw = lambda g: x._acum(g * y)
    return out


def concat(ts: Sequence[T], axis: int = -1) -> T:
    datas = [t.data for t in ts]
    out = T(np.concatenate(datas, axis=axis), tuple(ts))
    tam = [d.shape[axis] for d in datas]
    cortes = np.cumsum([0] + tam)

    def bw(g):
        for t, a, b in zip(ts, cortes[:-1], cortes[1:]):
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(a, b)
            t._acum(g[tuple(sl)])
    out._bw = bw
    return out


def softmax_crossentropy(logits: T, y: np.ndarray) -> T:
    """Pérdida de entropía cruzada con softmax fusionado (estable).
    logits (N, C), y enteros (N,). Devuelve la media."""
    z = logits.data
    zmax = z.max(axis=1, keepdims=True)
    ez = np.exp(z - zmax)
    p = ez / ez.sum(axis=1, keepdims=True)
    N = z.shape[0]
    nll = -np.log(p[np.arange(N), y] + 1e-12)
    out = T(nll.mean(), (logits,))

    def bw(g):
        gz = p.copy()
        gz[np.arange(N), y] -= 1.0
        logits._acum(g * gz / N)
    out._bw = bw
    return out


def mse(pred: T, target: np.ndarray) -> T:
    diff = pred - T(target)
    return (diff * diff).mean()


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZADOR ADAM
# ══════════════════════════════════════════════════════════════════════════════

class Adam:
    """Kingma & Ba (2015). Actualización con corrección de sesgo:
        m_t = β1 m + (1-β1) g ;  v_t = β2 v + (1-β2) g²
        θ ← θ - lr · m̂ / (√v̂ + ε)
    """

    def __init__(self, params: List[T], lr: float = 1e-2,
                 betas=(0.9, 0.999), eps: float = 1e-8,
                 clip: Optional[float] = 5.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.clip = clip
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        if self.clip is not None:
            norma = np.sqrt(sum(float((p.grad ** 2).sum())
                                for p in self.params if p.grad is not None))
            esc = self.clip / (norma + 1e-12) if norma > self.clip else 1.0
        else:
            esc = 1.0
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad * esc
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p.data -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ══════════════════════════════════════════════════════════════════════════════
# CAPAS / MLP DE CONVENIENCIA
# ══════════════════════════════════════════════════════════════════════════════

def inicializar_mlp(dims: List[int], rng: np.random.Generator) -> List[T]:
    """Pesos Xavier-Glorot: W_l ~ U(±sqrt(6/(fan_in+fan_out))). Devuelve la
    lista plana [W1, b1, W2, b2, ...] de tensores entrenables."""
    params: List[T] = []
    for a, b in zip(dims[:-1], dims[1:]):
        lim = np.sqrt(6.0 / (a + b))
        params.append(T(rng.uniform(-lim, lim, size=(a, b))))
        params.append(T(np.zeros(b)))
    return params


def mlp_forward(params: List[T], x: T, activacion=tanh,
                act_final=None) -> T:
    """Aplica el MLP (capas ocultas con `activacion`, última lineal o
    `act_final`)."""
    n_capas = len(params) // 2
    h = x
    for i in range(n_capas):
        W, b = params[2 * i], params[2 * i + 1]
        h = h @ W + b
        if i < n_capas - 1:
            h = activacion(h)
        elif act_final is not None:
            h = act_final(h)
    return h


def aplicar_mlp(params: List[T], X: np.ndarray) -> np.ndarray:
    """Forward puro NumPy (inferencia, sin grafo)."""
    n_capas = len(params) // 2
    h = np.asarray(X, dtype=np.float64)
    for i in range(n_capas):
        h = h @ params[2 * i].data + params[2 * i + 1].data
        if i < n_capas - 1:
            h = np.tanh(h)
    return h


def entrenar_autoencoder(params: List[T], X: np.ndarray, epochs: int = 150,
                         lr: float = 5e-3, batch: int = 256,
                         seed: int = 42, verbose: bool = False) -> List[float]:
    """Entrena el MLP como autoencoder (objetivo = entrada) con Adam."""
    rng = np.random.default_rng(seed)
    opt = Adam(params, lr=lr)
    N = len(X)
    historia = []
    for ep in range(epochs):
        idx = rng.permutation(N)
        perdida_ep = 0.0
        for ini in range(0, N, batch):
            sel = idx[ini: ini + batch]
            xb = X[sel]
            opt.zero_grad()
            rec = mlp_forward(params, T(xb))
            perdida = mse(rec, xb)
            perdida.backward()
            opt.step()
            perdida_ep += float(perdida.data) * len(sel)
        historia.append(perdida_ep / N)
        if verbose and (ep + 1) % 25 == 0:
            print(f"    AE epoch {ep+1}: mse={historia[-1]:.5f}")
    return historia


# ── Verificación numérica de gradientes ───────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def check(f, xs, nombre):
        ts = [T(x) for x in xs]
        out = f(*ts)
        out.backward()
        eps = 1e-6
        for ti, x in zip(ts, xs):
            num = np.zeros_like(x)
            it = np.nditer(x, flags=["multi_index"])
            while not it.finished:
                i = it.multi_index
                xp = x.copy(); xp[i] += eps
                xm = x.copy(); xm[i] -= eps
                fp = f(*[T(xp if a is x else b) for a, b in zip(xs, xs)]).data
                fm = f(*[T(xm if a is x else b) for a, b in zip(xs, xs)]).data
                num[i] = (fp - fm) / (2 * eps)
                it.iternext()
            err = np.abs(num - ti.grad).max()
            assert err < 1e-5, f"{nombre}: err={err}"
        print(f"  OK gradiente {nombre}")

    A = rng.normal(size=(3, 4))
    B = rng.normal(size=(4, 2))
    c = rng.normal(size=(2,))
    check(lambda a, b: ((a @ b) * (a @ b)).sum(), [A, B], "matmul")
    check(lambda a: tanh(a).sum(), [A], "tanh")
    check(lambda a: sigmoid(a * 0.5 + 1.0).mean(), [A], "sigmoid")
    check(lambda a, b, cc: (tanh(a @ b + cc) * tanh(a @ b + cc)).mean(),
          [A, B, c], "mlp-broadcast")
    y = np.array([0, 1, 1])
    L = rng.normal(size=(3, 2))
    tl = T(L)
    loss = softmax_crossentropy(tl, y)
    loss.backward()
    eps = 1e-6
    num = np.zeros_like(L)
    for i in np.ndindex(L.shape):
        Lp = L.copy(); Lp[i] += eps
        Lm = L.copy(); Lm[i] -= eps
        num[i] = (softmax_crossentropy(T(Lp), y).data
                  - softmax_crossentropy(T(Lm), y).data) / (2 * eps)
    assert np.abs(num - tl.grad).max() < 1e-5
    print("  OK gradiente softmax-crossentropy")
    print("OK autodiff: todos los gradientes verificados numericamente")
