"""
backend/pipeline_neuralde.py
============================
Entrena la progresión completa de modelos (RNN, GRU, NeuralODE, NeuralCDE,
NeuralRDE) sobre dos tareas de juguete y dos regímenes de muestreo, y exporta
todos los artefactos que consume el frontend.

Tareas
------
1. espirales : clasificación binaria del sentido de giro de espirales 2D con
   fase inicial aleatoria. La primera observación NO informa la clase: el
   modelo debe leer la evolución temporal. (El experimento canónico que
   separa NODE — solo ve x_0 — de CDE/RDE — controladas por todo el camino.)

2. it_mini   : clasificación normal/anómala de ventanas de telemetría de
   servidores (4 canales) del simulador IT — el puente con el tema de
   detección de anomalías, ahora con modelos de ecuaciones diferenciales.

Regímenes de muestreo
---------------------
- regular   : rejilla uniforme.
- irregular : cada serie con SU propia rejilla (uniforme ordenada para
  espirales; submuestreo aleatorio 50% para it_mini). Aquí se aprecia la
  ventaja estructural de NCDE/NRDE frente a RNN+Δt.

Exporta a outputs/neuralde/:
  resultados.json    — curvas de pérdida/accuracy + tabla comparativa
  vanderpol.json     — campo vectorial real vs aprendido (demo Neural ODE)
  trayectorias.json  — caminos muestra, estados ocultos z(t) del NCDE,
                       log-signaturas por ventana del NRDE (demo log-ODE),
                       demo de tipos de muestreo sobre una misma espiral
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from . import neuralde as nde
    from . import simuladores
    from . import signaturas as sigmod
except ImportError:  # pragma: no cover
    import neuralde as nde
    import simuladores
    import signaturas as sigmod

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "outputs" / "neuralde"

EPOCHS = 40
HIDDEN = 24
LR = 6e-3
BATCH = 64


# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def dataset_espirales(regimen: str, seed=0):
    irregular = regimen == "irregular"
    X, t, y = nde.datos_espirales(N=760, n_obs=40, ruido=0.05,
                                  irregular=irregular, seed=seed)
    return (X[:560], t[:560], y[:560]), (X[560:], t[560:], y[560:])


def dataset_it_mini(regimen: str, seed=0):
    ds = simuladores.simular_it(n_ventanas=400, frac_anom=0.10, seed=7)
    X, t, y_syn = ds["X"], ds["t"], ds["y_syn"]
    # submuestrear longitud 144 -> 48 y balancear clases
    X = X[:, ::3, :]
    t = t[:, ::3]
    idx_a = np.where(y_syn)[0]
    rng = np.random.default_rng(seed)
    idx_n = rng.choice(np.where(~y_syn)[0], size=len(idx_a), replace=False)
    idx = np.concatenate([idx_n, idx_a])
    rng.shuffle(idx)
    Xb, tb = X[idx], t[idx]
    yb = y_syn[idx].astype(int)
    # estandarizar canales (escalas muy distintas: %, MB/s, ms)
    mu = Xb.mean(axis=(0, 1), keepdims=True)
    sd = Xb.std(axis=(0, 1), keepdims=True) + 1e-9
    Xb = (Xb - mu) / sd
    if regimen == "irregular":
        Xb, tb = nde.submuestrear_irregular(Xb, tb, frac=0.5, seed=seed)
    n_tr = int(len(Xb) * 0.72)
    return (Xb[:n_tr], tb[:n_tr], yb[:n_tr]), (Xb[n_tr:], tb[n_tr:], yb[n_tr:])


DATASETS = {"espirales": dataset_espirales, "it_mini": dataset_it_mini}
DATASET_INFO = {
    "espirales": {
        "titulo": "Espirales 2D (sentido de giro)",
        "descripcion": "Clasificar si la espiral gira en sentido horario o antihorario. "
                       "Fase inicial aleatoria: x(0) no informa la clase.",
        "canales": ["x", "y"], "n_clases": 2,
    },
    "it_mini": {
        "titulo": "Telemetría IT (normal vs anómala)",
        "descripcion": "Detección supervisada de ventanas anómalas de servidores "
                       "(4 canales: cpu, mem, red, lat) — el puente entre ambos temas.",
        "canales": ["cpu", "mem", "red", "lat"], "n_clases": 2,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# ENTRENAMIENTO COMPARATIVO
# ══════════════════════════════════════════════════════════════════════════════

def correr_comparativa() -> dict:
    resultados = []
    modelos_entrenados = {}
    for ds_nombre, ds_fn in DATASETS.items():
        for regimen in ("regular", "irregular"):
            (Xtr, ttr, ytr), (Xva, tva, yva) = ds_fn(regimen)
            c_in = Xtr.shape[2]
            print(f"\n--- {ds_nombre} / {regimen}  "
                  f"(train={len(Xtr)}, val={len(Xva)}, n={Xtr.shape[1]}, c={c_in}) ---")
            for nombre, Cls in nde.MODELOS.items():
                kwargs = {"hidden": HIDDEN, "seed": 0}
                if nombre == "NeuralRDE":
                    kwargs["ventana"] = max(4, Xtr.shape[1] // 8)
                m = Cls(c_in=c_in, n_clases=2, **kwargs)
                h = nde.entrenar_clasificador(
                    m, Xtr, ttr, ytr, Xva, tva, yva,
                    epochs=EPOCHS, lr=LR, batch=BATCH, seed=0, verbose=False)
                acc_fin = float(np.mean(h["acc_val"][-5:]))
                print(f"  {nombre:10s} acc={acc_fin:.3f} "
                      f"params={h['n_params']} t={h['segundos']}s")
                resultados.append({
                    "modelo": nombre, "familia": Cls.familia,
                    "dataset": ds_nombre, "regimen": regimen,
                    "acc_final": round(acc_fin, 4),
                    "acc_max": round(float(np.max(h["acc_val"])), 4),
                    "n_params": h["n_params"],
                    "segundos": h["segundos"],
                    "loss": [round(x, 5) for x in h["loss"]],
                    "acc_val": [round(x, 4) for x in h["acc_val"]],
                })
                modelos_entrenados[(nombre, ds_nombre, regimen)] = m
    return {"resultados": resultados, "modelos": modelos_entrenados}


# ══════════════════════════════════════════════════════════════════════════════
# ARTEFACTOS PARA VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def exportar_trayectorias(modelos: dict) -> dict:
    rng = np.random.default_rng(3)
    out: dict = {}

    # 1) Espirales muestra (ambas clases, regular) para dibujar
    X, t, y = nde.datos_espirales(N=8, n_obs=60, ruido=0.03,
                                  irregular=False, seed=11)
    out["espirales_muestra"] = [
        {"x": np.round(X[i, :, 0], 4).tolist(),
         "y": np.round(X[i, :, 1], 4).tolist(),
         "t": np.round(t[i], 4).tolist(),
         "clase": int(y[i])}
        for i in range(8)
    ]

    # 2) Misma espiral, tres tipos de muestreo (demo de muestreo)
    Xf, tf, yf = nde.datos_espirales(N=1, n_obs=120, ruido=0.0,
                                     irregular=False, seed=5)
    base = {"x": Xf[0, :, 0], "y": Xf[0, :, 1], "t": tf[0]}
    idx_reg = np.linspace(0, 119, 24).astype(int)
    rng2 = np.random.default_rng(8)
    idx_irr = np.sort(np.concatenate(
        [[0], rng2.choice(np.arange(1, 119), 22, replace=False), [119]]))
    # por eventos: denso donde la curvatura (giro) es alta al final
    pesos = np.linspace(0.2, 1.8, 120) ** 2
    pesos /= pesos.sum()
    idx_ev = np.sort(np.concatenate(
        [[0], rng2.choice(np.arange(1, 119), 22, replace=False, p=pesos[1:119] / pesos[1:119].sum()), [119]]))
    out["muestreo_demo"] = {
        "continua": {k: np.round(v, 4).tolist() for k, v in base.items()},
        "regular": idx_reg.tolist(),
        "irregular": idx_irr.tolist(),
        "eventos": idx_ev.tolist(),
    }

    # 3) Trayectorias ocultas z(t) del NCDE (espirales, regular), una por clase
    m_ncde = modelos.get(("NeuralCDE", "espirales", "regular"))
    if m_ncde is not None:
        Xs, ts, ys = nde.datos_espirales(N=40, n_obs=40, ruido=0.04, seed=21)
        trazas = []
        usados = {0: 0, 1: 0}
        zs_all = []
        sel = []
        for i in range(len(Xs)):
            if usados[int(ys[i])] >= 2:
                continue
            usados[int(ys[i])] += 1
            sel.append(i)
            zs_all.append(m_ncde.trayectoria(Xs[i], ts[i]))
        Z = np.concatenate(zs_all, axis=0)
        Zc = Z - Z.mean(0)
        U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
        P = Vt[:3].T
        for j, i in enumerate(sel):
            zp = (zs_all[j] - Z.mean(0)) @ P
            trazas.append({
                "clase": int(ys[i]),
                "t": np.round(ts[i], 4).tolist(),
                "z": np.round(zp, 4).tolist(),
                "x": np.round(Xs[i, :, 0], 4).tolist(),
                "y": np.round(Xs[i, :, 1], 4).tolist(),
            })
        out["ncde_trayectorias"] = trazas
        out["ncde_var_explicada"] = [round(float(v), 4) for v in
                                     (S[:3] ** 2 / (S ** 2).sum())]

    # 4) Demo log-ODE del NRDE: log-signaturas por ventana de una espiral
    Xs, ts, ys = nde.datos_espirales(N=1, n_obs=48, ruido=0.03, seed=33)
    Xc = np.concatenate([ts[0][:, None], Xs[0]], axis=1)[None]  # (1, n, 3)
    s = 8
    n = Xc.shape[1]
    bordes = list(range(0, n - 1, s)) + [n - 1]
    ventanas = []
    for a, b in zip(bordes[:-1], bordes[1:]):
        ls = sigmod.logsig_nivel2_lote(Xc[:, a:b + 1, :])[0]
        ventanas.append({
            "rango": [int(a), int(b)],
            "logsig": np.round(ls, 4).tolist(),
        })
    out["nrde_logode_demo"] = {
        "x": np.round(Xs[0, :, 0], 4).tolist(),
        "y": np.round(Xs[0, :, 1], 4).tolist(),
        "t": np.round(ts[0], 4).tolist(),
        "ventanas": ventanas,
        "etiquetas_logsig": ["Δt", "Δx", "Δy", "A(t,x)", "A(t,y)", "A(x,y)"],
        "clase": int(ys[0]),
    }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=== 1/3 Comparativa de modelos ===")
    comp = correr_comparativa()
    (OUT / "resultados.json").write_text(
        json.dumps({"runs": comp["resultados"],
                    "datasets": DATASET_INFO,
                    "epochs": EPOCHS, "hidden": HIDDEN},
                   ensure_ascii=False), encoding="utf-8")

    print("\n=== 2/3 Demo Van der Pol (Neural ODE aprende el campo) ===")
    vdp = nde.entrenar_node_dinamica(mu=1.2, epochs=300, seed=0)
    (OUT / "vanderpol.json").write_text(json.dumps(vdp), encoding="utf-8")
    print(f"  perdida final: {vdp['perdidas'][-1]:.5f}")

    print("\n=== 3/3 Trayectorias y demos ===")
    tray = exportar_trayectorias(comp["modelos"])
    (OUT / "trayectorias.json").write_text(json.dumps(tray), encoding="utf-8")

    print(f"\nOK pipeline_neuralde ({time.time()-t0:.0f}s): outputs/neuralde/ completo")


if __name__ == "__main__":
    main()
