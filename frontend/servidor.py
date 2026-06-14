#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frontend/servidor.py
====================
Servidor de la SPA "Rough Paths Lab": detección de anomalías por signaturas
(AMI real + 3 contextos simulados multivariados) y ecuaciones diferenciales
neuronales (RNN → NODE → NCDE → NRDE).

Lee los artefactos pre-calculados de outputs/ y los fusiona en un único
datos.json servido estáticamente junto con la SPA.

Uso:
    python frontend/servidor.py          # puerto 8001, abre el navegador
    python frontend/servidor.py 8080     # puerto personalizado
    AMI_NO_BROWSER=1 ...                 # no abrir navegador
"""

import gzip
import json
import os
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

if getattr(sys, "frozen", False):
    WEB_DIR = sys._MEIPASS
    BASE_DIR = sys._MEIPASS
else:
    WEB_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(WEB_DIR)

TAB_DIR = os.path.join(BASE_DIR, "outputs", "tablas_framework")
SERIES_DIR = os.path.join(BASE_DIR, "outputs", "series")
SIM_DIR = os.path.join(BASE_DIR, "outputs", "simulados")
NDE_DIR = os.path.join(BASE_DIR, "outputs", "neuralde")
DATA_FILE = os.path.join(WEB_DIR, "datos.json")

# ─── Catálogos AMI ────────────────────────────────────────────────────────────
MODOS = ["single_house", "weekly_month", "monthly", "annual"]
MODOS_ES = {"single_house": "Casa Individual", "weekly_month": "Semana/Mes",
            "monthly": "Mensual", "annual": "Anual"}
TIPOS_ANOM = ["PartialBypass", "Smoothing", "FlipSchedule",
              "SuddenDrop", "SyntheticNoise", "FlatLine", "SpikeEvent"]
TIPO_LABEL = {"PartialBypass": "Derivación Parcial", "Smoothing": "Suavizado",
              "FlipSchedule": "Inversión Horario", "SuddenDrop": "Caída Brusca",
              "SyntheticNoise": "Ruido Sintético", "FlatLine": "Línea Plana",
              "SpikeEvent": "Picos de Evento"}
FAM_COLOR = {"A-Estadistico": "#1565c0", "B-Clustering": "#2e7d32",
             "C-ML": "#c62828", "D-Alternativo": "#7b1fa2",
             "Signatures": "#e65100"}
FAM_LABEL = {"A-Estadistico": "Estadístico", "B-Clustering": "Clustering",
             "C-ML": "Machine Learning", "D-Alternativo": "Alternativo",
             "Signatures": "Rough Path Signatures"}
DETECTORES = [
    "RobustZMAD", "PCAT2Q", "KDE", "GMM",
    "KMeans", "HDBSCAN", "LOF", "OPTICS",
    "IForest", "OCSVM", "Autoencoder", "RobustPCA", "Conformal",
    "SigKernel_d2", "SigKernel_d3", "SigKernel_d4",
    "SigMaHaKNN_d2_k3", "SigMaHaKNN_d2_k10", "SigMaHaKNN_d2_k20",
    "SigMaHaKNN_d3_k3", "SigMaHaKNN_d3_k10", "SigMaHaKNN_d3_k20",
    "SigMaHaKNN_d4_k3", "SigMaHaKNN_d4_k10", "SigMaHaKNN_d4_k20",
]
_fam = lambda d: ("Signatures" if "Sig" in d else
                  {"RobustZMAD": "A-Estadistico", "PCAT2Q": "A-Estadistico",
                   "KDE": "A-Estadistico", "GMM": "A-Estadistico",
                   "KMeans": "B-Clustering", "HDBSCAN": "B-Clustering",
                   "LOF": "B-Clustering", "OPTICS": "B-Clustering",
                   "IForest": "C-ML", "OCSVM": "C-ML", "Autoencoder": "C-ML",
                   "RobustPCA": "D-Alternativo", "Conformal": "D-Alternativo"
                   }.get(d, "Desconocido"))
DET_FAM = {d: _fam(d) for d in DETECTORES}
EXTRA_DETS = ["SigConformancia_d2", "LogSigMaHa_d2"]
for d in EXTRA_DETS:
    DET_FAM[d] = "Signatures"
ANOM_COLOR = {"PartialBypass": "#e65100", "Smoothing": "#1565c0",
              "FlipSchedule": "#6a1b9a", "SuddenDrop": "#c62828",
              "SyntheticNoise": "#2e7d32", "FlatLine": "#37474f",
              "SpikeEvent": "#f57f17"}
DESCRIP = {
    "PartialBypass": {"emoji": "🔌", "titulo": "Derivación Parcial (Energy Theft)",
        "texto": "Consumo reducido al <b>30–60%</b> del valor real (factor α aleatorio). Simula un bypass en el circuito del medidor. La <em>forma</em> del perfil semanal se conserva — solo escala la amplitud."},
    "Smoothing": {"emoji": "∿", "titulo": "Suavizado del Registro",
        "texto": "Mezcla ponderada entre el perfil original y su media semanal (β ∈ [0.10, 0.35]). Aplana los picos de consumo; posible manipulación del firmware del medidor."},
    "FlipSchedule": {"emoji": "🔄", "titulo": "Inversión de Horario",
        "texto": "Desplazamiento de <b>12 horas</b> aplicado día a día. El pico vespertino (≈19 h) se traslada a las 07 h. El área bajo la curva es <em>idéntica</em> a la normal: solo cambia la geometría temporal."},
    "SuddenDrop": {"emoji": "📉", "titulo": "Caída Brusca",
        "texto": "Toda la semana reducida a γ ∈ [5%, 15%] del nivel habitual. Corte de suministro, fuga o fraude extremo."},
    "SyntheticNoise": {"emoji": "📡", "titulo": "Ruido Sintético",
        "texto": "Perturbación gaussiana con σ proporcional a la varianza original. Interferencia electromagnética o medidor dañado."},
    "FlatLine": {"emoji": "➡", "titulo": "Línea Plana",
        "texto": "Tramo de 12–48 h congelado en un valor ≈ 0. Avería del medidor o pérdida de comunicación AMI."},
    "SpikeEvent": {"emoji": "⚡", "titulo": "Picos de Evento",
        "texto": "Bloque de 1–6 h multiplicado ×4–12. Arranque de maquinaria, soldadora o recarga de vehículo eléctrico."},
}


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if (v != v) else round(float(v), 6)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _df_records(path):
    df = pd.read_csv(path)
    return [{k: _clean(v) for k, v in r.items()} for r in df.to_dict("records")]


def _jaccard_block(directorio, tipos, sufijo):
    out = {}
    for t in tipos + ["general"]:
        path = os.path.join(directorio, f"jaccard_{t}_{sufijo}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        out[t] = {"dets": list(df.columns),
                  "mat": [[_clean(x) for x in r] for r in df.values.tolist()]}
    return out


def build_data():
    data = {
        "MODOS": MODOS, "MODOS_ES": MODOS_ES,
        "TIPOS_ANOM": TIPOS_ANOM, "TIPO_LABEL": TIPO_LABEL,
        "FAM_COLOR": FAM_COLOR, "FAM_LABEL": FAM_LABEL,
        "DETECTORES": DETECTORES, "DET_FAM": DET_FAM,
        "ANOM_COLOR": ANOM_COLOR, "DESCRIP": DESCRIP,
    }

    # ══ Bloque AMI (CSV pre-calculados del framework) ══════════════════════════
    try:
        data["metricas"] = {m: _df_records(os.path.join(TAB_DIR, f"metricas_pu_{m}.csv"))
                            for m in MODOS}
        data["jaccard"] = {}
        for m in MODOS:
            for t, blk in _jaccard_block(TAB_DIR, TIPOS_ANOM, m).items():
                data["jaccard"][f"{t}__{m}"] = blk

        pares_df = pd.read_csv(os.path.join(SERIES_DIR, "pares_series.csv"))
        hcols = [f"h{i}" for i in range(168)]
        data["pares"] = [
            {"casa_idx": int(r["casa_idx"]), "tipo": r["tipo"],
             "ID": str(r["ID"]), "mes": str(r["mes"]),
             "h": [_clean(r[c]) for c in hcols]}
            for _, r in pares_df.iterrows()]
        data["N_CASAS_REF"] = int(pares_df["casa_idx"].max()) + 1
        data["meta_pares"] = _df_records(os.path.join(SERIES_DIR, "meta_pares.csv"))
        data["detecciones"] = _df_records(os.path.join(SERIES_DIR, "detecciones_pares.csv"))
        print(f"  [OK] AMI: {len(data['pares'])} series, "
              f"{len(data['jaccard'])} matrices Jaccard")
    except Exception as e:
        print(f"  [WARN] bloque AMI incompleto: {e}")

    # ══ Bloque contextos simulados multivariados ═══════════════════════════════
    data["simulados"] = {}
    for ctx in ("it", "ambiental", "eeg"):
        try:
            meta_path = os.path.join(SIM_DIR, f"contexto_{ctx}.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            with open(os.path.join(SIM_DIR, f"muestras_{ctx}.json"),
                      encoding="utf-8") as f:
                muestras = json.load(f)
            blk = {
                "meta": meta,
                "metricas": _df_records(os.path.join(SIM_DIR, f"metricas_pu_{ctx}.csv")),
                "jaccard": _jaccard_block(SIM_DIR, meta["tipos"], ctx),
                "pares": muestras["pares"],
                "detecciones": _df_records(os.path.join(SIM_DIR, f"detecciones_{ctx}.csv")),
            }
            data["simulados"][ctx] = blk
            print(f"  [OK] simulado '{ctx}': {len(blk['pares'])} pares, "
                  f"{len(blk['metricas'])} filas de métricas")
        except Exception as e:
            print(f"  [WARN] contexto '{ctx}' incompleto: {e}")

    # ══ Bloque ecuaciones diferenciales neuronales ═════════════════════════════
    data["neuralde"] = {}
    for clave, archivo in (("resultados", "resultados.json"),
                           ("vanderpol", "vanderpol.json"),
                           ("trayectorias", "trayectorias.json")):
        path = os.path.join(NDE_DIR, archivo)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data["neuralde"][clave] = json.load(f)
    print(f"  [OK] neuralde: {list(data['neuralde'].keys())}")

    return data


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        # datos.json comprimido: 2 MB -> ~400 KB, enviado con Content-Length
        # exacto en una sola escritura (robusto frente a resets de conexión).
        if self.path.split("?")[0] == "/datos.json" and \
                "gzip" in self.headers.get("Accept-Encoding", ""):
            try:
                with open(DATA_FILE + ".gz", "rb") as f:
                    cuerpo = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(cuerpo)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(cuerpo)
                return
            except (OSError, BrokenPipeError, ConnectionResetError):
                pass
        return super().do_GET()

    def end_headers(self):
        # Servidor local de desarrollo: no cachear datos ni código estático,
        # así los cambios en .js/.css/.html se ven al recargar sin caché obsoleta.
        if self.path.split("?")[0].endswith((".json", ".js", ".css", ".html", "/")):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print("=" * 64)
    print("  Rough Paths Lab · Anomalías + Ecuaciones Dif. Neuronales")
    print("=" * 64)
    d = build_data()
    cuerpo = json.dumps(d, ensure_ascii=False)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        f.write(cuerpo)
    with gzip.open(DATA_FILE + ".gz", "wb", compresslevel=6) as f:
        f.write(cuerpo.encode("utf-8"))
    print(f"  [OK] datos.json escrito ({os.path.getsize(DATA_FILE)//1024} KB; "
          f"gz {os.path.getsize(DATA_FILE + '.gz')//1024} KB)")

    url = f"http://localhost:{port}"
    print(f"\n  >> Servidor activo en:  {url}")
    print("  >> Ctrl+C para detener\n" + "=" * 64)
    if not os.environ.get("AMI_NO_BROWSER"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    httpd = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  Servidor detenido.")
        httpd.server_close()


if __name__ == "__main__":
    main()
