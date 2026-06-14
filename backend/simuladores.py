"""
backend/simuladores.py
======================
Generadores de series de tiempo MULTIVARIADAS masivas con anomalías
etiquetadas, en tres contextos con TIPOS DE MUESTREO distintos:

  1. `simular_it`        — Telemetría de servidores (informático).
       Muestreo REGULAR cada 5 min; 4 canales correlacionados
       (cpu, memoria, red, latencia); ventanas de 12 h (144 puntos).

  2. `simular_ambiental` — Estación de sensores ambientales (temporal).
       Muestreo IRREGULAR dirigido por eventos (inter-arribos Gamma:
       el sensor reporta más cuando hay cambios); 3 canales
       (temperatura, humedad, presión); ventanas de 48 h (96 lecturas).

  3. `simular_eeg`       — Lectura de comportamiento neuronal (EEG).
       ALTA FRECUENCIA regular a 128 Hz; 6 canales (montaje 10-20:
       Fp1, Fp2, C3, C4, O1, O2); ventanas de 3 s (384 muestras).

Cada generador produce ventanas normales realistas (estacionalidad,
correlación entre canales, ruido AR/1-f) e inyecta 7 tipos de anomalías
específicas del dominio sobre copias de ventanas reales, devolviendo:

    {"X": (N, n, c), "t": (N, n), "y_syn": bool (N,), "y_tipo": str (N,),
     "base_idx": int (N,), "canales": [...], "tipos": [...],
     "descrip_tipos": {...}, "muestreo": {...}, ...}

El número de ventanas escala con `n_ventanas` (masividad configurable).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def _ar1(rng, n, c, rho=0.85, sigma=1.0):
    """Ruido AR(1) por canal: x_t = ρ x_{t-1} + ε."""
    e = rng.normal(0, sigma, size=(n, c))
    x = np.zeros((n, c))
    for k in range(1, n):
        x[k] = rho * x[k - 1] + e[k]
    return x


def _ruido_rosa(rng, n, c, alpha=1.0):
    """Ruido 1/f^alpha por síntesis espectral (típico en EEG)."""
    f = np.fft.rfftfreq(n, d=1.0)
    f[0] = f[1] if n > 1 else 1.0
    amp = 1.0 / f ** (alpha / 2.0)
    fases = rng.uniform(0, 2 * np.pi, size=(len(f), c))
    espectro = amp[:, None] * np.exp(1j * fases)
    x = np.fft.irfft(espectro, n=n, axis=0)
    return x / (x.std(axis=0, keepdims=True) + 1e-12)


def _ensamblar(X0, t0, transformaciones, n_por_tipo, rng):
    """Inyecta cada tipo sobre copias de ventanas aleatorias y concatena."""
    N = len(X0)
    Xs, ts, tipos, bases = [], [], [], []
    for tipo, fn in transformaciones.items():
        idx = rng.choice(N, size=n_por_tipo, replace=False)
        for i in idx:
            arr = X0[i].copy()
            tt = t0[i].copy()
            arr2, tt2 = fn(arr, tt, rng)
            Xs.append(arr2)
            ts.append(tt2)
            tipos.append(tipo)
            bases.append(int(i))
    X = np.concatenate([X0, np.stack(Xs)], axis=0)
    t = np.concatenate([t0, np.stack(ts)], axis=0)
    y_syn = np.concatenate([np.zeros(N, bool), np.ones(len(Xs), bool)])
    y_tipo = np.array(["original"] * N + tipos)
    base_idx = np.concatenate([np.arange(N), np.array(bases)])
    return X, t, y_syn, y_tipo, base_idx


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONTEXTO INFORMÁTICO — TELEMETRÍA DE SERVIDORES (muestreo regular)
# ══════════════════════════════════════════════════════════════════════════════

def simular_it(n_ventanas: int = 1600, n: int = 144, seed: int = 42,
               frac_anom: float = 0.018) -> Dict:
    """
    4 canales: cpu [%], mem [%], red [MB/s], lat [ms]. Paso = 5 min, 12 h.

    Modelo base: carga diurna sinusoidal + ráfagas de tráfico Poisson que
    elevan simultáneamente cpu/red/latencia (correlación realista),
    memoria con deriva lenta + liberaciones (GC), ruido AR(1).
    """
    rng = np.random.default_rng(seed)
    c = 4
    t_grid = np.linspace(0, 1, n)
    X0 = np.zeros((n_ventanas, n, c))
    for i in range(n_ventanas):
        fase = rng.uniform(0, 2 * np.pi)
        nivel = rng.uniform(0.3, 0.7)
        diurno = 0.5 + 0.45 * np.sin(2 * np.pi * t_grid + fase)
        carga = nivel * diurno
        # ráfagas de tráfico (Poisson) con decaimiento exponencial
        rafagas = np.zeros(n)
        for _ in range(rng.poisson(3)):
            t0i = rng.integers(0, n - 6)
            dur = rng.integers(3, 12)
            amp = rng.uniform(0.2, 0.7)
            seg = np.arange(min(dur, n - t0i))
            rafagas[t0i: t0i + len(seg)] += amp * np.exp(-seg / max(dur / 3, 1))
        ar = _ar1(rng, n, c, rho=0.8, sigma=0.04)
        cpu = np.clip(20 + 55 * (carga + 0.8 * rafagas) + 8 * ar[:, 0], 0, 100)
        gc = (np.cumsum(rng.random(n) < 0.02) % 2).astype(float)
        mem = np.clip(35 + 18 * carga + 10 * t_grid * rng.uniform(-0.3, 1.0)
                      - 6 * gc + 5 * ar[:, 1], 5, 98)
        red = np.clip(8 + 60 * (carga + rafagas) ** 1.3 + 4 * ar[:, 2], 0, None)
        lat = np.clip(12 + 35 * (carga + 1.4 * rafagas) ** 2 + 4 * ar[:, 3], 1, None)
        X0[i] = np.stack([cpu, mem, red, lat], axis=1)
    t0 = np.broadcast_to(t_grid, (n_ventanas, n)).copy()

    # ── 7 anomalías del dominio informático ───────────────────────────────────
    def ddos(a, t, r):
        ini = r.integers(n // 6, n // 2); dur = r.integers(n // 4, n // 2)
        s = slice(ini, min(ini + dur, n))
        a[s, 2] *= r.uniform(4, 9)              # red
        a[s, 3] *= r.uniform(5, 12)             # latencia
        a[s, 0] = np.clip(a[s, 0] * r.uniform(1.6, 2.4), 0, 100)
        return a, t

    def fuga_memoria(a, t, r):
        ini = r.integers(0, n // 3)
        rampa = np.zeros(n); rampa[ini:] = np.linspace(0, r.uniform(35, 55), n - ini)
        a[:, 1] = np.clip(a[:, 1] + rampa, 0, 99.5)
        a[:, 3] *= 1 + 0.6 * rampa / rampa.max()
        return a, t

    def caida_servicio(a, t, r):
        ini = r.integers(n // 4, 3 * n // 4); dur = r.integers(n // 8, n // 3)
        s = slice(ini, min(ini + dur, n))
        a[s, 0] = r.uniform(0.5, 3); a[s, 2] = r.uniform(0, 0.5)
        a[s, 3] = r.uniform(0.5, 2); a[s, 1] *= r.uniform(0.45, 0.7)
        return a, t

    def cripto_mineria(a, t, r):
        a[:, 0] = np.clip(r.uniform(88, 98) + r.normal(0, 1.0, n), 0, 100)
        return a, t

    def escaneo_puertos(a, t, r):
        pulsos = (r.random(n) < 0.25).astype(float)
        a[:, 2] += pulsos * r.uniform(10, 25)
        a[:, 3] += pulsos * r.uniform(8, 20)
        return a, t

    def exfiltracion(a, t, r):
        noche = np.exp(-((np.linspace(0, 1, n) - r.uniform(0.7, 0.9)) / 0.08) ** 2)
        a[:, 2] += noche * r.uniform(35, 70)
        return a, t

    def degradacion_disco(a, t, r):
        a[:, 3] *= 1 + np.linspace(0, r.uniform(2.5, 5), n)
        a[:, 0] = np.clip(a[:, 0] + np.linspace(0, 12, n), 0, 100)
        return a, t

    transform = {
        "DDoS": ddos, "FugaMemoria": fuga_memoria, "CaidaServicio": caida_servicio,
        "CriptoMineria": cripto_mineria, "EscaneoPuertos": escaneo_puertos,
        "Exfiltracion": exfiltracion, "DegradacionDisco": degradacion_disco,
    }
    n_por_tipo = max(20, int(n_ventanas * frac_anom))
    X, t, y_syn, y_tipo, base_idx = _ensamblar(X0, t0, transform, n_por_tipo, rng)

    return {
        "nombre": "it",
        "titulo": "Telemetría de servidores",
        "icono": "🖥️",
        "descripcion": ("Monitoreo de infraestructura: CPU, memoria, tráfico de red y "
                        "latencia de un parque de servidores. Ventanas de 12 horas con "
                        "paso de 5 minutos. Carga diurna, ráfagas de tráfico y ruido AR(1) "
                        "correlacionado entre canales."),
        "canales": ["cpu", "mem", "red", "lat"],
        "unidades": ["%", "%", "MB/s", "ms"],
        "muestreo": {"tipo": "regular",
                     "descripcion": "Rejilla regular: 1 lectura cada 5 minutos (144 por ventana). "
                                    "El canal de tiempo de la signatura es lineal: toda la "
                                    "información discriminante está en los valores."},
        "X": X, "t": t, "y_syn": y_syn, "y_tipo": y_tipo, "base_idx": base_idx,
        "tipos": list(transform.keys()),
        "descrip_tipos": {
            "DDoS": {"emoji": "🌊", "titulo": "Ataque DDoS",
                     "texto": "Inundación de tráfico: red ×4-9, latencia ×5-12 y CPU al doble durante un tramo sostenido. Firma multivariada simultánea en 3 canales."},
            "FugaMemoria": {"emoji": "🧠", "titulo": "Fuga de memoria",
                            "texto": "Rampa lineal de +35-55 puntos de memoria sin liberación, con degradación progresiva de latencia. Anomalía de DERIVA (lenta, no de pico)."},
            "CaidaServicio": {"emoji": "⛔", "titulo": "Caída de servicio",
                              "texto": "El proceso muere: CPU, red y latencia colapsan a ~0 durante un tramo. Equivalente informático del FlatLine de AMI."},
            "CriptoMineria": {"emoji": "⛏️", "titulo": "Criptominería",
                              "texto": "CPU clavada en 88-98% con varianza mínima todo el día, sin cambio en red ni latencia. Anomalía de DESACOPLE entre canales."},
            "EscaneoPuertos": {"emoji": "📡", "titulo": "Escaneo de puertos",
                               "texto": "Pulsos breves y frecuentes en red y latencia (25% de los instantes). Textura de alta frecuencia que infla el nivel 2 de la signatura."},
            "Exfiltracion": {"emoji": "🕳️", "titulo": "Exfiltración nocturna",
                             "texto": "Transferencia gaussiana suave de 35-70 MB/s centrada en horario valle. Magnitud anómala CONDICIONADA a la fase del ciclo diurno."},
            "DegradacionDisco": {"emoji": "💽", "titulo": "Degradación de disco",
                                 "texto": "La latencia crece multiplicativamente (hasta ×6) con CPU en aumento leve. Deriva multiplicativa, contraparte de la fuga de memoria."},
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. CONTEXTO TEMPORAL — SENSORES AMBIENTALES (muestreo irregular por eventos)
# ══════════════════════════════════════════════════════════════════════════════

def simular_ambiental(n_ventanas: int = 1500, n: int = 96, seed: int = 43,
                      frac_anom: float = 0.02) -> Dict:
    """
    3 canales: temperatura [°C], humedad [%], presión [hPa]. Ventana de 48 h.

    MUESTREO IRREGULAR: el datalogger reporta por eventos — inter-arribos
    Gamma(k≈2). Cada ventana tiene SU PROPIA rejilla temporal. El aumento
    temporal de la signatura y las Neural CDE/RDE consumen estos tiempos
    sin interpolación previa.
    """
    rng = np.random.default_rng(seed)
    X0 = np.zeros((n_ventanas, n, 3))
    t0 = np.zeros((n_ventanas, n))
    for i in range(n_ventanas):
        # tiempos por eventos: inter-arribos Gamma normalizados a [0,1] (48 h)
        gaps = rng.gamma(shape=2.0, scale=1.0, size=n - 1)
        tt = np.concatenate([[0.0], np.cumsum(gaps)])
        tt = tt / tt[-1]
        fase = rng.uniform(0, 2 * np.pi)
        amp_T = rng.uniform(4, 9)
        base_T = rng.uniform(12, 24)
        # 2 ciclos diarios en 48 h
        ciclo = np.sin(2 * np.pi * 2 * tt + fase)
        ar = _ar1(rng, n, 3, rho=0.9, sigma=0.05)
        temp = base_T + amp_T * ciclo + 1.2 * ar[:, 0]
        hum = np.clip(65 - 3.2 * amp_T * ciclo + 4 * ar[:, 1], 8, 100)
        pres = 1013 + 4 * np.sin(2 * np.pi * 0.7 * tt + rng.uniform(0, 6)) \
               + 1.5 * ar[:, 2]
        X0[i] = np.stack([temp, hum, pres], axis=1)
        t0[i] = tt

    def salto_calibracion(a, t, r):
        ch = r.integers(0, 3)
        ini = r.integers(n // 5, 4 * n // 5)
        salto = r.uniform(0.8, 2.0) * a[:, ch].std() * r.choice([-1, 1])
        a[ini:, ch] += salto
        return a, t

    def deriva_sensor(a, t, r):
        ch = r.integers(0, 3)
        a[:, ch] += np.linspace(0, r.uniform(2, 4) * a[:, ch].std(), n) * r.choice([-1, 1])
        return a, t

    def sensor_congelado(a, t, r):
        ch = r.integers(0, 3)
        ini = r.integers(0, n - n // 3)
        dur = r.integers(n // 4, n // 2)
        a[ini: ini + dur, ch] = a[ini, ch]
        return a, t

    def pico_transitorio(a, t, r):
        ch = r.integers(0, 3)
        for _ in range(r.integers(2, 5)):
            j = r.integers(1, n - 1)
            a[j, ch] += r.uniform(4, 8) * a[:, ch].std() * r.choice([-1, 1])
        return a, t

    def ruido_electronico(a, t, r):
        ch = r.integers(0, 3)
        a[:, ch] += r.normal(0, 1.6 * a[:, ch].std(), n)
        return a, t

    def desacople_canales(a, t, r):
        # la humedad deja de anti-correlacionar con la temperatura
        mu = a[:, 1].mean()
        a[:, 1] = mu + (a[:, 1] - mu) * -1.0 + r.normal(0, 1.0, n)
        return a, t

    def rafaga_muestreo(a, t, r):
        # el datalogger entra en modo ráfaga: 60% de las lecturas se
        # concentran en un 15% del tiempo (anomalía DEL MUESTREO, no del valor)
        centro = r.uniform(0.3, 0.7)
        m = int(n * 0.6)
        t_burst = np.clip(centro + r.normal(0, 0.04, m), 0, 1)
        t_resto = np.sort(r.uniform(0, 1, n - m))
        t_nuevo = np.sort(np.concatenate([t_burst, t_resto]))
        t_nuevo[0], t_nuevo[-1] = 0.0, 1.0
        # re-muestrear los valores sobre la nueva rejilla (interp del original)
        for ch in range(3):
            a[:, ch] = np.interp(t_nuevo, t, a[:, ch])
        return a, t_nuevo

    transform = {
        "SaltoCalibracion": salto_calibracion, "DerivaSensor": deriva_sensor,
        "SensorCongelado": sensor_congelado, "PicoTransitorio": pico_transitorio,
        "RuidoElectronico": ruido_electronico, "DesacopleCanales": desacople_canales,
        "RafagaMuestreo": rafaga_muestreo,
    }
    n_por_tipo = max(20, int(n_ventanas * frac_anom))
    X, t, y_syn, y_tipo, base_idx = _ensamblar(X0, t0, transform, n_por_tipo, rng)

    return {
        "nombre": "ambiental",
        "titulo": "Estación ambiental (muestreo irregular)",
        "icono": "🌡️",
        "descripcion": ("Datalogger meteorológico con reporte POR EVENTOS: inter-arribos "
                        "Gamma, cada ventana de 48 h tiene su propia rejilla de 96 lecturas. "
                        "Temperatura y humedad anti-correlacionadas con ciclo diario; "
                        "presión con onda lenta."),
        "canales": ["temp", "hum", "pres"],
        "unidades": ["°C", "%", "hPa"],
        "muestreo": {"tipo": "irregular",
                     "descripcion": "Muestreo dirigido por eventos (inter-arribos Gamma). El canal "
                                    "de tiempo de la signatura deja de ser lineal y codifica la "
                                    "densidad de muestreo; la anomalía RafagaMuestreo vive ahí."},
        "X": X, "t": t, "y_syn": y_syn, "y_tipo": y_tipo, "base_idx": base_idx,
        "tipos": list(transform.keys()),
        "descrip_tipos": {
            "SaltoCalibracion": {"emoji": "📏", "titulo": "Salto de calibración",
                                 "texto": "Offset abrupto y persistente en un canal tras un evento (recalibración fallida). Cambia el incremento neto (nivel 1)."},
            "DerivaSensor": {"emoji": "📈", "titulo": "Deriva de sensor",
                             "texto": "Tendencia lineal espuria de hasta 4σ por envejecimiento del transductor. Anomalía lenta: invisible para detectores de picos."},
            "SensorCongelado": {"emoji": "🧊", "titulo": "Sensor congelado",
                                "texto": "El canal repite el último valor durante 25-50% de la ventana (hielo o fallo del conversor). Análogo del FlatLine."},
            "PicoTransitorio": {"emoji": "⚡", "titulo": "Picos transitorios",
                                "texto": "2-4 lecturas aisladas a 4-8σ (descargas electrostáticas). Outliers puntuales clásicos."},
            "RuidoElectronico": {"emoji": "📻", "titulo": "Ruido electrónico",
                                 "texto": "La varianza de un canal se infla ×1.6σ adicional (interferencia EM). Textura, no nivel."},
            "DesacopleCanales": {"emoji": "🔗", "titulo": "Desacople de canales",
                                 "texto": "La humedad INVIERTE su anti-correlación con la temperatura. Solo visible en la estructura conjunta: las áreas de Lévy cruzadas A^(temp,hum) cambian de signo."},
            "RafagaMuestreo": {"emoji": "⏱️", "titulo": "Ráfaga de muestreo",
                               "texto": "El 60% de las lecturas se concentra en el 15% del tiempo. Los VALORES son normales: la anomalía está en la rejilla temporal — la detecta el canal t de la signatura."},
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. CONTEXTO NEURONAL — EEG SINTÉTICO (alta frecuencia)
# ══════════════════════════════════════════════════════════════════════════════

def simular_eeg(n_ventanas: int = 1500, fs: int = 128, dur_s: float = 3.0,
                seed: int = 44, frac_anom: float = 0.02) -> Dict:
    """
    6 canales (10-20): Fp1, Fp2 (frontales), C3, C4 (centrales), O1, O2
    (occipitales). fs=128 Hz, ventanas de 3 s (n=384).

    Modelo base: ritmos alfa (8-12 Hz, dominante occipital), beta (14-30 Hz,
    frontal), theta (4-8 Hz) con modulación de amplitud lenta + ruido rosa
    1/f + componente común (conductancia del cuero cabelludo).
    """
    rng = np.random.default_rng(seed)
    n = int(fs * dur_s)
    c = 6
    tt = np.arange(n) / fs
    pesos_alfa = np.array([0.3, 0.3, 0.6, 0.6, 1.2, 1.2])
    pesos_beta = np.array([0.8, 0.8, 0.5, 0.5, 0.25, 0.25])
    pesos_theta = np.array([0.5, 0.5, 0.45, 0.45, 0.35, 0.35])

    X0 = np.zeros((n_ventanas, n, c))
    for i in range(n_ventanas):
        f_a = rng.uniform(8, 12)
        f_b = rng.uniform(14, 28)
        f_t = rng.uniform(4, 7)
        mod = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * tt
                                 + rng.uniform(0, 6.28))
        alfa = np.sin(2 * np.pi * f_a * tt + rng.uniform(0, 6.28)) * mod
        beta = np.sin(2 * np.pi * f_b * tt + rng.uniform(0, 6.28))
        theta = np.sin(2 * np.pi * f_t * tt + rng.uniform(0, 6.28))
        rosa = _ruido_rosa(rng, n, c, alpha=1.0)
        comun = _ruido_rosa(rng, n, 1, alpha=1.2)
        amp = rng.uniform(8, 14)
        for ch in range(c):
            X0[i, :, ch] = amp * (pesos_alfa[ch] * alfa + pesos_beta[ch] * beta
                                  + pesos_theta[ch] * theta) \
                           + 6 * rosa[:, ch] + 4 * comun[:, 0]
    t_grid = tt / tt[-1]
    t0 = np.broadcast_to(t_grid, (n_ventanas, n)).copy()

    def crisis_epileptica(a, t, r):
        # complejo punta-onda a ~3 Hz, gran amplitud, generalizado
        ini = r.integers(0, n // 3)
        f = r.uniform(2.5, 3.5)
        fase = 2 * np.pi * f * tt
        punta = np.clip(np.sin(fase) ** 7, 0, None) * 3.0
        onda = np.sin(fase - 1.2)
        gan = np.zeros(n); gan[ini:] = np.linspace(0.4, 1.0, n - ini)
        for ch in range(c):
            a[ini:, ch] = a[ini:, ch] * 0.35 + 38 * gan[ini:] * (punta[ini:] + 0.7 * onda[ini:])
        return a, t

    def pop_electrodo(a, t, r):
        ch = r.integers(0, c)
        j = r.integers(n // 8, 7 * n // 8)
        a[j:, ch] += r.uniform(60, 120) * r.choice([-1, 1]) * np.exp(-(np.arange(n - j)) / (n / 6))
        return a, t

    def artefacto_muscular(a, t, r):
        ini = r.integers(0, n // 2); dur = r.integers(n // 5, n // 2)
        s = slice(ini, min(ini + dur, n))
        for ch in (0, 1):
            a[s, ch] += r.normal(0, 22, s.stop - s.start) * \
                        np.abs(np.sin(2 * np.pi * r.uniform(35, 55) * tt[s]))
        return a, t

    def supresion_brote(a, t, r):
        per = r.integers(n // 6, n // 3)
        patron = (np.arange(n) // per) % 2 == 0
        a[patron, :] *= r.uniform(0.05, 0.15)
        a[~patron, :] *= r.uniform(1.6, 2.2)
        return a, t

    def canal_desconectado(a, t, r):
        ch = r.integers(0, c)
        a[:, ch] = r.normal(0, 0.8, n)
        return a, t

    def parpadeo_ocular(a, t, r):
        for _ in range(r.integers(2, 4)):
            j = r.integers(0, n - n // 6)
            dur = r.integers(n // 12, n // 7)
            onda = np.hanning(dur) * r.uniform(45, 80)
            a[j: j + dur, 0] += onda
            a[j: j + dur, 1] += onda * r.uniform(0.85, 1.0)
        return a, t

    def interferencia_red(a, t, r):
        f_linea = 60.0
        amp = r.uniform(10, 18)
        for ch in range(c):
            a[:, ch] += amp * np.sin(2 * np.pi * f_linea * tt + r.uniform(0, 6.28))
        return a, t

    transform = {
        "CrisisEpileptica": crisis_epileptica, "PopElectrodo": pop_electrodo,
        "ArtefactoMuscular": artefacto_muscular, "SupresionBrote": supresion_brote,
        "CanalDesconectado": canal_desconectado, "ParpadeoOcular": parpadeo_ocular,
        "InterferenciaRed": interferencia_red,
    }
    n_por_tipo = max(20, int(n_ventanas * frac_anom))
    X, t, y_syn, y_tipo, base_idx = _ensamblar(X0, t0, transform, n_por_tipo, rng)

    return {
        "nombre": "eeg",
        "titulo": "EEG sintético (comportamiento neuronal)",
        "icono": "🧠",
        "descripcion": ("Electroencefalograma de 6 canales (montaje 10-20) a 128 Hz, "
                        "ventanas de 3 s. Ritmos alfa/beta/theta con topografía realista "
                        "(alfa occipital, beta frontal), modulación lenta de amplitud, "
                        "ruido rosa 1/f y componente común."),
        "canales": ["Fp1", "Fp2", "C3", "C4", "O1", "O2"],
        "unidades": ["µV"] * 6,
        "muestreo": {"tipo": "alta_frecuencia",
                     "descripcion": "128 Hz regulares (384 puntos/ventana). Las series son "
                                    "oscilatorias y 'rugosas': el régimen donde el método "
                                    "log-ODE (ventanas + log-signaturas) muestra su ventaja."},
        "X": X, "t": t, "y_syn": y_syn, "y_tipo": y_tipo, "base_idx": base_idx,
        "tipos": list(transform.keys()),
        "descrip_tipos": {
            "CrisisEpileptica": {"emoji": "🧠", "titulo": "Crisis epiléptica",
                                 "texto": "Complejo punta-onda generalizado a ~3 Hz con amplitud ×3-4 creciente. La firma clínica de una crisis de ausencia."},
            "PopElectrodo": {"emoji": "🔌", "titulo": "Pop de electrodo",
                             "texto": "Salto DC de 60-120 µV en UN canal con relajación exponencial (despegue del gel conductor). Anomalía local de un solo canal."},
            "ArtefactoMuscular": {"emoji": "💪", "titulo": "Artefacto muscular (EMG)",
                                  "texto": "Ráfaga de 35-55 Hz en canales frontales (apretar la mandíbula). Energía de alta frecuencia espacialmente localizada."},
            "SupresionBrote": {"emoji": "📉", "titulo": "Supresión de brotes",
                               "texto": "Alternancia patológica silencio (×0.1) / brote (×2): patrón de anestesia profunda o encefalopatía. Cambia la ESTRUCTURA temporal completa."},
            "CanalDesconectado": {"emoji": "🔇", "titulo": "Canal desconectado",
                                  "texto": "Un canal queda en ruido térmico (σ≈0.8 µV) sin señal neuronal. Detección por pérdida de correlación y de amplitud."},
            "ParpadeoOcular": {"emoji": "👁️", "titulo": "Parpadeo ocular",
                               "texto": "Ondas Hanning de 45-80 µV en Fp1/Fp2 (artefacto EOG). Lento, frontal, bilateral — confundible con theta patológico."},
            "InterferenciaRed": {"emoji": "⚡", "titulo": "Interferencia de red",
                                 "texto": "Sinusoide de 60 Hz (10-18 µV) en todos los canales: acoplamiento capacitivo con la red eléctrica. Periódica pura, sin modulación."},
        },
    }


SIMULADORES: Dict[str, Callable[..., Dict]] = {
    "it": simular_it,
    "ambiental": simular_ambiental,
    "eeg": simular_eeg,
}


if __name__ == "__main__":
    for nombre, fn in SIMULADORES.items():
        ds = fn(n_ventanas=200)
        X, t = ds["X"], ds["t"]
        print(f"{nombre:10s} X={X.shape} t={t.shape} "
              f"tipos={len(ds['tipos'])} anomalias={int(ds['y_syn'].sum())} "
              f"muestreo={ds['muestreo']['tipo']}")
        assert np.isfinite(X).all(), "valores no finitos"
        assert (np.diff(t, axis=1) >= -1e-12).all(), "tiempos no monotonos"
    print("OK simuladores: 3 contextos generan datos validos")
