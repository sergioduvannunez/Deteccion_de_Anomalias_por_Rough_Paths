# Rough Paths Lab

**Detección de anomalías por Rough Path Signatures** (muestra AMI real anonimizada +
3 contextos simulados multivariados) **y Ecuaciones Diferenciales Neuronales** con la
construcción completa **RNN → Neural ODE → Neural CDE → Neural RDE**.

> Proyecto deliberadamente **ligero**: pocos datos pero entendibles (~10 MB en total).
> Todo es reproducible desde los datos incluidos. Documentación completa del código
> en [`docs/instructivo_codigo.tex`](docs/instructivo_codigo.tex).

## Inicio rápido

```bash
# entorno: Python 3.10+ con numpy, pandas, scikit-learn, scipy, pyarrow
python frontend/servidor.py        # abre http://localhost:8001
```

En Windows también: doble clic en **`Iniciar Rough Paths Lab.bat`**.
El servidor fusiona los artefactos de `outputs/` en un `datos.json` y sirve la SPA
(Plotly + KaTeX **offline** — no necesita internet).

## Estructura de carpetas

```
├── backend/            ← TODO el código de cómputo (ambos temas)
│   ├── signaturas.py        Chen vectorizado por lotes, multivariado, log-signaturas
│   ├── registro.py          registry extensible: añadir un detector = decorar una clase
│   ├── detectores.py        13 clásicos + familia Signatures (27 en total, 5 familias)
│   ├── evaluacion.py        protocolo PU (DR/AR, τ=p90) + matrices de Jaccard
│   ├── simuladores.py       3 contextos multivariados: it / ambiental / eeg
│   ├── autodiff.py          motor de backprop NumPy (gradientes verificados)
│   ├── neuralde.py          RNN, GRU, NeuralODE, NeuralCDE, NeuralRDE desde cero
│   ├── cargador.py          datos AMI reales (parquet) + 4 modos de muestreo
│   ├── anomalias.py         inyector de las 7 anomalías AMI
│   ├── crear_muestra_ami.py   genera la muestra AMI anonimizada
│   ├── pipeline_anomalias.py  → outputs/simulados/
│   ├── pipeline_neuralde.py   → outputs/neuralde/
│   └── pipeline_ami.py        → demo AMI sobre la muestra (--demo)
│
├── frontend/           ← la SPA interactiva
│   ├── servidor.py          fusiona outputs/ → datos.json y sirve en :8001
│   ├── index.html           13 secciones (teoría, exploradores, resultados, lab)
│   ├── css/style.css        sistema de diseño (animaciones, responsive)
│   └── js/
│       ├── app.js           lógica reactiva de todas las secciones
│       ├── signatures.js    signaturas de Chen EN VIVO en el navegador
│       ├── anims.js         21 animaciones canvas (1 por algoritmo / etapa NDE)
│       ├── teoria.js        matemática rigurosa de cada detector y cada etapa
│       ├── plotly.min.js    offline
│       └── katex/           offline (js + css + fuentes)
│
├── outputs/            ← artefactos pre-calculados (los consume el frontend)
│   ├── tablas_framework/    métricas PU + Jaccard AMI (4 modos de muestreo)
│   ├── series/              pares normal/anómala AMI + detecciones
│   ├── simulados/           métricas + Jaccard + pares + detecciones (3 contextos)
│   └── neuralde/            curvas, Van der Pol, trayectorias z(t), log-signaturas
│
├── Raw_Processed/      ← MUESTRA AMI anonimizada (60 medidores, 1 mes, ~44 KB)
├── docs/instructivo_codigo.tex   ← documentación completa del código (LaTeX)
├── Iniciar Rough Paths Lab.bat   ← lanzador de doble clic (Windows)
├── Regenerar datos.bat           ← re-ejecuta los pipelines
└── build_exe.py                  ← empaqueta un .exe portable
```

## Reproducir los resultados

```bash
python -m backend.pipeline_anomalias    # 3 contextos simulados (~1 min)
python -m backend.pipeline_neuralde     # 20 entrenamientos + demos (~2 min)
python -m backend.pipeline_ami --demo   # demo AMI sobre la muestra incluida
```

> **Sobre los datos AMI.** Los resultados de AMI que muestra el frontend se
> calcularon con el dataset completo original (~7 GB, un año), **no incluido** por
> tamaño y privacidad. En su lugar se incluye una **muestra anonimizada** (la columna
> `ID` del medidor se sustituyó por códigos `AMI_xxxx`; ver `backend/crear_muestra_ami.py`)
> que ilustra la estructura y permite la demo `--demo`. Los 3 contextos simulados y los
> modelos neuronales son 100 % reproducibles desde el código.

## Los dos temas y su unión

1. **Detección de anomalías.** 27 detectores en 5 familias evaluados con protocolo
   PU-learning (τ = percentil 90 de los scores sobre originales ⇒ AR ≈ 10 %).
   La familia *Signatures* opera sobre la firma del camino con **aumento temporal**:
   el tipo de muestreo (regular / irregular por eventos / alta frecuencia) queda
   codificado en la geometría — la anomalía `RafagaMuestreo` se detecta con valores
   perfectamente normales.

2. **Ecuaciones diferenciales neuronales.** Construcción desde cero (motor autodiff
   propio): la RNN es un sistema dinámico discreto; su límite continuo es la Neural
   ODE (`dh/dt = f_θ(h)`), que **no** puede incorporar datos posteriores a t=0
   (Picard–Lindelöf); la Neural CDE (`dz = f_θ(z) dX`) convierte el dato en control;
   la Neural RDE aplica el **método log-ODE** de la teoría de caminos rugosos:
   ventanas resumidas por su log-signatura (incrementos + áreas de Lévy).

**El puente:** la log-signatura de nivel 2 es a la vez el mejor descriptor de
detección (`LogSigMaHa_d2`) y el motor del Neural RDE — un solo lenguaje matemático.

## Extensibilidad

```python
# backend/detectores.py — añadir un detector nuevo:
@registrar("MiDetector", familia="C-ML", vista="pca",
           descripcion="...")
class MiDetector(DetectorBase):
    def ajustar(self, X): ...
    def puntuar(self, X): ...   # score creciente en anomalía
```
Queda automáticamente disponible en la suite, los pipelines y el frontend.
Hiperparámetros: automáticos (`auto_hp`) pero siempre seleccionables a mano
(`SuiteDetectores(hiperparametros={"MiDetector": {...}})`).
