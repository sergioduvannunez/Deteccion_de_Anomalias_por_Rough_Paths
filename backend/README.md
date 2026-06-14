# backend/ — cómputo de ambos temas

Un único módulo Python (sin dependencias más allá de numpy/pandas/scikit-learn/
scipy/pyarrow) que cubre **detección de anomalías por signaturas** y
**ecuaciones diferenciales neuronales**.

## Mapa de módulos

| Módulo | Contenido | Verificación |
|---|---|---|
| `signaturas.py` | Identidad de Chen **vectorizada por lotes** para caminos en R^d (niveles 1–4), log-signatura nivel 2 en forma cerrada (incrementos + áreas de Lévy), logaritmo tensorial general, transformaciones de camino (aumento temporal, punto base, lead-lag), scores Mahalanobis-local / kernel-OCSVM / conformancia, interpretación de niveles | `python signaturas.py`: S(t)=1, identidad de shuffle, consistencia log-sig |
| `registro.py` | Patrón *registry*: `@registrar(nombre, familia, vista, auto_hp=...)` sobre clases con `ajustar`/`puntuar`. Política de hiperparámetros: defaults < automáticos < overrides del usuario | — |
| `detectores.py` | 13 clásicos (RobustZMAD, PCAT2Q, KDE, GMM, KMeans, HDBSCAN, LOF, OPTICS, IForest, OCSVM, **Autoencoder real** sobre autodiff, RobustPCA con IRLS-Huber, Conformal vectorizado) + familia Signatures (SigKernel_d{2,3,4}, SigMaHaKNN_d{2,3,4}_k{3,10,20}, SigConformancia_d2, LogSigMaHa_d2). `SuiteDetectores` construye las vistas de features una vez (shape/aug/pca/sig*/logsig2) y orquesta. Acepta series (N,n) o **(N,n,c) multivariadas** y tiempos irregulares `t` | smoke test vía pipelines |
| `evaluacion.py` | Protocolo PU: τ = cuantil q de scores sobre originales; DR/AR global y por tipo; matrices de Jaccard (general + por tipo). Genérico en la lista de tipos | — |
| `simuladores.py` | 3 contextos multivariados con 7 anomalías cada uno: **it** (4 canales, muestreo regular 5 min), **ambiental** (3 canales, muestreo irregular Gamma por eventos — incluye `RafagaMuestreo`, anomalía del *muestreo* y no de los valores), **eeg** (6 canales 10-20, 128 Hz, ritmos alfa/beta/theta + ruido 1/f) | `python simuladores.py` |
| `autodiff.py` | Tensores con gradiente, modo reverso, broadcasting con *unbroadcast*, ops (matmul, tanh, sigmoid, relu, concat, slice, softmax-crossentropy fusionada), Adam con clipping, helpers MLP | `python autodiff.py`: comparación con gradientes numéricos |
| `neuralde.py` | `ModeloRNN`, `ModeloGRU`, `ModeloNODE` (RK4, M=12), `ModeloNCDE` (control (t,x) lineal a trozos, paso RK2-midpoint), `ModeloNRDE` (log-ODE nivel 2 por ventanas). Datasets de juguete (espirales con fase aleatoria; submuestreo irregular). `entrenar_node_dinamica`: un Neural ODE aprende el campo de Van der Pol | `python neuralde.py` |
| `cargador.py` / `anomalias.py` | AMI real: lectura de parquet, clasificación residencial/comercial, ventanas semanales 168 h, 4 modos de muestreo; inyector de las 7 anomalías AMI | heredados del framework validado |

## Pipelines

```bash
python -m backend.pipeline_anomalias [--rapido]  # → outputs/simulados/
python -m backend.pipeline_neuralde              # → outputs/neuralde/
python -m backend.pipeline_ami [--quick]         # → outputs/tablas_framework/
```

`pipeline_anomalias`: por contexto — genera dataset, ajusta la suite SOLO con
originales (con `t` real: el muestreo irregular entra a la signatura), puntúa
todo, exporta métricas PU, Jaccard, pares normal/anómala y detecciones por par.

`pipeline_neuralde`: 5 modelos × 2 tareas (espirales, it_mini) × 2 regímenes
(regular, irregular) con Adam; exporta curvas, comparativa, campo de Van der Pol
(real vs aprendido), trayectorias ocultas z(t) del NCDE (PCA 3D) y las
log-signaturas por ventana que alimentan al NRDE.

## Decisiones de optimalidad (automatizada, seleccionable)

| Detector | Hiperparámetro automático | Justificación |
|---|---|---|
| KDE | h = N^(−1/(d+4)) | regla de Scott (óptimo AMISE) |
| GMM | k = argmin BIC | aproximación de Laplace a la evidencia |
| KMeans | k por codo-kneedle | máx. distancia a la cuerda de WCSS(k) |
| LOF | k = √N | balance sesgo-varianza estándar kNN |
| HDBSCAN | mcs = max(5, N/50) | clúster mínimo ≈ 2 % del dataset |
| OCSVM | ν por fracción MAD-Z>3.5 | propiedad ν (cota de outliers) |
| Conformal | k = 2·log₂N | localidad en alta dimensión |
| IForest | max_samples por N | descorrelación de árboles |
| SigMaHaKNN | PCA→60 si D>60 | colinealidad por identidad de shuffle |

Cualquiera se sobreescribe con
`SuiteDetectores(hiperparametros={"Detector": {"k": 15}})`.
