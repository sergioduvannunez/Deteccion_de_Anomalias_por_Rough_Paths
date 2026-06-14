# frontend/ — Rough Paths Lab (SPA)

Aplicación de una sola página, **100 % offline** (Plotly y KaTeX empaquetados),
servida por `servidor.py`.

```bash
python frontend/servidor.py          # http://localhost:8001 (abre el navegador)
python frontend/servidor.py 8080     # puerto personalizado
AMI_NO_BROWSER=1 python frontend/servidor.py
```

Al arrancar, fusiona `outputs/tablas_framework`, `outputs/series`,
`outputs/simulados` y `outputs/neuralde` en un único `datos.json` (~2 MB) y lo
sirve estáticamente (ThreadingHTTPServer + Cache-Control: no-store). Los bloques
ausentes se omiten con un aviso — el servidor funciona aunque solo exista parte
de los outputs.

## Secciones (13)

| Sección | Qué hace |
|---|---|
| Inicio | hero animado + tarjetas resumen auto-calculadas + mapa del sistema |
| Teoría de signaturas | definición, Chen, shuffle, unicidad, decaimiento factorial (KaTeX) + interpretación de los 4 niveles |
| Datos & muestreo | 4 fuentes de datos; demo de una espiral muestreada regular / irregular / por eventos |
| Series AMI | normal vs anómala por casa y tipo, con ventana deslizante enlazada al explorador |
| Contextos multivariados | visor multicanal (subplots por canal) de los pares normal/anómala de it / ambiental / eeg |
| **Catálogo de detectores** | al seleccionar un algoritmo: **animación canvas de su funcionamiento** + **construcción matemática rigurosa** + hiperparámetros y propiedades (27 algoritmos) |
| Resultados DR/AR | barras por familia, heatmap detector × tipo, top-5 — para AMI (4 modos) y los 3 contextos |
| Jaccard | heatmap 23–25×25 + lectura intra/inter-familia |
| Inspector | caso concreto: serie + score vs τ + margen score−τ de TODOS los detectores |
| **Explorador de signaturas** | cálculo de Chen EN VIVO en el navegador (js/signatures.js), animación del desarrollo del camino con normas por nivel, barras de coordenadas normal vs anómala, matriz de áreas de Lévy ΔA^ij, panel de interpretación por nivel |
| **De RNN a Neural RDE** | 4 etapas con animación canvas (recurrencia desplegada / campo con partículas / control dX / ventanas log-ODE) + construcción matemática completa |
| Laboratorio Neural DE | comparativa accuracy + convergencia (regular vs irregular), Van der Pol (campo real vs aprendido), trayectorias z(t) del NCDE en 3D, heatmap de log-signaturas por ventana |
| Síntesis | conclusiones y guía de extensibilidad |

## Arquitectura JS

- `app.js` — carga con reintentos, índices O(1), renderizadores por sección,
  IntersectionObserver para navegación y revelado.
- `signatures.js` — identidad de Chen multivariada (niveles 1–4) con productos
  tensoriales aplanados; etiquetas, áreas de Lévy, normas y distancias.
  Verificación en consola: `Sig.calcular(Sig.caminoDeSerie(y),3).niveles[0][0] === 1`.
- `anims.js` — `Animador` (rAF + DPR) y 21 funciones de dibujo: 16 de detectores
  + 4 etapas NDE + desarrollo de la signatura.
- `teoria.js` — contenido matemático (KaTeX) de cada detector y cada etapa NDE,
  e interpretación de niveles.

## Añadir contenido

- **Nuevo detector**: registra la clase en el backend; si quieres teoría propia
  en el frontend añade una entrada en `_KINDS` de `teoria.js` y (opcional) una
  animación en `anims.js`. Sin entrada propia, el detector aparece igualmente en
  resultados/Jaccard/inspector (todo es data-driven desde datos.json).
- **Nuevo contexto simulado**: añade la función en `backend/simuladores.py` y su
  nombre en `pipeline_anomalias.py`; el frontend lo descubre solo.
