# Cómo ejecutar — Rough Paths Lab

## Requisitos mínimos

- Python 3.9 o superior  
- Paquetes: `numpy pandas scikit-learn scipy pyarrow`

Si te faltan, instálalos con:

```
pip install -r requirements.txt
```

No necesita internet para funcionar: Plotly y KaTeX están incluidos en `frontend/`.

---

## Opción 1 — Un solo comando (cualquier PC)

```
python run.py
```

Comprueba dependencias, arranca el servidor y abre el navegador en
**http://localhost:8001** automáticamente.

Puerto distinto:

```
python run.py 8080
```

> En macOS/Linux puedes usar `python3 run.py` si `python` no está en el PATH.

---

## Opción 2 — Doble clic en Windows

Doble clic en **`Iniciar Rough Paths Lab.bat`**

Se abre una ventana negra, detecta Python automáticamente (incluyendo
entornos Conda) y arranca el servidor.

- Para **detener**: cierra la ventana negra o pulsa `Ctrl + C`.
- Si el navegador no se abre, entra manualmente a http://localhost:8001

---

## Opción 3 — Regenerar los datos del backend

Solo necesario si modificas algo en `backend/`. Ejecuta en orden:

```
python -m backend.pipeline_anomalias
python -m backend.pipeline_neuralde
```

O en Windows, doble clic en **`Regenerar datos.bat`** (tarda ~3-4 min).
Al terminar vuelve a lanzar el servidor con cualquiera de las opciones anteriores.

---

## Demo del análisis AMI

Corre el pipeline sobre la muestra anonimizada incluida en el repositorio:

```
python -m backend.pipeline_ami --demo
```

Deja los resultados en `outputs/ami_demo/` sin tocar los datos oficiales.

---

## Documentación del código

`docs/instructivo_codigo.tex` explica la matemática y el código en detalle.
Compílalo con MiKTeX/TeX Live o súbelo a [Overleaf](https://overleaf.com)
y ejecuta `pdflatex` dos veces.
