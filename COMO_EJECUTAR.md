# Cómo ejecutar — Rough Paths Lab

## 🚀 Opción 1 — Doble clic (lo más fácil)

Doble clic en **`Iniciar Rough Paths Lab.bat`**

Se abre una ventana negra, arranca el servidor y tu navegador se abre solo en
**http://localhost:8001** con todo el laboratorio.

- Para **detener**: cierra la ventana negra (o pulsa `Ctrl + C` en ella).
- Si el navegador no se abre solo, entra a mano a http://localhost:8001

## 🧮 Regenerar los datos (opcional)

Solo si modificas algo del `backend/`. Doble clic en **`Regenerar datos.bat`**
(tarda ~3-4 min). Al terminar, vuelve a abrir `Iniciar Rough Paths Lab.bat`.

## 📦 Opción 2 — Crear un .exe portable (para llevar a otro PC sin Python)

En una terminal, dentro de esta carpeta:

```
python build_exe.py
```

Genera **`RoughPathsLab.exe`** (~100-150 MB). Ese único archivo lleva dentro
Python, el servidor, la web y todos los datos: en cualquier Windows, doble clic
y funciona sin instalar nada. (Requiere `pip install pyinstaller` la primera vez.)

## 💻 Opción 3 — Línea de comandos

```powershell
cd "C:\Users\aipri\Desktop\Tesis Signaturas Junio 2026"
& "C:\Users\aipri\anaconda3\envs\ml_env\python.exe" frontend\servidor.py
```

Puerto distinto: añade el número al final (`... servidor.py 8080`).

## 📖 Documentación del código

`docs/instructivo_codigo.tex` explica cada parte del código (compílalo con
MiKTeX/TeX Live o súbelo a [Overleaf](https://overleaf.com): `pdflatex` dos
veces). El detalle del proyecto está en `README.md`.

## 🔬 Demo del análisis AMI con la muestra incluida

```powershell
& "C:\Users\aipri\anaconda3\envs\ml_env\python.exe" -m backend.pipeline_ami --demo
```

Corre el pipeline de anomalías sobre la muestra anonimizada de medidores
(`Raw_Processed/ACTIVE_Enero_0.parquet`) y deja los resultados en
`outputs/ami_demo/` sin tocar los oficiales.

---

**Requisitos** (ya cubiertos por el entorno `ml_env`): Python 3.10+ con
`numpy pandas scikit-learn scipy pyarrow`. No necesita internet: Plotly y
KaTeX están incluidos. Detalle completo del proyecto en `README.md`.
