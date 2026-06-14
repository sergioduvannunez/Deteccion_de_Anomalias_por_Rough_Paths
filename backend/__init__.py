"""
backend — Detección de anomalías por Rough Path Signatures
           + Ecuaciones Diferenciales Neuronales (RNN → NODE → NCDE → NRDE).

Módulos:
  signaturas   — Chen vectorizado, log-signaturas, transformaciones de camino
  registro     — registry extensible de detectores
  detectores   — 13 clásicos + familia Signatures sobre el registro
  evaluacion   — protocolo PU (DR/AR) + matrices de Jaccard
  simuladores  — 3 contextos multivariados (it, ambiental, eeg)
  autodiff     — motor de backpropagation NumPy
  neuralde     — RNN, GRU, NeuralODE, NeuralCDE, NeuralRDE
  cargador     — datos AMI reales (parquet) y 4 modos de muestreo
  anomalias    — inyector de 7 anomalías AMI univariadas

Pipelines ejecutables:
  python -m backend.pipeline_anomalias   # contextos simulados -> outputs/simulados
  python -m backend.pipeline_neuralde    # modelos neuronales  -> outputs/neuralde
  python -m backend.pipeline_ami         # regenera CSV AMI    -> outputs/tablas_framework
"""

__version__ = "2.0"
