#!/usr/bin/env bash
# run.sh — lanzador para macOS y Linux
#
# Uso:
#   bash run.sh          # puerto 8001 (por defecto)
#   bash run.sh 8080     # puerto personalizado

set -e

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUERTO="${1:-8001}"

# Buscar Python 3.9+ disponible en el sistema
PY=""
for candidate in python3 python; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" -c "import sys; print(sys.version_info >= (3,9))" 2>/dev/null)
        if [ "$version" = "True" ]; then
            PY="$candidate"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "[ERROR] No se encontró Python 3.9 o superior."
    echo "        Instálalo desde https://www.python.org/downloads/"
    exit 1
fi

echo "============================================================"
echo "   ROUGH PATHS LAB"
echo "   Detección de anomalías por signaturas + Neural ODE/CDE/RDE"
echo "============================================================"
echo ""
echo "   Iniciando servidor en http://localhost:$PUERTO"
echo "   Para DETENER: Ctrl+C"
echo "============================================================"
echo ""

"$PY" "$AQUI/run.py" "$PUERTO"
