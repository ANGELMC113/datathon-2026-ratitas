#!/usr/bin/env bash
# Mantiene el servicio vivo: si el proceso muere, vuelve a arrancar en 2 s.
# Uso: ./run_serve.sh [opciones de serve_v2.py]      (Ctrl+C dos veces para parar)
# Usa el Python del entorno virtual .venv (aquí o en la carpeta padre) si existe; si no, python3.
cd "$(dirname "$0")"
mkdir -p logs
if   [ -x .venv/bin/python ];    then PY=.venv/bin/python
elif [ -x ../.venv/bin/python ]; then PY=../.venv/bin/python
else PY=${PYTHON:-python3}; fi

if ! "$PY" -c "import numpy, pandas, requests, sklearn" 2>/dev/null; then
  echo "[run_serve] Faltan dependencias para $PY. Instálalas con:"
  echo "    python3 -m venv ../.venv && ../.venv/bin/pip install -r requirements-v2.txt"
  exit 1
fi

fast_fails=0
while true; do
  start=$(date +%s)
  "$PY" serve_v2.py "$@" 2>&1 | tee -a logs/serve-console.log
  if [ $(( $(date +%s) - start )) -lt 10 ]; then fast_fails=$((fast_fails + 1)); else fast_fails=0; fi
  if [ "$fast_fails" -ge 3 ]; then
    echo "[run_serve] ha fallado 3 veces seguidas al arrancar: lo paro. Mira el error de arriba."
    exit 1
  fi
  echo "[run_serve] el servicio terminó ($(date)); reinicio en 2 s"
  sleep 2
done
