#!/usr/bin/env bash
# Start Jupyter Lab in the background, bound to 127.0.0.1.
# Log + PID written to .jupyter/.  Stop with ./stop_jupyter_lab.sh.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/.jupyter"
PID_FILE="${RUNTIME_DIR}/lab.pid"
LOG_FILE="${RUNTIME_DIR}/lab.log"
PORT="${JUPYTER_PORT:-8888}"

mkdir -p "${RUNTIME_DIR}"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    pid=$(cat "${PID_FILE}")
    echo "Jupyter Lab already running (pid ${pid}) — http://127.0.0.1:${PORT}"
    echo "Log: ${LOG_FILE}"
    exit 0
fi

cd "${PROJECT_DIR}"

nohup uv run jupyter lab \
    --no-browser \
    --ip=127.0.0.1 \
    --port="${PORT}" \
    --ServerApp.token='' \
    --ServerApp.password='' \
    >"${LOG_FILE}" 2>&1 &

pid=$!
echo "${pid}" >"${PID_FILE}"
disown "${pid}" 2>/dev/null || true

sleep 2
if ! kill -0 "${pid}" 2>/dev/null; then
    echo "Jupyter Lab failed to start. Last log lines:"
    tail -n 30 "${LOG_FILE}"
    rm -f "${PID_FILE}"
    exit 1
fi

echo "Jupyter Lab started (pid ${pid}) — http://127.0.0.1:${PORT}/lab"
echo "Log: ${LOG_FILE}"
