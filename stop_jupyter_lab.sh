#!/usr/bin/env bash
# Stop the Jupyter Lab instance started by ./start_jupyter_lab.sh.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${PROJECT_DIR}/.jupyter/lab.pid"

if [[ ! -f "${PID_FILE}" ]]; then
    echo "No PID file at ${PID_FILE}."
    if pgrep -f "jupyter-lab.*--port" >/dev/null 2>&1; then
        echo "Found running jupyter-lab processes; terminating via pkill."
        pkill -f "jupyter-lab.*--port" || true
    else
        echo "Jupyter Lab is not running."
    fi
    exit 0
fi

pid=$(cat "${PID_FILE}")
if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    echo "SIGTERM sent to pid ${pid}; waiting for shutdown..."
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "${pid}" 2>/dev/null; then
        echo "Still alive; sending SIGKILL."
        kill -9 "${pid}" || true
    fi
    echo "Jupyter Lab stopped."
else
    echo "Process ${pid} not running; cleaning stale PID file."
fi

rm -f "${PID_FILE}"
