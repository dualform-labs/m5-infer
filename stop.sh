#!/bin/bash
# Stop the M5 Inference Engine
cd "$(dirname "$0")"
PID_FILE="logs/m5-infer.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping M5 Inference Engine (PID $PID)..."
        kill "$PID"
        rm -f "$PID_FILE"
        echo "Stopped."
    else
        echo "Server not running (stale PID file)."
        rm -f "$PID_FILE"
    fi
else
    echo "No PID file found. Server may not be running."
fi
