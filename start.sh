#!/bin/bash
# M5 MLX Inference Engine — start with log terminal
#
# Usage:
#   ./start.sh          # Start server + open log terminal
#   ./start.sh --fg     # Start in foreground (no separate terminal)

set -e
cd "$(dirname "$0")"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/m5-infer.log"
PID_FILE="$LOG_DIR/m5-infer.pid"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

if [ "$1" = "--fg" ]; then
    echo -e "${GREEN}M5 Inference Engine${NC} — foreground mode"
    echo -e "Logs: stdout"
    echo ""
    exec python3 -m app.api.server 2>&1 | tee "$LOG_FILE"
fi

# Kill existing server if running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Stopping existing server (PID $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$PID_FILE"
fi

# Resolve absolute paths (new Terminal window won't share cwd)
ABS_LOG_FILE="$(cd "$(dirname "$LOG_FILE")" && pwd)/$(basename "$LOG_FILE")"
ABS_PROJECT_DIR="$(pwd)"

# Start server in background with Super core priority
echo -e "${GREEN}Starting M5 Inference Engine (Super core priority)...${NC}"
taskpolicy -t 0 -l 0 python3 -m app.api.server > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
echo -e "Server PID: ${CYAN}$SERVER_PID${NC}"
echo -e "Log file:   ${CYAN}$ABS_LOG_FILE${NC}"

# Ensure log file exists before tailing
sleep 0.5
touch "$ABS_LOG_FILE"

# Open a new Terminal window that tails the log with absolute path
osascript <<APPLESCRIPT
tell application "Terminal"
    activate
    do script "clear; printf '\\\\033[1m🔍 M5 Inference Engine — Live Logs\\\\033[0m\\n\\n'; tail -f '${ABS_LOG_FILE}'"
end tell
APPLESCRIPT
if [ $? -ne 0 ]; then
    echo "(Could not open Terminal window — view logs with: tail -f $ABS_LOG_FILE)"
fi

# Wait for server to be ready
echo -n "Waiting for server..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:11436/health > /dev/null 2>&1; then
        echo -e " ${GREEN}ready!${NC}"
        echo ""
        curl -s http://127.0.0.1:11436/health | python3 -m json.tool 2>/dev/null || curl -s http://127.0.0.1:11436/health
        echo ""
        echo -e "API: ${CYAN}http://127.0.0.1:11436${NC}"
        echo -e "Stop: ${CYAN}kill $SERVER_PID${NC} or ${CYAN}./stop.sh${NC}"
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo " timeout (server may still be loading model)"
echo "Check logs: tail -f $LOG_FILE"
