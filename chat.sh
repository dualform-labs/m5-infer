#!/bin/bash
# m5-infer — Interactive Chat CLI
#
# Usage:
#   ./chat.sh                                               # default model
#   ./chat.sh mlx-community/Llama-3.2-3B-Instruct-4bit     # specific model
#
# Requires the engine to be running (./start.sh --fg) on port 11436.

cd "$(dirname "$0")"
exec python3 "$(dirname "$0")/app/cli/chat.py" "$@"
