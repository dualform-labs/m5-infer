#!/bin/bash
# M5 MLX Inference Engine — Interactive Chat
#
# Usage:
#   ./chat.sh                                               # default model
#   ./chat.sh mlx-community/Llama-3.2-3B-Instruct-4bit     # specific model

cd "$(dirname "$0")"
exec python3 "$(dirname "$0")/app/cli/chat.py" "$@"
