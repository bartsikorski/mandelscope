#!/usr/bin/env sh
# Mandelscope launcher for Linux / macOS-like shells
cd "$(dirname "$0")"
if command -v uv >/dev/null 2>&1; then
    exec uv run python -m mandelscope "$@"
else
    exec .venv/bin/python -m mandelscope "$@"
fi
