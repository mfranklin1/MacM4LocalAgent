#!/usr/bin/env bash
# scripts/turbo-install.sh — Download the MLX turbo model (~20 GB).
# Invoked by: make turbo-install
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/config/detected.env" 2>/dev/null || true

TURBO_MODEL="${TURBO_MODEL_ID:-mlx-community/Qwen2.5-Coder-32B-Instruct-4bit}"

echo "==> turbo-install: downloading $TURBO_MODEL"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found; cannot download MLX model." >&2
  exit 1
fi

if python3 -c "import mlx_lm" 2>/dev/null; then
  echo "  [mlx_lm] pulling $TURBO_MODEL ..."
  python3 -m mlx_lm.utils --model "$TURBO_MODEL" --local-dir models/turbo
else
  echo "  [huggingface-hub] mlx_lm not installed; falling back to huggingface-cli ..."
  if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "  Installing huggingface-hub ..."
    pip3 install -q huggingface-hub
  fi
  huggingface-cli download "$TURBO_MODEL" --local-dir models/turbo
fi

echo "==> turbo-install: done — model saved to $(pwd)/models/turbo"
echo "    Run 'make turbo-enable' then restart the proxy to activate."
