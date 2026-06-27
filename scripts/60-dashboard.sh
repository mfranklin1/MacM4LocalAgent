#!/usr/bin/env bash
# 60-dashboard.sh - Install dashboard deps into the litellm venv (shared
# Python env keeps memory low; both servers use FastAPI).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log() { printf "\033[1;34m[dashboard]\033[0m %s\n" "$*"; }

VENV="$REPO_ROOT/.venvs/litellm"
if [[ ! -d "$VENV" ]]; then
  echo "litellm venv missing; run scripts/40-litellm.sh first" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

uv pip install --upgrade \
  "fastapi>=0.110" \
  "uvicorn>=0.30" \
  "jinja2>=3.1" \
  "httpx>=0.27" \
  "python-multipart>=0.0.9"

# Ensure cost.db exists with schema applied.
PYTHONPATH="$REPO_ROOT" python -c "from cost.ingest import connect; connect().close(); print('cost.db ready')"

mkdir -p "$REPO_ROOT/.logs"

# Render launchd plists with absolute paths and per-user values.
# shellcheck disable=SC1091
source "$REPO_ROOT/config/detected.env"
# Source the root .env (one level up from REPO_ROOT) to pick up user-configured
# values: ANTHROPIC_API_KEY, CLAUDE_AUTH_MODE, etc.  setdefault semantics:
# already-exported vars win; this is a fallback for values not yet in the shell.
ROOT_ENV="$REPO_ROOT/../.env"
if [[ -f "$ROOT_ENV" ]]; then
  while IFS= read -r _line; do
    [[ "$_line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${_line// }" ]] && continue
    _key="${_line%%=*}"
    [[ -z "${!_key+x}" ]] && export "$_line" 2>/dev/null || true
  done < "$ROOT_ENV"
fi

MLX_LOCAL_DIR="${MLX_LOCAL_DIR:-$REPO_ROOT/models/mlx}"

: "${KV_CACHE_TYPE:=q4_0}"
: "${OLLAMA_NUM_PARALLEL:=1}"
: "${OLLAMA_KEEP_ALIVE:=30m}"
: "${OLLAMA_PORT:=11434}"
: "${OLLAMA_TAG:=qwen3-coder-next:q4_K_M}"
: "${CLAUDE_AUTH_MODE:=subscription}"
: "${ANTHROPIC_API_KEY:=}"

for src in "$REPO_ROOT"/launchd/*.plist; do
  case "$src" in *.rendered.plist) continue;; esac
  out="${src%.plist}.rendered.plist"
  sed -e "s|@@REPO_ROOT@@|$REPO_ROOT|g" \
      -e "s|@@MLX_LOCAL_DIR@@|$MLX_LOCAL_DIR|g" \
      -e "s|@@KV_CACHE_TYPE@@|$KV_CACHE_TYPE|g" \
      -e "s|@@OLLAMA_NUM_PARALLEL@@|$OLLAMA_NUM_PARALLEL|g" \
      -e "s|@@OLLAMA_KEEP_ALIVE@@|$OLLAMA_KEEP_ALIVE|g" \
      -e "s|@@OLLAMA_PORT@@|$OLLAMA_PORT|g" \
      -e "s|@@OLLAMA_TAG@@|$OLLAMA_TAG|g" \
      -e "s|@@HOME@@|$HOME|g" \
      -e "s|@@USER@@|$USER|g" \
      -e "s|@@TMPDIR@@|${TMPDIR:-/tmp}|g" \
      -e "s|@@CLAUDE_AUTH_MODE@@|$CLAUDE_AUTH_MODE|g" \
      -e "s|@@ANTHROPIC_API_KEY@@|$ANTHROPIC_API_KEY|g" \
      "$src" > "$out"
  log "rendered $(basename "$out") (KV=$KV_CACHE_TYPE, parallel=$OLLAMA_NUM_PARALLEL, keep_alive=$OLLAMA_KEEP_ALIVE)"
done

log "done"
