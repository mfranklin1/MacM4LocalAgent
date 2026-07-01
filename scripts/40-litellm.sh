#!/usr/bin/env bash
# 40-litellm.sh - Install LiteLLM proxy + size-based router callback.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config/detected.env"

log() { printf "\033[1;34m[litellm]\033[0m %s\n" "$*"; }

VENV="$REPO_ROOT/.venvs/litellm"
if [[ ! -d "$VENV" ]]; then
  log "creating venv at $VENV"
  uv venv --python 3.12 "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

log "installing litellm[proxy] + deps"
uv pip install --upgrade pip >/dev/null
uv pip install --upgrade \
  "litellm[proxy]>=1.89,<1.90" \
  "anthropic>=0.40" \
  "tiktoken>=0.7" \
  "fastapi>=0.110" \
  "uvicorn>=0.30" \
  "httpx>=0.27" \
  "sqlite-utils>=3.36" \
  "jinja2>=3.1"

# Render the LiteLLM config from the template, substituting detected values.
TEMPLATE="$REPO_ROOT/config/litellm-config.yaml"
if [[ ! -f "$TEMPLATE" ]]; then
  log "no config/litellm-config.yaml yet (will be in repo); creating default"
fi

# The actually-used config gets the resolved Ollama tag from detected.env
# so renames in the backend don't strand the proxy.
RENDERED="$REPO_ROOT/config/litellm-config.rendered.yaml"
: "${LOCAL_LONG_CTX:?LOCAL_LONG_CTX missing from detected.env; rerun make detect}"
# TURBO_MODEL_LOCAL_DIR is optional — only needed when TURBO_ENABLED=1.
# Default to a sensible path under the repo's models/ directory so the
# rendered config is syntactically valid even before the turbo model is downloaded.
TURBO_MODEL_LOCAL_DIR="${TURBO_MODEL_LOCAL_DIR:-$HOME/Documents/GitHub/MacM4LocalAgent/models/mlx-community_Qwen2.5-Coder-32B-Instruct-4bit}"
sed -e "s|@@OLLAMA_TAG@@|$OLLAMA_TAG|g" \
    -e "s|@@LOCAL_LONG_CTX@@|$LOCAL_LONG_CTX|g" \
    -e "s|@@OLLAMA_PORT@@|$OLLAMA_PORT|g" \
    -e "s|@@TURBO_MODEL_LOCAL_DIR@@|${TURBO_MODEL_LOCAL_DIR}|g" \
    "$TEMPLATE" > "$RENDERED"
log "rendered config: $RENDERED"

# LiteLLM's callback loader resolves dotted module paths relative to the
# config file's directory when config_file_path is set. Symlink router/ and
# cost/ into config/ so `router.route_by_size.proxy_handler_instance` resolves.
ln -sf "../router" "$REPO_ROOT/config/router"
ln -sf "../cost"   "$REPO_ROOT/config/cost"
log "linked config/router -> ../router and config/cost -> ../cost"

# Sanity-check that the router module imports.
PYTHONPATH="$REPO_ROOT" python -c "from router.route_by_size import proxy_handler_instance; print('router ok')"

log "done"
