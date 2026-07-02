#!/usr/bin/env bash
# download-ollama-from-hf.sh
#
# Download a GGUF model from a US-hosted Hugging Face repo using
# hf_transfer (8 parallel HTTPS connections per file) and register it
# with the local Ollama daemon via `ollama create`.
#
# Why: pulling from registry.ollama.ai is throttled to ~1 MB/s per
# connection by Cloudflare's US edge for our network. HF is
# Cloudfront-fronted (also US, also LAX) but supports
# multi-connection scaling — `hf_transfer` opens 8 parallel range
# requests and aggregate throughput jumps to 8-10 MB/s.
#
# Usage:
#   bash scripts/download-ollama-from-hf.sh \
#     <hf_repo_id> <gguf_filename_or_glob> <ollama_tag>
#
# Examples:
#   bash scripts/download-ollama-from-hf.sh \
#     bartowski/Qwen_Qwen3-Coder-Next-GGUF \
#     'Qwen_Qwen3-Coder-Next-Q4_K_M.gguf' \
#     qwen3-coder-next:q4_K_M
#
# Defaults are the bartowski Qwen3-Coder-Next Q4_K_M variant if no
# arguments are provided.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config/detected.env"

HF_REPO="${1:-bartowski/Qwen_Qwen3-Coder-Next-GGUF}"
HF_GLOB="${2:-Qwen_Qwen3-Coder-Next-Q4_K_M.gguf}"
OLLAMA_TAG_NEW="${3:-qwen3-coder-next:q4_K_M}"

LOG_DIR="$REPO_ROOT/.logs"
LOG="$LOG_DIR/install-ollama-from-hf.log"
TARGET_DIR="$REPO_ROOT/models/${HF_REPO//\//_}"
mkdir -p "$LOG_DIR" "$TARGET_DIR"

log()  { printf "\033[1;34m[hf-ollama]\033[0m %s\n" "$*" | tee -a "$LOG"; }
warn() { printf "\033[1;33m[hf-ollama]\033[0m %s\n" "$*" | tee -a "$LOG" >&2; }
err()  { printf "\033[1;31m[hf-ollama]\033[0m %s\n" "$*" | tee -a "$LOG" >&2; }

if ! command -v ollama >/dev/null 2>&1; then
  err "ollama CLI not found on PATH"; exit 1
fi
if ! curl -fsS --max-time 5 "http://127.0.0.1:${OLLAMA_PORT:-11434}/api/version" >/dev/null; then
  err "ollama daemon not responding on port ${OLLAMA_PORT:-11434}; start it with: launchctl load launchd/com.local.ollama.rendered.plist"
  exit 1
fi

VENV="$REPO_ROOT/.venvs/mlx"
if [[ ! -x "$VENV/bin/python" ]]; then
  err "MLX venv not found at $VENV. Run scripts/30-mlx.sh first."; exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

log "==== HF -> Ollama import ===="
log "repo:      $HF_REPO"
log "filename:  $HF_GLOB"
log "ollama tag: $OLLAMA_TAG_NEW"
log "target:    $TARGET_DIR"
log "log:       $LOG"

# Force IPv4 (Cloudflare IPv6 reset issues observed during testing).
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DOWNLOAD_TIMEOUT=60

DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-7200}"   # 2 hours
STALL_WINDOW="${STALL_WINDOW:-300}"            # 5 min of zero growth = abort

log "downloading via hf_transfer (8 parallel streams), timeout=${DOWNLOAD_TIMEOUT}s, stall_window=${STALL_WINDOW}s"

python - <<PY 2>&1 | tee -a "$LOG"
import os, sys, threading, time, pathlib

target = "$TARGET_DIR"
repo   = "$HF_REPO"
glob   = "$HF_GLOB"
timeout = $DOWNLOAD_TIMEOUT
stall_window = $STALL_WINDOW

print(f"[python] hf_transfer={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}", flush=True)

from huggingface_hub import snapshot_download

done = threading.Event()

def total_bytes_in_target():
    # Sum bytes across all files in target_dir, INCLUDING the
    # .cache/huggingface/download/<hash>.incomplete files where hf_transfer
    # writes its in-flight payload before atomic-renaming on completion.
    size = 0
    p = pathlib.Path(target)
    if not p.exists():
        return 0
    for f in p.rglob("*"):
        try:
            if f.is_file():
                size += f.stat().st_size
        except OSError:
            pass
    return size

def watchdog():
    last_size = -1
    last_change = time.time()
    last_print = last_change
    deadline = last_change + timeout
    while not done.is_set():
        time.sleep(5)
        size = total_bytes_in_target()
        if size != last_size:
            last_size = size
            last_change = time.time()
        now = time.time()
        if now - last_print >= 15:
            gb = size / 1024 / 1024 / 1024
            stale = int(now - last_change)
            print(f"[watchdog] target_size={gb:.2f} GB  since_last_growth={stale}s", flush=True)
            last_print = now
        if now - last_change > stall_window:
            print(f"STALL: no progress for {stall_window}s; aborting", flush=True)
            os._exit(2)
        if now > deadline:
            print(f"TIMEOUT: download exceeded {timeout}s", flush=True)
            os._exit(3)

threading.Thread(target=watchdog, daemon=True).start()

try:
    snapshot_download(
        repo_id=repo,
        local_dir=target,
        allow_patterns=[glob],
        max_workers=8,
    )
    print("DOWNLOAD_OK", flush=True)
except Exception as e:
    print(f"DOWNLOAD_FAIL: {e}", flush=True)
    sys.exit(1)
finally:
    done.set()
PY

# Locate the downloaded GGUF file.
GGUF_PATH=$(find "$TARGET_DIR" -type f -name "$HF_GLOB" 2>/dev/null | head -1)
if [[ -z "$GGUF_PATH" || ! -f "$GGUF_PATH" ]]; then
  err "GGUF file not found after download (looked for $HF_GLOB in $TARGET_DIR)"
  exit 1
fi

GGUF_SIZE=$(stat -f "%z" "$GGUF_PATH")
GGUF_GB=$(awk -v s="$GGUF_SIZE" 'BEGIN { printf "%.1f", s/1024/1024/1024 }')
log "GGUF on disk: $GGUF_PATH (${GGUF_GB} GB)"

# Build a Modelfile that points Ollama at the GGUF and sets the long-context
# parameters we need. The KV cache type comes from the daemon's launchd env
# (OLLAMA_KV_CACHE_TYPE=tq3) so we don't need to set it here.
#
# The TEMPLATE block matters: bartowski's GGUF for Qwen3-Coder-Next ships
# WITHOUT a chat template baked in, and Ollama's default fallback when no
# TEMPLATE is set is `{{ .Prompt }}` (raw passthrough, no role markers).
# When LiteLLM forwards OpenAI-shaped role messages through Ollama under
# that empty template, Ollama crudely serializes them as
# `### User: ... ### Assistant: ...` markdown headers. The model is
# trained on canonical ChatML (`<|im_start|>role\n...<|im_end|>\n`), so
# it sees the markdown-header form as out-of-distribution and tends to
# hallucinate `### User:` continuations mid-response (observed in Cline
# turn-3: 359 output tokens with fake "### Assistant:" preamble and
# fake completion). Setting the canonical Qwen3 ChatML template here
# fixes that at the source: Cline turn-3 drops from 359 -> 93 tokens
# and the hallucinated role-markers disappear.
#
# The eos token for Qwen3 is <|im_end|>; we add it as a stop parameter
# so the model halts cleanly at the end of an assistant turn even if
# the client doesn't pass its own stop list.
#
# RENDERER/PARSER (2026-07-02): use Ollama's built-in qwen3-coder
# renderer + parser (Ollama >= 0.12) instead of a handwritten TEMPLATE.
# The renderer emits the full canonical chat format INCLUDING the tools
# section, and the parser turns the model's native tool-call output into
# structured OpenAI `tool_calls`. A handwritten ChatML TEMPLATE (the
# previous approach) leaves the model registered with
# `Capabilities: completion` only -- Ollama then rejects `tools` on the
# OpenAI endpoint and every tool call reaches the harness as raw JSON
# text that Cline can't execute (the "tool-call-as-text" stall).
MODELFILE="$TARGET_DIR/Modelfile"
cat > "$MODELFILE" <<'MODELFILE_EOF'
FROM __GGUF_PATH__
RENDERER qwen3-coder
PARSER qwen3-coder
PARAMETER num_ctx __LOCAL_LONG_CTX__
PARAMETER num_predict 8192
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
MODELFILE_EOF
sed -i.bak \
    -e "s|__GGUF_PATH__|$GGUF_PATH|" \
    -e "s|__LOCAL_LONG_CTX__|${LOCAL_LONG_CTX:-131072}|" \
    "$MODELFILE"
rm -f "${MODELFILE}.bak"

log "Modelfile:"
sed 's/^/    /' "$MODELFILE" | tee -a "$LOG"

log "registering with ollama as: $OLLAMA_TAG_NEW"
if ollama create "$OLLAMA_TAG_NEW" -f "$MODELFILE" 2>&1 | tee -a "$LOG"; then
  log "OK: $OLLAMA_TAG_NEW registered"
else
  err "ollama create failed"; exit 1
fi

# Verify it's listed.
log "ollama list:"
ollama list | tee -a "$LOG"

# Persist the new tag back into config/detected.env.
sed -i.bak "s|^OLLAMA_TAG=.*|OLLAMA_TAG=\"$OLLAMA_TAG_NEW\"|" "$REPO_ROOT/config/detected.env"
rm -f "$REPO_ROOT/config/detected.env.bak"
log "updated config/detected.env: OLLAMA_TAG=\"$OLLAMA_TAG_NEW\""

log "==== done ===="
