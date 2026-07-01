#!/usr/bin/env bash
# 00-detect.sh - Scan Mac capabilities and emit config/detected.env
# Drives downstream installers: picks model quant, KV cache type, and tiering.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$REPO_ROOT/config"
OUT="$CONFIG_DIR/detected.env"
mkdir -p "$CONFIG_DIR"

# Preserve fields that downstream scripts (20-ollama.sh) write back into
# detected.env so re-running `make detect` doesn't unpin the
# already-installed model tags.
PREV_OLLAMA_TAG=""
if [[ -f "$OUT" ]]; then
  # shellcheck disable=SC1090
  source "$OUT" 2>/dev/null || true
  PREV_OLLAMA_TAG="${OLLAMA_TAG:-}"
fi

log() { printf "\033[1;34m[detect]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[detect]\033[0m %s\n" "$*" >&2; }

if [[ "$(uname)" != "Darwin" ]]; then
  echo "This installer targets macOS (Apple Silicon). Detected: $(uname)" >&2
  exit 1
fi

CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
ARCH="$(uname -m)"
CORES_TOTAL="$(sysctl -n hw.ncpu)"
CORES_PERF="$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 0)"
CORES_EFF="$(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo 0)"
RAM_BYTES="$(sysctl -n hw.memsize)"
RAM_GB=$(( RAM_BYTES / 1024 / 1024 / 1024 ))

GPU_CORES="$(system_profiler SPDisplaysDataType 2>/dev/null \
  | awk -F': ' '/Total Number of Cores/ {print $2; exit}')"
GPU_CORES="${GPU_CORES:-unknown}"

DISK_FREE_GB="$(df -g / | awk 'NR==2 {print $4}')"

# Pick quant tier based on RAM (Qwen3-Coder-Next sizes: q8 ~85GB, q4 ~46GB).
# These are the *defaults* for a fresh install. If detected.env already
# pins an OLLAMA_TAG (because we previously downloaded it), keep that pin
# - the user's bandwidth was already spent, and re-pulling a different
# quant on every `make detect` would be hostile.
# 96 GB threshold (not 128 GB): accounts for ~30 GB OS + system overhead on M-series 128 GB machines.
if (( RAM_GB >= 96 )); then
  QUANT_TIER="q8"
  OLLAMA_TAG_DEFAULT="qwen3-coder-next:q8_0"
  LOCAL_LONG_CTX=131072
elif (( RAM_GB >= 48 )); then
  QUANT_TIER="q4"
  OLLAMA_TAG_DEFAULT="qwen3-coder-next:q4_K_M"
  LOCAL_LONG_CTX=65536
else
  QUANT_TIER="q4-small"
  OLLAMA_TAG_DEFAULT="qwen3-coder:30b"
  LOCAL_LONG_CTX=32768
  warn "Only ${RAM_GB}GB RAM detected; falling back to qwen3-coder:30b. 64GB+ recommended."
fi
if [[ -n "$PREV_OLLAMA_TAG" ]]; then
  OLLAMA_TAG="$PREV_OLLAMA_TAG"
  if [[ "$OLLAMA_TAG" != "$OLLAMA_TAG_DEFAULT" ]]; then
    warn "Keeping installed OLLAMA_TAG='$OLLAMA_TAG' (default for this RAM tier would be '$OLLAMA_TAG_DEFAULT'; delete config/detected.env to reset)."
    # Recompute QUANT_TIER / LOCAL_LONG_CTX from the *pinned*
    # tag so all derived values stay self-consistent.
    # For q4 variants we use RAM_GB to decide context: ≥96 GB can afford
    # 131072 (adds ~4.6 GiB KV cache on top of the ~45 GiB model weights),
    # while 48-95 GB machines stay at 65536 to leave room for the OS.
    case "$OLLAMA_TAG" in
      *:q8_0) QUANT_TIER="q8"; LOCAL_LONG_CTX=131072 ;;
      *:q4*)
        QUANT_TIER="q4"
        if (( RAM_GB >= 96 )); then LOCAL_LONG_CTX=131072
        else LOCAL_LONG_CTX=65536; fi ;;
      *) QUANT_TIER="q4-small"; LOCAL_LONG_CTX=32768 ;;
    esac
  fi
else
  OLLAMA_TAG="$OLLAMA_TAG_DEFAULT"
fi

# KV cache compression. We pick the strongest type that is *actually
# supported by the running Ollama daemon*, in this priority order:
#   tq3  - Google TurboQuant 3-bit (~5-6x compression). Not yet in any
#          released Ollama as of 2026-04. Tracked by ollama/ollama#15090
#          (closed pending MLX upstream) and ml-explore/mlx#3328.
#   q4_0 - Standard 4-bit block quant (~4x compression). Stable, requires
#          OLLAMA_FLASH_ATTENTION=1 (we set it).
#   q8_0 - 8-bit (~2x compression). Stable, requires Flash Attention.
#   f16  - No compression. Default fallback.
#
# To avoid the silent-fallback footgun where unsupported values like 'tq3'
# revert to f16 without warning, we resolve the strongest supported value
# at detect time and stash it in detected.env.
detect_kv_cache_type() {
  local supported
  if ! command -v ollama >/dev/null 2>&1; then
    echo "q4_0"; return
  fi
  # `ollama serve --help` doesn't list valid values, but the daemon binary
  # contains them as string literals. Grep is a pragmatic probe.
  supported="$(strings "$(command -v ollama)" 2>/dev/null \
    | grep -Eo '\b(tq3|tq4|q4_0|q8_0|f16)\b' | sort -u | tr '\n' ' ')"
  for cand in tq3 q4_0 q8_0 f16; do
    if [[ " $supported " == *" $cand "* ]]; then
      echo "$cand"; return
    fi
  done
  echo "f16"
}
KV_CACHE_TYPE="${KV_CACHE_TYPE_OVERRIDE:-$(detect_kv_cache_type)}"
case "$KV_CACHE_TYPE" in
  tq3|tq4) KV_CACHE_NOTE="TurboQuant rotation-quant (5-6x, real)" ;;
  q4_0)    KV_CACHE_NOTE="standard Q4_0 block quant (~4x, stable)" ;;
  q8_0)    KV_CACHE_NOTE="standard Q8_0 block quant (~2x, stable)" ;;
  *)       KV_CACHE_NOTE="uncompressed f16 fallback" ;;
esac

# Routing threshold (tokens).
# ROUTE_LONG_MAX MUST equal LOCAL_LONG_CTX: any prompt that fits in the
# local Ollama KV cache goes to local-long; anything larger escalates to
# Claude. Mismatching these (e.g. ROUTE_LONG_MAX=128000 with
# LOCAL_LONG_CTX=65536) causes Ollama to silently truncate prompts that
# land in the gap — the bug this line was added to prevent.
ROUTE_LONG_MAX=$LOCAL_LONG_CTX

# Ollama runtime tunables.
#  OLLAMA_NUM_PARALLEL=1: dedicate the full Metal pipeline to one request
#    so long-prompt prefill isn't sharing the KV-cache budget with a
#    second slot. Cursor sends one request at a time per session anyway.
#    Bump back to 2+ if you start running multi-pane parallel agents.
#  OLLAMA_KEEP_ALIVE=30m: keep the model resident for half an hour after
#    the last call. Default 5m evicts mid-session and forces a 60-90s
#    cold reload + fresh KV cache. 30m matches typical agent cadence.
OLLAMA_NUM_PARALLEL=1
OLLAMA_KEEP_ALIVE=30m

# Note: there is intentionally no LITELLM_MASTER_KEY. The LiteLLM proxy is
# bound to 127.0.0.1 only (see scripts/run_litellm.py / launchd plist), so
# the loopback bind IS the security boundary -- a bearer-token gate over
# loopback adds no protection any local process couldn't already bypass by
# reading detected.env. See config/litellm-config.yaml for re-enable
# instructions if you ever expose the proxy off-host.

cat > "$OUT" <<EOF
# Auto-generated by scripts/00-detect.sh on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Do not edit by hand; rerun \`make detect\`.
CHIP="$CHIP"
ARCH="$ARCH"
CORES_TOTAL=$CORES_TOTAL
CORES_PERF=$CORES_PERF
CORES_EFF=$CORES_EFF
RAM_GB=$RAM_GB
GPU_CORES="$GPU_CORES"
DISK_FREE_GB=$DISK_FREE_GB

QUANT_TIER="$QUANT_TIER"
OLLAMA_TAG="$OLLAMA_TAG"

KV_CACHE_TYPE="$KV_CACHE_TYPE"
LOCAL_LONG_CTX=$LOCAL_LONG_CTX

ROUTE_LONG_MAX=$ROUTE_LONG_MAX

OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL
OLLAMA_KEEP_ALIVE="$OLLAMA_KEEP_ALIVE"

LITELLM_PORT=4000
OLLAMA_PORT=11434
DASHBOARD_PORT=4001
CLAUDE_PROXY_PORT=4002
CLAUDE_PROXY_LARGE_CTX_MODE=passthrough
EOF

log "Hardware: $CHIP, ${RAM_GB}GB RAM, ${GPU_CORES}-core GPU, ${DISK_FREE_GB}GB free"
log "Quant tier: $QUANT_TIER ($OLLAMA_TAG)"
log "KV cache:   $KV_CACHE_TYPE ($KV_CACHE_NOTE)"
log "Context:    long=$LOCAL_LONG_CTX"
log "Wrote $OUT"
