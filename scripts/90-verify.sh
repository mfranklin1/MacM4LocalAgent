#!/usr/bin/env bash
# 90-verify.sh - Health-check every endpoint and run a 3-prompt smoke matrix.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config/detected.env"

# Auth header array for curl calls that hit authenticated endpoints.
# /v1/models is open (loopback-only, no auth gate); /v1/chat/completions requires the master key.
HEADERS=(-H "Authorization: Bearer ${LITELLM_MASTER_KEY:-}")

PASS=0
FAIL=0

ok()   { printf "  \033[1;32mPASS\033[0m  %s\n" "$*"; PASS=$((PASS+1)); }
fail() { printf "  \033[1;31mFAIL\033[0m  %s\n" "$*"; FAIL=$((FAIL+1)); }

probe_port() {
  local port="$1" name="$2"
  if lsof -nP -iTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1; then
    ok "$name on :$port is listening"
  else
    fail "$name on :$port is NOT listening"
  fi
}

echo
echo "== Port health =="
probe_port "$OLLAMA_PORT"                       "ollama"
probe_port "$MLX_PORT"                          "mlx_lm.server"
probe_port "$LITELLM_PORT"                      "litellm"
probe_port "$DASHBOARD_PORT"                    "dashboard"
probe_port "${CLAUDE_PROXY_PORT:-4002}"         "claude-proxy"

echo
echo "== Ollama KV cache =="
# Honest probe: don't trust the env var alone, ask the running daemon.
# `launchctl getenv` shows what was set, not what Ollama accepted. If we
# asked for an unsupported type (e.g. tq3 against stable 0.21.x), Ollama
# silently falls back to f16 - this section catches that.
REQUESTED_KV="${KV_CACHE_TYPE:-$(launchctl getenv OLLAMA_KV_CACHE_TYPE 2>/dev/null || true)}"
EFFECTIVE_KV=""
# 1) Strings-grep the running ollama binary for the value being requested.
#    If the literal isn't present in the daemon binary, the daemon cannot
#    honor that value -> it will silently degrade to f16.
OLLAMA_BIN="$(command -v ollama 2>/dev/null || echo /opt/homebrew/bin/ollama)"
if [[ -x "$OLLAMA_BIN" ]] && strings "$OLLAMA_BIN" 2>/dev/null \
   | grep -Eo '\b(tq3|tq4|q4_0|q8_0|f16)\b' | grep -qx "$REQUESTED_KV"; then
  EFFECTIVE_KV="$REQUESTED_KV"
else
  EFFECTIVE_KV="f16 (silent fallback - $REQUESTED_KV not supported by this Ollama build)"
fi

case "$EFFECTIVE_KV" in
  tq3|tq4)
    ok "KV cache: $EFFECTIVE_KV (TurboQuant rotation-quant, ~5-6x compression)" ;;
  q4_0)
    ok "KV cache: q4_0 (~4x compression, stable)" ;;
  q8_0)
    ok "KV cache: q8_0 (~2x compression, stable)" ;;
  f16)
    ok "KV cache: f16 (no compression - intentional)" ;;
  *)
    fail "KV cache: requested='$REQUESTED_KV' effective='$EFFECTIVE_KV'" ;;
esac

# Sanity: requested vs effective must match if user explicitly chose a
# compressed type. This is the check that would have caught our tq3 silent
# fallback during the original install.
if [[ -n "$REQUESTED_KV" && "$REQUESTED_KV" != f16 && "$EFFECTIVE_KV" != "$REQUESTED_KV" ]]; then
  fail "KV cache mismatch: requested '$REQUESTED_KV' but daemon does not support it (effective: '$EFFECTIVE_KV'). Run \`make detect && make finalize\` to re-pick a supported type."
fi

# Check OLLAMA_FLASH_ATTENTION from the running process env first, then
# fall back to reading the launchd plist (the env var is set per-plist, not
# globally, so `launchctl getenv` misses it).
_FLASH_VAL="${OLLAMA_FLASH_ATTENTION:-$(launchctl getenv OLLAMA_FLASH_ATTENTION 2>/dev/null)}"
if [[ -z "$_FLASH_VAL" ]]; then
  _OLLAMA_PLIST=~/Library/LaunchAgents/com.local.ollama.plist
  if [[ -f "$_OLLAMA_PLIST" ]]; then
    _FLASH_VAL="$(plutil -extract EnvironmentVariables.OLLAMA_FLASH_ATTENTION raw "$_OLLAMA_PLIST" 2>/dev/null || true)"
  fi
fi
if [[ "$_FLASH_VAL" == "1" ]]; then
  ok "OLLAMA_FLASH_ATTENTION=1 (required for q4_0/q8_0/tq3)"
else
  case "$EFFECTIVE_KV" in
    f16) ;;  # not required when uncompressed
    *) fail "OLLAMA_FLASH_ATTENTION not set; $EFFECTIVE_KV needs it to apply" ;;
  esac
fi

echo
echo "== claude-proxy health =="
PROXY_PORT="${CLAUDE_PROXY_PORT:-4002}"
PROXY_RESP="$(curl -fsS "http://127.0.0.1:${PROXY_PORT}/health" 2>/dev/null || true)"
if [[ -n "$PROXY_RESP" ]]; then
  PROXY_STATUS="$(echo "$PROXY_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','?'))" 2>/dev/null || true)"
  PROXY_MODE="$(echo "$PROXY_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('large_ctx_mode','?'))" 2>/dev/null || true)"
  if [[ "$PROXY_STATUS" == "ok" ]]; then
    ok "claude-proxy /health → status=ok large_ctx_mode=${PROXY_MODE}"
    if [[ "$PROXY_MODE" == "passthrough" ]]; then
      ok "claude-proxy large-ctx uses Team OAuth (no ANTHROPIC_API_KEY needed)"
    elif [[ "$PROXY_MODE" == "apikey" ]]; then
      if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        ok "claude-proxy large-ctx uses ANTHROPIC_API_KEY (apikey mode)"
      else
        fail "claude-proxy large_ctx_mode=apikey but ANTHROPIC_API_KEY is not set"
      fi
    fi
  else
    fail "claude-proxy /health returned unexpected status: $PROXY_STATUS"
  fi
else
  fail "could not reach claude-proxy /health on :${PROXY_PORT}"
fi

echo
echo "== LiteLLM model registry =="
# The proxy is loopback-only (127.0.0.1) with no auth gate; no bearer token needed.
RESP="$(curl -fsS "http://127.0.0.1:${LITELLM_PORT}/v1/models" || true)"
if [[ -n "$RESP" ]]; then
  for m in local-fast local-long claude-code hybrid-auto; do
    if echo "$RESP" | grep -q "\"$m\""; then ok "model '$m' registered"; else fail "model '$m' missing"; fi
  done
else
  fail "could not reach litellm /v1/models"
fi

smoke() {
  local model="$1" prompt="$2" label="$3"
  body="$(jq -n --arg m "$model" --arg p "$prompt" \
    '{model:$m, messages:[{role:"user", content:$p}], max_tokens:64}')"
  resp="$(curl -fsS -m 120 "${HEADERS[@]}" -H "Content-Type: application/json" \
    -X POST "http://127.0.0.1:${LITELLM_PORT}/v1/chat/completions" -d "$body" 2>/dev/null || true)"
  if [[ -n "$resp" ]] && echo "$resp" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    ok "smoke: $label"
  else
    fail "smoke: $label (no valid response)"
  fi
}

echo
echo "== Smoke matrix =="
smoke "local-fast"  "Write a one-line Python function that returns x+1." "local-fast 1k tokens"
# Use a prompt that is clearly below ROUTE_FAST_MAX and contains no complexity-classifier keywords.
smoke "hybrid-auto" "What is the capital of France?" "hybrid-auto tiny -> local-fast"

# 7000 lines × ~2.5 tok/line ≈ 17500 tokens > ROUTE_FAST_MAX(16000) → routes to local-long.
# (5000 lines only reached ~12500 tokens, below the threshold.)
LONG="$(python3 -c "print('// noise\n' * 7000)")"
smoke "hybrid-auto" "$LONG"$'\n\nSummarize the above noisy file in one sentence.' "hybrid-auto 17k tokens -> local-long"

# claude-code only if ANTHROPIC_API_KEY is set.
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  smoke "claude-code" "Say hi in 3 words." "claude-code minimal"
else
  printf "  \033[1;33mSKIP\033[0m  claude-code (set ANTHROPIC_API_KEY to test)\n"
fi

echo
echo "== Summary =="
echo "  pass: $PASS"
echo "  fail: $FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
