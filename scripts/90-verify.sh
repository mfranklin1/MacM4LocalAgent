#!/usr/bin/env bash
# 90-verify.sh - Health-check every endpoint and run a 3-prompt smoke matrix.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config/detected.env"

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
probe_port "$OLLAMA_PORT"    "ollama"
probe_port "$MLX_PORT"       "mlx_lm.server"
probe_port "$LITELLM_PORT"   "litellm"
probe_port "$DASHBOARD_PORT" "dashboard"

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

if [[ "${OLLAMA_FLASH_ATTENTION:-$(launchctl getenv OLLAMA_FLASH_ATTENTION 2>/dev/null)}" == "1" ]]; then
  ok "OLLAMA_FLASH_ATTENTION=1 (required for q4_0/q8_0/tq3)"
else
  case "$EFFECTIVE_KV" in
    f16) ;;  # not required when uncompressed
    *) fail "OLLAMA_FLASH_ATTENTION not set; $EFFECTIVE_KV needs it to apply" ;;
  esac
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
smoke "hybrid-auto" "Refactor this small snippet: def f(x): return x*2" "hybrid-auto small -> local-fast"

# Build a long-ish prompt to probe local-long routing without hammering Claude.
LONG="$(python3 -c "print('// noise\n' * 5000)")"
smoke "hybrid-auto" "$LONG"$'\n\nSummarize the above noisy file in one sentence.' "hybrid-auto 18k tokens -> local-long"

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
