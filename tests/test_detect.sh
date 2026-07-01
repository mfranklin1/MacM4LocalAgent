#!/usr/bin/env bash
# tests/test_detect.sh - assert scripts/00-detect.sh writes a sane detected.env.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Point CONFIG_DIR at a tmp dir by working in a sandbox copy.
WORK="$TMP/repo"
mkdir -p "$WORK"
cp -R "$REPO_ROOT/scripts" "$WORK/"

PASS=0; FAIL=0
ok()   { printf "  PASS  %s\n" "$*"; PASS=$((PASS+1)); }
fail() { printf "  FAIL  %s\n" "$*"; FAIL=$((FAIL+1)); }

cd "$WORK"
if bash scripts/00-detect.sh >/dev/null 2>"$TMP/detect.err"; then
  ok "scripts/00-detect.sh exit 0"
else
  echo "----- detect.err -----"; cat "$TMP/detect.err"
  fail "scripts/00-detect.sh nonzero exit"
fi

ENV_FILE="$WORK/config/detected.env"
[[ -f "$ENV_FILE" ]] && ok "detected.env exists" || fail "detected.env missing"

if [[ -f "$ENV_FILE" ]]; then
  for key in CHIP RAM_GB GPU_CORES QUANT_TIER OLLAMA_TAG KV_CACHE_TYPE \
             LOCAL_LONG_CTX ROUTE_LONG_MAX \
             LITELLM_PORT OLLAMA_PORT DASHBOARD_PORT; do
    if grep -q "^$key=" "$ENV_FILE"; then ok "key $key present"; else fail "key $key missing"; fi
  done

  # shellcheck disable=SC1090
  source "$ENV_FILE"
  # KV_CACHE_TYPE must be one we *actually* support. tq3/tq4 are accepted
  # too because the detector will pick them automatically once Ollama
  # ships native TurboQuant. f16 is the safe fallback when nothing better
  # is available and is also acceptable.
  case "$KV_CACHE_TYPE" in
    tq3|tq4|q4_0|q8_0|f16) ok "KV_CACHE_TYPE=$KV_CACHE_TYPE (supported)" ;;
    *)                     fail "KV_CACHE_TYPE='$KV_CACHE_TYPE' not a supported Ollama KV cache type" ;;
  esac
  if (( ROUTE_LONG_MAX == LOCAL_LONG_CTX )); then ok "ROUTE_LONG_MAX == LOCAL_LONG_CTX"; else fail "ROUTE_LONG_MAX must equal LOCAL_LONG_CTX"; fi
fi

echo "  pass=$PASS fail=$FAIL"
[[ "$FAIL" -eq 0 ]]
