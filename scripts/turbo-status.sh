#!/usr/bin/env bash
# scripts/turbo-status.sh — Check turbo backend readiness.
# Reports: model downloaded?, mlx-lm turbo_kv_bits support?, launchd plist status.
# Invoked by: make turbo-status
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/config/detected.env" 2>/dev/null || true

PASS=0; FAIL=0; SKIP=0
ok()   { printf "  \033[1;32mPASS\033[0m  %s\n" "$*"; PASS=$((PASS+1)); }
fail() { printf "  \033[1;31mFAIL\033[0m  %s\n" "$*"; FAIL=$((FAIL+1)); }
skip() { printf "  \033[1;33mSKIP\033[0m  %s\n" "$*"; SKIP=$((SKIP+1)); }

echo "== Turbo backend status =="
echo

# 1) Model downloaded?
TURBO_MODEL_DIR="$REPO_ROOT/models/turbo"
if [[ -d "$TURBO_MODEL_DIR" ]] && [[ -n "$(ls -A "$TURBO_MODEL_DIR" 2>/dev/null)" ]]; then
  ok "model directory exists: $TURBO_MODEL_DIR"
else
  fail "model directory missing or empty ($TURBO_MODEL_DIR) — run 'make turbo-install'"
fi

# 2) mlx_lm installed with turbo_kv_bits support?
if python3 -c "import mlx_lm; import inspect; src=inspect.getsource(mlx_lm); assert 'kv_bits' in src" 2>/dev/null; then
  ok "mlx_lm has kv_bits (quantised KV-cache) support"
elif python3 -c "import mlx_lm" 2>/dev/null; then
  fail "mlx_lm installed but kv_bits not found — update mlx_lm: pip install -U mlx-lm"
else
  fail "mlx_lm not installed — run: pip install mlx-lm"
fi

# 3) TURBO_ENABLED flag
TURBO_ENABLED="${TURBO_ENABLED:-0}"
if [[ "$TURBO_ENABLED" == "1" ]]; then
  ok "TURBO_ENABLED=1 in detected.env"
else
  skip "TURBO_ENABLED=0 (disabled) — run 'make turbo-enable' to activate"
fi

# 4) launchd plist status for turbo-256k
PLIST_256="$REPO_ROOT/launchd/com.local.turbo-256k.rendered.plist"
if launchctl list 2>/dev/null | grep -q "com.local.turbo-256k"; then
  ok "launchd: com.local.turbo-256k is loaded"
elif [[ -f "$PLIST_256" ]]; then
  skip "launchd: com.local.turbo-256k plist present but not loaded"
else
  skip "launchd: com.local.turbo-256k plist not rendered yet — run 'make reconfigure'"
fi

# 5) 256k turbo HTTP health
TURBO_PORT="${TURBO_256K_PORT:-8082}"
if curl -fsS -m 3 "http://127.0.0.1:${TURBO_PORT}/health" >/dev/null 2>&1; then
  ok "turbo-256k HTTP health at :${TURBO_PORT}"
else
  skip "turbo-256k not responding on :${TURBO_PORT} (start with 'make turbo-start-256')"
fi

echo
echo "  pass=$PASS fail=$FAIL skip=$SKIP"
[[ "$FAIL" -eq 0 ]] || exit 1
