#!/usr/bin/env bash
# tests/test_scripts.sh - syntax-check every shell script and Makefile target list.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0; FAIL=0
ok()   { printf "  PASS  %s\n" "$*"; PASS=$((PASS+1)); }
fail() { printf "  FAIL  %s\n" "$*"; FAIL=$((FAIL+1)); }

# 1) bash -n on all shell scripts and the install wrapper.
shopt -s nullglob
for f in "$REPO_ROOT"/scripts/*.sh "$REPO_ROOT"/install.sh "$REPO_ROOT"/tests/*.sh; do
  if bash -n "$f" 2>/dev/null; then ok "syntax: ${f#$REPO_ROOT/}"; else fail "syntax: ${f#$REPO_ROOT/}"; fi
done

# 2) All install scripts use `set -euo pipefail`.
for f in "$REPO_ROOT"/scripts/*.sh; do
  if grep -qE 'set -[a-z]*e[a-z]*' "$f" && grep -q 'pipefail' "$f"; then
    ok "strict mode: ${f#$REPO_ROOT/}"
  else
    fail "strict mode missing: ${f#$REPO_ROOT/}"
  fi
done

# 3) Makefile target presence. We parse the Makefile directly to avoid ANSI noise.
TARGETS_REQUIRED=(help detect install start stop restart status dashboard verify report compare clean nuke test)
MAKEFILE="$REPO_ROOT/Makefile"
for t in "${TARGETS_REQUIRED[@]}"; do
  if grep -qE "^$t:" "$MAKEFILE"; then
    ok "make target: $t"
  else
    fail "make target missing: $t"
  fi
done

# 4) launchd plists are well-formed XML.
# Skip *.rendered.plist — they may contain unsubstituted @@ tokens when turbo is disabled.
for p in "$REPO_ROOT"/launchd/*.plist; do
  case "$p" in *.rendered.plist) continue;; esac
  if plutil -lint "$p" >/dev/null 2>&1; then ok "plist: ${p##*/}"; else fail "plist: ${p##*/}"; fi
done

# 5) Plist placeholders are recognised by the renderer.
# Only check the source templates; *.rendered.plist files are intentionally
# substituted and won't contain @@ markers anymore.
for p in "$REPO_ROOT"/launchd/*.plist; do
  case "$p" in *.rendered.plist) continue;; esac
  if grep -q '@@REPO_ROOT@@' "$p"; then ok "placeholder @@REPO_ROOT@@ in ${p##*/}"; else fail "no @@REPO_ROOT@@ in ${p##*/}"; fi
done

echo "  pass=$PASS fail=$FAIL"
[[ "$FAIL" -eq 0 ]]
