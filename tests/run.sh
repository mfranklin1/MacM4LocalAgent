#!/usr/bin/env bash
# tests/run.sh - run python + shell test suites and print a summary.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VENV="$REPO_ROOT/.venvs/test"

ensure_pytest() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found; install with: brew install uv" >&2
    exit 1
  fi
  if [[ ! -d "$VENV" ]]; then
    uv venv --python 3.12 "$VENV"
  fi
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  uv pip install --quiet --upgrade \
    "pytest>=8.0" "pytest-cov>=5.0" "pytest-asyncio>=0.23" \
    "fastapi>=0.110" "httpx>=0.27" "jinja2>=3.1" "python-multipart>=0.0.9" \
    "pyyaml>=6.0"
}

PASS=0; FAIL=0
section() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
pass()    { printf "  \033[1;32mPASS\033[0m  %s\n" "$*"; PASS=$((PASS+1)); }
fail()    { printf "  \033[1;31mFAIL\033[0m  %s\n" "$*"; FAIL=$((FAIL+1)); }

section "Python suite (pytest)"
ensure_pytest
if PYTHONPATH="$REPO_ROOT" pytest -q tests; then
  pass "pytest"
else
  fail "pytest"
fi

section "Shell: scripts + plists"
if bash "$REPO_ROOT/tests/test_scripts.sh"; then pass "test_scripts.sh"; else fail "test_scripts.sh"; fi

section "Shell: detect"
if bash "$REPO_ROOT/tests/test_detect.sh"; then pass "test_detect.sh"; else fail "test_detect.sh"; fi

section "Summary"
echo "  pass=$PASS  fail=$FAIL"
[[ "$FAIL" -eq 0 ]] || exit 1
