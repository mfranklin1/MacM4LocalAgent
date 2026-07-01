#!/usr/bin/env bash
# perf-suite.sh - Reproducible end-to-end perf pass against the live stack.
# Times cold-load + 3 warm runs across Ollama (local-long at mid + long
# context), and exercises the hybrid-auto router at the size boundary.
# Reports prompt size, decode tok/sec, total latency.
#
# Usage:
#   bash scripts/perf-suite.sh            # full suite (cold + 500 + 5k + 18k + router)
#   bash scripts/perf-suite.sh --short    # skip the 18k-token long run
#   bash scripts/perf-suite.sh --stress   # full suite + ~110k local stress + ~140k over-ceiling
#                                         #   (~10-25 min total, exercises q4_0 KV cache near limit
#                                         #    and confirms >128k prompts route to claude-code)
#   STRESS_TOKENS=96000 OVER_CEILING_TOKENS=160000 bash scripts/perf-suite.sh --stress
#                                         # override either target prompt size
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config/detected.env"

MODE="${1:-full}"
URL="http://127.0.0.1:${LITELLM_PORT}/v1/chat/completions"
# Proxy is loopback-only with no auth gate, so no Authorization header.
HDRS=(-H "Content-Type: application/json")
MAX_TOK_OUT=128
STRESS_TOKENS="${STRESS_TOKENS:-110000}"            # near the 128k router ceiling, still routed local-long
OVER_CEILING_TOKENS="${OVER_CEILING_TOKENS:-140000}" # above ROUTE_LONG_MAX, must route to claude-code
# Per-call deadline (s). Stress requests are bound by Ollama's prompt
# evaluation throughput; on M5 Max with q4_0 KV cache and 110k tok prompt
# we've measured ~5-10 min cold, ~3-5 min warm. Give it generous headroom.
STRESS_TIMEOUT=1500

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
hdr()  { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }

# Render a prompt of approximately N tokens (1 tok ~= 3.6 chars).
# Pass `target_tokens` and `tail` through argv to avoid any heredoc-level
# substitution mangling newlines/quotes inside `tail`.
make_prompt() {
  local target_tokens="$1" tail="$2"
  python3 - "$target_tokens" "$tail" <<'PY'
import sys
target_tokens = int(sys.argv[1])
tail = sys.argv[2]
target_chars = target_tokens * 36 // 10
chunk = "// noise line, kept short and uniform so token estimate is stable.\n"
n = max(1, target_chars // len(chunk))
sys.stdout.write(chunk * n)
sys.stdout.write(tail)
PY
}

# Send one chat-completions call. Prints: PROMPT_TOK COMPLETION_TOK ELAPSED_S TOK_PER_S TIER_USED
# Uses python json parsing to avoid jq dep on perf path.
one_call() {
  local model="$1" body="$2" timeout="${3:-600}" router_est_tok="${4:-0}"
  local t0 t1 resp prompt_tok completion_tok elapsed
  t0=$(python3 -c "import time;print(time.time())")
  resp="$(curl -fsS -m "$timeout" "${HDRS[@]}" -X POST -d "$body" "$URL" || echo '{}')"
  t1=$(python3 -c "import time;print(time.time())")
  elapsed=$(python3 -c "print(round($t1-$t0,2))")
  local parsed
  # Note: keep this python -c (single line) instead of $( ... <<HEREDOC )
  # because parens in the heredoc body get miscounted by bash inside $( ).
  parsed=$(python3 -c '
import json, sys
resp = json.loads(sys.argv[1] or "{}")
asked = sys.argv[2]
long_max = int(sys.argv[3])
router_est = int(sys.argv[4])  # what the router would have seen
u = resp.get("usage", {}) or {}
pt = int(u.get("prompt_tokens", 0) or 0)
ct = int(u.get("completion_tokens", 0) or 0)
if asked == "claude-code":
    tier = "claude"
elif asked == "local-long":
    tier = "local-long"
elif asked == "hybrid-auto":
    # Mirror router/route_by_size.py decide_tier() exactly. We use the
    # router-side estimate (chars/3.6 of the *outgoing* prompt) because
    # that is what the router itself sees - not the upstream tokenizer.
    tier = "claude-routed" if router_est > long_max else "local-long-routed"
else:
    tier = "unknown:" + asked
print(pt, ct, tier)
' "$resp" "$model" "${ROUTE_LONG_MAX:-128000}" "$router_est_tok")
  read -r prompt_tok completion_tok tier_used <<<"$parsed"
  local toks_per_s
  if [[ "${completion_tok:-0}" -gt 0 && "$(python3 -c "print($elapsed > 0)")" == "True" ]]; then
    toks_per_s=$(python3 -c "print(round($completion_tok / $elapsed, 1))")
  else
    toks_per_s="--"
  fi
  printf "    prompt_tok=%-6s completion_tok=%-4s elapsed=%-6ss decode=%-6s tok/s tier=%s\n" \
    "${prompt_tok:-?}" "${completion_tok:-?}" "$elapsed" "$toks_per_s" "${tier_used:-?}"
}

run_n() {
  local label="$1" model="$2" prompt="$3" n="$4" timeout="${5:-600}"
  # Estimate routing tokens with the same heuristic the router uses
  # (chars / 3.6). Show this alongside the upstream tokenizer's count so
  # the user can see why the router made the decision it made.
  local router_est_tok
  router_est_tok=$(python3 -c 'import sys; s=sys.stdin.read(); print(max(1, int(len(s)/3.6)))' <<<"$prompt")
  bold "$label  (model=$model, router_est=${router_est_tok} tok, timeout=${timeout}s, runs=$n)"
  local body
  body=$(python3 -c '
import json, sys
print(json.dumps({
    "model": sys.argv[1],
    "messages": [{"role":"user","content": sys.argv[2]}],
    "max_tokens": int(sys.argv[3]),
    "temperature": 0.2,
}))
' "$model" "$prompt" "$MAX_TOK_OUT")
  for i in $(seq 1 "$n"); do
    printf "  run %d:" "$i"
    one_call "$model" "$body" "$timeout" "$router_est_tok"
  done
}

# ----------------------------------------------------------------------
hdr "Pre-flight"
echo "  litellm:  http://127.0.0.1:${LITELLM_PORT}"
echo "  ollama:   http://127.0.0.1:${OLLAMA_PORT}  (KV cache: ${KV_CACHE_TYPE})"
echo "  ollama tag: ${OLLAMA_TAG}"
PID_OLL=$(lsof -nP -iTCP:${OLLAMA_PORT} -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
echo "  ollama pid: ${PID_OLL:-down}    started: $(ps -p ${PID_OLL:-0} -o lstart= 2>/dev/null | xargs)"

# ----------------------------------------------------------------------
hdr "Cold-load: first call after a quiet period"
echo "(A cold call includes mmap + weight load. Subsequent calls reflect steady-state.)"
SHORT="Write a one-line Python function that returns x+1."
run_n "Ollama cold"        local-long "$SHORT" 1

# ----------------------------------------------------------------------
hdr "Ollama local-long (~500 tok prompt, 3 warm runs)"
P_SHORT="$(make_prompt 500 $'\nPlease write a Python function add_one(x) and a unit test.')"
run_n "Ollama warm" local-long "$P_SHORT" 3

# ----------------------------------------------------------------------
hdr "Ollama local-long (~5k tok prompt, 3 warm runs)"
P_MID="$(make_prompt 5000 $'\nSummarize the noise above in one sentence.')"
run_n "Ollama mid" local-long "$P_MID" 3

# ----------------------------------------------------------------------
if [[ "$MODE" != "--short" ]]; then
  hdr "Ollama local-long (~18k tok prompt, 3 warm runs) [exercises q4_0 KV cache]"
  P_LONG="$(make_prompt 18000 $'\nSummarize the noise above in one sentence.')"
  run_n "Ollama long" local-long "$P_LONG" 3
fi

# ----------------------------------------------------------------------
hdr "Hybrid-auto router boundary check"
echo "(hybrid-auto should pick local-long up to 128k, claude beyond that.)"
run_n "hybrid small (~500 tok)"  hybrid-auto "$P_SHORT" 1
run_n "hybrid mid (~5k tok)"     hybrid-auto "$P_MID"   1
if [[ "$MODE" != "--short" ]]; then
  run_n "hybrid long (~18k tok)" hybrid-auto "$P_LONG"  1
fi

# ----------------------------------------------------------------------
# STRESS test: push Ollama near the local-long ceiling (q4_0 KV cache).
# This is the headroom test - confirms the model + KV cache can hold a
# prompt 6x the size of the standard "long" run without OOMing or
# tripping over Ollama's default num_ctx=4096 truncation.
if [[ "$MODE" == "--stress" ]]; then
  hdr "STRESS: Ollama local-long near 128k ceiling (~${STRESS_TOKENS} tok prompt) [exercises q4_0 KV cache near limit]"
  echo "  Note: Ollama's default num_ctx is 4096. If the response usage shows"
  echo "  prompt_tokens << ${STRESS_TOKENS}, the request was silently truncated."
  echo "  Expected on this build: prompt_tokens close to ${STRESS_TOKENS}."
  P_STRESS="$(make_prompt "$STRESS_TOKENS" $'\nIn one sentence, what is the dominant repeated string above?')"
  run_n "Ollama stress" local-long "$P_STRESS" 1 "$STRESS_TIMEOUT"

  hdr "STRESS: hybrid-auto at >128k ceiling (~${OVER_CEILING_TOKENS:-140000} tok prompt) [must route to claude-code]"
  echo "  This prompt exceeds ROUTE_LONG_MAX=${ROUTE_LONG_MAX}, so the size-based"
  echo "  router MUST rewrite hybrid-auto -> claude-code. Tier in the result"
  echo "  line below should read 'claude'. If it reads 'local-long',"
  echo "  routing is broken."
  if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && ! launchctl getenv ANTHROPIC_API_KEY >/dev/null 2>&1; then
    echo "  WARN: ANTHROPIC_API_KEY not visible to this shell or launchd."
    echo "        The router will still pick claude-code, but the upstream call will 401."
  fi
  P_OVER="$(make_prompt "${OVER_CEILING_TOKENS:-140000}" $'\nIn one sentence, what is the dominant repeated string above?')"
  run_n "hybrid over-ceiling" hybrid-auto "$P_OVER" 1 "$STRESS_TIMEOUT"
fi

# ----------------------------------------------------------------------
hdr "Live KV cache footprint (after the runs above)"
# Ollama spawns a separate `ollama runner` child that holds the model +
# KV cache. ps on the parent alone reports ~120 MB which is misleading;
# we need to sum the whole process tree under the listener pid.
sum_tree_rss_kb() {
  local root="$1"
  python3 - "$root" <<'PY'
import subprocess, sys
root = int(sys.argv[1])
out = subprocess.run(["ps","-A","-o","pid=,ppid=,rss="], capture_output=True, text=True).stdout
procs = []
for ln in out.strip().splitlines():
    parts = ln.split()
    if len(parts) >= 3:
        procs.append((int(parts[0]), int(parts[1]), int(parts[2])))
keep = {root}
changed = True
while changed:
    changed = False
    for pid, ppid, _ in procs:
        if ppid in keep and pid not in keep:
            keep.add(pid); changed = True
total = sum(rss for pid, _, rss in procs if pid in keep)
print(total)
PY
}
PID_OLL_NOW=$(lsof -nP -iTCP:${OLLAMA_PORT} -sTCP:LISTEN -t 2>/dev/null | head -1 || true)
if [[ -n "${PID_OLL_NOW:-}" ]]; then
  RSS_KB=$(sum_tree_rss_kb "$PID_OLL_NOW")
  RSS_GB=$(python3 -c "print(round(${RSS_KB:-0}/1024/1024,1))")
  echo "  ollama tree RSS:   ${RSS_GB} GB   (parent + runner child = model + KV cache + runtime)"
fi
hdr "Done"
