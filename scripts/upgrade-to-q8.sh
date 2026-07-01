#!/usr/bin/env bash
# upgrade-to-q8.sh — Switch the local Ollama model from the pinned q4 tag to
# q8_0 on a 96+ GB machine.  Run this once after initial setup; the detection
# script already picks q8_0 for fresh installs but preserves the prior tag on
# re-runs to avoid re-downloading.
#
# What this does:
#   1. Verifies RAM_GB >= 96 (q8_0 requires ~80 GB VRAM for Qwen3-Coder-Next).
#   2. Pulls qwen3-coder-next:q8_0 from Ollama registry.
#   3. Rewrites QUANT_TIER and OLLAMA_TAG in config/detected.env.
#   4. Re-renders launchd plists via scripts/60-dashboard.sh.
#   5. Bounces the Ollama daemon to pick up the new model.
#
# Usage:
#   bash scripts/upgrade-to-q8.sh
#   make upgrade-to-q8
#
# To revert: re-run scripts/00-detect.sh with PREV_OLLAMA_TAG unset (or deleted
# from detected.env) and set FORCE_QUANT_TIER=q4 in the environment.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETECTED_ENV="$REPO_ROOT/config/detected.env"

log()  { printf "\033[1;34m[upgrade-q8]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[upgrade-q8]\033[0m %s\n" "$*" >&2; }
ok()   { printf "\033[1;32m[upgrade-q8]\033[0m %s\n" "$*"; }
die()  { printf "\033[1;31m[upgrade-q8]\033[0m ERROR: %s\n" "$*" >&2; exit 1; }

if [[ ! -f "$DETECTED_ENV" ]]; then
  die "config/detected.env not found. Run \`make detect\` first."
fi

# shellcheck disable=SC1090
source "$DETECTED_ENV"

if [[ "${QUANT_TIER:-}" == "q8" ]]; then
  ok "Already on q8_0 (QUANT_TIER=q8 in config/detected.env). Nothing to do."
  exit 0
fi

if (( RAM_GB < 96 )); then
  die "RAM_GB=${RAM_GB} < 96. q8_0 requires at least 96 GB for Qwen3-Coder-Next-80B."
fi

TARGET_TAG="qwen3-coder-next:q8_0"

log "Machine: ${CHIP} ${RAM_GB}GB — eligible for q8_0"
log "Current: OLLAMA_TAG=${OLLAMA_TAG:-<unset>}"
log "Target:  OLLAMA_TAG=${TARGET_TAG}"
echo ""
printf "\033[1;33mThis will pull ~80 GB from the Ollama registry. Proceed? [y/N] \033[0m"
read -r reply
[[ "$reply" =~ ^[Yy]$ ]] || { log "Aborted."; exit 0; }

log "Pulling ${TARGET_TAG} …"
ollama pull "$TARGET_TAG"
ok "Pull complete."

log "Rewriting config/detected.env …"
sed -i.bak \
  -e 's|^QUANT_TIER=.*|QUANT_TIER="q8"|' \
  -e "s|^OLLAMA_TAG=.*|OLLAMA_TAG=\"${TARGET_TAG}\"|" \
  "$DETECTED_ENV"
rm -f "${DETECTED_ENV}.bak"
ok "config/detected.env updated."

log "Re-rendering launchd plists …"
bash "$REPO_ROOT/scripts/60-dashboard.sh" >/dev/null

AGENT_PLIST="$HOME/Library/LaunchAgents/com.local.ollama.plist"
if [[ -f "$AGENT_PLIST" ]]; then
  log "Bouncing Ollama launchd service …"
  launchctl bootout  "gui/$(id -u)" "$AGENT_PLIST" 2>/dev/null || true
  sleep 2
  launchctl bootstrap "gui/$(id -u)" "$AGENT_PLIST"
  sleep 3
  ok "Ollama restarted with ${TARGET_TAG}."
else
  warn "LaunchAgent plist not found at $AGENT_PLIST — restart Ollama manually."
  warn "  ollama serve &"
fi

echo ""
ok "Upgrade complete. Run \`make verify\` to confirm the stack is healthy."
