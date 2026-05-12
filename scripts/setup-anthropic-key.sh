#!/usr/bin/env bash
#
# setup-anthropic-key.sh
# ----------------------
# Persist ANTHROPIC_API_KEY across reboots in a way that is visible to every
# launchd-spawned process on this machine: MacM4 services (litellm, dashboard,
# mlx, ollama), GUI apps (Cursor, Claude Desktop), and freshly-opened Terminal
# sessions. macOS does not propagate shell env (.zshrc/.zshenv) to launchd, so
# the standard pattern is a tiny LaunchAgent that runs `launchctl setenv` at
# login. This script installs that agent.
#
# The script prompts for the key with `read -s` so the value never echoes to
# the terminal, never lands in shell history, and never appears in any chat
# transcript. The value does end up in plain text inside
#   ~/Library/LaunchAgents/com.local.setenv-anthropic.plist
# at mode 600 (user-only). If you need stronger at-rest protection, swap this
# script for a Keychain-backed variant -- see RUNBOOK.md.
#
# Usage:
#   bash scripts/setup-anthropic-key.sh
#
# Re-run safely: each invocation overwrites the plist and reloads the agent.

set -euo pipefail

PLIST="$HOME/Library/LaunchAgents/com.local.setenv-anthropic.plist"
LABEL="com.local.setenv-anthropic"
LITELLM_LABEL="com.local.litellm"

# --- 1. read the key silently ------------------------------------------------

printf "Paste your Anthropic API key (starts with sk-ant-): "
# -s = silent, -r = raw (no backslash escape), -p prompt would echo so we used printf
read -rs KEY
printf "\n"

if [[ -z "${KEY}" ]]; then
  echo "ERROR: empty key, aborting." >&2
  exit 2
fi
if [[ ! "${KEY}" =~ ^sk-ant- ]]; then
  echo "WARN: key does not start with 'sk-ant-'. Proceeding anyway."
fi

# Escape XML special chars in case the key ever contains <, >, &, ", '.
xml_escape() {
  printf '%s' "$1" \
    | sed -e 's/&/\&amp;/g' \
          -e 's/</\&lt;/g' \
          -e 's/>/\&gt;/g' \
          -e 's/"/\&quot;/g' \
          -e "s/'/\\&apos;/g"
}
ESCAPED_KEY=$(xml_escape "${KEY}")

# --- 2. write the LaunchAgent plist -----------------------------------------

mkdir -p "$(dirname "${PLIST}")"
umask 077  # forces 600 on subsequent creations
cat > "${PLIST}" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/launchctl</string>
    <string>setenv</string>
    <string>ANTHROPIC_API_KEY</string>
    <string>${ESCAPED_KEY}</string>
  </array>
  <key>StandardOutPath</key>
  <string>/tmp/${LABEL}.out.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/${LABEL}.err.log</string>
</dict>
</plist>
PLIST_EOF
chmod 600 "${PLIST}"
echo "wrote ${PLIST} (mode 600)"

# --- 3. (re)load the agent --------------------------------------------------

# bootout is idempotent-ish: errors if not loaded, which we ignore.
launchctl bootout "gui/${UID}/${LABEL}" 2>/dev/null || true
launchctl bootstrap "gui/${UID}" "${PLIST}"
launchctl kickstart -k "gui/${UID}/${LABEL}"

# Give launchd a beat to run the one-shot setenv before we probe it.
sleep 1

# --- 4. verify --------------------------------------------------------------

CURRENT=$(launchctl getenv ANTHROPIC_API_KEY || true)
if [[ -z "${CURRENT}" ]]; then
  echo "ERROR: launchctl getenv returned empty after bootstrap." >&2
  echo "       Check /tmp/${LABEL}.err.log for details." >&2
  exit 3
fi
echo "launchctl getenv ANTHROPIC_API_KEY: present (length=${#CURRENT}, prefix=${CURRENT:0:8})"

# --- 5. restart litellm so it picks up the new env --------------------------
#
# litellm can be in three states:
#   (a) Managed by launchd via ~/Library/LaunchAgents/com.local.litellm.plist,
#       healthy. We just `launchctl kickstart -k` it.
#   (b) Managed by launchd, but the plist is in a flap loop (EX_CONFIG / TCC
#       issues with ~/Documents/). We bootout the failing service and fall
#       through to the manual-start path.
#   (c) Running manually as a nohup background process (the common workaround).
#       We SIGTERM it, then start a new one with the fresh env.
#
# In every case we want to end with a single litellm process bound to :4000
# whose ANTHROPIC_API_KEY came from the new launchctl setenv.

litellm_is_healthy_under_launchd() {
  # Returns 0 if launchd manages it AND its last exit code is not EX_CONFIG.
  local print_out
  print_out=$(launchctl print "gui/${UID}/${LITELLM_LABEL}" 2>/dev/null) || return 1
  if echo "${print_out}" | grep -qE 'last exit code = 78: EX_CONFIG'; then
    return 1
  fi
  return 0
}

start_litellm_manually() {
  local repo=""
  if [[ -f "${HOME}/Documents/GitHub/MacM4LocalAgent/scripts/run_litellm.py" ]]; then
    repo="${HOME}/Documents/GitHub/MacM4LocalAgent"
  elif [[ -f "$(pwd)/scripts/run_litellm.py" ]]; then
    repo="$(pwd)"
  else
    echo "WARN: could not locate MacM4LocalAgent repo; not starting litellm." >&2
    return 1
  fi
  (
    cd "${repo}"
    # Pull the key out of launchctl rather than relying on this shell's env --
    # the script may have been invoked from a session that pre-dates the
    # setenv call.
    export ANTHROPIC_API_KEY="$(launchctl getenv ANTHROPIC_API_KEY)"
    mv .logs/litellm.out.log .logs/litellm.out.log.prev 2>/dev/null || true
    mv .logs/litellm.err.log .logs/litellm.err.log.prev 2>/dev/null || true
    nohup .venvs/litellm/bin/python scripts/run_litellm.py \
      --config config/litellm-config.rendered.yaml \
      --host 127.0.0.1 --port 4000 \
      > .logs/litellm.out.log 2> .logs/litellm.err.log &
    disown
  )
  # Wait up to 30s for :4000 to bind.
  for _ in $(seq 1 30); do
    if lsof -i :4000 -nP -sTCP:LISTEN >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "WARN: litellm did not bind :4000 within 30s." >&2
  return 1
}

if litellm_is_healthy_under_launchd; then
  echo "litellm is healthy under launchd; kickstarting so it inherits the new env"
  launchctl kickstart -k "gui/${UID}/${LITELLM_LABEL}"
else
  # Either launchd doesn't have it, or it's stuck in EX_CONFIG flap. Either
  # way, kill any existing :4000 listener and (re)start manually.
  if launchctl print "gui/${UID}/${LITELLM_LABEL}" >/dev/null 2>&1; then
    echo "launchd's ${LITELLM_LABEL} is in EX_CONFIG flap; booting it out"
    launchctl bootout "gui/${UID}/${LITELLM_LABEL}" 2>/dev/null || true
  fi
  EXISTING_PID=$(lsof -t -i :4000 -sTCP:LISTEN 2>/dev/null | head -1 || true)
  if [[ -n "${EXISTING_PID}" ]]; then
    echo "SIGTERMing existing :4000 listener (pid=${EXISTING_PID}) so we can"
    echo "respawn with the new env"
    kill "${EXISTING_PID}" 2>/dev/null || true
    # Wait for it to release :4000.
    for _ in $(seq 1 10); do
      lsof -i :4000 -nP -sTCP:LISTEN >/dev/null 2>&1 || break
      sleep 1
    done
  fi
  echo "starting litellm manually via nohup (TCC workaround for ~/Documents/)"
  start_litellm_manually
fi

# --- 6. probe a cloud tier to confirm the proxy now has the key -------------

# Defense in depth: scrub any sk-ant- token that might slip into a response or
# log line. This guards against future regressions where an upstream error
# echoes the key back (older litellm builds have done this).
redact() {
  sed -E 's/sk-ant-[A-Za-z0-9_-]+/sk-ant-***REDACTED***/g'
}

sleep 3
echo ""
echo "probing claude-haiku-4-5 through the proxy..."
RESP=$(curl -sS --max-time 15 -X POST http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer noop' \
  -d '{"model":"claude-haiku-4-5","messages":[{"role":"user","content":"ping"}],"max_tokens":4}' \
  2>&1 || true)
if echo "${RESP}" | grep -q '"choices"'; then
  echo "OK: claude-haiku-4-5 returned a completion."
elif echo "${RESP}" | grep -q '"error".*[Aa]uthentication'; then
  echo "STILL 401 -- litellm did not pick up the env. Restart litellm:"
  echo "  cd ~/Documents/GitHub/MacM4LocalAgent && bash scripts/setup-anthropic-key.sh"
  echo "  (this script will detect the EX_CONFIG flap and fall back to manual start)"
  echo "Full response (redacted):"
  printf '%s' "${RESP}" | redact
elif echo "${RESP}" | grep -q 'Failed to connect.*4000'; then
  echo "litellm is NOT listening on :4000 -- the manual-start path didn't bind."
  echo "Check the latest log:"
  echo "  tail -50 ~/Documents/GitHub/MacM4LocalAgent/.logs/litellm.err.log"
else
  echo "Unexpected response (first 500 chars, redacted):"
  printf '%s' "${RESP}" | redact | head -c 500
fi

echo ""
echo "done. The ANTHROPIC_API_KEY LaunchAgent persists across reboots, so the"
echo "env will reappear in launchd at next login automatically. Note that the"
echo "litellm process itself currently starts manually (TCC issues with the"
echo "launchd-managed plist when ~/Documents/ is the program path); after a"
echo "reboot you'll still need 'make start' or to rerun this script."
