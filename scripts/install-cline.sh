#!/usr/bin/env bash
# install-cline.sh - Install the MacM4-patched Cline extension into
# whichever supported IDE is on this Mac.
#
# Source preference (in order):
#   1. GCS artifact  gs://cline-repo/cline-macm4/cline-macm4-latest.vsix
#      (built from the integration/macm4-enhancements branch; publisher
#       martinfr-certifyos; includes Ollama tag fix + MacM4 provider)
#   2. Upstream marketplace  saoudrizwan.claude-dev  (fallback, no MacM4 patches)
#
# IDE detection order:
#   1. VS Code  (preferred for local MacM4 stack -- no Cursor SSRF gateway)
#   2. Cursor
#
# Idempotent: re-running reinstalls / upgrades to the latest VSIX.
set -euo pipefail

GCS_BUCKET="cline-repo"
GCS_VSIX_PATH="gs://${GCS_BUCKET}/cline-macm4/cline-macm4-latest.vsix"
EXT_ID_PATCHED="martinfr-certifyos.claude-dev"
EXT_ID_UPSTREAM="saoudrizwan.claude-dev"
MARKETPLACE_URL="https://marketplace.visualstudio.com/items?itemName=${EXT_ID_UPSTREAM}"

log()  { printf "\033[1;34m[cline]\033[0m %s\n" "$*"; }
ok()   { printf "\033[1;32m[cline]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[cline]\033[0m %s\n" "$*" >&2; }

# Resolve the CLI binary for an IDE. Order tries:
#   (a) bare command on PATH (set up via the IDE's Shell Command palette)
#   (b) the bundled binary inside the .app
resolve_cli() {
  local app_name="$1"
  local bin_name="$2"
  local bundled_path="/Applications/${app_name}.app/Contents/Resources/app/bin/${bin_name}"
  if command -v "$bin_name" >/dev/null 2>&1; then
    command -v "$bin_name"
  elif [[ -x "$bundled_path" ]]; then
    echo "$bundled_path"
  else
    echo ""
  fi
}

CURSOR_CLI="$(resolve_cli "Cursor" "cursor")"
VSCODE_CLI="$(resolve_cli "Visual Studio Code" "code")"

# Prefer VS Code for the local MacM4 stack (no Cursor SSRF gateway issues).
# User can override with IDE=cursor|code.
case "${IDE:-auto}" in
  cursor) PICKED="$CURSOR_CLI"; IDE_NAME="Cursor" ;;
  code|vscode)
          PICKED="$VSCODE_CLI"; IDE_NAME="Visual Studio Code" ;;
  auto)
    if [[ -n "$VSCODE_CLI" ]]; then
      PICKED="$VSCODE_CLI"; IDE_NAME="Visual Studio Code"
    elif [[ -n "$CURSOR_CLI" ]]; then
      PICKED="$CURSOR_CLI"; IDE_NAME="Cursor"
    else
      PICKED=""
    fi
    ;;
  *) warn "IDE=$IDE not recognized (use cursor or code); falling back to auto"
     PICKED="${VSCODE_CLI:-$CURSOR_CLI}"
     IDE_NAME="$( [[ -n "$VSCODE_CLI" ]] && echo "Visual Studio Code" || echo Cursor )"
     ;;
esac

if [[ -z "$PICKED" ]]; then
  warn "Couldn't find a VS Code or Cursor CLI on this Mac."
  cat <<EOF >&2

  Neither of these worked:
    - 'code' on PATH   (or /Applications/Visual Studio Code.app/Contents/Resources/app/bin/code)
    - 'cursor' on PATH (or /Applications/Cursor.app/Contents/Resources/app/bin/cursor)

  Install VS Code (preferred), then re-run \`make cline\`:
    Download: https://code.visualstudio.com

  Or install Cline manually from the marketplace GUI:
    1. Open the IDE -> Extensions tab (Cmd+Shift+X).
    2. Search for "Cline".
    3. Click Install.
    Marketplace listing: ${MARKETPLACE_URL}

EOF
  exit 1
fi

log "using ${IDE_NAME} CLI: $PICKED"

# ── Try GCS VSIX first (MacM4-patched build) ──────────────────────────────
VSIX_TMP=""
if command -v gsutil >/dev/null 2>&1; then
  VSIX_TMP="$(mktemp /tmp/cline-macm4-XXXXXX.vsix)"
  log "downloading MacM4 VSIX from ${GCS_VSIX_PATH} ..."
  if gsutil cp "${GCS_VSIX_PATH}" "${VSIX_TMP}" 2>/dev/null; then
    log "installing MacM4-patched Cline (${EXT_ID_PATCHED}) into ${IDE_NAME} ..."
    if "$PICKED" --install-extension "${VSIX_TMP}"; then
      ok "MacM4-patched Cline installed in ${IDE_NAME}."
      rm -f "${VSIX_TMP}"
      INSTALLED_ID="${EXT_ID_PATCHED}"
    else
      warn "VSIX install failed; falling back to marketplace."
      rm -f "${VSIX_TMP}"; VSIX_TMP=""
    fi
  else
    warn "GCS download failed (gsutil auth issue?); falling back to marketplace."
    rm -f "${VSIX_TMP}"; VSIX_TMP=""
  fi
else
  warn "gsutil not found; installing upstream Cline from marketplace (no MacM4 patches)."
fi

# ── Marketplace fallback ──────────────────────────────────────────────────
if [[ -z "${INSTALLED_ID:-}" ]]; then
  log "installing upstream ${EXT_ID_UPSTREAM} from marketplace ..."
  if "$PICKED" --install-extension "${EXT_ID_UPSTREAM}"; then
    ok "Upstream Cline installed in ${IDE_NAME}."
    INSTALLED_ID="${EXT_ID_UPSTREAM}"
  else
    warn "Extension install failed."
    echo "  Try the marketplace GUI instead: ${MARKETPLACE_URL}" >&2
    exit 1
  fi
fi

echo ""
echo "  Installed: ${INSTALLED_ID}"
echo ""
echo "  Next steps:"
echo "    1. Quit and relaunch ${IDE_NAME} so the activation hook fires."
echo "    2. Click the Cline icon in the left sidebar (robot/chat bubble)."
echo "    3. Gear icon -> API Configuration:"
echo "         API Provider : OpenAI Compatible"
echo "         Base URL     : http://127.0.0.1:4000/v1"
echo "         API Key      : any non-empty string  (e.g. not-needed)"
echo "         Model ID     : gpt-hybrid-auto"
echo "         Native Tool Calling: OFF"
echo "    4. Try: 'Read README.md and summarize it in one sentence.'"
echo ""
echo "  Full walkthrough: docs/RUNBOOK-cline-setup.md"
