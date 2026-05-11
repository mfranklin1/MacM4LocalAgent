#!/usr/bin/env bash
# install-cline.sh - Install the Cline extension (saoudrizwan.claude-dev)
# into whichever supported IDE is on this Mac.
#
# Detection order:
#   1. Cursor   (preferred -- this project's documented integration path)
#   2. VS Code
#
# Falls back to printing manual GUI / marketplace instructions if neither
# CLI is on PATH or in /Applications. Idempotent: re-running upgrades to
# the latest published version of the extension.
set -euo pipefail

EXT_ID="saoudrizwan.claude-dev"
MARKETPLACE_URL="https://marketplace.visualstudio.com/items?itemName=${EXT_ID}"

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

# Pick the first IDE we found. User can override with IDE=cursor|code.
case "${IDE:-auto}" in
  cursor) PICKED="$CURSOR_CLI"; IDE_NAME="Cursor" ;;
  code|vscode)
          PICKED="$VSCODE_CLI"; IDE_NAME="Visual Studio Code" ;;
  auto)
    if [[ -n "$CURSOR_CLI" ]]; then
      PICKED="$CURSOR_CLI"; IDE_NAME="Cursor"
    elif [[ -n "$VSCODE_CLI" ]]; then
      PICKED="$VSCODE_CLI"; IDE_NAME="Visual Studio Code"
    else
      PICKED=""
    fi
    ;;
  *) warn "IDE=$IDE not recognized (use cursor or code); falling back to auto"
     PICKED="${CURSOR_CLI:-$VSCODE_CLI}"
     IDE_NAME="$( [[ -n "$CURSOR_CLI" ]] && echo Cursor || echo "Visual Studio Code" )"
     ;;
esac

if [[ -z "$PICKED" ]]; then
  warn "Couldn't find a Cursor or VS Code CLI on this Mac."
  cat <<EOF >&2

  Neither of these worked:
    - 'cursor' on PATH   (or /Applications/Cursor.app/Contents/Resources/app/bin/cursor)
    - 'code' on PATH     (or /Applications/Visual Studio Code.app/Contents/Resources/app/bin/code)

  Install one of these, then re-run \`make cline\`:

    Cursor (recommended):
      Download:        https://cursor.com
      After install, no extra setup needed -- the .app ships the CLI.

    VS Code:
      Download:        https://code.visualstudio.com
      After install:   open VS Code, Cmd+Shift+P,
                       'Shell Command: Install code command in PATH'

  Or install Cline manually via the marketplace GUI:
    1. Open the IDE -> Extensions tab (Cmd+Shift+X).
    2. Search for "Cline".
    3. Click Install on "Cline" by saoudrizwan.
    Marketplace listing: ${MARKETPLACE_URL}

EOF
  exit 1
fi

log "using ${IDE_NAME} CLI: $PICKED"
log "installing $EXT_ID ..."
if "$PICKED" --install-extension "$EXT_ID"; then
  ok "Cline installed/upgraded in ${IDE_NAME}."
  echo ""
  echo "  Next steps:"
  echo "    1. Quit and relaunch ${IDE_NAME} so the activation hook fires."
  echo "    2. Click the Cline icon in the left sidebar (robot/chat bubble)."
  echo "    3. Gear icon -> API Configuration:"
  echo "         API Provider : OpenAI Compatible"
  echo "         Base URL     : http://127.0.0.1:4000/v1"
  echo "         API Key      : any non-empty string  (e.g. not-needed)"
  echo "         Model ID     : gpt-hybrid-auto"
  echo "    4. Try: 'Read README.md and summarize it in one sentence.'"
  echo ""
  echo "  Full walkthrough: docs/RUNBOOK-cline-setup.md"
else
  warn "extension install failed via CLI."
  echo "  Try the marketplace GUI instead: ${MARKETPLACE_URL}" >&2
  exit 1
fi
