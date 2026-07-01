#!/usr/bin/env bash
# 50-cursor.sh - Drop a Cursor rule (so any Cursor-native chat / Ask /
# Plan-mode interactions know the routing semantics) and print the
# Cline-integration instructions, since that's the supported way to drive
# the proxy. Cursor BYOK (Override OpenAI Base URL) is kept for the Ask /
# Plan-mode use case but is NOT the primary integration -- see
# docs/RUNBOOK-cline-setup.md for the architectural explanation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/config/detected.env"

log() { printf "\033[1;34m[cursor]\033[0m %s\n" "$*"; }

RULE_DIR="$REPO_ROOT/.cursor/rules"
mkdir -p "$RULE_DIR"
cat > "$RULE_DIR/hybrid-routing.mdc" <<EOF
---
description: When to use local-long vs claude-code
alwaysApply: true
---

# Hybrid local + cloud routing

This workspace runs a hybrid local+cloud LLM setup behind LiteLLM
(\`http://127.0.0.1:${LITELLM_PORT}\`). Pick the right tier for the prompt.

## Models exposed

- **local-long** — Ollama + ${KV_CACHE_TYPE} KV cache, ≤${LOCAL_LONG_CTX} ctx, free.
- **claude-code** — default Claude tier, currently mapped to Opus 4.7 (\$5 in / \$25 out per MTok, 1M context).
- **claude-haiku-4-5** / **claude-sonnet-4-6** / **claude-opus-4-7** — pinned to a specific Claude model.
- **hybrid-auto** — let the router decide based on size + heuristic complexity.

## Decision tree

\`\`\`
Is the prompt explicitly tagged at the start?
├─ "[local] ..."  ────────────────► local-long  (absolute opt-out)
├─ "[haiku] ..."  ────────────────► claude-haiku-4-5
├─ "[sonnet] ..." ────────────────► claude-sonnet-4-6
├─ "[opus] ..."   ────────────────► claude-opus-4-7
└─ "[claude] ..." ────────────────► claude-code (default Claude tier)

Else, by content:
├─ Architectural / multi-file / deep reasoning ──► claude-code
├─ ≤128k tokens                                  ──► local-long
└─ >128k tokens                                  ──► claude-code
\`\`\`

## Worked examples

| Prompt                                              | Tier               | Why                                |
| --------------------------------------------------- | ------------------ | ---------------------------------- |
| "Rename \`foo\` to \`bar\` in this file."           | local-long         | tiny, single-file                  |
| "Summarize this 60k-token codebase dump."           | local-long         | size only, not architectural       |
| "Refactor the auth subsystem across 12 services."   | claude-code (Opus) | architectural + multi-file         |
| "[local] Refactor across multiple files."           | local-long         | absolute opt-out wins              |
| "[haiku] What is 2+2?"                              | claude-haiku-4-5   | explicit cheap-Claude pin          |
| "[opus] design a billing service"                   | claude-opus-4-7    | explicit Opus pin                  |
| "[claude] hello world"                              | claude-code (Opus) | default Claude tier                |

## Cost model

Every request is logged to \`cost/cost.db\` with both \`actual_cost\` (real
USD; 0 for local) and \`shadow_cost\` (what Claude would have charged).
Dashboard: <http://127.0.0.1:${DASHBOARD_PORT}>. CLI: \`make report\`.

## Defaults

- Set **hybrid-auto** as your default model in Cursor; the router picks the
  cheapest tier that can carry the prompt.
- Switch to **claude-code** explicitly when you want the strongest output
  (architecture reviews, gnarly bugs, designs).
- Use **local-long** for any "read-this-large-thing" task — staying local
  is free and fast on Apple Silicon.

## Don't

- Don't paste secrets into prompts. \`local-*\` keeps them on-device, but
  \`claude-code\` will send them to Anthropic.
- Don't disable the proxy and call providers directly; you'll lose cost
  tracking and the savings dashboard will show 0.
EOF
log "wrote $RULE_DIR/hybrid-routing.mdc"

cat <<EOF

==================================================================
  Cline wiring (the supported integration path)
==================================================================

  1. Install the Cline extension into Cursor (one-time):

       /Applications/Cursor.app/Contents/Resources/app/bin/cursor \\
         --install-extension saoudrizwan.claude-dev

     Then quit & relaunch Cursor so Cline's activation hooks fire.

  2. Open Cursor -> click the Cline icon in the left sidebar
     (robot / chat-bubble shape) -> gear icon -> API Configuration.

  3. Fill in:

       API Provider : OpenAI Compatible
       Base URL     : http://127.0.0.1:${LITELLM_PORT}/v1
       API Key      : any non-empty string (e.g. "not-needed")
                      The proxy is loopback-only with no auth gate, but
                      Cline won't save a blank key field.
       Model ID     : gpt-hybrid-auto   (recommended; router picks tier)

  4. Click Done. Try a prompt:

       Read README.md and summarize it in one sentence.

  5. Watch traffic:

       make report        # CLI summary
       make dashboard     # http://127.0.0.1:${DASHBOARD_PORT}

  Inline routing overrides (prefix the prompt, leading-only):

       [local]   force local-long  (absolute opt-out from Claude)
       [haiku]   force claude-haiku-4-5
       [sonnet]  force claude-sonnet-4-6
       [opus]    force claude-opus-4-7
       [claude]  force whichever model claude-code aliases (= opus)

  Full walkthrough: docs/RUNBOOK-cline-setup.md

==================================================================
  Cursor BYOK wiring (LEGACY -- works only for Ask / Plan modes)
==================================================================

  Cursor Agent mode does NOT pass tool-call deltas through BYOK
  providers, so this path can't drive a coding agent. It is kept for
  the Ask / Plan-mode use case only.

  1. Cursor -> Settings (Cmd+,) -> Models
  2. Toggle "Override OpenAI Base URL" ON
  3. Base URL:  http://127.0.0.1:${LITELLM_PORT}/v1
  4. API Key:   any non-empty string (e.g. "not-needed")
  5. Add models: local-long, claude-code, hybrid-auto
  6. Pick hybrid-auto as default.

  See docs/RUNBOOK-cursor-setup.md for the SSRF / CGNAT background.
EOF
