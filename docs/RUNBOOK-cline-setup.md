# Runbook — Cline (in Cursor) against your local hybrid stack

**Why this exists:** Cursor 3.1.17's BYOK provider routes requests through
`api2.cursor.sh`, whose SSRF policy blocks RFC 1918, loopback, *and*
CGNAT (`100.64.0.0/10`, the Tailscale range). Worse, Cursor's Agent
mode does not pass tool-call deltas through BYOK providers at all
— it is locked to first-party models. See the **STOP** callout in
`RUNBOOK-cursor-setup.md` for the full architectural finding.

**The fix:** drive the local LiteLLM proxy from a different agent
host. Cline is a VSCode extension that installs cleanly into Cursor
(Cursor is a VSCode fork) and **makes its OpenAI calls directly from
your Mac to the proxy** — no Cursor cloud round-trip, no SSRF gateway,
no agent-mode limitation.

This document captures the working configuration on this machine.

---

## 0. Prerequisites (already done, verify only)

| Component | Where | How to check |
| --- | --- | --- |
| LiteLLM proxy | `127.0.0.1:4000` listening on loopback | `lsof -nP -iTCP:4000 -sTCP:LISTEN` shows `TCP 127.0.0.1:4000 (LISTEN)` |
| `local-agent` model | `llama3.1:8b-instruct-q8_0` pulled in Ollama | `ollama list \| grep llama3.1` |
| LiteLLM model alias | `gpt-local-agent` exposed | `grep "model_name: gpt-local-agent" config/litellm-config.rendered.yaml` |
| Cursor IDE | 3.1.x | Cursor menu → About |
| Cline extension | Installed in Cursor's extension list | `ls ~/.cursor/extensions/saoudrizwan.claude-dev*` |

> **Tailscale is NOT required for Cline.** Cline runs inside Cursor's
> extension host (a Node.js process on your Mac) and calls the proxy
> directly via loopback. There is no cloud round-trip, so Cursor's
> SSRF policy never sees the request and a public hostname is
> unnecessary. Earlier versions of this runbook recommended pointing
> Cline at the Tailscale IP — that worked but added latency
> (~3 ms / call), an extra dependency on `tailscaled` being up, and
> exposed the proxy to every device on the tailnet. Loopback is
> simpler, faster, and tighter.

If any of those is missing, fix it before continuing — Cline can't
work around an offline proxy or a missing model.

---

## 1. Install the Cline extension into Cursor

```bash
/Applications/Cursor.app/Contents/Resources/app/bin/cursor \
  --install-extension saoudrizwan.claude-dev
```

This pulls Cline 3.81.0 (or current) from the marketplace and lands
it in `~/.cursor/extensions/saoudrizwan.claude-dev-<version>-universal/`.

> **Why not VSCode?** We considered installing a separate VSCode just
> for Cline. Skipped because Cursor is itself a VSCode fork and runs
> Cline cleanly. One IDE on disk; no second toolchain. The marketplace
> Cursor uses mirrors most VSCode extensions, including Cline.

After install, **fully quit and relaunch Cursor** so the extension's
activation hooks fire on the next start.

---

## 2. Configure Cline

Cline's settings are not in `settings.json` — they live in its own
panel.

1. In Cursor, click the **Cline icon** in the left sidebar (a small
   robot / chat-bubble shape that appeared after install).
2. Click the **gear icon** in the Cline panel's top bar, or press
   `Cmd+Shift+P` → "Cline: Open Settings".
3. Configure the API provider:

   | Field | Value |
   | --- | --- |
   | API Provider | **OpenAI Compatible** |
   | Base URL | `http://127.0.0.1:4000/v1` |
   | API Key | `sk-litellm-REDACTED-PRE-ROTATION` |
   | Model ID | `gpt-local-long` *(see model-selection note below)* |

4. Click **Done** / **Save**.

> **Model choice for Cline matters more than for chat.** We initially
> configured Cline against `gpt-local-agent` (llama3.1-8B) because we
> picked that model for its reliable structured `tool_calls[]`
> support. **Cline doesn't use OpenAI's structured tools field at
> all** — it bakes its 10+ tool catalog into a 54 KB system prompt
> and parses XML-fenced tool uses (`<read_file>...</read_file>`,
> `<attempt_completion>...</attempt_completion>`) from the assistant
> message text. Smaller local models (8B, 14B, 32B) often produce
> the correct *content* for a turn but emit it as plain prose without
> the closing `<attempt_completion>` wrapper. Cline then rejects the
> turn with `[ERROR] You did not use a tool in your previous
> response!` and the loop continues until the budget is exhausted.
>
> See §5 below for the empirical 5-model benchmark — only
> `gpt-local-long` (Qwen3-Coder-Next 80B q4_K_M) reliably emits the
> wrapper from the local stack. Use Claude (`gpt-claude-code`) only
> when you need maximum quality + minimum latency and don't mind
> paying.

---

## 3. Sanity test

In the Cline chat panel:

> Read the file `/Users/martinfr/Documents/GitHub/MacM4LocalAgent/README.md`
> and summarize it in one sentence.

> **Don't pick an empty file.** Cline's `read_file` tool returns the
> literal string `(tool did not return anything)` for 0-byte files
> (it conflates empty content with a tool-error result via JS falsy
> coercion in `extension.js`). A small local model cannot distinguish
> "the file is genuinely empty" from "the tool failed", and will
> either hallucinate a plausible summary from surrounding context or
> ask for clarification. Both are correct-ish answers for an
> unactionable signal, but they make Cline look broken when it is
> actually the empty-file edge case biting. We hit this with
> `spec.txt` (0 bytes) and the dump trace in
> `.logs/cline-dumps/req-*.json` confirmed the tool result was the
> Cline-side string, not an empty assistant message.

What you should see:

- Cline renders a **"Read File"** tool step with a green checkmark and
  the requested path.
- The assistant message contains a one-sentence summary of the file.

What you should see in the proxy log
(`tail -f .logs/litellm.out.log`):

```
INFO:     127.0.0.1:NNNNN - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

The source IP being `127.0.0.1` confirms the request came **directly
from this machine** through loopback, not from Cursor's cloud.

---

## 4. Pre-flight verification (already passed during setup)

Before installing Cline, we confirmed the proxy + model combination
emits structured tool calls, not fenced text. Reproduce with:

```bash
KEY=$(grep -E '^LITELLM_MASTER_KEY=' config/detected.env | cut -d= -f2- | tr -d '"')

curl -sS -m 60 \
  -H "Authorization: Bearer ${KEY}" \
  -H "Content-Type: application/json" \
  -X POST "http://127.0.0.1:4000/v1/chat/completions" \
  -d '{
    "model":"gpt-local-agent",
    "messages":[{"role":"user","content":"What is 7 times 8? Use the multiply tool."}],
    "tools":[{"type":"function","function":{"name":"multiply","description":"Multiply two integers","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}}],
    "tool_choice":"auto",
    "max_tokens":256,
    "stream":false
  }' | python3 -m json.tool
```

Expected (working) shape:

```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": "",
      "tool_calls": [{
        "function": {
          "name": "multiply",
          "arguments": "{\"a\": 7, \"b\": 8}"
        }
      }]
    }
  }]
}
```

If `tool_calls` is missing and `content` contains fenced JSON, the
upstream model is not honoring the tools field. That's a model issue,
not a Cline issue.

---

## 5. Model selection benchmark (May 2026)

We replayed the same captured Cline conversation state — `system + task
+ <read_file> + read_file_result(README.md, 9 KB)` — against five
different upstream models via the proxy and asked each to produce the
final assistant turn that closes the task. Only models that emit a
proper `<attempt_completion><result>...</result></attempt_completion>`
wrapper close cleanly. Anything else triggers Cline's
`[ERROR] You did not use a tool` retry loop.

Replay was via `scripts/replay_cline_turn.py` against the dump
`.logs/cline-dumps/req-1777660153436-004.json`. Each model received
the exact same 4-message conversation, no `tools` field, `temperature
0.2`. Wall times include first-call cold start (model load); see
"warm" note for `local-long`.

| Model alias | Backing model | Wall time | Output tokens | Closed `<attempt_completion>`? | Quality |
| --- | --- | --- | --- | --- | --- |
| `gpt-claude-code` | Anthropic Sonnet 4.6 | **5.5 s** | 174 | ✅ Yes | Best — captures the SQLite cost ledger + dashboard angle |
| `gpt-local-long` | Qwen3-Coder-Next 80B q4_K_M (Ollama) | 51 s cold, **7.8 s warm** | 244 | ✅ Yes | Excellent — explicit MLX vs Ollama+TurboQuant routing |
| `gpt-local-coder-32b` | Qwen2.5-Coder 32B (Ollama) | 140 s | 121 | ❌ No | Accurate but slow + minor hallucination on TurboQuant attribution |
| `gpt-local-coder-14b` | Qwen2.5-Coder 14B (Ollama) | 67 s | 99 | ❌ No | Concise + accurate, but harness-rejected |
| `gpt-local-agent` | Llama3.1 8B q8 (Ollama) | 13 s | 108 | ❌ No | Concise + accurate, but harness-rejected |

**Conclusions:**

1. **For local-only Cline, use `gpt-local-long`.** It is the only
   local model in our stack that follows Cline's harness conventions
   correctly (emits the closing `<attempt_completion>` wrapper). Once
   the model is loaded into Ollama RAM, it answers in ~8 seconds —
   only ~40 % slower than Claude. First-turn cold-start is ~50 s
   while the 48 GB model swaps in.
2. **Smaller local models (8B, 14B, 32B) are not viable for Cline.**
   Even the 32B Qwen2.5-Coder produced correct *content* but failed
   to wrap it. This is a model-size + instruction-following ceiling,
   not a model-family issue — both Llama and Qwen2.5-Coder failed.
3. **Cost vs latency is reframed.** Claude is 1.4× faster (warm) but
   costs ~$0.06/turn at this prompt size. Over a 30-turn task that
   is ~$2; `local-long` does the same work for $0.00 in marginal
   cost. Use Claude when latency genuinely matters; use local-long
   for everything else.
4. **The benchmark is reproducible.** Re-run with:
   ```bash
   .venvs/litellm/bin/python scripts/replay_cline_turn.py \
       --dump .logs/cline-dumps/req-1777660153436-004.json \
       --cut-after-msg 3 \
       --models gpt-local-agent,gpt-local-coder-14b,gpt-local-coder-32b,gpt-local-long,gpt-claude-code
   ```

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Cline shows "Connection refused" or hangs on first message | LiteLLM not running | `launchctl list \| grep litellm` and `make verify` |
| Cline returns 401 | Wrong API key | Re-paste; key is in `config/detected.env` |
| Tool steps render as raw text instead of tool-step UI | Model didn't emit `tool_calls` (sometimes happens with Qwen models) | Switch Cline's Model ID to `gpt-local-agent` (llama3.1, tool-capable) |
| Cline can read but not write files | Cline's auto-approve setting is off | Cline panel → settings → enable auto-approval for the tools you want |
| Assistant message contains `<\|im_end\|>` artifacts | Local model leaked its chat-template stop token | Cosmetic only; or switch to `gpt-claude-code` for a one-off call where it matters |
| Cline reports "Task completed" but the file on disk is unchanged; UI shows `Error executing replace_in_file: Failed to open diff editor` and/or `User closed text editor, unable to edit file` | Cline-side bug: the diff-editor pop-up failed to open, but `attempt_completion` was still emitted by the model so Cline mis-reports success | Re-run the prompt; if it persists across runs, prefer prompts that nudge the model toward `write_to_file` (e.g. "rewrite the file as follows") over `replace_in_file`. Auditing approach: `git diff <path>` to confirm whether the edit actually applied. **This is independent of the model — Claude Sonnet hits the same UI bug.** |

> **Edit-task validation (May 2026):** We tested an edit task against a
> seeded `spec.txt` with the prompt *"Add a single line to the end…
> preserve the existing content."* `gpt-local-long` correctly chose
> `replace_in_file` (the surgical apply-diff tool, not the destructive
> `write_to_file`), generated a syntactically valid diff envelope,
> and emitted `<attempt_completion>` cleanly. **The model behavior
> was correct and fast** (read=8.6 s, replace_in_file=14.2 s, both
> warm). However Cline's diff-editor pop-up failed to render twice
> (`Failed to open diff editor`, then `User closed text editor`),
> the file was never modified, and Cline's UI mis-reported success.
> Conclusion: read+summarize is end-to-end reliable; edit tasks
> through `replace_in_file` are model-correct but UI-fragile.
> Recommend `git diff` after every edit to verify the change
> actually landed.

---

## 7. What we lose vs. native Cursor agent mode

- No Cursor "checkpoints" (Cline has its own task history but it's
  not file-system-snapshot-based the way Cursor's is).
- No "Composer" multi-file edit mode — Cline does file edits one
  at a time through `apply_diff` / `write_to_file` tools.
- Tab autocomplete still uses Cursor's built-in models, not Cline.

---

## 8. What we gain

- Local model drives a real agent loop with rendered tool steps.
- Zero per-token cost when running on `gpt-local-agent` /
  `gpt-local-long`.
- Full visibility into the harness: every request goes through the
  LiteLLM proxy, gets logged in `cost/cost.db`, gets the over-gen
  controls applied.
- Works **without** a public tunnel (cloudflared / Funnel) because
  the request originates from the same machine the proxy is on.

---

## 9. Reverting

```bash
/Applications/Cursor.app/Contents/Resources/app/bin/cursor \
  --uninstall-extension saoudrizwan.claude-dev
```

Then in Cursor: command palette → "Developer: Reload Window".
