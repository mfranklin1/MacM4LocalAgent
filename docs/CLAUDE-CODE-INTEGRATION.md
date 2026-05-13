# Claude Code Integration with MacM4LocalAgent

## Overview

MacM4LocalAgent supports **Claude Code** (via VS Code or Cursor) as a second client alongside Cline. This guide explains how to configure Claude Code to use the local Ollama/MLX stack for small-to-medium contexts, with automatic fallback to Anthropic for larger tasks.

**Key benefits:**
- Small tasks (≤128k tokens) route to **free local models** (Ollama/Qwen3 or MLX)
- Large tasks route to Anthropic with **configurable billing** (Team subscription OAuth or API key)
- Reduces Claude Team plan overages and usage limits
- Avoids the localhost blocking issue that affects Cursor's native agent mode

---

## Architecture

Claude Code connects to MacM4 via a dedicated thin proxy on `:4002`:

```
Claude Code
  ↓ (Anthropic format: /v1/messages)
http://127.0.0.1:4002 (claude_proxy)
  ├─ ≤128k tokens → routes to :4000 LiteLLM → Ollama/MLX (FREE)
  └─ >128k tokens → [configurable]
      ├─ passthrough mode: forwards to api.anthropic.com with Claude Code's OAuth token (Team subscription)
      └─ apikey mode: replaces auth with ANTHROPIC_API_KEY (platform.claude.com API account)
```

**Why separate from Cline?**
- **Cline** uses LiteLLM on `:4000` (OpenAI format), API key only
- **Claude Code** uses new proxy on `:4002` (Anthropic format), supports OAuth pass-through
- **Architectural isolation:** Prevents Cline's API key from being routed through Team subscription OAuth (which would violate terms of service)

---

## Terms of Service Compliance

**ToS Research Summary:**
Anthropic's consumer terms, AUP, and commercial terms were reviewed. None explicitly prohibit transparent proxying or request interception at the infrastructure level.

**This implementation's compliance:**

| Client | Large-context behavior | ToS basis |
|--------|------------------------|-----------|
| **Claude Code** | Pass-through OAuth OR API key | Claude Code is Anthropic's own product. Forwarding its own requests to Anthropic unchanged is not "powering another service." No prohibition found. |
| **Cline** | API key only (no OAuth pass-through) | Cline is a third-party app. Using Claude Team subscription to service Cline requests would violate the spirit of "automated/non-human use" and "powering other services" clauses. |

**Enforcement:** The architecture prevents misuse by design — Claude Code connects to `:4002` (which supports pass-through), while Cline only reaches `:4000` (which never uses pass-through).

---

## Setup

### Prerequisites

1. **MacM4LocalAgent services running** — Ollama, MLX, LiteLLM all healthy
   ```bash
   make verify
   ```

2. **Claude Code installed** — available as VS Code extension or CLI
   - Install in Cursor or VS Code
   - Sign in with your claude.ai account (no separate setup needed)

3. **ANTHROPIC_API_KEY set** (optional, for apikey mode)
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   launchctl setenv ANTHROPIC_API_KEY "$ANTHROPIC_API_KEY"
   ```

### Configuration

#### Step 1: Detect and initialize services

The claude_proxy service is auto-initialized on first `make detect`:

```bash
cd ~/MacM4LocalAgent
make detect
# Emits CLAUDE_PROXY_PORT=4002 and CLAUDE_PROXY_LARGE_CTX_MODE=passthrough to config/detected.env
```

#### Step 2: Start the claude_proxy service

```bash
make start
# Starts all services including claude-proxy on :4002
```

Verify it's running:
```bash
curl -s http://127.0.0.1:4002/health
# {"status": "ok", "route_long_max": 128000, "large_ctx_mode": "passthrough"}
```

#### Step 3: Configure Claude Code to use the proxy

Add to your shell profile (`~/.zshrc` or `~/.bash_profile`):

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:4002"
launchctl setenv ANTHROPIC_BASE_URL "http://127.0.0.1:4002"
```

Then restart your IDE (Cursor or VS Code with Claude Code extension).

#### Step 4: Verify connection

In Claude Code, ask it a simple question (e.g., "What files are in the current directory?").

Check the logs:
```bash
tail -f .logs/claude-proxy.out.log
# Should show: "token_estimate=XXX mode=local route_reason=..."
```

---

## Configuration Options

### Large-context fallback mode

Control how Claude Code handles requests exceeding 128k tokens via `config/detected.env`:

```bash
# Mode 1: Pass-through with Team OAuth (default, no API key needed)
CLAUDE_PROXY_LARGE_CTX_MODE=passthrough
# → Billing against your Team subscription (flat fee)
# → Avoids API token costs but uses Team plan quota/limits

# Mode 2: Use API key (requires ANTHROPIC_API_KEY set)
CLAUDE_PROXY_LARGE_CTX_MODE=apikey
# → Billing against platform.claude.com account (pay-per-token)
# → Independent of Team plan, no overages or limits
```

Apply changes:
```bash
make restart  # Reloads claude-proxy with new config
```

**Which mode to choose?**

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Avoiding Team plan overages | `passthrough` | Team subscription already covers large contexts |
| Heavy API usage (many large tasks) | `apikey` | Separate billing; avoid exhausting Team plan quota |
| Testing / exploration | `passthrough` | Simpler; uses existing Team subscription |

---

## Monitoring & Cost Tracking

### Dashboard

The cost dashboard at `http://localhost:4001` tracks all requests:

- **Local-routed turns** (Claude Code on Ollama/MLX) → $0 cost, tier shows `local-long` or `local-fast`
- **Pass-through turns** (Claude Code >128k, passthrough mode) → $0 cost in cost.db (billed to Team subscription separately)
- **API-key turns** (Claude Code >128k, apikey mode) → shows actual cost in cost.db

Note: Pass-through turns don't generate cost rows in the cost database because billing happens at Anthropic, not in MacM4.

### Manual inspection

Check claude-proxy logs for routing decisions:

```bash
tail -f .logs/claude-proxy.out.log

# Local route example:
# 2026-05-13T10:23:45Z model=claude-sonnet-4-6 tokens=45000 mode=local → hybrid-auto

# Pass-through route example:
# 2026-05-13T10:24:10Z model=claude-opus-4-7 tokens=185000 mode=passthrough → api.anthropic.com

# API-key route example:
# 2026-05-13T10:25:00Z model=claude-opus-4-7 tokens=175000 mode=apikey → api.anthropic.com
```

### Monthly cost estimate

If you're using passthrough mode:

```
Local-routed cost = $0
Pass-through cost = whatever your Claude Team plan covers (flat fee)
Total = Team subscription fee (unchanged)
```

If you're using apikey mode:

```
Local-routed cost = $0
API-key routed cost = (input_tokens × rate) + (output_tokens × rate)
Total = Team fee + API charges
```

---

## Known Limitations

### 1. Tool call translation

Claude Code sends Anthropic-format `tool_use` content blocks. These are translated to OpenAI `tool_calls` for local models (Ollama/MLX) and back. This translation **may be lossy**.

**Symptoms:** Tool calls fail or behave unexpectedly on local-routed turns.

**Mitigation:** Test file I/O and code edits on small, local-routed tasks first. If tools consistently fail, either:
- Disable local routing for tool-heavy sessions (manually use `[claude]` tag to force large-context path)
- Switch to apikey mode and accept the per-token cost

### 2. Model capability differences

Qwen3-Coder-Next (local) and Claude Sonnet/Opus (API) have different strengths:

- **Qwen3:** Excellent at straightforward code tasks, refactoring, inline edits
- **Claude:** Better at multi-file architecture, complex reasoning, edge cases

**Symptom:** Local model produces poor output on complex tasks.

**Mitigation:** Use `[claude]` tag in your task prompt to force escalation to Claude for complex work.

### 3. Context window asymmetry

Claude Code's system prompt is ~13k tokens, which means almost every request will route to `local-long` (Ollama) at minimum. The `local-fast` tier (MLX) is structurally unreachable.

This is expected and fine — Ollama is still free.

---

## Troubleshooting

### Claude Code doesn't reach the proxy

**Symptom:** Claude Code times out or returns "network error"

**Diagnosis:**
```bash
curl -s http://127.0.0.1:4002/health
# If this fails, the proxy isn't running
```

**Fix:**
```bash
make restart
# Verify:
make verify  # Should show "claude-proxy on :4002 is listening"
```

### Requests always go to Anthropic (not routing locally)

**Symptom:** All requests show in claude-proxy logs as `mode=passthrough` or `mode=apikey`, never `mode=local`

**Diagnosis:** Check token count estimate
```bash
grep "tokens=" .logs/claude-proxy.out.log | tail -5
# If all show tokens > 128000, that's normal (Claude Code's system prompt is large)
```

**Expected behavior:** Claude Code requests almost always exceed 16k (MLX threshold) and often exceed 128k (Ollama threshold) due to system prompt size. Expect most turns to route to local-long (Ollama) for $0, not MLX.

### Tool calls fail on local-routed turns

**Symptom:** Claude Code tries to read/write files but the tool result shows an error

**Diagnosis:** Check for Anthropic-to-OpenAI translation issues
```bash
grep -i "tool" .logs/claude-proxy.out.log
grep -i "error" .logs/litellm.out.log  # LiteLLM may have translation logs
```

**Mitigation:**
1. Try the same task again (transient issue)
2. Force large-context path with `[claude]` tag in your prompt
3. If consistent, open an issue with the tool call content (redacted) for investigation

### Cannot use Team subscription (apikey mode only)

**Symptom:** You need `CLAUDE_PROXY_LARGE_CTX_MODE=apikey` but don't have an ANTHROPIC_API_KEY set

**Reason:** You're using Claude Team subscription, which doesn't provide an API key.

**Options:**
1. Create an API key on `platform.claude.com` with a separate Anthropic account (different from your Team user)
2. Or stay in `passthrough` mode (default) to use Team subscription

---

## FAQ

### Q: Can I use Claude Code and Cline at the same time?

**A:** Yes! They use separate connections:
- Claude Code → `:4002` (claude-proxy)
- Cline → `:4000` (LiteLLM)

Both can be running in the same IDE and will share the same local model pool (Ollama/MLX) without contention. The dashboard combines costs from both.

### Q: Does passthrough mode expose my Team subscription credentials?

**A:** No. The proxy forwards the original OAuth token that Claude Code already has from signing in to claude.ai. This is no different from Claude Code calling Anthropic directly. The proxy is transparent.

### Q: Can I switch modes without restarting?

**A:** Partially. If you change `CLAUDE_PROXY_LARGE_CTX_MODE` in `config/detected.env`:
```bash
make restart  # Full restart for clean state
```

If you just set `ANTHROPIC_API_KEY` as an environment variable, the proxy will read it on the next request (no restart needed). But the proxy process needs `make restart` if the config file changes.

### Q: Will local models improve in the future?

**A:** Ollama regularly releases new quantizations and models. You can update with:
```bash
ollama pull qwen3-coder-next:q2_K_M  # Lower quant, smaller memory footprint
ollama pull qwen3-coder-next:q5_K_M  # Higher quant, better quality
```

Then update `config/litellm-config.yaml` to point at the new tag. The proxy doesn't need changes.

---

## Next steps

- Read [routing.md](routing.md) for details on how context size drives routing decisions
- Check [operations.md](operations.md) for service management and logs
- See [troubleshooting.md](troubleshooting.md) for additional issues
