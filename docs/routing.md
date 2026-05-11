# Routing

`hybrid-auto` is a virtual model registered with LiteLLM. Every request to it
goes through `SizeBasedRouter.async_pre_call_hook`, which rewrites
`data["model"]` to one of `local-fast`, `local-long`, or `claude-code`.

This page documents the rules and shows worked examples.

## Inputs the router uses

| Signal            | Source                                  | Notes                                       |
| ----------------- | --------------------------------------- | ------------------------------------------- |
| Estimated tokens  | concatenated `messages[*].content`      | `chars / 3.6` heuristic, fast and good enough |
| Complexity flag   | `router/complexity_classifier.py`       | Regex match on architectural / multi-file language |
| Explicit override | `[claude]` or `[local]` tag in prompt   | User wins                                   |

## Decision tree

```
                              ┌─────────────────┐
                              │ data["model"]   │
                              │  == hybrid-auto?│
                              └─────┬───────────┘
                            no      │ yes
                                    ▼
                           ┌─────────────────────┐
                           │ Estimate tokens (T) │
                           │ Classify complexity │
                           └─────┬───────────────┘
                                 │
            ┌────────────────────┼─────────────────────┐
            │                    │                     │
   prompt has [local]?   prompt has [claude]?   T > ROUTE_LONG_MAX (128k)?
            │ yes              │ yes                   │ yes
            ▼                  ▼                       ▼
       ┌──────────┐       ┌──────────┐           ┌──────────┐
       │local-fast│       │claude    │           │claude    │
       │(if T fits)│      │-code     │           │-code     │
       └──────────┘       └──────────┘           └──────────┘

  classifier flagged complex?  ──yes──►  claude-code

  T <= ROUTE_FAST_MAX (16k)?   ──yes──►  local-fast

  ROUTE_FAST_MAX < T <= ROUTE_LONG_MAX  ──►  local-long
```

`ROUTE_FAST_MAX` and `ROUTE_LONG_MAX` are emitted by `scripts/00-detect.sh`
based on installed RAM. On a 128 GB M5 Max they default to 16k and 128k.

## Worked examples

### 1. Tiny refactor

> **Prompt:** _Rename `getCwd` to `getCurrentWorkingDirectory` in this file._
>
> Tokens: ~30. Complexity: false.
>
> **Routed to:** `local-fast` → MLX, ~50ms first-token latency, $0.

### 2. Long-context summarization

> **Prompt:** _\<paste of a 60k-token codebase\> Summarize the data flow._
>
> Tokens: ~60,000. Complexity: false.
>
> **Routed to:** `local-long` → Ollama with TurboQuant. Stays on-device.

### 3. Architecture review

> **Prompt:** _Refactor the architecture across multiple files. Think
> step by step._
>
> Tokens: ~80. Complexity: true (matches both "Refactor.*architecture" and
> "multiple files").
>
> **Routed to:** `claude-code` even though the prompt is small, because the
> classifier expects deeper reasoning than the local model handles.

### 4. Forced local

> **Prompt:** _[local] Refactor the architecture across multiple files._
>
> The `[local]` tag wins; complexity heuristic is suppressed.
>
> **Routed to:** `local-fast` (since the prompt itself is short).

### 5. Forced cloud

> **Prompt:** _[claude] generate a hello-world script._
>
> The `[claude]` tag wins; size is irrelevant.
>
> **Routed to:** `claude-code`.

### 6. Beyond local capacity

> **Prompt:** _A 200,000-token monorepo dump. Find all dead code._
>
> Tokens > `ROUTE_LONG_MAX` (128k).
>
> **Routed to:** `claude-code`. Local-long would exceed even the `tq3`
> KV-cache budget for 80B models.

## Token estimation: why `chars / 3.6`?

It's deliberately cheap. We need an approximate token count *before* the
request goes out, and `tiktoken`-style encoders take 5–50ms per call. The
3.6 ratio is the empirical mean for English code + comments and was chosen
by sampling our own LiteLLM logs.

If you want exact counts, swap in `tiktoken` inside `_estimate_tokens` —
the rest of the routing logic stays the same.

## Offline mode

When `OFFLINE=1` is set (real env or `config/detected.env`) or the
1.5s TCP probe to `api.anthropic.com:443` fails, the router never
returns a Claude tier — every branch that would have escalated is
rewritten to `local-long` and stamped with `route_reason="offline-downgrade: ..."`.
This applies uniformly to:

- `hybrid-auto` complexity / size escalations.
- Explicit `[claude]` / `[opus]` / `[sonnet]` / `[haiku]` tags
  (originally-requested tag preserved in the route reason).
- Cline tool-result failure rescues (Python tracebacks, Rust
  panics, JS stacks) on turn 2+.
- Direct `claude-*` model selections that never went through
  `decide_tier()` at all — caught by a chokepoint in
  `async_pre_call_hook`.

`OFFLINE_STRICT=1` flips the silent downgrade to a 503 for the
"explicit Claude" cases. Toggle with `make offline` / `make online`
/ `make offline-status`. Full runbook + Cline context-clearing
recipe in [offline-mode.md](offline-mode.md).

## Modifying the rules

- **Move the boundaries:** edit `ROUTE_FAST_MAX` / `ROUTE_LONG_MAX` in
  `config/detected.env`. No code change needed.
- **Tighten the complexity classifier:** edit
  `router/complexity_classifier.py`. The regex list is the entire policy.
- **Add a new tier:** add a new entry under `model_list` in
  `config/litellm-config.yaml`, register it in the router's tier-mapping, and
  add it to the `decide_tier` function. See [contributing.md](contributing.md).

## Observability

Every routed request stores `route_reason` in `requests.route_reason`, so the
dashboard can show *why* a request landed where it did. The CLI `make report`
already groups by `tier`; for finer breakdowns query the DB directly:

```sql
SELECT route_reason, COUNT(*) AS n, AVG(latency_ms) AS avg_ms
  FROM requests
 GROUP BY route_reason
 ORDER BY n DESC;
```
