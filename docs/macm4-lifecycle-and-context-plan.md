# MacM4LocalAgent — Lifecycle Switching, Context Compression & Context Janitor Plan

> **Status:** design / proposal — not yet implemented. Captured from a prior
> planning conversation and added here for tracking. Names, ports, and config
> values are proposals; reconcile against the live stack before implementing.

This is the full plan discussed for making long local coding sessions
sustainable on Apple Silicon. It has three layers that work together:

1. **TurboQuant lifecycle switching** — dynamically load/unload quantized
   local backends across context tiers (128k → 256k → 512k) instead of
   keeping everything resident, with user-visible switch status and pending
   request persistence. *(performance / memory headroom)*
2. **Context compression before backend switching (Headroom)** — mechanically
   compress and deduplicate noisy context (tool output, repeated reads, old
   diffs) so a turn fits a smaller backend before any model switch.
3. **Context Janitor** — use the 128k local model to make *semantic* cleanup
   decisions (keep / summarize / archive / discard) and maintain durable
   project memory, plus a Cline-fork integration plan and the Qwen model-tier
   decision for which model runs the janitor.

Core principle:

> Headroom compresses. The local janitor decides what matters. The router
> switches models only if the curated context still does not fit.
>
> (and: *important does not always mean recent.*)

## Contents

- [Reconciliation notes](#reconciliation-notes)
- Part 1 — TurboQuant Lifecycle Switching Plan (incl. the Context Janitor and Context Compression updates)
- Part 2 — Qwen Model-Tier Decision
- Part 3 — Cline Context Janitor Integration Plan

## Reconciliation notes

Two reconciliation points the plan must address:

- The repo already uses the name "TurboQuant" for an **Ollama KV-cache quantization** track (`tq3`/`tq4`, `make turboquant-*`, `docs/turboquant.md`, experimental llama.cpp server on **:8082**). Your spec's "TurboQuant MLX-LM tiers" are a *different* mechanism (MLX-LM serving with quantized KV cache / `turbo_kv_bits`). **Resolved below**: the backend registry now uses ports 8084/8085 to avoid the existing `:8082` collision.
- `config/detected.env` on this host currently shows `q4` / `qwen3-coder-next:q4`, not the Q8 your model-tier decision assumes. **Resolved below**: the Current Baseline now reflects the actual detected tier (q4). Whether to force Q8 on 128 GB machines is an open policy question — see notes in Current Baseline.

---

# Part 1 — TurboQuant Lifecycle Switching Plan

_(Includes the Context Janitor and Context Compression / Headroom updates further down.)_

> **Status of MLX-LM `turbo_kv_bits`**: As of June 2026, `turbo_kv_bits` is **not yet shipped in a released version of mlx-lm**. This mirrors the Ollama tq3 situation (see `docs/turboquant.md`). The 256k/512k tier architecture is designed and ready to wire up; the lifecycle manager, backend registry, and launchd plists can be built now, but the MLX-LM serve path needs upstream support before the turbo backends go live. Track the relevant MLX-LM PR and add a `make turbo-status` target similar to `make turboquant-status`.

## Objective

Add experimental TurboQuant-powered local context tiers to MacM4LocalAgent while preserving the current stable 128k workflow.

The system must support dynamic local escalation:

```text
local-fast -> local-long-128k -> turbo-256k -> turbo-512k -> Claude/external
```

The key design constraint is that large model backends cannot all remain resident on a 128GB MacBook Max. Switching between large local backends may take seconds or minutes, so the router must manage lifecycle, user-visible status, request queuing, and context preservation automatically.

---

## Current Baseline

Current repo behavior:

- `local-fast`: MLX short-context fast path (MLX_PORT=8081, `mlx-community/Qwen3-Coder-Next-4bit`).
- `local-long`: Ollama long-context path (OLLAMA_PORT=11434, `qwen3-coder-next:q4`).
- On this 128 GB M5 Max, `config/detected.env` currently selects `QUANT_TIER=q4` / `qwen3-coder-next:q4` and `LOCAL_LONG_CTX=131072`. **Planned change**: upgrade to `q8_0` on 128 GB machines — better output quality; the KV cache is ~2× larger but fits. Requires updating `scripts/00-detect.sh` to prefer `q8_0` at ≥ 128 GB RAM before the lifecycle work begins.
- Cline traffic escalates to Claude when the full request exceeds 85% of ROUTE_LONG_MAX (~111k tokens, via saturation check in `router/route_by_size.py`).
- Requests beyond the local long context route to Claude or external Anthropic.

Target behavior:

- Keep current 128k path unchanged by default.
- Add optional TurboQuant MLX-LM tiers for 256k and 512k local contexts.
- Route to Claude only after all enabled local options are exhausted or fail health checks.

---

## Critical Memory Constraint

Do not run all large local model servers concurrently.

A 128GB MacBook Max should keep lightweight services always running, but only one large local model backend should be resident at a time.

Always-on services:

```text
LiteLLM router
Dashboard
Claude proxy
Ollama daemon
Small MLX fast model, if lightweight enough
```

Mutually exclusive large backends:

```text
qwen3-coder-next:q8_0 128k
TurboQuant 256k backend
TurboQuant 512k backend
```

Policy:

```text
Only one large local backend may be loaded at any time.
```

---

## Why Lifecycle Management Is Required

Switching large models is not instantaneous.

Expected switching costs:

- stopping current large backend: seconds
- freeing memory / waiting for pressure to drop: seconds to tens of seconds
- starting target backend: seconds to minutes
- model load / warmup: tens of seconds to minutes
- first prefill at large context: potentially slow

Therefore, routing cannot simply return a new upstream target. It must coordinate a state transition.

---

## User Experience Requirement

When a large backend switch is required, the user must be told what is happening.

Example status message:

```text
This request needs a larger local context window. Switching from the 128k local model to the 256k TurboQuant backend. This may take a minute; your conversation context is being preserved.
```

For 512k escalation:

```text
The current conversation has exceeded the 256k local window. Switching to the 512k TurboQuant backend instead of routing externally. This may take longer; context is being preserved.
```

If local escalation fails:

```text
The larger local backend could not be loaded safely due to memory pressure or startup failure. Falling back to Claude/external for this request.
```

---

## Context Preservation Requirement

The router must preserve the full incoming request while switching backends.

Requirements:

1. Accept the request.
2. Estimate token count.
3. Determine target tier.
4. If target tier is not loaded, place request into a pending queue.
5. Persist the full request payload to disk before stopping/loading services.
6. Switch backend.
7. Verify backend health.
8. Replay the preserved request to the new backend.
9. Stream the result back to the original caller.

The request must not be dropped if the backend switch takes time.

---

## State Machine

Implement a backend lifecycle manager with explicit states.

```text
IDLE
ACTIVE_FAST
ACTIVE_LONG_128K
SWITCHING_TO_TURBO_256K
ACTIVE_TURBO_256K
SWITCHING_TO_TURBO_512K
ACTIVE_TURBO_512K
FALLING_BACK_EXTERNAL
FAILED
```

State transitions must be logged and surfaced in the dashboard.

---

## Backend Registry

Create a registry describing each backend.

Example:

```yaml
backends:
  local-long-128k:
    kind: ollama
    model: qwen3-coder-next:q8_0
    port: 11434
    max_context: 131072
    resident_policy: warm_optional
    startup_timeout_seconds: 180
    idle_timeout_seconds: 1800

  local-turbo-256k:
    kind: mlx-turbo
    # Model family TBD: Qwen2.5-Coder-32B-Instruct-4bit (plan original) vs
    # Qwen3-Coder-Next-4bit (current family). Verify MLX-LM turbo_kv_bits
    # support before committing to either. See open questions below.
    model_path: /Users/martinfr/Documents/GitHub/MacM4LocalAgent/models/Qwen2.5-Coder-32B-Instruct-4bit
    port: 8084  # 8082 is owned by turboquant-experimental-serve (llama.cpp); 8083 reserved as buffer
    max_context: 262144
    turbo_kv_bits: 3
    turbo_fp16_layers: 2
    resident_policy: on_demand
    startup_timeout_seconds: 240
    idle_timeout_seconds: 900

  local-turbo-512k:
    kind: mlx-turbo
    model_path: /Users/martinfr/Documents/GitHub/MacM4LocalAgent/models/Qwen2.5-Coder-32B-Instruct-4bit
    port: 8085
    max_context: 524288
    turbo_kv_bits: 3
    turbo_fp16_layers: 4
    resident_policy: on_demand
    startup_timeout_seconds: 360
    idle_timeout_seconds: 600
```

---

## Routing Rules

```text
<= 16k        -> local-fast
16k-128k      -> local-long-128k
128k-256k     -> local-turbo-256k
256k-512k     -> local-turbo-512k
> 512k        -> Claude/external
```

If TurboQuant is disabled:

```text
<= 16k        -> local-fast
16k-128k      -> local-long-128k
> 128k        -> Claude/external
```

---

## Dynamic Escalation Behavior

If the current request exceeds the active backend limit, the system should escalate locally before using Claude.

Example:

```text
Current backend: local-long-128k
Request size: 180k
Action: switch to local-turbo-256k
```

Example:

```text
Current backend: local-turbo-256k
Request size: 330k
Action: switch to local-turbo-512k
```

Example:

```text
Current backend: local-turbo-512k
Request size: 650k
Action: route to Claude/external
```

---

## Service Lifecycle Algorithm

Pseudo-code:

```python
async def route_request(request):
    token_count = estimate_tokens(request)
    target_backend = choose_backend(token_count)

    if target_backend == active_backend and backend_healthy(target_backend):
        return await forward(request, target_backend)

    request_id = persist_pending_request(request)
    notify_client_status(request_id, switching_message(active_backend, target_backend))

    async with backend_switch_lock:
        if target_backend != active_backend:
            await stop_large_backend(active_backend)
            await wait_for_memory_release()
            await start_backend(target_backend)
            await wait_for_health(target_backend)
            await warm_backend(target_backend)
            set_active_backend(target_backend)

    restored_request = load_pending_request(request_id)
    return await forward(restored_request, target_backend)
```

---

## Switch Locking

Only one backend switch may occur at a time.

If multiple requests arrive while switching:

- queue them
- coalesce compatible requests to the same target backend
- reject or external-route requests only if the queue timeout is exceeded

Add:

```text
BACKEND_SWITCH_LOCK
PENDING_REQUEST_QUEUE
MAX_SWITCH_WAIT_SECONDS
```

---

## Pending Request Persistence

Persist pending requests under:

```text
.runtime/pending_requests/<request_id>.json
```

Each record should include:

```json
{
  "request_id": "...",
  "created_at": "...",
  "estimated_tokens": 0,
  "target_backend": "local-turbo-256k",
  "client": "cline",
  "payload": {}
}
```

Delete after successful completion.

Keep failed records for debugging under:

```text
.runtime/failed_requests/
```

---

## User-Visible Status Streaming

For Cline/OpenAI-compatible clients, inject assistant-visible progress events when possible.

If native event streaming is not safe, return an early text chunk:

```text
[MacM4LocalAgent] Switching to larger local context backend: local-turbo-256k. Context is preserved. This may take a minute.
```

Then continue streaming the model response after the backend is ready.

For Claude Code / Anthropic-format proxy, emit a similar content block before forwarding.

If the client cannot accept interim chunks, log status and expose it through dashboard + `/status` endpoint.

---

## New Status Endpoint

Add:

```text
GET /backend/status
```

Return:

```json
{
  "active_backend": "local-turbo-256k",
  "state": "SWITCHING_TO_TURBO_512K",
  "switch_started_at": "...",
  "target_backend": "local-turbo-512k",
  "pending_requests": 1,
  "memory_pressure": "medium",
  "last_status_message": "Switching to 512k TurboQuant backend"
}
```

---

## Memory Pressure Gate

Before starting a large backend, check memory.

Required checks:

```text
available memory
swap usage
memory pressure
currently resident model processes
Metal / MLX allocation failures
```

If memory pressure is high:

1. stop idle large backend
2. wait for memory to clear
3. retry once
4. if still unsafe, fall back external

---

## Idle Timeout Policy

Large backends should unload when idle.

Suggested defaults:

```text
local-long-128k idle timeout: 30 minutes
local-turbo-256k idle timeout: 15 minutes
local-turbo-512k idle timeout: 10 minutes
```

When idle timeout fires:

```text
stop backend
mark state IDLE
keep router/dashboard/proxy alive
```

---

## Make Targets

Add:

```makefile
turbo-install
turbo-start-256
turbo-start-512
turbo-stop
turbo-status
turbo-bench
turbo-verify
turbo-enable
turbo-disable
backend-status
backend-stop-large
```

---

## Configuration Variables

Add to `config/detected.env` or generated config:

```bash
TURBO_ENABLED=0
TURBO_MODEL_REPO="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
TURBO_MODEL_LOCAL_DIR="/Users/martinfr/Documents/GitHub/MacM4LocalAgent/models/mlx-community_Qwen2.5-Coder-32B-Instruct-4bit"
TURBO_KV_BITS=3
TURBO_FP16_LAYERS_256=2
TURBO_FP16_LAYERS_512=4
LOCAL_TURBO_256_CTX=262144
LOCAL_TURBO_512_CTX=524288
MAX_SWITCH_WAIT_SECONDS=420
PENDING_REQUEST_DIR=".runtime/pending_requests"
FAILED_REQUEST_DIR=".runtime/failed_requests"
```

---

## Dashboard Updates

Display:

- active backend
- current state
- target backend during switch
- pending request count
- memory pressure
- last switch duration
- last switch failure reason
- local vs external routing count

---

## Benchmark Requirements

Benchmark both steady-state and cold-switch behavior.

Test matrix:

```text
128k current Qwen3-Coder-Next Q8
256k TurboQuant Qwen2.5-Coder-32B
512k TurboQuant Qwen2.5-Coder-32B
```

Measure:

```text
cold start time
backend switch time
memory release time
first token latency
prefill time
decode tokens/sec
RAM peak
swap usage
quality on coding tasks
failure rate
```

---

## Failure Modes To Handle

1. Target backend fails to start.
2. Health check times out.
3. Memory pressure remains high.
4. Request queue timeout exceeded.
5. Client disconnects during switch.
6. Persisted request cannot be replayed.
7. TurboQuant output quality is degraded.
8. 512k context is too slow for useful interactive work.

Each failure must produce:

- log entry
- dashboard status
- user-visible message if client still connected
- clean fallback or clean failure

---

## Fallback Rules

If 512k local fails:

```text
route to Claude/external
```

If 256k local fails and request <=128k:

```text
retry local-long-128k
```

If 256k local fails and request >128k:

```text
route to Claude/external
```

If user has forced offline mode:

```text
fail clearly instead of routing external
```

Offline failure message:

```text
This request exceeds the available local context and offline mode prevents external fallback.
```

---

## Cline-Specific Acceptance Criteria

Cline should observe one of the following during a switch:

1. A streamed status message before model output, or
2. A clear delayed response after backend switch, with no dropped context, or
3. A structured error explaining why local escalation failed.

The user should never see a silent hang with no explanation.

---

## Implementation Phases

### Phase 0: Prerequisite — upgrade detection to q8_0 on 128 GB machines

Deliverables:

```text
scripts/00-detect.sh — prefer q8_0 when RAM_GB >= 128
config/detected.env — regenerated after detect
make detect && make verify
```

This must land before any lifecycle work. The lifecycle manager's memory-pressure gate needs to know the actual resident model size; using q4 estimates with a q8 model will produce wrong gate thresholds.

### Phase 1: Add backend registry and config

Deliverables:

```text
config/backend-registry.yaml
config/detected.env additions
router backend selection unit tests
```

### Phase 2: Add lifecycle manager

Deliverables:

```text
router/backend_lifecycle.py
single-switch lock
backend start/stop commands
memory pressure checks
```

### Phase 3: Add pending request persistence

Deliverables:

```text
.runtime/pending_requests
.runtime/failed_requests
request replay logic
client disconnect handling
```

### Phase 4: Add user-visible status

Deliverables:

```text
OpenAI-compatible status chunk support
Anthropic-compatible status block support
/status and /backend/status endpoints
```

### Phase 5: Add TurboQuant services

Deliverables:

```text
launchd turbo 256 plist\ launchd turbo 512 plist
Make targets
health checks
```

### Phase 6: Add dashboard + benchmark support

Deliverables:

```text
dashboard backend state card
turbo-bench
cold-switch benchmark report
```

---

## Final Target Behavior

Current behavior:

```text
128k local -> Claude
```

Target behavior:

```text
128k local -> switch to 256k local -> switch to 512k local -> Claude only if needed
```

The switch must be automatic, context-preserving, user-visible, and safe under memory pressure.

---

## Update: Use the 128k Local Model as a Context Janitor

Add a dedicated Context Janitor stage that uses the existing 128k local model to make semantic cleanup decisions before routing to larger local backends.

This should help. Headroom can compress token-heavy material, but the 128k model can judge importance: what must stay active, what can be summarized, what should be archived for retrieval, and what is safe to drop.

Core principle:

`Headroom compresses. The local janitor decides what matters. The router switches models only if the curated context still does not fit.`

### Why this matters

Long Cline sessions grow because they accumulate logs, tool output, repeated file reads, failed attempts, old diffs, plans, and partial decisions. A pure compressor may reduce size, but it may not know that an old user constraint is still important.

The janitor model protects against the key failure mode:

`important does not always mean recent.`

### New pipeline

`Cline request or tool result -> persist raw event -> Headroom compression if noisy -> 128k janitor review -> update project ledger -> rebuild active context pack -> estimate curated tokens -> route locally -> switch to 256k or 512k only if still needed.`

### Janitor artifacts

The janitor should maintain these files:

- Active Context Pack: current goal, current task, latest blocker, relevant files, current diffs, required constraints.
- Project Ledger: durable decisions, user preferences, architecture choices, failed approaches, files changed, open issues.
- Retrieval Manifest: references to archived logs, full tool outputs, old diffs, file snapshots, and prior checkpoints.
- Cleanup Report: what was kept, summarized, archived, or discarded.

### Trigger policy

Run the janitor when:

- raw context exceeds 64k tokens;
- context grows by more than 20k tokens since the last checkpoint;
- a large tool output is added;
- a new error, test failure, or stack trace appears;
- the user changes goals or architecture direction;
- before any backend escalation decision.

Do not wait until the session reaches 128k. Cleanup should happen continuously.

### Keep, summarize, archive, discard

Keep verbatim:

- explicit user instructions;
- hard constraints;
- current failing errors;
- stack traces being debugged;
- file paths and line numbers;
- commands that changed state;
- API contracts and architecture decisions.

Summarize:

- long discussions after a decision is made;
- old plans superseded by newer plans;
- repeated explanations;
- successful command output;
- implementation history that is no longer active.

Archive retrievably:

- full logs;
- large JSON payloads;
- old terminal output;
- old file snapshots;
- full test output after failures are summarized;
- older diffs no longer directly relevant.

Discard only high-confidence noise:

- duplicate boilerplate;
- repeated success lines;
- progress spinners;
- package install noise;
- identical repeated tool output.

If the janitor is unsure, it should archive retrievably instead of discarding.

### Interaction with Headroom

Use both layers:

- Headroom handles mechanical compression and deduplication.
- The 128k local janitor handles semantic classification and project memory updates.
- The prompt builder assembles the final active context from curated state, not from the full append-only transcript.

### Janitor-aware routing examples

Example 1:

`raw transcript: 240k -> Headroom: 170k -> janitor curated prompt: 92k -> stay on 128k backend.`

Example 2:

`raw transcript: 420k -> Headroom: 310k -> janitor curated prompt: 180k -> switch to 256k backend, not 512k.`

### New configuration

Add these variables:

- CONTEXT_JANITOR_ENABLED=1
- CONTEXT_JANITOR_MODEL=local-long-128k  # default; Part 2 recommends Qwen3-Coder-30B-A3B once benchmarked
- CONTEXT_JANITOR_TRIGGER_TOKENS=64000
- CONTEXT_JANITOR_GROWTH_TRIGGER_TOKENS=20000
- CONTEXT_JANITOR_MAX_LATENCY_SECONDS=45
- CONTEXT_JANITOR_OUTPUT_DIR=.runtime/context_janitor
- CONTEXT_LEDGER_PATH=.runtime/context_janitor/project-ledger.json
- ACTIVE_CONTEXT_PACK_PATH=.runtime/context_janitor/active-context-pack.md
- RETRIEVAL_MANIFEST_PATH=.runtime/context_janitor/retrieval-manifest.json

### New Make targets

Add:

- janitor-enable
- janitor-disable
- janitor-status
- janitor-run
- janitor-bench
- janitor-show-ledger
- janitor-show-active-context
- janitor-reset

### Dashboard additions

Show:

- janitor enabled status;
- last janitor run;
- raw tokens before janitor;
- curated tokens after janitor;
- tokens removed from active context;
- items kept, summarized, archived, and discarded;
- whether a backend switch was avoided.

### Acceptance criteria

1. The janitor reduces active context growth during long Cline sessions.
2. Explicit user constraints and current failures are preserved.
3. Archived content remains retrievable.
4. Routing uses curated prompt size, not raw transcript size.
5. Backend switching is avoided when safe cleanup brings the request under the active backend limit.
6. Discard decisions must be conservative.
7. The user should never lose current errors, current diffs, open tasks, or explicit constraints silently.

### Expected benefit

Using the 128k local model as a janitor should reduce unnecessary 256k and 512k switches, reduce external fallbacks, lower latency, preserve important old decisions, and stop Cline from repeatedly rediscovering project state.

---

## Update: Context Compression Before Backend Switching

Add a compression and context-reduction layer before triggering any large backend switch.

The core principle is:

```text
compress first, switch models second, route external last
```

This matters because switching from the 128k backend to a 256k or 512k TurboQuant backend may take seconds or minutes. If context compression can reduce the request below the active backend's limit, the system should avoid a cold model switch entirely.

---

## Headroom Evaluation

Evaluate Headroom as the first candidate for the context-reduction layer.

Repository:

```text
https://github.com/headroomlabs-ai/headroom
```

Hypothesis:

Headroom is useful because it addresses prompt growth, not KV-cache memory. TurboQuant increases how much context a model can physically hold; Headroom reduces how much context needs to be sent in the first place.

Target placement:

```text
incoming Cline / Claude Code request
  -> token estimate
  -> Headroom compression / context shaping
  -> token estimate after compression
  -> route to active backend if possible
  -> switch backend only if still too large
  -> external fallback only if local options fail or are exceeded
```

---

## Revised Routing Pipeline

Replace the original direct routing path with a compression-aware path.

Old pipeline:

```text
token estimate -> choose backend -> switch if needed -> forward
```

New pipeline:

```text
accept request
  -> persist original payload
  -> estimate raw token count
  -> run context compression if raw count exceeds current backend threshold
  -> estimate compressed token count
  -> choose cheapest viable local backend
  -> avoid switching if compressed request fits active backend
  -> switch local backend only if compression is insufficient
  -> route external only if local tiers cannot handle the compressed request
```

---

## Compression-Aware Routing Rules

Use raw and compressed token counts.

```text
raw_tokens = estimate(original_request)
compressed_request = compress_if_needed(original_request, target_budget)
compressed_tokens = estimate(compressed_request)
```

Routing should be based on compressed tokens, while retaining the original request for replay, audit, and retrieval.

```text
compressed <= 16k      -> local-fast
compressed <= 128k     -> local-long-128k
compressed <= 256k     -> local-turbo-256k
compressed <= 512k     -> local-turbo-512k
compressed > 512k      -> Claude/external or fail if offline
```

---

## Compression Budgets

The router should attempt progressively larger budgets before switching.

Example when active backend is `local-long-128k`:

```text
raw request: 180k
try compress to: 120k
if successful: stay on 128k backend
if unsuccessful: switch to 256k backend
```

Example when active backend is `local-turbo-256k`:

```text
raw request: 330k
try compress to: 240k
if successful: stay on 256k backend
if unsuccessful: switch to 512k backend
```

Example when active backend is `local-turbo-512k`:

```text
raw request: 650k
try compress to: 480k
if successful: stay on 512k backend
if unsuccessful: external fallback or offline failure
```

---

## Headroom Integration Modes To Evaluate

Evaluate these integration modes in order:

1. Library mode inside the router.
2. Local proxy sidecar between Cline and LiteLLM.
3. MCP/tool wrapper mode for compressing tool outputs before they enter the transcript.

Preferred initial integration:

```text
router-side library integration
```

Reason:

The router already controls token estimation, backend selection, pending request persistence, and fallback behavior. Keeping compression inside the router makes policy enforcement easier.

---

## Context Reduction Techniques To Support

The implementation should support these techniques, whether via Headroom or internal adapters.

### Tool Output Compression

Compress large logs, JSON payloads, test output, stack traces, grep results, and build output before appending to the model context.

Preserve:

```text
errors
warnings
stack traces
failed test names
file paths
line numbers
commands run
non-zero exits
changed values
```

Drop or summarize:

```text
repeated successful test lines
unchanged JSON fields
large duplicate arrays
verbose dependency output
low-value boilerplate
```

### Conversation Compaction

Replace older turns with structured summaries when the conversation grows.

Summary must preserve:

```text
user goals
explicit decisions
current plan
open questions
files modified
commands run
failures encountered
constraints and preferences
```

### Retrieval-Backed Context

Store original dropped content locally and make it retrievable.

```text
.runtime/context_store/<conversation_id>/
.runtime/context_index/<conversation_id>/
```

The model should receive a compact note that additional original detail is available if needed.

### Code-Aware Compression

For source files, prefer:

```text
function/class signatures
changed hunks
imports
interfaces
failing functions
neighboring context around edits
```

Avoid sending entire unchanged files when a diff or focused hunk is enough.

### Diff-Based Context

For repo work, prefer:

```text
git diff
git status
changed files list
relevant hunks
recent test failures
```

instead of full repository snapshots.

### Prefix Cache Alignment

Keep stable prompt prefixes stable.

```text
system prompt
routing rules
tool instructions
project rules
```

Move volatile items toward the tail of the prompt so prompt caching is more effective where supported.

### Atomic Tool-Call Pruning

If pruning old tool output, remove or compact complete tool call/result pairs together. Do not keep dangling tool results without the corresponding request, or vice versa.

---

## Compression Metadata

Every compressed request should carry metadata.

```json
{
  "compression_applied": true,
  "raw_tokens": 180000,
  "compressed_tokens": 118000,
  "compression_ratio": 0.66,
  "compression_strategy": "headroom-tool-output-plus-history-compaction",
  "dropped_items_count": 14,
  "retrievable_items_count": 14,
  "context_store_path": ".runtime/context_store/<conversation_id>/"
}
```

Persist this next to pending request records.

---

## User-Visible Compression Status

If compression avoids a backend switch, surface that clearly.

Example:

```text
[MacM4LocalAgent] Compressed the working context from ~180k to ~118k tokens, so this request can stay on the current 128k local backend instead of switching models.
```

If compression is insufficient:

```text
[MacM4LocalAgent] Compressed context from ~330k to ~285k tokens, but it still exceeds the 256k local window. Switching to the 512k TurboQuant backend; context is preserved.
```

If compression fails:

```text
[MacM4LocalAgent] Context compression failed or was unsafe. Proceeding with backend escalation.
```

---

## Revised Service Lifecycle Algorithm

```python
async def route_request(request):
    request_id = persist_original_request(request)
    raw_tokens = estimate_tokens(request)

    active_limit = backend_limit(active_backend)

    if raw_tokens > active_limit:
        compressed = await try_context_compression(
            request,
            target_budget=active_limit * 0.92,
        )
        compressed_tokens = estimate_tokens(compressed)
        persist_compression_metadata(request_id, raw_tokens, compressed_tokens)
    else:
        compressed = request
        compressed_tokens = raw_tokens

    target_backend = choose_backend(compressed_tokens)

    if target_backend == active_backend and backend_healthy(target_backend):
        notify_if_compressed(request_id, raw_tokens, compressed_tokens)
        return await forward(compressed, target_backend)

    notify_client_status(
        request_id,
        switching_or_compression_message(raw_tokens, compressed_tokens, active_backend, target_backend),
    )

    persist_pending_request(request_id, compressed, original_request_ref=True)

    async with backend_switch_lock:
        if target_backend != active_backend:
            await stop_large_backend(active_backend)
            await wait_for_memory_release()
            await start_backend(target_backend)
            await wait_for_health(target_backend)
            await warm_backend(target_backend)
            set_active_backend(target_backend)

    restored_request = load_pending_request(request_id)
    return await forward(restored_request, target_backend)
```

---

## New Configuration Variables

Add:

```bash
CONTEXT_COMPRESSION_ENABLED=1
CONTEXT_COMPRESSION_ENGINE="headroom"
CONTEXT_COMPRESSION_MODE="router-library"
CONTEXT_COMPRESSION_TARGET_HEADROOM_PERCENT=8
CONTEXT_STORE_DIR=".runtime/context_store"
CONTEXT_INDEX_DIR=".runtime/context_index"
CONTEXT_COMPRESSION_MIN_TOKENS=64000
CONTEXT_COMPRESSION_MAX_LATENCY_SECONDS=30
CONTEXT_COMPRESSION_ALLOW_RETRIEVAL=1
```

---

## New Make Targets

Add:

```makefile
context-compression-install
context-compression-enable
context-compression-disable
context-compression-status
context-compression-bench
headroom-verify
```

---

## Dashboard Additions

Display:

```text
raw token estimate
compressed token estimate
compression ratio
compression engine
compression latency
backend switch avoided: yes/no
retrievable context items
context store size
last compression failure
```

---

## Updated Benchmark Matrix

Benchmark four scenarios:

```text
no compression + current 128k backend
compression + current 128k backend
compression + turbo 256k backend
compression + turbo 512k backend
```

Measure:

```text
compression latency
compression ratio
quality regression
backend switches avoided
external fallbacks avoided
first token latency
total wall-clock time
RAM peak
user-visible delay
```

---

## Updated Acceptance Criteria

1. Compression must never silently discard critical user instructions.
2. Original context must remain retrievable when compressed out.
3. Compression must run before backend switching when token count exceeds active backend budget.
4. If compression brings the request under the active backend limit, no model switch should occur.
5. If compression is insufficient, backend switching proceeds with preserved compressed context and original context reference.
6. Dashboard must show raw tokens, compressed tokens, and switch-avoidance status.
7. User must receive a status message when compression materially changes routing behavior.

---

## Revised Final Target Behavior

```text
raw request
  -> context compression / retrieval shaping
  -> stay on active local backend if compressed context fits
  -> switch to 256k local only if needed
  -> switch to 512k local only if needed
  -> external only if local compressed context still does not fit or local health fails
```

This makes TurboQuant escalation the second line of defense. Headroom/context compression becomes the first line of defense.

---

# Part 2 — Qwen Model-Tier Decision

Add this decision to the Cline Context Janitor integration plan.

## Model selection update

The current MacM4LocalAgent default on a 128GB MacBook Max should remain Qwen3-Coder-Next Q8 at 128k context.

Qwen3-Coder-Next is the best current main local coding model in this architecture because it is large enough for strong agentic coding but still practical on 128GB Apple Silicon.

## Add Qwen3-Coder-30B-A3B as an experimental tier

Add Qwen3-Coder-30B-A3B, sometimes referred to as the smaller or flash Qwen3-Coder variant, as an experimental model candidate.

Recommended roles:

- fast local coding tier;
- always-on Context Janitor model;
- cheaper semantic cleanup model;
- fallback model when the main 128k backend is unloaded;
- benchmark comparison against Qwen3-Coder-Next for janitor quality.

Do not replace Qwen3-Coder-Next as the main model until benchmarks show that the smaller model preserves coding quality and janitor quality.

## Exclude Qwen3-Coder-480B-A35B from local targets

Do not target Qwen3-Coder-480B-A35B for local MacBook execution. It is too large for the 128GB MacBook Max once weights, KV cache, runtime overhead, and long context are included.

It can remain an external or cloud comparison target, but not a local backend target.

## Revised model roles

Main coding backend:

- Qwen3-Coder-Next Q8
- 128k local context
- default local backend for serious coding sessions

Experimental fast/janitor backend:

- Qwen3-Coder-30B-A3B
- 64k to 128k initial context target
- evaluate as always-on janitor model
- evaluate as fast coding model

Experimental long-context backend:

- Qwen3-Coder-Next or Qwen3-Coder-30B-A3B with KV-cache compression
- 256k first target
- 512k only after benchmarks

Not local:

- Qwen3-Coder-480B-A35B
- external/cloud only

## Benchmark requirements

Add model comparison benchmarks:

1. Qwen3-Coder-Next Q8 as main 128k coding model.
2. Qwen3-Coder-30B-A3B as Context Janitor.
3. Qwen3-Coder-30B-A3B as fast coding tier.
4. Qwen3-Coder-Next versus Qwen3-Coder-30B-A3B on cleanup decisions.
5. Qwen3-Coder-Next versus Qwen3-Coder-30B-A3B on long Cline coding sessions.

Measure:

- tokens per second;
- first token latency;
- RAM usage;
- maximum stable context;
- quality of code edits;
- quality of context cleanup;
- missed constraints;
- bad discard decisions;
- backend switches avoided;
- external fallbacks avoided.

## Updated recommendation

Keep Qwen3-Coder-Next as the main stable local model. Add Qwen3-Coder-30B-A3B as an experimental, lower-RAM model for the Context Janitor and fast tier. Do not attempt to run the 480B Qwen3-Coder locally.

---

# Part 3 — Cline Context Janitor Integration Plan

## Target

Fork: https://github.com/mfranklin1/cline (remote: `origin`, branch: `main`)

Local backend project: https://github.com/mfranklin1/MacM4LocalAgent
*(Note: push target is the mfranklin1 personal account — never push to martinfr-certifyos org.)*

Note: the GitHub connector can read this fork but does not have push permission, so this is a Cline/Codex/manual implementation plan rather than a committed change.

## Objective

Integrate Headroom and a local-model-driven Context Janitor directly into the Cline fork that talks to the M4 local agent stack.

The goal is to stop long Cline sessions from becoming append-only transcripts. Cline should continuously curate context before sending requests to MacM4LocalAgent.

Target flow: Cline task event -> checkpoint-aware event capture -> Headroom compression for noisy payloads -> 128k local model semantic cleanup -> active context pack rebuild -> MacM4LocalAgent routing.

## Why integrate inside Cline

The router sees only the final model request. Cline sees richer task intent: user messages, tool calls, tool results, terminal output, file edits, checkpoints, restore actions, editor state, and task history.

Therefore Cline should own semantic task memory. MacM4LocalAgent should own backend routing and model lifecycle.

## Reuse Cline checkpoints

Cline already has checkpoints using a shadow Git repository. Reuse them as the file-state backbone for the janitor.

Do not duplicate checkpointing.

The janitor should attach semantic summaries to checkpoint hashes. A cleanup report should say what changed, why it matters, which files were touched, and which checkpoint can restore that state.

Checkpoints are for file rollback. The Context Janitor is for semantic memory and prompt cleanup.

## Proposed Cline subsystem

Add a new context subsystem under the VS Code app source tree.

Suggested components:

- ContextJanitorService
- ContextLedger
- ActiveContextPack
- RetrievalManifest
- HeadroomAdapter
- JanitorModelClient
- ContextBudgeter
- ContextPromptBuilder

Responsibilities:

- capture task events;
- compress noisy payloads;
- classify what should stay active;
- maintain durable task memory;
- link summaries to Cline checkpoints;
- build curated prompts;
- expose janitor status to the UI.

## Integration points

### Task lifecycle

Hook into task creation, user message submission, tool result handling, file edit results, command results, and task completion.

The janitor should see each event before the final prompt is assembled.

### Checkpoint events

When a checkpoint is created, attach the checkpoint hash to the current event batch. The janitor should update the ledger and active context pack with any relevant file changes.

### Provider request assembly

This is the most important insertion point.

Before Cline sends a request to the selected provider, replace append-only task history with a curated prompt when janitor mode is enabled.

The prompt should include system rules, current user request, active context pack, durable constraints, relevant checkpoint summaries, current diff or failure summary, retrieval manifest, and only the recent messages that still matter.

### Settings UI

Add toggles for Context Janitor, Headroom compression, janitor model provider, trigger token count, maximum janitor latency, status messages, and local archive retention.

## Headroom role

Headroom should be used as the mechanical compression layer.

Prioritize terminal output, test output, install logs, large JSON, repeated tool output, large file reads, and search results.

Headroom reduces token-heavy inputs. It should not own durable project memory. The Cline janitor decides semantic importance.

## Local 128k model role

Use the existing MacM4LocalAgent 128k endpoint as the janitor model.

The janitor model should not produce user-facing code. It should produce structured cleanup decisions: keep verbatim, summarize, archive retrievably, discard, durable constraints, ledger updates, active context updates, risk flags, and confidence.

Rules:

- if unsure, archive retrievably rather than discard;
- never discard explicit user instructions;
- never discard current errors, active diffs, or open tasks;
- preserve failed approaches if they prevent repeated mistakes;
- link file-state summaries to checkpoint hashes.

## Context storage

Use Cline global or workspace storage, not tracked project files.

Store project ledger, active context pack, retrieval manifest, cleanup reports, event batches, and archived raw context.

Storage should persist across editor restarts, similar to checkpoints.

## Routing contract with MacM4LocalAgent

Cline should send metadata to MacM4LocalAgent when possible:

- raw token estimate;
- curated token estimate;
- janitor enabled;
- Headroom enabled;
- task id;
- workspace hash;
- current checkpoint hash.

The local agent can use this to avoid unnecessary 256k or 512k backend switches and to show dashboard metrics.

## User experience

Cline should show a small status message only when cleanup materially changes behavior.

Examples:

- context was curated enough to stay on the 128k local model;
- context still exceeds 128k and requires 256k;
- old terminal output was archived but remains retrievable.

Avoid noisy per-turn messages.

## Initial defaults

Keep disabled by default until benchmarked.

Recommended defaults:

- Context Janitor disabled initially (enable via settings toggle; flip to on once first benchmark pass is complete);
- runs automatically in the background — no per-turn approval prompt, surfaces only a summary message when cleanup materially changes routing;
- Headroom enabled when Janitor is enabled;
- janitor model is local-long-128k (upgrade to Qwen3-Coder-30B-A3B once Ollama tag is confirmed);
- trigger at roughly 64k tokens;
- growth trigger at roughly 20k additional tokens;
- max janitor latency around 45 seconds;
- archive raw context locally.

## Implementation phases

### Phase 1: Discover insertion points

Map the current fork and the M4 integration branch. Identify the MacM4 provider implementation, request assembly path, task history storage, tool result handling, checkpoint event path, and settings UI definitions.

Deliverable: docs/context-janitor/cline-insertion-points.md

### Phase 2: Add context storage and ledger

Implement project ledger, active context pack, retrieval manifest, cleanup report storage, and checkpoint hash linkage. No model calls yet.

### Phase 3: Add Headroom adapter

Compress noisy outputs before they become permanent task context. Start with terminal output and test output only.

Acceptance: original output is archived, compressed output preserves errors and file paths, and token savings are recorded.

### Phase 4: Add janitor model client

Call the local 128k model for semantic cleanup decisions. Use conservative rules and structured output validation.

Acceptance: ledger updates work, explicit constraints are preserved, and uncertain material is archived retrievably.

### Phase 5: Add budgeted prompt builder

Intercept provider request assembly. When janitor mode is enabled, send curated context rather than full append-only task history.

### Phase 6: Add UI and observability

Add settings toggles, status chip, token stats, cleanup report viewer, and links from cleanup reports to checkpoint compare.

### Phase 7: Integrate with MacM4LocalAgent routing

Send curated token metadata to the local agent. Backend switching should happen only after cleanup.

### Phase 8: Benchmark

Benchmark long terminal logs, repeated test failures, multi-file refactors, architecture changes, and failed-then-corrected approaches.

Measure raw tokens, Headroom-only tokens, janitor-curated tokens, switches avoided, external fallbacks avoided, latency added, missed constraints, and final code correctness.

## Acceptance criteria

1. Cline behaves exactly as before when janitor mode is disabled.
2. Cline checkpoints remain the source of truth for file rollback.
3. Janitor summaries link back to checkpoint hashes.
4. Headroom compresses noisy outputs before they bloat task context.
5. The local 128k model produces conservative cleanup decisions.
6. The prompt sent to MacM4LocalAgent is curated when janitor mode is enabled.
7. Explicit user instructions, active failures, active diffs, and open tasks are never silently discarded.
8. Raw archived context remains retrievable.
9. Backend switching decreases in long sessions.
10. UI makes cleanup visible without being noisy.

## Key design decision

Cline owns semantic task memory. MacM4LocalAgent owns backend routing and lifecycle. Cline should send the cleanest possible request; MacM4LocalAgent should choose the cheapest safe backend.

## Open questions

1. ~~Which branch contains the M4 plugin integration?~~ **Answered**: `main` branch of `mfranklin1/cline`.
2. ~~Is the M4 integration implemented as an API provider, MCP integration, or custom proxy target?~~ **Answered**: Standard `litellm` API provider type (OpenAI-compatible endpoint at `localhost:4000`). Confirmed in `src/shared/api.ts`.
3. Should janitor storage live under VS Code global storage or workspace storage?
4. Should cleanup reports appear in the chat transcript or only in diagnostics?
5. Should cleanup run automatically or require explicit approval during early testing?
6. *(New)* Which MLX model should the turbo tiers use — `Qwen2.5-Coder-32B-Instruct-4bit` (as originally specced) or `Qwen3-Coder-Next-4bit` (the current model family already in use)? Needs verification that the target model has an MLX-LM quantised release and that MLX-LM actually supports `turbo_kv_bits`.
7. *(New)* What is the exact Ollama model tag for Qwen3-Coder-30B-A3B? Needed to configure the janitor model and add it to the backend registry.
