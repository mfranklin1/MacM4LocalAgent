# Offline mode

The hybrid router auto-detects when Anthropic is unreachable and
transparently rewrites every Claude selection to `local-long`. You
can also force it explicitly (`make offline`) for an airplane,
tethered hotspot, secure-lab session, or any other time you want
a guaranteed zero-cloud workflow.

This page documents:

1. How the proxy decides "offline vs online".
2. How to toggle / inspect it (`make offline` / `online` / `offline-status`).
3. **How to clear Cline context cleanly when you go offline mid-task.**
   This is the bit most people get wrong: if your earlier turns rode
   Claude, the local model will keep trying to mimic that style and
   produce slightly off-key output until you reset the conversation.
4. Strict mode for tests / CI (`OFFLINE_STRICT=1`).
5. How to audit downgrades after the fact.

---

## 1. Decision tree

```
                       ┌────────────────────────┐
                       │  request comes in      │
                       │  model = claude-* OR   │
                       │  hybrid-auto -> claude │
                       └─────────┬──────────────┘
                                 │
                       ┌─────────▼───────────┐
                       │ OFFLINE env var?    │
                       └─────────┬───────────┘
              ┌──────────────────┼──────────────────┐
              │                  │                  │
           OFFLINE=1         OFFLINE=0         OFFLINE=auto
        (or absent + file)   (forced online)   (default if unset)
              │                  │                  │
              ▼                  ▼                  ▼
        ┌──────────┐        ┌──────────┐    ┌─────────────────┐
        │downgrade │        │use Claude│    │TCP-probe        │
        │to        │        │as normal │    │api.anthropic.com│
        │local-long│        │          │    │:443 (1.5s)      │
        └──────────┘        └──────────┘    └────┬────────────┘
                                                 │
                                  probe ok       │ probe fail
                                       ┌─────────┴────────┐
                                       ▼                  ▼
                                  ┌──────────┐      ┌──────────┐
                                  │use Claude│      │downgrade │
                                  └──────────┘      │to        │
                                                    │local-long│
                                                    └──────────┘
```

Precedence (highest first):

1. `OFFLINE=1` in real env — offline, skip probe entirely.
2. `OFFLINE=0` in real env — online, skip probe.
3. `OFFLINE=auto` (or unset) in real env — probe.
4. Same keys in `config/detected.env` — same semantics, lower priority.
5. Nothing set anywhere — probe.

The probe is bounded (1.5s connect timeout), threaded, and cached
(30s positive, 10s negative). A routing decision never blocks for
more than ~1.5s and a flaky link doesn't thrash on every turn.

---

## 2. Toggle / inspect

| Command | Effect |
| --- | --- |
| `make offline` | Writes `OFFLINE=1` to `config/detected.env`. Router re-reads per call — no proxy restart needed. |
| `make online` | Writes `OFFLINE=auto`. Router goes back to probing on every routing decision. |
| `make offline-status` | Prints the current `OFFLINE` / `OFFLINE_STRICT` values and runs a live probe (same exact code path the router uses). |

Sample `make offline-status` while sitting at a café with wifi:

```text
OFFLINE=auto
OFFLINE_STRICT=0
live probe: ONLINE
```

Same machine in flight mode, wifi off:

```text
OFFLINE=auto
OFFLINE_STRICT=0
live probe: OFFLINE (timeout: timed out)
```

The router doesn't need that command to be run — it reaches the same
conclusion on its own — but it's handy for confirming you're on a
captive portal that *looks* like wifi but actually isn't reaching
the internet.

---

## 3. Clearing Cline context before / during an offline session

**This is the bit you actually want.** When you've been working with
Claude (Opus / Sonnet / Haiku) and then go offline, the local model
inherits the entire prior conversation — including assistant turns
that the cloud-tier produced. It will then try to continue in that
same style on a model that's structurally different. Symptoms:

- Cline shows the local model "thinking" but the answers are
  noticeably worse than the prior Claude turns.
- The local model invents content that referenced specifics from a
  Claude-tier response 10 turns ago.
- Tool calls succeed but the synthesis is shallow.

The fix is to **start a fresh Cline task** so the conversation
history resets. The router warning (printed once per process to
`stderr` and appended to `.logs/offline-events.log`) explicitly
recommends this — but it can't reach into Cline's task store to do
it for you.

### Quick reference — three ways to clear

| Method | When to use |
| --- | --- |
| **`Cmd+Shift+P` → "Cline: New Task"** | Fastest. Wipes the in-flight task and lands you on an empty prompt. |
| Trash icon in the Cline panel header | Same effect, mouse-driven. The button only shows when a task is open. |
| Cline gear icon → **History** → swipe-left or click the × on the task row | Permanently deletes a task (and any associated files Cline cached). Use after the session if you don't want it surfaced by future `[history search]` tool calls. |

After clearing, **re-paste any reference files** you still need —
the new task has no memory of `read_file` results from the old one.
For airplane sessions this is usually a feature, not a bug: you get
a clean local-only workspace with no Claude residue.

### Verifying the reset took effect

Run a one-line probe from inside Cline:

> What is the model behind this assistant turn? Reply with one word.

A reset, offline Cline session will say something like
`qwen3-coder-next` or `local-long` (depending on how literally the
model reads the question). A *not-reset* session can still surface
Claude-shaped boilerplate from the prior context — especially in
the first one or two turns — even though the actual upstream call
is local now.

### When you don't need to reset

If your prior turns are all `local-long` already (i.e. the router
never escalated to Claude), nothing carries over from the cloud
tier and resetting just costs you context. Check the route log:

```bash
sqlite3 cost/cost.db \
  "SELECT datetime(ts,'unixepoch','localtime') AS t, tier, route_reason \
     FROM requests \
     WHERE task_id = (SELECT task_id FROM requests ORDER BY ts DESC LIMIT 1) \
     ORDER BY ts;"
```

If every row in your current task is `local-fast` or `local-long`,
keep the context.

---

## 4. Strict mode (`OFFLINE_STRICT=1`)

By default, an offline downgrade is **silent**: the proxy quietly
substitutes `local-long` and stamps `offline_downgrade=true` in the
request log. That's friendly for everyday flight-mode use.

For tests / CI / production runs where a silent downgrade would
mask a misconfiguration (e.g. `ANTHROPIC_API_KEY` accidentally
unset, DNS broken in a container), set strict mode:

```bash
make offline                # or OFFLINE=1 in env
./scripts/offline-mode.sh strict on
```

With `OFFLINE_STRICT=1`, an **explicit** Claude request (direct
`claude-*` model name, `gpt-claude-*` Cursor alias, or a leading
`[claude]`/`[opus]`/`[sonnet]`/`[haiku]` tag) raises an HTTP 503
to the client instead of being silently downgraded:

```text
HTTP/1.1 503 Service Unavailable
{
  "error": "Offline mode is active (network unreachable (probe
   api.anthropic.com:443 failed)) and OFFLINE_STRICT=1. Refusing to
   attempt a Claude call. Drop OFFLINE_STRICT or retry with a
   [local] tag / local-long model."
}
```

`hybrid-auto` requests still silently downgrade in strict mode —
the user didn't ask for Claude by name, so a quiet local fallback
is the right call.

---

## 5. Auditing downgrades

Every downgrade is recorded in three places:

| Where | What |
| --- | --- |
| `cost.db.requests.route_reason` | Starts with `offline-downgrade: ...` so SQL filters work. |
| `.logs/offline-events.log` | One JSON line per event: `{ts, kind, requested_alias, resolved_model, fallback, reason}`. |
| `stderr` (proxy log) | One-time-per-process banner the moment we first detect offline. |

Find every offline downgrade in the last 24h:

```bash
sqlite3 cost/cost.db "
  SELECT datetime(ts,'unixepoch','localtime') AS t,
         tier, model, route_reason
    FROM requests
   WHERE route_reason LIKE 'offline-downgrade%'
      OR route_reason LIKE 'cline+offline-downgrade%'
      OR route_reason LIKE 'cline-mode: cline+offline-downgrade%'
   ORDER BY ts DESC
   LIMIT 50;
"
```

Or just `tail .logs/offline-events.log`:

```text
{"ts": 1778001234, "kind": "downgrade", "requested_alias": "hybrid-auto",
 "resolved_model": "claude-code", "fallback": "local-long",
 "reason": "network unreachable (probe api.anthropic.com:443 failed)",
 "explicit_claude": false}
```

---

## 6. What stays the same when offline

- **`local-fast` / `local-long` / `local-agent`** all keep working
  exactly as online — they were never going to make a network call.
- **`[local]` tag** still wins absolute precedence (cost safety,
  unchanged behavior).
- **Cline detection / over-generation controls** unchanged. Cline
  traffic still gets its tailored stop sequences and route logic;
  the only difference is that the Claude-tier branches are
  unreachable.
- **`cost.db`** still records every call, with `actual_cost=0` for
  the local-long downgrades (matching the rest of the local-tier
  bookkeeping).

## 7. What changes when offline

- **`[claude]` / `[opus]` / `[sonnet]` / `[haiku]` tags** are
  ignored, with the original tag preserved in `route_reason`
  (`offline-downgrade: ...; ignored [opus] tag`).
- **`hybrid-auto` decisions that would have gone to Claude** route
  to `local-long` instead.
- **Direct `claude-*` model selections** route to `local-long`
  instead (unless `OFFLINE_STRICT=1`).
- **Cline tool-result Claude rescues** are suppressed — a Python
  traceback in a tool result will not escalate to Claude while
  offline, even if it normally would.
- **Sticky escalations** are not cleared by going offline; they
  hibernate. When the network returns within the 30-minute TTL,
  the sticky task picks up where it left off.

---

## 8. Quick recipes

```bash
# Going on a flight in 5 minutes
make offline
# (work normally — every Claude tier silently routes to local-long)
make offline-status     # sanity-check before takeoff

# Back on the ground
make online
make offline-status     # confirms: live probe: ONLINE

# CI / test runs where Claude must not be silently substituted
OFFLINE=1 OFFLINE_STRICT=1 make test

# Investigate after a session
tail -20 .logs/offline-events.log
sqlite3 cost/cost.db "SELECT COUNT(*) FROM requests \
  WHERE route_reason LIKE '%offline-downgrade%';"
```

## See also

- [docs/routing.md](routing.md) — base routing tree (online behavior).
- [docs/RUNBOOK-cline-setup.md](RUNBOOK-cline-setup.md) — full Cline
  install + configuration.
- [docs/cost-model.md](cost-model.md) — how `route_reason` ties into
  the cost ledger.
