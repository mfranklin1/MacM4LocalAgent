"""LiteLLM custom callback that routes requests by prompt size + complexity
and records every call (with shadow Claude cost) into cost/cost.db.

Wired in as both `callbacks` (for pre-call routing) and `success_callback`
(for post-call cost ingestion) in config/litellm-config.yaml.

Routing rules (thresholds come from config/detected.env):
  - <=ROUTE_LONG_MAX tokens, not complex  -> local-long (Ollama)
  - >ROUTE_LONG_MAX tokens OR complex     -> claude-code

If the user explicitly picks a real model name (local-long or claude-code)
we never override. We only intercept the magical `hybrid-auto` alias
defined in litellm-config.yaml.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import sqlite3
import sys
import threading
import time
from typing import Any, Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cost.pricing import (  # noqa: E402
    actual_claude_cost,
    maybe_run_pricing_check,
    shadow_cost as _shadow_cost_fn,
    sonnet_rate,
)
from router.complexity_classifier import classify  # noqa: E402
from router.offline_mode import (  # noqa: E402
    DEFAULT_OFFLINE_FALLBACK,
    OfflineStrictReject,
    is_offline,
    maybe_downgrade,
    offline_reason,
)
from router.overgeneration_control import (  # noqa: E402
    CLAUDE_THINKING_EFFORT_DEFAULT,
    _looks_like_cline,
    _model_supports_think,
    apply_claude_thinking_params,
    apply_multi_turn_tighten,
    apply_static_guardrail,
    inject_qwen3_think_directive,
)

try:
    from litellm.integrations.custom_logger import CustomLogger as _LiteLLMCustomLogger
except Exception:  # tests / standalone runs don't need LiteLLM installed
    class _LiteLLMCustomLogger:  # type: ignore[no-redef]
        pass

# Backwards-compat shims for the few external callers (and tests) that
# still import these constants directly. They mirror the canonical
# Sonnet 4.6 rates from cost/pricing.py and are kept in sync via
# sonnet_rate(). New code should use actual_claude_cost() and the
# shadow_cost helper from cost.pricing.
CLAUDE_INPUT_PER_TOKEN = sonnet_rate().input
CLAUDE_OUTPUT_PER_TOKEN = sonnet_rate().output

# Kick off the (non-blocking, best-effort) pricing freshness check on
# import. Safe to call: it bails immediately if it ran in the last 24h
# or if PRICING_STARTUP_CHECK=0.
maybe_run_pricing_check()

DB_PATH = REPO_ROOT / "cost" / "cost.db"


def _read_env_file(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


_ENV = _read_env_file(REPO_ROOT / "config" / "detected.env")
ROUTE_LONG_MAX = int(_ENV.get("ROUTE_LONG_MAX", "128000"))


def _control_flag(name: str, default: str = "1") -> bool:
    """Read a bool flag from env or detected.env. We let real env vars
    win over the file so a quick `OVERGEN_STATIC=0 make restart`
    flips the strategy without editing detected.env."""
    raw = os.environ.get(name)
    if raw is None:
        raw = _ENV.get(name, default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# Strategy A (static guardrail) is on by default; Strategy C is on by
# default too. Disable with OVERGEN_STATIC=0 / OVERGEN_MULTI_TURN=0.
ENABLE_STATIC_GUARDRAIL = _control_flag("OVERGEN_STATIC", "1")
ENABLE_MULTI_TURN_TIGHTEN = _control_flag("OVERGEN_MULTI_TURN", "1")
# Qwen3 /think injection. Legacy switch kept for back-compat: when set,
# /think is injected on forced-local turns even if the broader thinking
# mode (ROUTER_THINKING) is off.
ENABLE_THINK_INJECTION = _control_flag("ROUTER_THINK_INJECTION", "1")

# Tool-call routing. Cline executes ONLY native OpenAI `tool_calls`; it never
# parses tool calls out of text. When a request carries `tools` but routing
# landed on a local tier whose model can't emit parseable tool_calls, redirect
# it to a tool-native local tier. Set CLINE_TOOL_TIER="" to disable.
#
# History: local-long (qwen3-coder-next) was the non-tool-native tier until
# 2026-07-02, when its Ollama import was recreated with the qwen3-coder
# RENDERER/PARSER and moved to ollama_chat/ -- it is now BOTH the strongest
# local tier and tool-native, so it became the redirect target instead of
# the redirect source. The qwen2.5-coder tiers stay non-tool-native: the
# model declares the `tools` capability but writes calls as raw JSON text
# (0/2 in live tests) instead of the <tool_call> envelope Ollama parses.
TOOL_NATIVE_LOCAL_TIER = (os.environ.get("CLINE_TOOL_TIER") or _ENV.get("CLINE_TOOL_TIER", "local-long")).strip()
# Local tiers whose underlying model does NOT reliably emit parseable
# tool_calls. Comma-separated aliases.
_NON_TOOL_NATIVE_LOCAL = {
    t.strip()
    for t in (
        os.environ.get("CLINE_NON_TOOL_TIERS")
        or _ENV.get("CLINE_NON_TOOL_TIERS", "local-coder-14b,local-coder-32b")
    ).split(",")
    if t.strip()
}
# Upstream model-id substrings known to fail tool-call emission regardless
# of which alias or route served them (used by _is_tool_native_model for
# post-hoc fallback annotation, where only the upstream id is available).
_NON_TOOL_NATIVE_MODEL_SUBSTRINGS = ("qwen2.5-coder",)

# What the `claude-code` alias resolves to upstream. MUST stay in sync with
# the claude-code entry in config/litellm-config.yaml. Default is Haiku 4.5
# -- the only model the Team subscription OAuth token is permitted to call
# on the raw API (verified 2026-07-02; Sonnet/Opus policy-429). Controls
# whether thinking-mode params are injected: Haiku 4.5 rejects adaptive
# thinking and output_config.effort with a 400, so the injection guard must
# see the UPSTREAM model, not the alias. The planned apikey escalation
# setting will override this (env or detected.env) when claude-code is
# repointed at sonnet/opus/fable.
CLAUDE_CODE_UPSTREAM_MODEL = (
    os.environ.get("CLAUDE_CODE_MODEL")
    or _ENV.get("CLAUDE_CODE_MODEL", "claude-haiku-4-5")
).strip()


def _resolved_claude_model(model_name: str | None) -> str:
    """Resolve a Claude-tier alias to its upstream model id for capability
    checks. Only `claude-code` (and its gpt- mirror) is indirect; the
    per-model aliases (claude-haiku-4-5, claude-sonnet-5, ...) already name
    their upstream model."""
    name = model_name or ""
    canonical = name[len("gpt-"):] if name.startswith("gpt-") else name
    if canonical == "claude-code":
        return CLAUDE_CODE_UPSTREAM_MODEL
    return name

# Thinking mode: stream a live reasoning trace into the harness (Cline)
# for BOTH tiers. When on:
#   - Qwen3 local tiers get /think injected on every turn (so the model
#     emits a <think> trace, which the streaming hook below relays as
#     reasoning_content), gated on the resolved model actually being a
#     Qwen3 model via _model_supports_think().
#   - Claude tiers get Anthropic extended thinking enabled, which LiteLLM
#     surfaces to the client as reasoning_content deltas.
# Always-on by design (see the feat/thinking-mode discussion); flip off
# globally with ROUTER_THINKING=0. Costs ~10-30% more output tokens and
# roughly doubles per-turn latency.
ENABLE_THINKING_MODE = _control_flag("ROUTER_THINKING", "1")

# Turbo escalation: when TURBO_ENABLED=1, context-saturated requests are
# first compressed (via router.context_compression) and, if the compressed
# token count fits a local turbo backend (router.backend_registry), routed
# there instead of escalating to claude-code. Set TURBO_ENABLED=0 (or omit
# from detected.env) to restore the original behaviour exactly.
#
# TODO: add "local-turbo-256k" and "local-turbo-512k" aliases to
#       config/litellm-config.yaml before enabling this in production.
TURBO_ENABLED = _control_flag("TURBO_ENABLED", "0")


def _thinking_effort() -> str:
    """Claude adaptive-thinking effort level (env override).

    One of low|medium|high|xhigh|max; unknown values fall back to the
    default. `high` is Anthropic's default ("almost always thinks").
    """
    raw = os.environ.get("ROUTER_THINKING_EFFORT") or _ENV.get(
        "ROUTER_THINKING_EFFORT", CLAUDE_THINKING_EFFORT_DEFAULT
    )
    return str(raw).strip().lower() or CLAUDE_THINKING_EFFORT_DEFAULT


# When /think is injected on a local Qwen3 turn the <think> reasoning
# trace and the final answer share the same max_tokens budget -- and the
# over-generation static guardrail has just clamped that to ~6k. Without
# headroom the trace eats the answer. Re-assert a higher floor on think
# turns. Env override: ROUTER_THINKING_LOCAL_MAX.
LOCAL_THINK_MAX_TOKENS_DEFAULT = 12288


def _local_think_max_tokens() -> int:
    raw = os.environ.get("ROUTER_THINKING_LOCAL_MAX") or _ENV.get(
        "ROUTER_THINKING_LOCAL_MAX", str(LOCAL_THINK_MAX_TOKENS_DEFAULT)
    )
    try:
        return max(2048, int(raw))
    except (TypeError, ValueError):
        return LOCAL_THINK_MAX_TOKENS_DEFAULT


# ---- Qwen3 <think> stream parsing -------------------------------------------
# Qwen3 emits its reasoning trace inline as `<think> ... </think>` in the
# normal content stream. Cline only renders a live "Thinking" section for
# text that arrives on the `reasoning_content` delta channel, so we split
# the content stream: text inside <think>...</think> is re-routed to
# reasoning_content, everything else stays on content.
_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _partial_tag_suffix_len(s: str, tag: str) -> int:
    """Length of the longest suffix of `s` that is a proper prefix of
    `tag` (so we can hold it back in case the tag is split across stream
    chunks). 0 if none."""
    for k in range(min(len(s), len(tag) - 1), 0, -1):
        if s.endswith(tag[:k]):
            return k
    return 0


def _split_think_stream(buffer: str, in_think: bool) -> tuple[str, str, str, bool]:
    """Incremental <think> splitter.

    Given the accumulated `buffer` and whether we're currently inside a
    think block, return (content_out, reasoning_out, carry, in_think):
      - content_out: text to emit on the content channel this step
      - reasoning_out: text to emit on the reasoning_content channel
      - carry: trailing bytes held back (possible split tag) for next call
      - in_think: updated state
    """
    content_out: list[str] = []
    reasoning_out: list[str] = []
    while True:
        if in_think:
            idx = buffer.find(_THINK_CLOSE)
            if idx == -1:
                break
            reasoning_out.append(buffer[:idx])
            buffer = buffer[idx + len(_THINK_CLOSE):]
            in_think = False
        else:
            idx = buffer.find(_THINK_OPEN)
            if idx == -1:
                break
            content_out.append(buffer[:idx])
            buffer = buffer[idx + len(_THINK_OPEN):]
            in_think = True
    tag = _THINK_CLOSE if in_think else _THINK_OPEN
    hold = _partial_tag_suffix_len(buffer, tag)
    safe = buffer[: len(buffer) - hold] if hold else buffer
    carry = buffer[len(buffer) - hold:] if hold else ""
    if in_think:
        reasoning_out.append(safe)
    else:
        content_out.append(safe)
    return "".join(content_out), "".join(reasoning_out), carry, in_think


def _is_claude_model_name(model: str | None) -> bool:
    """True if the resolved model name belongs to a Claude tier."""
    if not isinstance(model, str):
        return False
    m = model.lower()
    return "claude" in m or "anthropic" in m


def _is_local_model_name(model: str | None) -> bool:
    """True if the resolved model name belongs to the local tier."""
    if not isinstance(model, str):
        return False
    canonical = model[len("gpt-"):] if model.startswith("gpt-") else model
    return canonical in ("local-long", "local-agent") or canonical.startswith("local-coder-")

# When OVERGEN_TRACE=1, append a one-line summary to .logs/overgen-trace.log
# every time a control fires. Used to verify Cursor traffic is being
# constrained correctly. Off by default so we don't pollute disk in normal
# use.
OVERGEN_TRACE = _control_flag("OVERGEN_TRACE", "0")
_TRACE_PATH = REPO_ROOT / ".logs" / "overgen-trace.log"

# When CLINE_TRACE=1, dump every inbound /v1/chat/completions request body
# (BEFORE any routing or control mutation) to a per-request JSON file under
# .logs/cline-dumps/. Bounded by CLINE_TRACE_MAX so an idle Cline session
# can't fill the disk. Used to investigate why Cline-style agent requests
# misbehave on small local models (e.g. tool-call confusion).
#
# REMOVE OR DISABLE AFTER INVESTIGATION -- these dumps contain the full
# system prompt + messages + tool definitions, which can be sensitive.
CLINE_TRACE = _control_flag("CLINE_TRACE", "0")
# Read from real env first, then detected.env (same precedence as _control_flag).
CLINE_TRACE_MAX = int(
    os.environ.get("CLINE_TRACE_MAX")
    or _ENV.get("CLINE_TRACE_MAX")
    or "20"
)
_CLINE_DUMP_DIR = REPO_ROOT / ".logs" / "cline-dumps"
_cline_dump_count = 0  # process-local; resets on proxy restart


# Keys we know LiteLLM injects that contain circular refs or arbitrary
# Python objects (like internal callback handles). We project the request
# down to only the OpenAI-API-shaped fields that matter for analysis.
_DUMP_KEEP_KEYS = (
    "model",
    "messages",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "stream",
    "response_format",
    "user",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
)


def _dump_full_request(data: dict[str, Any]) -> None:
    """Best-effort dump of the inbound request to .logs/cline-dumps/.

    Bounded by CLINE_TRACE_MAX. Never raises -- a dumper that crashes
    requests is worse than no dumper at all.

    We project `data` down to OpenAI-API-shaped fields only; LiteLLM
    augments the dict with internal handles (callbacks, deployment
    objects) that contain circular references and aren't JSON-serializable.
    """
    global _cline_dump_count
    try:
        if _cline_dump_count >= CLINE_TRACE_MAX:
            return
        _CLINE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        path = _CLINE_DUMP_DIR / f"req-{ts}-{_cline_dump_count:03d}.json"
        projected = {k: data[k] for k in _DUMP_KEEP_KEYS if k in data}
        # default=str catches any stragglers (e.g. enum values).
        path.write_text(json.dumps(projected, indent=2, default=str))
        _cline_dump_count += 1
    except Exception as e:  # pragma: no cover -- diagnostic during investigation
        print(f"[router] cline-dumper failed: {type(e).__name__}: {e}", file=sys.stderr)


def _trace_overgen(**kw: Any) -> None:
    """Best-effort one-line append for live observation. Never raises."""
    try:
        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = (
            f"{int(time.time())} "
            f"req={kw.get('requested')} "
            f"resolved={kw.get('resolved')} "
            f"max_tokens={kw.get('pre_max')}->{kw.get('post_max')} "
            f"stop_set={bool(kw.get('post_stop'))} "
            f"n_msgs={kw.get('pre_n_msgs')}->{kw.get('post_n_msgs')} "
            f"first_role={kw.get('pre_first_role')}->{kw.get('post_first_role')}\n"
        )
        with _TRACE_PATH.open("a") as f:
            f.write(line)
    except Exception:
        pass


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Run migration BEFORE schema -- schema.sql declares an index on
    # task_id which would fail to create on databases predating that
    # column. The migration is a no-op on fresh databases. We delegate
    # to cost.ingest's helper to keep the migration logic in one place.
    from cost.ingest import _migrate_requests_columns
    _migrate_requests_columns(conn)
    schema = (REPO_ROOT / "cost" / "schema.sql").read_text()
    conn.executescript(schema)
    return conn


# Tiktoken encoder is loaded lazily so the import cost (~30ms) is paid
# once per process, not per request. We use cl100k_base because:
#   - It's the closest publicly available BPE to the actual Qwen
#     tokenizer (within ~5% on code-heavy English prompts in our bench).
#   - tiktoken is already installed by scripts/40-litellm.sh.
# A module-level singleton + lock keeps the lazy init thread-safe under
# LiteLLM's worker pool.
_TIKTOKEN_ENCODER = None
_TIKTOKEN_LOCK = threading.Lock()
_TIKTOKEN_DISABLED = os.environ.get("ROUTER_DISABLE_TIKTOKEN") == "1"


def _get_tiktoken_encoder():
    """Return a cl100k_base encoder, or None if tiktoken is unavailable
    or explicitly disabled."""
    global _TIKTOKEN_ENCODER
    if _TIKTOKEN_DISABLED:
        return None
    if _TIKTOKEN_ENCODER is not None:
        return _TIKTOKEN_ENCODER
    with _TIKTOKEN_LOCK:
        if _TIKTOKEN_ENCODER is not None:
            return _TIKTOKEN_ENCODER
        try:
            import tiktoken
            _TIKTOKEN_ENCODER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Any failure (offline, missing package, network sandbox)
            # falls back to the heuristic. Mark disabled so we don't
            # re-attempt on every call.
            _TIKTOKEN_ENCODER = None
            globals()["_TIKTOKEN_DISABLED"] = True
    return _TIKTOKEN_ENCODER


def _heuristic_tokens_from_chars(total_chars: int) -> int:
    """The original chars/3.6 estimate. Kept as a fast first-pass and
    as a fallback when tiktoken can't be loaded."""
    return max(1, int(total_chars / 3.6))


# Routing precision matters most near a tier boundary: if the
# heuristic estimate says "20K tokens" and the boundary is 16K, the
# routing decision is sensitive to estimator error. Outside this band
# the heuristic is good enough (and roughly 20-50x faster than running
# the prompt through tiktoken). Boundary band is symmetric in chars
# rather than tokens so we don't have to tokenize first.
#
# `_TIER_BOUNDARY_BAND_FRACTION = 0.2` means: if the heuristic
# estimate is within 20% of ROUTE_LONG_MAX we re-tokenize with tiktoken
# to make a more accurate routing call. Outside the band, we trust the
# heuristic.
_TIER_BOUNDARY_BAND_FRACTION = float(
    os.environ.get("ROUTER_BOUNDARY_BAND", "0.2")
)


def _near_tier_boundary(heuristic_tokens: int) -> bool:
    """True if the heuristic estimate is close enough to the tier
    boundary that estimator error could flip the routing decision."""
    if ROUTE_LONG_MAX <= 0:
        return False
    band = ROUTE_LONG_MAX * _TIER_BOUNDARY_BAND_FRACTION
    return abs(heuristic_tokens - ROUTE_LONG_MAX) <= band


def _collect_text(messages: Iterable[dict[str, Any]] | None) -> tuple[int, str]:
    """Walk message contents once, returning (total_chars, flat_text).
    Used by both the heuristic and the accurate path so we don't iterate
    the messages list twice."""
    total_chars = 0
    parts: list[str] = []
    if not messages:
        return 0, ""
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
            parts.append(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total_chars += len(part["text"])
                    parts.append(part["text"])
    return total_chars, "\n".join(parts)


def _estimate_tokens(messages: Iterable[dict[str, Any]] | None) -> int:
    """Token estimate used for routing decisions.

    Strategy: cheap chars/3.6 heuristic everywhere except in the
    ±20% band around a tier boundary. Inside the band we run tiktoken
    (cl100k_base) for a more accurate count, because that's where
    estimator error actually flips the routing decision.

    Rationale: tiktoken is roughly 20-50x slower than the heuristic
    (still <5ms for a typical Cline turn), and most prompts are NOT
    near a boundary. Pay the cost only when it matters.
    """
    if not messages:
        return 0
    total_chars, flat = _collect_text(messages)
    heuristic = _heuristic_tokens_from_chars(total_chars)
    if not _near_tier_boundary(heuristic):
        return heuristic
    enc = _get_tiktoken_encoder()
    if enc is None:
        return heuristic
    try:
        return max(1, len(enc.encode(flat, disallowed_special=())))
    except Exception:
        return heuristic


def _flat_prompt(messages: Iterable[dict[str, Any]] | None) -> str:
    """Return all message text joined with newlines. Kept for callers
    that need the flat text without the char count side-channel."""
    _, flat = _collect_text(messages)
    return flat


def decide_tier(messages: Iterable[dict[str, Any]] | None) -> tuple[str, str, int]:
    """Return (model_name, reason, estimated_tokens).

    Offline awareness: if `is_offline()` returns True at the point of
    decision, we never return a Claude tier. Explicit `[claude]/[opus]/
    [sonnet]/[haiku]` tags are still recognized but downgraded with a
    descriptive reason so the cost ledger reflects WHY we ignored the
    user's pin. The async_pre_call_hook applies the same guard as a
    final chokepoint for direct claude-* model selections that never
    pass through decide_tier()."""
    tokens = _estimate_tokens(messages)
    prompt = _flat_prompt(messages)
    is_complex, why = classify(prompt)
    offline_now = is_offline()

    # Per-tier Claude override beats size + complexity for non-Cline
    # callers too (CLI, curl, benches). `[local]` is still handled by
    # classify() above and surfaces as is_complex=False, so the size
    # rules below apply -- which is the documented behavior.
    override_alias, override_tag = _extract_model_override(prompt)
    if override_alias is not None and why != "explicit [local] tag":
        if offline_now:
            return (
                DEFAULT_OFFLINE_FALLBACK,
                f"offline-downgrade ({offline_reason()}); ignored [{override_tag}] tag",
                tokens,
            )
        return (override_alias, f"explicit [{override_tag}] tag", tokens)

    if is_complex:
        if offline_now:
            return (
                DEFAULT_OFFLINE_FALLBACK,
                f"offline-downgrade ({offline_reason()}); would have been complex: {why}",
                tokens,
            )
        return ("claude-code", f"complex: {why}", tokens)
    if tokens > ROUTE_LONG_MAX:
        if offline_now:
            return (
                DEFAULT_OFFLINE_FALLBACK,
                f"offline-downgrade ({offline_reason()}); tokens {tokens} > {ROUTE_LONG_MAX}",
                tokens,
            )
        return ("claude-code", f"tokens {tokens} > {ROUTE_LONG_MAX}", tokens)
    return ("local-long", f"tokens {tokens} <= {ROUTE_LONG_MAX}", tokens)


# ----- Cline-aware routing --------------------------------------------------
#
# Why this exists:
# Cline ships a ~13.5K-token system prompt encoding its tool catalogue as
# XML, which drowns out the actual complexity of the user's task if the
# size-based `decide_tier()` classifier scans the whole raw prompt. Worse,
# the complexity classifier scans the FLAT prompt, and Cline's
# system prompt happens to contain phrases like "[local development]"
# that match our `[local]` opt-out tag -- so naive size-based routing
# would misclassify nearly every Cline turn.
#
# The Cline-aware path:
#   1. Detects Cline traffic via the existing fingerprint
#      (`_looks_like_cline`).
#   2. Extracts the user's actual task from `<task>...</task>` envelope.
#      That's literally the only text Cline-the-extension-host wraps
#      around the user's prompt.
#   3. Classifies on the extracted task ONLY -- the harness can't
#      drown the signal.
#   4. Defaults to `local-long`.
#   5. Escalates to `claude-code` on:
#        a. Explicit `[claude]` tag in the task.
#        b. Architecture / multi-file / deep-reasoning keywords.
#        c. Hairy-debugging signal in the LATEST tool result
#           (Python `Traceback`, JS stack, Rust panic, > 3 `error:`
#            lines). Big file dumps are NOT a signal -- those are
#           Cline's normal `environment_details` payload, not a real
#           failure.
#   6. Stickiness: once a task fingerprint has escalated to claude
#      this session, every subsequent turn from the same task stays
#      on claude. Prevents flapping; max one escalation per task. The
#      fingerprint is a SHA256 of the normalized `<task>...</task>`
#      text and it lives in a TTL'd in-memory dict (proxy restart
#      wipes it -- acceptable, the next first-turn re-classifies
#      anyway).

_CLINE_TASK_RE = re.compile(r"<task>\s*(.*?)\s*</task>", re.DOTALL | re.IGNORECASE)

# ---- inline per-turn model override tags ----
#
# Map a leading `[tag]` on the user's task text to a specific Claude
# alias defined in litellm-config.yaml. Lets a user say
#   [opus] refactor the auth subsystem
# and force Opus 4.7 for that single turn, regardless of what the
# router would otherwise pick.
#
# Precedence within the routing pipeline (highest first):
#   1. `[local]` -- absolute opt-out, beats EVERYTHING (cost safety)
#   2. `[haiku]` / `[sonnet]` / `[opus]` -- specific Claude model
#   3. `[claude]` -- the default Claude tier (currently Opus 4.7)
#   4. complexity heuristics
#   5. tool-result failure signal (Cline only)
#   6. local-long default
#
# Matching is LEADING-ONLY: the tag must be the first non-whitespace
# token. This avoids false positives from Cline's tool docs that
# happen to contain bracketed words mid-prompt, and matches the
# intuitive "tag your task at the start" user model documented in
# the runbook.
_MODEL_OVERRIDE_TAGS: dict[str, str] = {
    "haiku":  "claude-haiku-4-5",
    "sonnet": "claude-sonnet-5",
    "opus":   "claude-opus-4-8",
}
_OVERRIDE_TAG_RE = re.compile(
    r"^\s*\[(haiku|sonnet|opus)\]\s*",
    re.IGNORECASE,
)


def _extract_model_override(task_text: str) -> tuple[str | None, str | None]:
    """Detect a leading `[haiku]` / `[sonnet]` / `[opus]` tag.

    Returns (model_alias, tag_seen) where:
      - model_alias is the litellm-config.yaml alias the request
        should be rewritten to (e.g. "claude-opus-4-7"), or None
        if no recognized tag is present at the leading position.
      - tag_seen is the lowercase tag we matched ("opus"), used
        for the route_reason log line; None when no match.

    Whitespace before the tag is allowed; anything between the
    closing bracket and the rest of the task is consumed too. We
    intentionally do NOT mutate task_text -- the original survives
    so the task fingerprint stays stable across turns even if the
    user only added the override on turn 1.
    """
    if not task_text:
        return (None, None)
    m = _OVERRIDE_TAG_RE.match(task_text)
    if not m:
        return (None, None)
    tag = m.group(1).lower()
    return (_MODEL_OVERRIDE_TAGS[tag], tag)


# Latest-tool-result error signatures. Match a Python traceback frame,
# a JS stack-frame `at func (...)`, a Rust panic banner, or 3+ lines
# starting with `error:` / `Error:` / `ERROR:` regardless of language.
_PY_TRACEBACK_RE = re.compile(r"^\s*Traceback \(most recent call last\)", re.MULTILINE)
_JS_STACK_RE = re.compile(r"^\s+at\s+\S+\s+\(", re.MULTILINE)
_RUST_PANIC_RE = re.compile(r"thread '.+?' panicked at", re.IGNORECASE)
_GENERIC_ERROR_LINE_RE = re.compile(r"^\s*(error|Error|ERROR):\s", re.MULTILINE)

# Sticky escalation tracker.
#
# Maps task-fingerprint -> (timestamp, reason, remaining_turns).
#
# Two stickiness flavors share this dict:
#
#   - TIME-BOUNDED: timestamp is when the entry was created;
#     remaining_turns is _UNBOUNDED. Entry persists until it hits
#     _STICKY_TTL_SECONDS and is then evicted on read. Used for
#     explicit user tags ([claude]/[opus]/etc.) and complexity-keyword
#     escalations -- both are strong signals where the user has
#     either explicitly accepted Claude pricing or the task content
#     itself looks Claude-shaped. Re-evaluating those mid-task would
#     just flap.
#
#   - TURN-BOUNDED: timestamp is still set so the entry can also
#     time-expire as a safety net, but the primary expiry is
#     remaining_turns -- each sticky-driven escalation decrements
#     it by one, and when it hits zero the entry is evicted on
#     read. Used for tool-result failure escalations, where the
#     signal is heuristic ("we saw a stack trace in the latest
#     tool response") and we want the proxy to re-evaluate after
#     a short window. Without this, a single transient panic in
#     a tool result would lock a task to Claude for 30 minutes
#     even after the local model could have recovered.
#
# Memory bound: ~40 tasks/session * ~96 bytes/entry ~= 4 KB.
# Not worth tuning.
_STICKY_TTL_SECONDS = 30 * 60  # 30 min: covers a typical Cline session
_TOOL_RESULT_STICKY_TURNS = 3  # tool-result escalations stick for this many follow-up turns
_UNBOUNDED = -1  # sentinel: no per-turn budget; expires only on TTL
_sticky_escalations: dict[str, tuple[float, str, int]] = {}


def _extract_user_task(messages: Iterable[dict[str, Any]] | None) -> str | None:
    """For Cline-shaped requests, return the user's task text.

    Legacy Cline builds wrap the task in a `<task>...</task>` envelope;
    the FIRST envelope found in any user message wins. Newer builds
    (native tool-calling protocol, observed v4.x) drop the envelope and
    send the task as the first user message's plain text -- fall back to
    that, minus any trailing `<environment_details>` block so the task
    fingerprint stays stable across turns of the same task (Cline
    refreshes environment_details every turn).

    Returns None when no user text is found -- caller falls back to
    legacy size-based routing.
    """
    if not messages:
        return None
    first_user_text: str | None = None
    for m in messages:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(
                p.get("text", "") for p in c if isinstance(p, dict)
            )
        if not isinstance(c, str):
            continue
        match = _CLINE_TASK_RE.search(c)
        if match:
            return match.group(1).strip()
        if first_user_text is None:
            first_user_text = c
    if first_user_text is None:
        return None
    # No-envelope protocol: the first user message IS the task.
    task = first_user_text.split("<environment_details>", 1)[0].strip()
    return task or None


def _latest_tool_result_text(messages: Iterable[dict[str, Any]] | None) -> str:
    """Return the content of the trailing user message (Cline's
    tool-result envelope), or empty string if none. We deliberately
    look only at the LATEST message, not the whole history, so that
    a single early failure doesn't keep escalating turns later in
    the same task.
    """
    if not messages:
        return ""
    msgs = list(messages)
    if not msgs:
        return ""
    last = msgs[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return ""
    c = last.get("content")
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c if isinstance(c, str) else ""


def _looks_like_failure(text: str) -> tuple[bool, str]:
    """Return (is_failure, reason).

    Conservative on purpose. We want to distinguish "the local model
    just produced a broken artefact" (worth escalating) from "the
    project happens to surface a stack trace in passing" (NOT worth
    escalating to Claude because qwen3-coder-next can absolutely keep
    handling the task structurally -- the tool result just contains
    one piece of error-shaped text).

    Requires either:
      - >=2 instances within a single high-confidence category
        (>=2 distinct rust panic banners OR >=2 distinct python
        traceback frames OR >=3 JS stack frames)
      - OR >=2 distinct error CATEGORIES at the same time
        (e.g. a panic AND >=3 error: lines, or a python traceback
        AND a JS stack hit)

    A single panic / single traceback / single Error: line is no
    longer enough by itself. This change was made after observing
    a 35K-token architectural-overview task escalate to Claude (and
    the rest of the task get stuck on Claude via stickiness) when
    Cline's tool result merely contained one Rust panic that Qwen3
    could have handled.
    """
    if not text:
        return (False, "")

    panic_hits  = len(_RUST_PANIC_RE.findall(text))
    py_hits     = len(_PY_TRACEBACK_RE.findall(text))
    js_hits     = len(_JS_STACK_RE.findall(text))
    err_hits    = len(_GENERIC_ERROR_LINE_RE.findall(text))

    # Categories present (at any non-zero count). Used to decide
    # whether several distinct error sources reinforce each other.
    categories: list[str] = []
    if panic_hits:
        categories.append(f"rust panic x{panic_hits}")
    if py_hits:
        categories.append(f"python traceback x{py_hits}")
    if js_hits:
        categories.append(f"js stack x{js_hits}")
    # The error: line counter is noisy on its own (a single grep -i
    # 'error:' against many codebases finds matches), so we require
    # >=3 of them to even register the category. This is the same
    # threshold the previous version used to count it as "real".
    if err_hits >= 3:
        categories.append(f"{err_hits} error: lines")

    # Strong single-category signals (high confidence): repeated
    # frames within one category usually means a real meltdown.
    if panic_hits >= 2:
        return (True, f"rust panic x{panic_hits} in tool result")
    if py_hits >= 2:
        return (True, f"python traceback x{py_hits} in tool result")
    if js_hits >= 3:
        return (True, f"js stack ({js_hits} frames) in tool result")

    # Cross-category corroboration: two different signals in one
    # tool result means the failure is unlikely to be a stray.
    if len(categories) >= 2:
        return (True, "multiple error signals in tool result: " + ", ".join(categories))

    # Single weak signal -- decline to escalate. The local model
    # gets another turn.
    return (False, "")


def _task_fingerprint(task: str) -> str:
    """SHA256 of the normalized task text. Whitespace-collapse so that
    Cline's idiosyncratic indentation in the <task> envelope doesn't
    cause false-misses on the second turn."""
    norm = " ".join(task.split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def _check_sticky(fingerprint: str) -> str | None:
    """If the given task is already on the sticky-escalated list and
    the entry is still fresh (TTL not expired AND turn budget, if
    any, not depleted), return the original reason; else None.

    Has the side effect of decrementing the turn budget by 1 each
    time it returns a non-None reason for a turn-bounded entry --
    i.e. each sticky-driven escalation costs one turn from the
    budget. When the budget hits zero we evict the entry so the
    next call re-evaluates from scratch.

    Opportunistically evicts TTL-expired entries on each read.
    """
    now = time.time()
    expired = [
        k for k, (ts, _, _) in _sticky_escalations.items()
        if now - ts > _STICKY_TTL_SECONDS
    ]
    for k in expired:
        _sticky_escalations.pop(k, None)

    entry = _sticky_escalations.get(fingerprint)
    if entry is None:
        return None

    ts, reason, remaining = entry
    if remaining == _UNBOUNDED:
        # Time-bounded entry: just return the reason; TTL handles
        # cleanup separately above.
        return reason

    # Turn-bounded: this read consumes one turn. We return the
    # reason for the CURRENT turn (so the caller can route to
    # Claude this time), then decrement -- if that takes us to
    # zero, the entry is evicted so the NEXT turn re-evaluates.
    if remaining <= 1:
        _sticky_escalations.pop(fingerprint, None)
    else:
        _sticky_escalations[fingerprint] = (ts, reason, remaining - 1)
    return reason


def _mark_sticky(fingerprint: str, reason: str) -> None:
    """Time-bounded sticky entry. Persists until _STICKY_TTL_SECONDS.

    Use for high-confidence signals where the user has either
    opted in (explicit `[claude]`/`[opus]`/etc. tags) or the task
    content itself looks Claude-shaped (architecture / multi-file
    keywords). Re-evaluating those mid-task would flap.
    """
    _sticky_escalations[fingerprint] = (time.time(), reason, _UNBOUNDED)


def _mark_sticky_turns(fingerprint: str, reason: str, turns: int) -> None:
    """Turn-bounded sticky entry. Stays sticky for at most `turns`
    subsequent escalation reads, after which the entry self-evicts
    and routing re-evaluates.

    Use for heuristic signals (e.g. tool-result error patterns)
    where we want to give Claude a short rescue window but NOT
    pay Claude pricing for a half-hour just because one tool
    response contained a stack trace.
    """
    _sticky_escalations[fingerprint] = (time.time(), reason, max(1, turns))


def _try_turbo_escalation(
    token_count: int,
    fingerprint: str,
    reason: str,
) -> tuple[str, str] | None:
    """Attempt to route a context-saturated request to a local turbo backend
    instead of claude-code.

    Algorithm:
      1. Call context_compression.compress() on the current messages to obtain
         a compressed token estimate.
      2. Ask backend_registry.pick_turbo_backend(compressed_tokens) for a
         suitable turbo model ("local-turbo-256k" or "local-turbo-512k").
      3. If a backend is available, return (model, updated_reason).
         Otherwise return None so the caller falls through to claude-code.

    Only called when TURBO_ENABLED=1. Any import or runtime error is caught
    and logged; the function returns None so the original escalation path
    is preserved.

    The `token_count` argument is the FULL (pre-compression) request size.
    `fingerprint` is the task fingerprint used for sticky tracking.
    `reason` is the saturation reason string built by the caller; we annotate
    it with the compression outcome before returning.
    """
    try:
        from router.context_compression import compress  # noqa: PLC0415
        from router.backend_registry import pick_turbo_backend  # noqa: PLC0415

        compressed_tokens = compress(token_count)
        model = pick_turbo_backend(compressed_tokens)
        if model is None:
            return None
        turbo_reason = (
            f"{reason}; turbo: compressed {token_count}->{compressed_tokens} tok, "
            f"routed to {model}"
        )
        return (model, turbo_reason)
    except Exception as exc:  # pragma: no cover -- guard against missing modules
        print(
            f"[router] _try_turbo_escalation failed ({type(exc).__name__}: {exc}); "
            "falling back to claude-code",
            file=sys.stderr,
        )
        return None


def decide_tier_cline(
    messages: Iterable[dict[str, Any]] | None,
) -> tuple[str, str, int]:
    """Cline-aware tier decision. Returns (model, reason, task_tokens).

    The third element is an APPROXIMATE token count of the user's
    extracted task -- NOT the full request -- because routing is by
    task complexity, not harness size. Useful for cost-attribution
    in the request log: "this turn billed against a 12-token task".
    """
    task = _extract_user_task(messages)
    if task is None:
        # Should not happen -- caller already verified _looks_like_cline.
        # Fall back safely.
        return decide_tier(messages)

    task_tokens = max(1, len(task) // 4)
    fingerprint = _task_fingerprint(task)
    offline_now = is_offline()

    sticky_reason = _check_sticky(fingerprint)
    if sticky_reason is not None:
        if offline_now:
            # Sticky entry would have sent us to Claude, but we're
            # offline now. Don't evict the entry -- it's still
            # informative for when the network returns -- just
            # downgrade THIS turn.
            return (
                DEFAULT_OFFLINE_FALLBACK,
                f"cline+offline-downgrade({fingerprint}): {offline_reason()}; "
                f"would have been: {sticky_reason}",
                task_tokens,
            )
        return (
            "claude-code",
            f"cline+sticky({fingerprint}): {sticky_reason}",
            task_tokens,
        )

    is_complex, why = classify(task)
    # `[local]` opt-out wins over EVERYTHING, including tool-result
    # failure detection. If the user wrote `[local]`, they want a
    # local model even on hard tasks or repeated failures -- they're
    # deliberately exercising the local stack and don't want the
    # proxy quietly escalating spend behind their back.
    if why == "explicit [local] tag":
        return (
            "local-long",
            "cline+override: explicit [local] tag",
            task_tokens,
        )
    # Per-tier Claude override: `[haiku]` / `[sonnet]` / `[opus]`.
    # Sits below [local] (cost-safety wins) but above the complexity
    # classifier and the [claude] default-Claude tag, so the user
    # can force a specific model regardless of how trivial or
    # complex the task looks. Marked sticky so subsequent turns of
    # the same task stay on the same model (otherwise turn 2 could
    # downgrade to local-long once the [opus] tag is no longer in
    # the task envelope).
    override_alias, override_tag = _extract_model_override(task)
    if override_alias is not None:
        reason = f"explicit [{override_tag}] tag"
        if offline_now:
            # Don't mark sticky -- when we come back online we want
            # the [opus] tag to take effect properly. The user
            # explicitly asked for Claude; this turn just can't
            # honor it.
            return (
                DEFAULT_OFFLINE_FALLBACK,
                f"cline+offline-downgrade({fingerprint}): {offline_reason()}; "
                f"ignored {reason}",
                task_tokens,
            )
        _mark_sticky(fingerprint, reason)
        return (
            override_alias,
            f"cline+override({fingerprint}): {reason}",
            task_tokens,
        )
    if is_complex:
        if offline_now:
            return (
                DEFAULT_OFFLINE_FALLBACK,
                f"cline+offline-downgrade({fingerprint}): {offline_reason()}; "
                f"would have been complex: {why}",
                task_tokens,
            )
        _mark_sticky(fingerprint, why)
        return ("claude-code", f"cline+task({fingerprint}): {why}", task_tokens)

    # Context saturation check. Cline accumulates the full message
    # history (system prompt + every tool result + every assistant
    # turn) and resends it on each call, so a long task hits the
    # local-long 131K ceiling well before the user notices.
    #
    # When the FULL request size approaches the local-long ceiling we
    # escalate to claude-code (200K window) rather than letting the
    # local stack truncate mid-turn. Sticky-marked so subsequent
    # turns of the same long-running task stay on Claude instead of
    # bouncing back and forth between truncated local-long calls and
    # full-context claude-code calls.
    #
    # Threshold rationale:
    #   - Default 0.85 of ROUTE_LONG_MAX (default 128K), so ~108800.
    #   - Local-long's actual num_ctx is 131K but model quality drops
    #     visibly past ~70% (the ContextManager truncation threshold
    #     on the Cline side, separately).
    #   - Override with ROUTER_SATURATION_FRACTION env var.
    full_request_tokens = _estimate_tokens(messages)
    saturation_fraction = float(
        os.environ.get("ROUTER_SATURATION_FRACTION", "0.85")
    )
    saturation_limit = int(ROUTE_LONG_MAX * saturation_fraction)
    if full_request_tokens > saturation_limit:
        sat_reason = (
            f"saturation: {full_request_tokens} > {saturation_limit} "
            f"({int(saturation_fraction * 100)}% of {ROUTE_LONG_MAX})"
        )
        # Lifecycle-aware escalation: when TURBO_ENABLED=1, attempt context
        # compression and route to a local turbo backend BEFORE paying Claude
        # pricing. The sticky entry uses the original sat_reason so a
        # subsequent turn that also saturates gets the same treatment without
        # re-running compression. When turbo is disabled or unavailable the
        # block is a no-op and the original claude-code path is taken.
        if TURBO_ENABLED:
            turbo = _try_turbo_escalation(full_request_tokens, fingerprint, sat_reason)
            if turbo is not None:
                turbo_model, turbo_reason = turbo
                _mark_sticky(fingerprint, sat_reason)
                return (
                    turbo_model,
                    f"cline+saturation({fingerprint}): {turbo_reason}",
                    task_tokens,
                )
        _mark_sticky(fingerprint, sat_reason)
        return (
            "claude-code",
            f"cline+saturation({fingerprint}): {sat_reason}",
            task_tokens,
        )

    # Tool-result-driven escalation. Only fires on turns 2+ since
    # turn-1 has no tool result yet. The trigger now requires
    # corroborating signals (see `_looks_like_failure`) so a single
    # incidental panic doesn't move the whole task to Claude.
    #
    # Stickiness is TURN-BOUNDED at _TOOL_RESULT_STICKY_TURNS (3)
    # rather than time-bounded: after a 3-turn rescue window the
    # entry expires and routing re-evaluates. This avoids the
    # failure mode where one bad tool result locks 30 minutes of
    # the same task to Claude even after the local model could
    # have recovered.
    msg_count = len(list(messages or []))
    if msg_count >= 4:  # system + task + asst + tool_result(s)
        tool_text = _latest_tool_result_text(messages)
        is_failure, fail_reason = _looks_like_failure(tool_text)
        if is_failure:
            if offline_now:
                # Tool-result failure rescue requires Claude; we
                # can't do that offline. Stay on local-long and
                # log the would-have-been reason for the audit
                # trail.
                return (
                    DEFAULT_OFFLINE_FALLBACK,
                    f"cline+offline-downgrade({fingerprint}): {offline_reason()}; "
                    f"would have been: {fail_reason}",
                    task_tokens,
                )
            _mark_sticky_turns(
                fingerprint, fail_reason, _TOOL_RESULT_STICKY_TURNS,
            )
            return (
                "claude-code",
                f"cline+turn({fingerprint}): {fail_reason}",
                task_tokens,
            )

    # Default: local-long.
    return ("local-long", f"cline+default: task={task_tokens} tok", task_tokens)


# ----- In-flight request registry -------------------------------------------
#
# We keep a tiny in-memory dict of currently-running requests on the
# SizeBasedRouter instance, populated in async_pre_call_hook and drained
# in the success/failure hooks. The dashboard runs in a *separate*
# process from LiteLLM, so we mirror the dict to a file (.logs/active.json)
# after every mutation; the dashboard reads that file. This keeps the
# bridge IPC-free and matches the existing pattern of cost.db being the
# single shared source of truth across processes.
ACTIVE_PATH = REPO_ROOT / ".logs" / "active.json"

# Anything older than this is treated as a leaked / abandoned call and
# swept on the next snapshot. Empirically LiteLLM always fires either a
# success or failure hook, but a crashed worker would leave entries
# behind without this guardrail.
ACTIVE_TTL_SEC = 600


def _model_to_tier(model: str) -> str:
    """Classify a (possibly upstream-shaped) model id into `claude` or
    `local-long`. Pulled out of `_record` so the pre-call registration
    can use the same logic without rewriting it. Mirrors the rules
    documented in `_record` -- if you change one, change both."""
    m_lower = model.lower()
    if "claude" in m_lower or "anthropic" in m_lower:
        return "claude"
    return "local-long"


def _is_tool_native_model(model: str) -> bool:
    """Whether the (alias- or upstream-shaped) model id reliably emits
    structured `tool_calls`.

    Claude tiers are always tool-native. Local tiers are tool-native only
    when served through Ollama's OpenAI-compat layer (`ollama_chat/`
    upstream ids); the legacy `ollama/` route bypasses tool parsing and
    returns tool calls as raw JSON text. Alias names are checked against
    _NON_TOOL_NATIVE_LOCAL, and upstream ids additionally against
    _NON_TOOL_NATIVE_MODEL_SUBSTRINGS (models that declare the tools
    capability but don't emit the envelope, e.g. qwen2.5-coder)."""
    if _model_to_tier(model) == "claude":
        return True
    m_lower = model.lower()
    if m_lower.startswith("ollama/"):
        return False
    if any(s in m_lower for s in _NON_TOOL_NATIVE_MODEL_SUBSTRINGS):
        return False
    canonical = model[len("gpt-"):] if model.startswith("gpt-") else model
    return canonical not in _NON_TOOL_NATIVE_LOCAL


def _annotate_router_fallback(
    model: str,
    route_decision: str | None,
    tools_present: bool,
    reason: str,
) -> str:
    """Append a loud marker to route_reason when LiteLLM's router-level
    fallback moved the request off the tier the pre-call hook decided.

    Router fallbacks (router_settings.fallbacks in the YAML) run AFTER
    async_pre_call_hook, which only fires once per request -- so the
    tool-native redirect cannot correct them. If a fallback lands a
    tool-carrying request on a non-tool-native local model, the model
    emits the tool call as raw JSON text and the harness (Cline) stalls
    showing the blob. That failure was silent in the ledger until this
    marker: the row kept the original route_reason and looked like a
    normal local turn. Pure function -- unit-testable without LiteLLM."""
    decision = (route_decision or "").strip()
    if not decision:
        return reason
    decision_tier = _model_to_tier(decision)
    served_tier = _model_to_tier(model)
    if decision_tier == served_tier:
        return reason
    # The hook decided Claude but a local model served the response (or
    # vice versa, which shouldn't happen): a router fallback fired.
    marker = f"ROUTER-FALLBACK({decision}->{model})"
    if tools_present and not _is_tool_native_model(model):
        # The exact failure mode this exists to surface.
        marker += " TOOLS-AS-TEXT-RISK"
        print(
            f"[router] WARNING: fallback landed a tool-carrying request on "
            f"non-tool-native model {model!r} (decided: {decision!r}); the "
            f"harness will see raw JSON instead of tool_calls",
            file=sys.stderr,
        )
    return f"{reason} | {marker}" if reason else marker


# ----- LiteLLM callback shim ------------------------------------------------
#
# LiteLLM looks for a class with `async_pre_call_hook` and/or `log_success_event`.
# We implement both. The class is referenced by dotted path in the YAML config.

class SizeBasedRouter(_LiteLLMCustomLogger):
    """Subclass LiteLLM's CustomLogger so all post-call/streaming/failure hooks
    are no-ops by default; we only override the two we care about
    (pre-call routing and success logging)."""

    user_api_key_cache: dict[str, Any] = {}

    def __init__(self) -> None:
        try:
            super().__init__()
        except Exception:
            pass
        self._conn = _ensure_db()
        # Currently-running requests, keyed by litellm_call_id. Mutated
        # under self._active_lock; mirrored to disk via _flush_active().
        self._active: dict[str, dict[str, Any]] = {}
        self._active_lock = threading.Lock()
        # Best-effort: clear any stale snapshot from a previous proxy
        # process so the dashboard doesn't show ghost rows on restart.
        self._flush_active()

    # ------------------ tool-native redirect -------------------------------
    def _maybe_redirect_tools_to_native(self, data: dict[str, Any]) -> None:
        """Redirect a tool-carrying request off a local tier that can't emit
        native tool_calls onto the tool-native local tier.

        Cline (and any OpenAI-native harness) only executes structured
        `tool_calls`; the qwen2.5-coder tiers emit them as raw JSON text,
        so the turn stalls. When the request carries `tools` and the
        resolved model is a non-tool-native local tier, rewrite it to
        TOOL_NATIVE_LOCAL_TIER (default local-long: qwen3-coder-next via
        ollama_chat/, tool-native since the 2026-07-02 template fix) and
        stamp the reason so the redirect is visible in the monitor / cost
        ledger.

        No-op when: the redirect is disabled, the turn carries no tools, the
        model is a Claude tier, or the local tier is already tool-native
        (e.g. local-long, local-agent). Never raises -- the caller's
        try/except already guards the whole pre-call hook."""
        if not TOOL_NATIVE_LOCAL_TIER:
            return
        tools = data.get("tools")
        if not isinstance(tools, list) or not tools:
            return
        current = data.get("model", "") or ""
        # gpt- mirror aliases are already stripped upstream, but be defensive.
        canonical = current[len("gpt-"):] if current.startswith("gpt-") else current
        if canonical not in _NON_TOOL_NATIVE_LOCAL:
            return
        if canonical == TOOL_NATIVE_LOCAL_TIER:
            return
        data["model"] = TOOL_NATIVE_LOCAL_TIER
        meta = data.setdefault("metadata", {})
        prev_reason = meta.get("route_reason") or canonical
        meta["route_decision"] = TOOL_NATIVE_LOCAL_TIER
        meta["route_reason"] = (
            f"tool-native redirect ({canonical}->{TOOL_NATIVE_LOCAL_TIER}): {prev_reason}"
        )
        meta["tool_native_redirect"] = True

    # ------------------ pre-call: rewrite hybrid-auto -> tier ---------------
    async def async_pre_call_hook(  # type: ignore[override]
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any] | None:
        try:
            # FIRST: optional full-request dump for investigations. Runs
            # before any of our routing or control mutation so the dump
            # is exactly what the client sent.
            if CLINE_TRACE:
                _dump_full_request(data)

            requested = data.get("model", "")
            # Cursor's "Verify" button rejects non-OpenAI-shaped names, so the
            # YAML config exposes mirror aliases prefixed with `gpt-`. Strip
            # that prefix here so downstream routing/decision logic sees the
            # canonical name (`hybrid-auto`, `local-long`, etc.) and behaves
            # identically regardless of which alias the client used.
            if requested.startswith("gpt-") and requested[len("gpt-"):] in (
                "hybrid-auto",
                "local-long",
                "local-agent",
                "claude-code",
                "claude-haiku-4-5",
                "claude-sonnet-5",
                "claude-opus-4-8",
            ):
                canonical = requested[len("gpt-"):]
                data["model"] = canonical
                requested = canonical

            if requested == "hybrid-auto":
                msgs = data.get("messages")
                # Cline traffic uses the task-aware path; other clients
                # (CLI, raw curl, benchmarks) keep the legacy size-based
                # logic. Detected via the same fingerprint that drives
                # over-generation control, so the two stay aligned.
                cline_mode = _looks_like_cline(msgs)
                if cline_mode:
                    model, reason, tokens = decide_tier_cline(msgs)
                    reason = f"cline-mode: {reason}"
                    # decide_tier_cline's token count is just the
                    # <task>-envelope estimate (surfaced in the reason as
                    # "task=N tok") because routing is by task complexity,
                    # not harness size. But the in-flight dashboard row and
                    # its cost estimate want the ACTUAL prompt size being
                    # processed -- otherwise a 60K-token Cline turn shows a
                    # frozen "111 (est)" for minutes, then jumps to the real
                    # number only when the turn completes. Recompute the full
                    # request estimate for display/cost; the routing decision
                    # itself already used the task estimate above.
                    tokens = _estimate_tokens(msgs)
                else:
                    model, reason, tokens = decide_tier(msgs)
                data["model"] = model
                meta = data.setdefault("metadata", {})
                meta["route_decision"] = model
                meta["route_reason"] = reason
                meta["route_tokens_estimated"] = tokens
                # For Cline traffic, stamp the task fingerprint and a
                # truncated copy of the user's task text so the success
                # callback can persist them. Both default to None for
                # non-Cline traffic, which makes downstream task-grouped
                # views naturally exclude CLI/curl callers (they aren't
                # part of an agent task).
                if cline_mode:
                    task_text = _extract_user_task(msgs)
                    if task_text is not None:
                        meta["task_id"] = _task_fingerprint(task_text)
                        # Truncate to keep `requests` rows lean; the
                        # full task text is never displayed in full,
                        # only as a preview on /tasks.
                        meta["task_text"] = task_text[:500]

            # Offline-mode chokepoint. Runs AFTER routing so it catches
            # both `hybrid-auto`-driven Claude decisions AND direct
            # claude-* model selections that never went through
            # decide_tier(). The helper mutates `data` in place: it
            # rewrites `data["model"]` to local-long and stamps
            # `metadata.offline_downgrade` for the cost ledger.
            #
            # `requested` here is the post-gpt-stripped alias the user
            # asked for ("claude-code", "claude-opus-4-7", "hybrid-auto",
            # ...). We pass it as `requested_alias` so the user-visible
            # warning reads naturally even when the routing rewrote the
            # model already.
            #
            # `explicit_claude` flips strict-mode rejection on, so a
            # user who said `claude-opus-4-7` by name on an airplane
            # gets a clear 503 instead of a silent (and possibly
            # confusing) local result.
            explicit_claude = (
                requested.startswith("claude-")
                or requested == "claude-code"
            )
            downgraded, error_msg = maybe_downgrade(
                data,
                requested_alias=requested,
                explicit_claude=explicit_claude,
            )
            if downgraded and error_msg:
                # Strict-mode rejection. Raising a sentinel subclass
                # of RuntimeError lets the outer try/except below
                # re-raise this specifically (so LiteLLM returns a
                # 503 to the client) instead of swallowing it like
                # an unexpected routing error.
                raise OfflineStrictReject(error_msg)

            # Tool-native redirect runs AFTER routing AND the offline
            # chokepoint, so it also catches turns that offline-mode just
            # downgraded to local-long. Must run BEFORE thinking-mode and
            # the over-gen controls below so they see the corrected model
            # (e.g. no /think injection for llama3.1).
            self._maybe_redirect_tools_to_native(data)

            # Over-generation controls run AFTER the hybrid-auto rewrite
            # so the controls can see the resolved model name. Both
            # strategies are no-ops for non-local models and for
            # request shapes they don't recognize, so it's safe to call
            # them unconditionally even when the user picked Claude.
            #
            # Snapshot the inbound request so we can log a delta if any
            # control fires. This is cheap (the request dict is small)
            # and only matters when the trace log is enabled.
            pre_max = data.get("max_tokens")
            pre_stop = data.get("stop")
            pre_n_msgs = len(data.get("messages") or [])
            pre_first_role = ((data.get("messages") or [{}])[0] or {}).get("role")

            # ---- Thinking mode --------------------------------------
            # Goal: stream a live reasoning trace into the harness
            # (Cline) for BOTH tiers, instead of an empty "thinking" tag.
            #
            # Local (Qwen3): inject /think so the model emits a <think>
            # trace inline; async_post_call_streaming_iterator_hook below
            # relays that as reasoning_content. Gated on the resolved
            # model actually being a Qwen3 model (env-derived) so we
            # never inject the literal "/think " into llama3.1 /
            # qwen2.5-coder, which don't have the switch.
            #
            # Claude: enable Anthropic extended thinking; LiteLLM
            # surfaces it to the client as reasoning_content deltas.
            #
            # ROUTER_THINKING is always-on by design but kept behind a
            # kill switch. The legacy ENABLE_THINK_INJECTION path (fire
            # only on the explicit [local] tag) is preserved for when
            # thinking mode is globally off.
            model_name = data.get("model")
            if ENABLE_THINKING_MODE and _model_supports_think(model_name):
                inject_qwen3_think_directive(data)
                meta = data.setdefault("metadata", {})
                meta["qwen3_think_injected"] = True
            elif ENABLE_THINK_INJECTION and _is_local_model_name(model_name):
                # Back-compat: only on the explicit [local] opt-out tag.
                route_reason = (
                    data.get("metadata", {}).get("route_reason") or ""
                )
                if "[local]" in route_reason:
                    inject_qwen3_think_directive(data)
                    meta = data.setdefault("metadata", {})
                    meta["qwen3_think_injected"] = True

            # Adaptive thinking is unsupported on Haiku tiers, so skip them
            # (injecting it 400s: Haiku 4.5 rejects thinking.type=adaptive
            # AND output_config.effort). The check must resolve aliases to
            # their upstream model -- `claude-code` currently maps to
            # claude-haiku-4-5 (subscription policy limit), so checking the
            # alias string alone would inject thinking and 400 every
            # escalated turn.
            if (
                ENABLE_THINKING_MODE
                and _is_claude_model_name(model_name)
                and "haiku" not in _resolved_claude_model(model_name).lower()
            ):
                apply_claude_thinking_params(data, effort=_thinking_effort())
                meta = data.setdefault("metadata", {})
                meta["claude_thinking_enabled"] = True

            if ENABLE_STATIC_GUARDRAIL:
                apply_static_guardrail(data)
                meta = data.setdefault("metadata", {})
                meta["overgen_static_applied"] = True
            if ENABLE_MULTI_TURN_TIGHTEN:
                pre_max_mt = data.get("max_tokens")
                apply_multi_turn_tighten(data)
                post_max_mt = data.get("max_tokens")
                if pre_max_mt != post_max_mt:
                    meta = data.setdefault("metadata", {})
                    meta["overgen_multi_turn_applied"] = True

            # The over-generation guardrails above just clamped
            # max_tokens for local turns. On a /think turn the reasoning
            # trace shares that budget with the answer, so re-assert a
            # floor AFTER the guardrails to stop the trace truncating the
            # response. (Claude's floor is set by apply_claude_thinking_params
            # and the guardrails are local-only no-ops, so this only
            # affects local think turns.)
            if data.get("metadata", {}).get("qwen3_think_injected"):
                floor = _local_think_max_tokens()
                cur = data.get("max_tokens")
                if not isinstance(cur, int) or cur < floor:
                    data["max_tokens"] = floor

            if OVERGEN_TRACE and (ENABLE_STATIC_GUARDRAIL or ENABLE_MULTI_TURN_TIGHTEN):
                _trace_overgen(
                    requested=requested,
                    resolved=data.get("model"),
                    pre_max=pre_max,
                    post_max=data.get("max_tokens"),
                    pre_stop=pre_stop,
                    post_stop=data.get("stop"),
                    pre_n_msgs=pre_n_msgs,
                    post_n_msgs=len(data.get("messages") or []),
                    pre_first_role=pre_first_role,
                    post_first_role=((data.get("messages") or [{}])[0] or {}).get("role"),
                )

            # Register this call as in-flight so the dashboard can show
            # it. Done last in pre-call so the model/tier we record is
            # the post-routing one. Failure here must not break the
            # request -- the inner method already swallows exceptions.
            self._register_active(data)
        except OfflineStrictReject:
            # Strict-mode rejection is intentional: re-raise so
            # LiteLLM returns a 503 to the client. The metadata was
            # already stamped on `data` by maybe_downgrade(), and
            # _log_event() recorded the rejection in the audit log.
            raise
        except Exception as e:  # never break user requests
            print(f"[router] pre-call hook error: {e}", file=sys.stderr)
        return data

    # ------------------ post-call: stream <think> -> reasoning_content ------
    async def async_post_call_streaming_iterator_hook(  # type: ignore[override]
        self,
        user_api_key_dict: Any,
        response: Any,
        request_data: dict,
    ):
        """Re-route Qwen3's inline `<think>...</think>` trace onto the
        `reasoning_content` delta channel so Cline renders it as a live
        reasoning section instead of dumping it into the message body.

        Only active for turns where we injected /think (flagged in
        metadata by the pre-call hook). For every other response -- Claude
        (LiteLLM already emits reasoning_content), non-think local turns,
        tool-call deltas -- chunks pass through untouched. Never raises:
        on any error we fall back to yielding the original chunk.
        """
        meta = (request_data or {}).get("metadata") or {}
        if not meta.get("qwen3_think_injected"):
            async for chunk in response:
                yield chunk
            return

        buffer = ""
        in_think = False
        async for chunk in response:
            try:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    yield chunk
                    continue
                delta = getattr(choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta is not None else None
                if delta is None or not isinstance(content, str) or content == "":
                    yield chunk
                    continue
                buffer += content
                out_content, out_reasoning, buffer, in_think = _split_think_stream(
                    buffer, in_think
                )
                delta.content = out_content or None
                if out_reasoning:
                    delta.reasoning_content = out_reasoning
            except Exception:
                pass
            yield chunk
        # Flush any held-back tail (e.g. a partial tag that never completed).
        if buffer:
            try:
                from litellm.types.utils import (  # local import: optional dep
                    Delta,
                    ModelResponseStream,
                    StreamingChoices,
                )

                delta = Delta()
                if in_think:
                    delta.reasoning_content = buffer
                else:
                    delta.content = buffer
                yield ModelResponseStream(
                    choices=[StreamingChoices(index=0, delta=delta)]
                )
            except Exception:
                pass

    # ------------------ post-call: record actual + shadow cost --------------
    def log_success_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        try:
            self._record(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            print(f"[router] log_success_event error: {e}", file=sys.stderr)

    async def async_log_success_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        try:
            self._record(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            print(f"[router] async_log_success_event error: {e}", file=sys.stderr)

    def _record(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        # Drop the in-flight registration first so the dashboard stops
        # showing this row, even if the rest of _record errors out.
        self._drop_active(kwargs)

        model = kwargs.get("model") or kwargs.get("litellm_params", {}).get("model") or "unknown"

        usage = {}
        if hasattr(response_obj, "usage") and response_obj.usage:
            usage = (
                response_obj.usage.model_dump()
                if hasattr(response_obj.usage, "model_dump")
                else dict(response_obj.usage)
            )
        elif isinstance(response_obj, dict):
            usage = response_obj.get("usage", {}) or {}

        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)

        # Determine "tier": claude vs local-long. The `model` string
        # LiteLLM hands us depends on where in the lifecycle we are:
        #   - From async_pre_call_hook we already rewrote `data["model"]`
        #     to the alias (local-long / claude-code), and LiteLLM
        #     sometimes echoes that back as `model` to the callback.
        #   - For direct alias calls (no hybrid-auto) the model can be the
        #     alias itself.
        #   - In other paths LiteLLM passes the upstream id (ollama/...,
        #     anthropic/claude-...).
        # The shared `_model_to_tier` helper handles all of these and is
        # also used by the in-flight registration path.
        tier = _model_to_tier(model)

        # Actual cost: only Claude calls cost real money. We look up the
        # rate by the actual model id LiteLLM hands us (e.g.
        # 'anthropic/claude-opus-4-7') so Opus, Haiku, and Sonnet are
        # priced correctly. Unknown models fall back to Sonnet rates
        # with a one-time stderr warning -- see cost.pricing.claude_rate.
        actual_cost = 0.0
        if tier == "claude":
            actual_cost = actual_claude_cost(model, in_tok, out_tok)

        # Shadow cost: pinned to Sonnet 4.6 so savings comparisons stay
        # apples-to-apples across time, even if the router starts using
        # Haiku or Opus for some requests. See cost.pricing.shadow_cost.
        shadow_cost = _shadow_cost_fn(in_tok, out_tok)

        try:
            latency_ms = int((float(end_time) - float(start_time)) * 1000)
        except Exception:
            try:
                latency_ms = int((end_time - start_time).total_seconds() * 1000)
            except Exception:
                latency_ms = 0

        # LiteLLM relocates the metadata we set in async_pre_call_hook
        # into kwargs["litellm_params"]["metadata"] by the time it
        # reaches the success callback. The top-level kwargs["metadata"]
        # is also valid in some lifecycle paths, so we check both --
        # litellm_params first because that's where Cline-aware routing
        # decisions actually land.
        litellm_params = kwargs.get("litellm_params") or {}
        nested_meta = litellm_params.get("metadata") or {}
        top_meta = kwargs.get("metadata") or {}
        reason = (
            nested_meta.get("route_reason")
            or top_meta.get("route_reason")
            or ""
        )
        # Surface router-level fallbacks in the ledger. The pre-call hook
        # stamped route_decision with the tier it chose; if the model that
        # actually served the response maps to a different tier, LiteLLM's
        # fallback chain fired (e.g. claude-code 429 -> local fallback) --
        # and if the request carried tools while the fallback target can't
        # emit structured tool_calls, that's the tools-as-text stall.
        # `tools` lives in different kwargs slots depending on the LiteLLM
        # lifecycle path, so check each known location.
        route_decision = nested_meta.get("route_decision") or top_meta.get(
            "route_decision"
        )
        optional_params = kwargs.get("optional_params") or {}
        tools_present = bool(
            kwargs.get("tools")
            or optional_params.get("tools")
            or (litellm_params.get("optional_params") or {}).get("tools")
        )
        reason = _annotate_router_fallback(
            model, route_decision, tools_present, reason
        )
        # task_id / task_text are only set for Cline traffic; both
        # NULL otherwise. SQLite stores NULL natively for None bindings.
        task_id = nested_meta.get("task_id") or top_meta.get("task_id")
        task_text = nested_meta.get("task_text") or top_meta.get("task_text")

        self._conn.execute(
            """
            INSERT INTO requests
              (ts, model, tier, input_tok, output_tok, actual_cost, shadow_cost,
               latency_ms, route_reason, task_id, task_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(time.time()),
                model,
                tier,
                in_tok,
                out_tok,
                actual_cost,
                shadow_cost,
                latency_ms,
                reason,
                task_id,
                task_text,
            ),
        )
        self._conn.commit()

    # ------------------ failure: drain in-flight registration ---------------
    def log_failure_event(  # type: ignore[override]
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Sync failure hook. Drops the in-flight entry so a 4xx/5xx
        from upstream doesn't leak into the dashboard's Active list.
        We don't write a row to `requests` for failures -- that's
        consistent with prior behavior (only successes are billed)."""
        try:
            self._drop_active(kwargs)
        except Exception as e:
            print(f"[router] log_failure_event error: {e}", file=sys.stderr)

    async def async_log_failure_event(  # type: ignore[override]
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        try:
            self._drop_active(kwargs)
        except Exception as e:
            print(f"[router] async_log_failure_event error: {e}", file=sys.stderr)

    # ------------------ in-flight registry helpers --------------------------
    @staticmethod
    def _extract_call_id(source: dict[str, Any]) -> str | None:
        """Find the LiteLLM call id in any of the locations LiteLLM uses
        across its lifecycle. None if the request predates call-id
        plumbing (older LiteLLM) -- in that case we generate a synthetic
        one in `_register_active` so the row still shows up."""
        cid = source.get("litellm_call_id")
        if cid:
            return cid
        meta = source.get("metadata") or {}
        cid = meta.get("litellm_call_id")
        if cid:
            return cid
        params = source.get("litellm_params") or {}
        cid = params.get("litellm_call_id")
        if cid:
            return cid
        nested_meta = params.get("metadata") or {}
        return nested_meta.get("litellm_call_id")

    def _register_active(self, data: dict[str, Any]) -> None:
        """Add an entry to the in-flight registry. Called at the end of
        the pre-call hook, after routing + over-gen control have run, so
        the model/tier we record is the post-routing one.

        Never raises -- a failure here must not break a user request."""
        try:
            cid = self._extract_call_id(data) or f"anon-{time.time_ns()}"
            meta = data.get("metadata") or {}
            model = data.get("model", "")
            entry = {
                "started": time.time(),
                "model": model,
                "tier": _model_to_tier(model),
                "in_tok_est": int(meta.get("route_tokens_estimated", 0) or 0),
                "task_id": meta.get("task_id"),
                "task_text_short": (meta.get("task_text") or "")[:80],
                "route_reason": meta.get("route_reason", ""),
            }
            with self._active_lock:
                self._active[cid] = entry
            self._flush_active()
        except Exception as e:
            print(f"[router] _register_active error: {e}", file=sys.stderr)

    def _drop_active(self, kwargs: dict[str, Any]) -> None:
        """Remove an entry from the in-flight registry. Called from
        success and failure hooks. Tolerates missing call-ids and
        already-dropped entries silently."""
        try:
            cid = self._extract_call_id(kwargs)
            if not cid:
                return
            with self._active_lock:
                self._active.pop(cid, None)
            self._flush_active()
        except Exception as e:
            print(f"[router] _drop_active error: {e}", file=sys.stderr)

    def snapshot_active(self) -> list[dict[str, Any]]:
        """Return a stable list of currently-running calls. Sweeps any
        entry older than ACTIVE_TTL_SEC -- defensive in case LiteLLM
        ever drops a call without firing either hook (a crashed worker,
        for instance)."""
        now = time.time()
        with self._active_lock:
            stale = [
                cid for cid, v in self._active.items()
                if now - v.get("started", now) > ACTIVE_TTL_SEC
            ]
            for cid in stale:
                self._active.pop(cid, None)
            out = [
                {**v, "call_id": cid, "elapsed_sec": now - v.get("started", now)}
                for cid, v in self._active.items()
            ]
        if stale:
            # Persist the post-sweep state so the dashboard reflects it
            # on the very next poll.
            self._flush_active()
        return out

    def _flush_active(self) -> None:
        """Mirror the in-memory dict to ACTIVE_PATH so the dashboard
        process (separate launchd plist) can read it. Atomic via
        write-rename so a concurrent reader never sees a half-written
        file. Never raises."""
        try:
            ACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with self._active_lock:
                payload = [
                    {**v, "call_id": cid}
                    for cid, v in self._active.items()
                ]
            tmp = ACTIVE_PATH.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload))
            os.replace(tmp, ACTIVE_PATH)
        except Exception as e:
            print(f"[router] active flush failed: {e}", file=sys.stderr)


# Module-level instance so LiteLLM can import either the class or this object.
proxy_handler_instance = SizeBasedRouter()

# Backend status endpoints (/backend/status etc.) are mounted in
# scripts/run_litellm.py via app.add_api_route() — done there because
# LiteLLM wraps include_router() in _IncludedRouter and the routes don't
# survive the proxy middleware chain when added at import time here.


if __name__ == "__main__":
    # Tiny self-test.
    msgs = [{"role": "user", "content": "Refactor this function across multiple files please."}]
    print(decide_tier(msgs))
    msgs2 = [{"role": "user", "content": "x = 1\n" * 5000}]
    print(decide_tier(msgs2))
