"""Server-side over-generation controls for the local models.

The local Qwen3-Coder-Next model has a documented behavioral quirk on the
quotas_feature benchmark: when asked to fix a small bug it tends to
re-emit the entire affected feature rather than a minimal patch. That
inflates wall time (more decode tokens) and occasionally causes the
model to run into its `max_tokens` ceiling mid-rewrite.

Cursor talks to LiteLLM at /v1/chat/completions exactly like an OpenAI
client would. We don't want to ask Cursor to change its behavior --
we want every request to local-long to be quietly tightened on the
proxy side. This module produces the patches.

Two strategies, each pure-functional and testable in isolation:

A. Static guardrail (`apply_static_guardrail`):
   - Cap `max_tokens` at a sane code-generation ceiling.
   - Add a stop sequence that catches the "second file fence after a
     completed first one" pattern.
   - Prepend a short system-message nudge unless the user already has
     one.
   Applied to every local-* request unconditionally.

B. Multi-turn-aware tightening (`apply_multi_turn_tighten`):
   - Detect a fix-up turn (multi-message conversation with a prior
     assistant message that already contains code).
   - On those, clamp `max_tokens` harder and append a one-line
     reminder to emit a minimal patch.
   No-ops for single-turn requests.

Both strategies are idempotent and never raise -- a control that
crashes the request path is worse than no control at all.
"""

from __future__ import annotations

import os
import re
from typing import Any, Iterable

# --- Tunables ----------------------------------------------------------
# Code generation rarely needs more than ~6k output tokens for a single
# file. Leaving headroom for legitimate larger responses, we still cap
# well under the 16k Ollama default to interrupt runaway rewrites.
LOCAL_MAX_TOKENS_DEFAULT = 6144

# When we detect a fix-up turn, we expect a focused patch. Anything
# beyond ~3k tokens of output on a fix-up is almost certainly the model
# re-rewriting an unrelated file. This is the bound we observed Claude
# operating under (~2k tokens per fix-up) plus a margin.
LOCAL_MAX_TOKENS_FIXUP = 3072

# Cline's Plan mode produces a single `<plan_mode_respond>` block of
# conversational text. There is no code to emit, no file edit, no
# tool chain. A well-formed plan response is typically 200-800 tokens.
# Cap at 1024 to leave a small margin while preventing the model from
# padding the response with implementation code it isn't supposed to
# write yet.
LOCAL_MAX_TOKENS_PLAN = 1024

# The "you have already started a second file fence" stop pattern. The
# grader format we use is ```python:<filename>\n...\n```. After the
# first complete file, a second `\n```python:` is the model deciding to
# emit another file. Stopping there forces it to either (a) be done or
# (b) explicitly justify the second file in another turn. Servers
# usually accept up to 4 stop sequences.
LOCAL_STOP_SEQUENCES = ["\n```python:", "\n```py:"]

# Cline-specific stop sequences. Cline encodes its tool catalogue as
# XML inside the system prompt and expects exactly ONE tool call per
# assistant turn. Without these stops, even Qwen3-Coder-Next will
# happily continue past `</replace_in_file>` and hallucinate the rest
# of a multi-turn conversation (`### User:`, fake tool results, fake
# <attempt_completion>) inside a single response. Cline parses only
# the FIRST tool tag, so the hallucinated continuation either confuses
# the harness or wastes tokens and wall time.
#
# OpenAI/LiteLLM/Ollama accept up to 4 stop sequences reliably, so we
# pick the four highest-value Cline tool-close tags. Ordering reflects
# the cost of an over-generation past each tag:
#
#   1. </attempt_completion> -- the task-end signal. Anything generated
#      AFTER this is hallucinated next-turn content (fake user message,
#      fake tool result, fake next assistant). This is by far the
#      worst-case over-generation: it inflates output tokens 3-5x and
#      pollutes Cline's parser. Always stop here.
#
#   2. </replace_in_file> -- the most-frequent edit tool. Cline parses
#      only the FIRST tool tag; anything after the closing </replace>
#      is wasted decode budget.
#
#   3. </write_to_file> -- bulk file write. Same parser dynamics as
#      replace_in_file; over-generation past the close tag is purely
#      wasted output.
#
#   4. </execute_command> -- shell command tool. Over-generation past
#      this tag has a worse outcome than </read_file>: a runaway
#      execute_command can suggest follow-up shell commands the model
#      did not actually invoke, which can mislead a user reading the
#      assistant message. read_file over-generation is harmless prose;
#      execute_command over-generation looks authoritative and isn't.
#
# Dropped from the previous set: </read_file>. Cline rarely chains a
# second read_file inside a single response, the over-generation
# pattern is benign (just verbose), and the cap is hard at 4 entries.
CLINE_STOP_SEQUENCES = [
    "</attempt_completion>",
    "</replace_in_file>",
    "</write_to_file>",
    "</execute_command>",
]

LOCAL_SYSTEM_NUDGE = (
    "You are running on a local Apple-Silicon model with a hard output "
    "budget. When asked to fix or modify code, emit ONLY the minimal "
    "set of changed files. If a file is already correct, do not "
    "re-emit it. Prefer a single file in the response unless the task "
    "genuinely requires multiple."
)

LOCAL_FIXUP_NUDGE = (
    "[Fix-up turn] Re-emit at most ONE file -- the file containing the "
    "actual bug. Do not include unchanged files. Stop generation as "
    "soon as the fix is complete."
)

# Qwen3-Coder-Next supports a runtime "thinking mode" switch via the
# /think and /no_think pseudo-directives placed at the start of the
# user message. When enabled, the model emits an internal reasoning
# trace (visible to the harness but not the user) before its final
# answer, at the cost of higher latency and ~10-30% more output
# tokens.
#
# Why this constant exists: the router escalates *complex* tasks to
# claude-code by default, so the local model normally doesn't have to
# handle reasoning-heavy work. But when a user explicitly tags a
# complex task `[local]` (opting out of cloud escalation), or when
# context saturation forces a normally-claude task back to local
# because of explicit overrides, the local model benefits from the
# extra reasoning budget. We inject /think to give Qwen3 a fair shot.
QWEN3_THINK_PREFIX = "/think "

# Substring that identifies a model family that actually understands the
# Qwen3 /think runtime switch. We check the *resolved* upstream model id
# (or the env that the tier alias maps to) rather than the tier alias
# itself: `local-long` is only Qwen3 because detected.env currently
# points it at Qwen3-Coder-Next, and that can drift. If we gated on the
# alias and OLLAMA_TAG got repointed to a non-Qwen3 model we'd silently
# inject the literal string "/think " as junk into a model that has no
# thinking switch. Gating on the real model id makes "use the right
# model for thinking" something the code enforces.
_THINK_CAPABLE_SIGNATURE = "qwen3"


def _model_supports_think(model: str | None) -> bool:
    """True only if the resolved local tier is backed by a Qwen3 model.

    `local-long` resolves (via LiteLLM config) to the Ollama tag in
    OLLAMA_TAG, which must carry the Qwen3 signature for /think to do
    anything. `local-agent` (llama3.1) and `local-coder-*`
    (qwen2.5-coder) never support the switch.
    """
    if not isinstance(model, str) or not model:
        return False
    m = model.lower()
    if "claude" in m or "anthropic" in m:
        return False
    canonical = m[len("gpt-"):] if m.startswith("gpt-") else m
    # Tiers we know are NOT Qwen3 regardless of env.
    if canonical == "local-agent" or canonical.startswith("local-coder-"):
        return False
    if canonical == "local-long":
        return _THINK_CAPABLE_SIGNATURE in os.environ.get("OLLAMA_TAG", "").lower()
    # Upstream-shaped ids passed through verbatim (e.g. "ollama/qwen3-...").
    return _THINK_CAPABLE_SIGNATURE in canonical


# Claude adaptive-thinking defaults. Anthropic streams a `thinking`
# content block (surfaced by LiteLLM as `delta.reasoning_content`) only
# when the request asks for extended thinking. Current models (Opus 4.6+,
# Sonnet 4.6+, and Opus 4.7/4.8 where it is the ONLY mode) use
# `thinking={"type":"adaptive"}` plus an optional `output_config.effort`
# knob; the legacy `{"type":"enabled","budget_tokens":N}` form returns a
# 400 on Opus 4.7+. Thinking is incompatible with custom temperature/top_k,
# so we pin temperature=1 and drop top_p/top_k. effort defaults to "high"
# (Anthropic's own default -- "almost always thinks") so Cline reliably
# sees a reasoning trace. max_tokens is a hard cap over thinking + answer,
# so we floor it to keep a tiny inbound cap from starving the trace.
CLAUDE_THINKING_EFFORT_DEFAULT = "high"
_CLAUDE_THINKING_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})
CLAUDE_THINKING_MAX_TOKENS_FLOOR = 8192

# Models we treat as "local" for the purposes of these controls.
# Includes both the alias names and common upstream id prefixes that
# LiteLLM might pass through. Claude-anything is excluded.
#
# `local-agent` (and the `ollama_chat/` prefix it uses) is intentionally
# included: even though the agent model emits structured tool_calls
# (where output is bounded by the tool argument shape), we still want
# the static guardrail to keep a sane max_tokens cap and the stop
# sequence in case the model falls back to prose+code in chat mode.
LOCAL_MODEL_TOKENS = (
    "local-long",
    "local-agent",
    "ollama/",
    "ollama_chat/",
    "openai/mlx-",
    "mlx-community/",
    "qwen3-coder-next",
    "llama3.1",
)


def _is_local(model: str | None) -> bool:
    if not model:
        return False
    m = model.lower()
    if "claude" in m or "anthropic" in m:
        return False
    return any(tok.lower() in m for tok in LOCAL_MODEL_TOKENS)


# Substrings that uniquely identify a Cline-shaped request. We look at
# the system prompt because Cline's harness embeds its full XML tool
# catalogue there and uses a stable, distinctive opening line. Matching
# on the system prompt lets us turn on Cline-specific behavior
# regardless of which model alias the user picked, and without false
# positives on benchmark traffic that just happens to use the same
# model.
_CLINE_SYSTEM_FINGERPRINTS = (
    "You are Cline,",
    "<replace_in_file>",
    "<attempt_completion>",
)

# Cline's Plan mode is detected via the system-prompt mention of the
# plan_mode_respond tool. When Cline is in Plan mode it constrains
# itself to only emit a `<plan_mode_respond>` block; when in Act mode
# it can use any tool. The plan_mode_respond tool description only
# appears in the system prompt when Plan mode is active, so a single
# substring match is a reliable detector.
_CLINE_PLAN_MODE_FINGERPRINT = "plan_mode_respond"


def _looks_like_cline(messages: Any) -> bool:
    """Return True if the request shape matches Cline's harness.

    Cheap O(1) string scan over the first system-role message. Returns
    False for any non-list / empty / non-Cline input.
    """
    if not isinstance(messages, list) or not messages:
        return False
    first = messages[0]
    if not isinstance(first, dict) or first.get("role") != "system":
        return False
    text = _content_text(first.get("content"))
    if not text:
        return False
    return any(fp in text for fp in _CLINE_SYSTEM_FINGERPRINTS)


def _looks_like_cline_plan_mode(messages: Any) -> bool:
    """Return True if a Cline request is in Plan mode (vs Act mode).

    Detected via the system-prompt mention of `plan_mode_respond` --
    the tool is only documented in the system prompt when Plan mode
    is active. Returns False for any non-Cline request shape.

    Why this matters: Plan mode produces only conversational text
    inside a `<plan_mode_respond>` block, never code, never edits.
    Tighter `max_tokens` here saves wall time without truncating
    legitimate output.
    """
    if not _looks_like_cline(messages):
        return False
    assert isinstance(messages, list)  # narrowed by _looks_like_cline
    text = _content_text(messages[0].get("content"))
    return _CLINE_PLAN_MODE_FINGERPRINT in text


def _content_text(content: Any) -> str:
    """Flatten an OpenAI-style message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                out.append(part["text"])
        return "\n".join(out)
    return ""


_PYTHON_FENCE_RE = re.compile(r"```(?:python|py)\b", re.IGNORECASE)


def _looks_like_fixup_turn(messages: Iterable[dict[str, Any]] | None) -> bool:
    """Heuristic: this is a fix-up turn if the conversation contains at
    least one prior assistant message that itself contains a Python
    code fence. That matches our cursor_loop.py shape (user-prompt,
    assistant-with-code, user-feedback) and Cursor's own follow-up
    pattern.

    Single-turn requests (just one user message) always return False.
    """
    if not messages:
        return False
    msgs = list(messages)
    if len(msgs) < 3:
        return False
    saw_assistant_code = False
    for m in msgs:
        if m.get("role") != "assistant":
            continue
        text = _content_text(m.get("content"))
        if _PYTHON_FENCE_RE.search(text):
            saw_assistant_code = True
            break
    if not saw_assistant_code:
        return False
    # The most recent message must be a user message (otherwise we are
    # not actually about to generate a fix-up; the assistant turn was
    # the last thing).
    return msgs[-1].get("role") == "user"


# --- Public API: pure transforms on the request dict -------------------


def apply_static_guardrail(
    data: dict[str, Any],
    *,
    max_tokens: int = LOCAL_MAX_TOKENS_DEFAULT,
    stop_sequences: list[str] | None = None,
    system_nudge: str | None = LOCAL_SYSTEM_NUDGE,
    only_for_local: bool = True,
) -> dict[str, Any]:
    """Apply Strategy A. Mutates and returns `data`.

    Rules:
      - If the caller already pinned `max_tokens` to a value <= our cap,
        we leave it alone. Otherwise we clamp.
      - We extend `stop` rather than replace it, so callers can add
        their own stop sequences.
      - For Cline-shaped traffic (detected via system-prompt
        fingerprint) we swap the python-fence stops for the Cline
        tool-close-tag stops -- the python-fence stops are useless to
        Cline and the tool-close stops prevent the model from
        hallucinating a multi-turn conversation in one response.
      - The system nudge is prepended only if there is no existing
        system message. Cline already has a ~13K-token system prompt
        of its own, so we don't add ours on top of that.
    """
    try:
        if only_for_local and not _is_local(data.get("model")):
            return data

        messages = data.get("messages")
        is_cline = _looks_like_cline(messages)
        is_plan = is_cline and _looks_like_cline_plan_mode(messages)

        # Clamp max_tokens. Plan mode gets a tighter cap because plan
        # responses are conversational, never contain code, and rarely
        # exceed 800 tokens of legitimate output.
        effective_cap = LOCAL_MAX_TOKENS_PLAN if is_plan else max_tokens
        existing_max = data.get("max_tokens")
        if existing_max is None or int(existing_max) > effective_cap:
            data["max_tokens"] = int(effective_cap)

        # Extend stop sequences. Cline traffic gets the tool-close-tag
        # set; everything else gets the python-fence default.
        if stop_sequences is not None:
            seqs = list(stop_sequences)
        elif is_cline:
            seqs = list(CLINE_STOP_SEQUENCES)
        else:
            seqs = list(LOCAL_STOP_SEQUENCES)
        existing_stop = data.get("stop")
        if isinstance(existing_stop, str):
            existing_stop = [existing_stop]
        elif existing_stop is None:
            existing_stop = []
        elif not isinstance(existing_stop, list):
            existing_stop = list(existing_stop)
        merged_stop: list[str] = list(existing_stop)
        for s in seqs:
            if s not in merged_stop:
                merged_stop.append(s)
        # OpenAI/LiteLLM accepts up to 4 stop strings reliably.
        if merged_stop:
            data["stop"] = merged_stop[:4]

        # Prepend a system message if none present. Cline already has
        # a sizable system prompt of its own; injecting ours on top
        # would only confuse the agent harness.
        if system_nudge and not is_cline:
            messages = data.get("messages") or []
            if not any(
                isinstance(m, dict) and m.get("role") == "system"
                for m in messages
            ):
                data["messages"] = [
                    {"role": "system", "content": system_nudge},
                    *messages,
                ]
    except Exception:
        # Swallowing: a control bug must never break the request.
        pass
    return data


def apply_multi_turn_tighten(
    data: dict[str, Any],
    *,
    max_tokens: int = LOCAL_MAX_TOKENS_FIXUP,
    fixup_nudge: str | None = LOCAL_FIXUP_NUDGE,
    only_for_local: bool = True,
) -> dict[str, Any]:
    """Apply Strategy C. Mutates and returns `data`.

    Detects a fix-up turn and tightens further:
      - clamps max_tokens to a smaller ceiling
      - appends a brief reminder to the LAST user message (not as a new
        message, so the cache prefix stays maximally reusable)
    No effect on single-turn requests or non-local models. Cline
    traffic is also exempted: its harness expects a strict
    role/content shape, and silently appending a nudge to the user
    message can be parsed as part of the tool-result envelope and
    confuse the model. The static guardrail's stop-sequence swap is
    enough to fix Cline over-generation on its own.
    """
    try:
        if only_for_local and not _is_local(data.get("model")):
            return data
        messages = data.get("messages") or []
        if _looks_like_cline(messages):
            return data
        if not _looks_like_fixup_turn(messages):
            return data

        existing_max = data.get("max_tokens")
        if existing_max is None or int(existing_max) > max_tokens:
            data["max_tokens"] = int(max_tokens)

        if fixup_nudge:
            # Append to the trailing user message in place. Mutating a
            # message object inside the list is fine for LiteLLM's
            # downstream code paths.
            last = messages[-1]
            content = last.get("content")
            suffix = "\n\n" + fixup_nudge
            if isinstance(content, str):
                last["content"] = content + suffix
            elif isinstance(content, list):
                # Find the last text part and extend it.
                for part in reversed(content):
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        part["text"] = part["text"] + suffix
                        break
                else:
                    content.append({"type": "text", "text": fixup_nudge})
            else:
                last["content"] = fixup_nudge
    except Exception:
        pass
    return data


def _has_think_directive(text: str) -> bool:
    """True if the user message already starts with /think or
    /no_think. Both are Qwen3 runtime switches and we treat either as
    "user already decided", so we don't override."""
    stripped = text.lstrip()
    return stripped.startswith("/think") or stripped.startswith("/no_think")


def inject_qwen3_think_directive(data: dict[str, Any]) -> dict[str, Any]:
    """Prepend `/think ` to the trailing user message so Qwen3-Coder-Next
    engages extended-reasoning mode for this turn.

    Idempotent: a /think or /no_think directive already at the start
    of the user message is left in place. Mutates and returns `data`
    so it can be chained with apply_static_guardrail / apply_all.

    Only intended for local-tier calls. The caller is responsible for
    gating on `_is_local(data.get("model"))`; we apply unconditionally
    here for testability.
    """
    try:
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            return data
        # Find the LAST user message and prepend the directive there.
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                if _has_think_directive(content):
                    return data
                msg["content"] = QWEN3_THINK_PREFIX + content
            elif isinstance(content, list):
                # Walk to the FIRST text part and prepend there.
                for part in content:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        if _has_think_directive(part["text"]):
                            return data
                        part["text"] = QWEN3_THINK_PREFIX + part["text"]
                        break
            else:
                msg["content"] = QWEN3_THINK_PREFIX.rstrip()
            return data
    except Exception:
        pass
    return data


def apply_claude_thinking_params(
    data: dict[str, Any],
    effort: str = CLAUDE_THINKING_EFFORT_DEFAULT,
) -> dict[str, Any]:
    """Enable Anthropic *adaptive* extended thinking on a Claude-tier request.

    Sets `thinking={"type":"adaptive"}` and an `output_config.effort` knob
    (the model decides when/how deeply to think; effort is soft guidance),
    forces the Anthropic-required sampling params (temperature must be 1 and
    top_p/top_k must be unset when thinking is on), and floors `max_tokens`
    so a tiny inbound cap can't starve the reasoning trace.

    Uses adaptive rather than the legacy `{"type":"enabled","budget_tokens"}`
    form, which 400s on Opus 4.7/4.8. Unknown effort values fall back to the
    default. Idempotent and never raises -- a control that crashes the
    request path is worse than no thinking. Mutates and returns `data`.
    """
    try:
        data["thinking"] = {"type": "adaptive"}
        eff = effort if effort in _CLAUDE_THINKING_EFFORTS else CLAUDE_THINKING_EFFORT_DEFAULT
        output_config = data.get("output_config")
        if not isinstance(output_config, dict):
            output_config = {}
        output_config["effort"] = eff
        data["output_config"] = output_config
        # Anthropic rejects temperature != 1 / any top_p|top_k when thinking is on.
        data["temperature"] = 1
        data.pop("top_p", None)
        data.pop("top_k", None)
        current = data.get("max_tokens")
        if not isinstance(current, int) or current < CLAUDE_THINKING_MAX_TOKENS_FLOOR:
            data["max_tokens"] = CLAUDE_THINKING_MAX_TOKENS_FLOOR
    except Exception:
        pass
    return data


def apply_all(
    data: dict[str, Any],
    *,
    enable_static: bool = True,
    enable_multi_turn: bool = True,
) -> dict[str, Any]:
    """Convenience: apply both strategies in order. Order matters --
    the static guardrail prepends the system message, then the
    multi-turn tightening adjusts the trailing user message."""
    if enable_static:
        apply_static_guardrail(data)
    if enable_multi_turn:
        apply_multi_turn_tighten(data)
    return data
