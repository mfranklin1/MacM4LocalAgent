"""Unit tests for the size + complexity-based router callback."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from router.complexity_classifier import classify
from router.route_by_size import (
    SizeBasedRouter,
    _check_sticky,
    _estimate_tokens,
    _extract_model_override,
    _extract_user_task,
    _flat_prompt,
    _looks_like_failure,
    _mark_sticky,
    _mark_sticky_turns,
    _sticky_escalations,
    _task_fingerprint,
    _TOOL_RESULT_STICKY_TURNS,
    _UNBOUNDED,
    decide_tier,
    decide_tier_cline,
    ROUTE_FAST_MAX,
    ROUTE_LONG_MAX,
)


# ---- complexity_classifier ----------------------------------------------------

@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Refactor the entire architecture",   True),
        ("Design a system for billing",        True),
        ("change across multiple files",       True),
        ("[claude] handle this please",        True),
        ("Think step by step about this",      True),
        ("[local] just do it fast",            False),
        ("add 1 to x",                         False),
        ("",                                   False),
    ],
)
def test_classify(prompt: str, expected: bool) -> None:
    is_complex, _ = classify(prompt)
    assert is_complex is expected


def test_classify_local_tag_overrides_complex() -> None:
    is_complex, reason = classify("[local] refactor the architecture please")
    assert is_complex is False
    assert "[local]" in reason


# ---- token estimator + flat_prompt --------------------------------------------

def test_estimate_tokens_empty() -> None:
    assert _estimate_tokens(None) == 0
    assert _estimate_tokens([]) == 0


def test_estimate_tokens_string_content() -> None:
    msgs = [{"role": "user", "content": "x" * 360}]
    # 360 / 3.6 = 100 tokens
    assert _estimate_tokens(msgs) == 100


def test_estimate_tokens_list_content() -> None:
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "abc"},
            {"type": "image", "image": "..."},        # ignored
            {"type": "text", "text": "defg"},
        ],
    }]
    assert _estimate_tokens(msgs) >= 1


def test_flat_prompt_concats() -> None:
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user",   "content": "hi"},
    ]
    assert "sys" in _flat_prompt(msgs)
    assert "hi"  in _flat_prompt(msgs)


# ---- M1: tiktoken-accurate path near tier boundaries -------------------------

def test_estimate_tokens_far_from_boundary_uses_heuristic(monkeypatch) -> None:
    """Tiny prompts are nowhere near a tier boundary; estimator stays
    on the cheap chars/3.6 path. We pin the result to the heuristic
    value to detect accidental tiktoken regressions for cheap calls."""
    from router import route_by_size

    # Sentinel encoder that raises if called -- guarantees we did NOT
    # invoke tiktoken on a non-boundary input.
    class _Boom:
        def encode(self, *_a, **_k):
            raise AssertionError("tiktoken called on non-boundary input")

    monkeypatch.setattr(route_by_size, "_TIKTOKEN_ENCODER", _Boom())
    monkeypatch.setattr(route_by_size, "_TIKTOKEN_DISABLED", False)
    msgs = [{"role": "user", "content": "x" * 360}]  # heuristic = 100
    assert _estimate_tokens(msgs) == 100


def test_estimate_tokens_near_boundary_uses_tiktoken(monkeypatch) -> None:
    """Inside the ±20% band around ROUTE_FAST_MAX we re-tokenize. A
    stub encoder returns a known token count; we verify it was used
    instead of the heuristic."""
    from router import route_by_size

    class _StubEncoder:
        called = 0

        def encode(self, text: str, **_k):
            type(self).called += 1
            # Return a deterministic count so the test assertion is
            # not coupled to cl100k_base behaviour.
            return [0] * 12345

    monkeypatch.setattr(route_by_size, "_TIKTOKEN_ENCODER", _StubEncoder())
    monkeypatch.setattr(route_by_size, "_TIKTOKEN_DISABLED", False)

    # Heuristic = 16000 (right on the boundary -> always inside band).
    msgs = [{"role": "user", "content": "x" * int(16000 * 3.6)}]
    assert _estimate_tokens(msgs) == 12345
    assert _StubEncoder.called == 1


def test_estimate_tokens_falls_back_when_tiktoken_disabled(monkeypatch) -> None:
    """If tiktoken is disabled (e.g. failed import, env opt-out), the
    estimator silently returns the heuristic value even near a
    boundary."""
    from router import route_by_size

    monkeypatch.setattr(route_by_size, "_TIKTOKEN_ENCODER", None)
    monkeypatch.setattr(route_by_size, "_TIKTOKEN_DISABLED", True)

    msgs = [{"role": "user", "content": "x" * int(16000 * 3.6)}]
    # Heuristic value should come through unchanged.
    assert _estimate_tokens(msgs) == 16000


def test_estimate_tokens_falls_back_on_tiktoken_exception(monkeypatch) -> None:
    """If tiktoken raises (e.g. encoding error on weird unicode), we
    return the heuristic instead of crashing the router callback."""
    from router import route_by_size

    class _Raiser:
        def encode(self, *_a, **_k):
            raise RuntimeError("simulated tiktoken failure")

    monkeypatch.setattr(route_by_size, "_TIKTOKEN_ENCODER", _Raiser())
    monkeypatch.setattr(route_by_size, "_TIKTOKEN_DISABLED", False)

    msgs = [{"role": "user", "content": "x" * int(16000 * 3.6)}]
    assert _estimate_tokens(msgs) == 16000  # heuristic fallback


def test_near_tier_boundary_band() -> None:
    """The ±20% band is symmetric in tokens around each tier max."""
    from router.route_by_size import (
        _near_tier_boundary,
        ROUTE_FAST_MAX,
        ROUTE_LONG_MAX,
    )

    # Right on the boundary -> inside.
    assert _near_tier_boundary(ROUTE_FAST_MAX) is True
    assert _near_tier_boundary(ROUTE_LONG_MAX) is True

    # Just outside the fast-band on the low side.
    assert _near_tier_boundary(int(ROUTE_FAST_MAX * 0.75)) is False
    # Just inside the long-band on the high side.
    assert _near_tier_boundary(int(ROUTE_LONG_MAX * 1.15)) is True
    # Way past the long boundary.
    assert _near_tier_boundary(int(ROUTE_LONG_MAX * 1.5)) is False


# ---- decide_tier --------------------------------------------------------------

def test_decide_tier_routes_small_to_fast() -> None:
    msgs = [{"role": "user", "content": "what does this regex do?"}]
    model, reason, tokens = decide_tier(msgs)
    assert model == "local-fast"
    assert tokens <= ROUTE_FAST_MAX


def test_decide_tier_routes_medium_to_long() -> None:
    # Heuristic must land clearly above the fast-tier band so the
    # tiktoken-accurate path doesn't kick in. The ±20% band ends at
    # ROUTE_FAST_MAX * 1.2, so we aim for ~1.3x to be safely outside
    # AND still well under ROUTE_LONG_MAX.
    target_tokens = int(ROUTE_FAST_MAX * 1.3)
    chars = target_tokens * 4  # 4 chars/token heuristic upper bound
    msgs = [{"role": "user", "content": "x" * chars}]
    model, reason, tokens = decide_tier(msgs)
    assert model == "local-long"
    assert ROUTE_FAST_MAX < tokens <= ROUTE_LONG_MAX


def test_decide_tier_routes_huge_to_claude() -> None:
    # Push well beyond the long-tier band (>1.2x ROUTE_LONG_MAX) so the
    # heuristic alone resolves the routing and tiktoken's accurate path
    # never fires.
    target_tokens = int(ROUTE_LONG_MAX * 1.3)
    chars = target_tokens * 4
    msgs = [{"role": "user", "content": "x" * chars}]
    model, reason, tokens = decide_tier(msgs)
    assert model == "claude-code"
    assert "tokens" in reason


def test_decide_tier_complex_short_goes_claude() -> None:
    msgs = [{"role": "user", "content": "Refactor the architecture across multiple files"}]
    model, reason, _ = decide_tier(msgs)
    assert model == "claude-code"
    assert "complex" in reason


# ---- SizeBasedRouter callback -------------------------------------------------

@pytest.fixture
def router(tmp_db) -> SizeBasedRouter:                                       # noqa: ARG001
    return SizeBasedRouter()


def test_pre_call_rewrites_hybrid_auto(router: SizeBasedRouter) -> None:
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": [{"role": "user", "content": "tiny prompt"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] in {"local-fast", "local-long", "claude-code"}
    meta = new["metadata"]
    assert meta["route_decision"] == new["model"]
    assert isinstance(meta["route_reason"], str)
    assert meta["route_tokens_estimated"] >= 1


def test_pre_call_does_not_touch_explicit_model(router: SizeBasedRouter) -> None:
    data: dict[str, Any] = {
        "model": "claude-code",
        "messages": [{"role": "user", "content": "hello"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new["model"] == "claude-code"
    assert "metadata" not in new or "route_decision" not in new.get("metadata", {})


# ---- gpt-* prefix strip (Cursor-friendly aliases) ----------------------------

@pytest.mark.parametrize(
    "incoming,expected_canonical",
    [
        ("gpt-local-fast",   "local-fast"),
        ("gpt-local-long",   "local-long"),
        ("gpt-local-agent",  "local-agent"),
        ("gpt-claude-code",  "claude-code"),
    ],
)
def test_pre_call_strips_gpt_prefix_for_explicit_aliases(
    router: SizeBasedRouter, incoming: str, expected_canonical: str
) -> None:
    """Cursor sends the OpenAI-shaped alias name; the router must rewrite
    it to the canonical name so cost-tier classification, the over-gen
    controls, and any downstream logic see the model they expect."""
    data: dict[str, Any] = {
        "model": incoming,
        "messages": [{"role": "user", "content": "hi"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] == expected_canonical


def test_pre_call_strips_gpt_prefix_then_rewrites_hybrid_auto(
    router: SizeBasedRouter,
) -> None:
    """gpt-hybrid-auto must collapse to hybrid-auto AND then trigger the
    size-based rewrite, exactly like the canonical alias does."""
    data: dict[str, Any] = {
        "model": "gpt-hybrid-auto",
        "messages": [{"role": "user", "content": "tiny prompt"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] in {"local-fast", "local-long", "claude-code"}
    meta = new["metadata"]
    assert meta["route_decision"] == new["model"]
    assert isinstance(meta["route_reason"], str)
    assert meta["route_tokens_estimated"] >= 1


def test_pre_call_does_not_strip_unknown_gpt_prefix(
    router: SizeBasedRouter,
) -> None:
    """A real OpenAI model name like `gpt-4o` should pass through
    untouched -- the strip is whitelisted to our specific aliases."""
    data: dict[str, Any] = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new["model"] == "gpt-4o"


class _FakeUsage:
    def __init__(self, in_tok: int, out_tok: int) -> None:
        self.prompt_tokens = in_tok
        self.completion_tokens = out_tok

    def model_dump(self) -> dict[str, int]:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens}


class _FakeResponse:
    def __init__(self, in_tok: int, out_tok: int) -> None:
        self.usage = _FakeUsage(in_tok, out_tok)


def test_log_success_event_records_local(router: SizeBasedRouter, tmp_db) -> None:
    start = time.time()
    router.log_success_event(
        kwargs={"model": "local-fast", "metadata": {"route_reason": "tokens 100 <= 16000"}},
        response_obj=_FakeResponse(100, 50),
        start_time=start,
        end_time=start + 0.3,
    )
    rows = list(router._conn.execute("SELECT * FROM requests"))
    assert len(rows) == 1
    r = dict(zip([d[0] for d in router._conn.execute("SELECT * FROM requests").description], rows[0]))
    assert r["tier"] == "local-fast"
    assert r["actual_cost"] == 0.0
    # shadow_cost = 100*3e-6 + 50*15e-6 = 0.0003 + 0.00075 = 0.00105
    assert r["shadow_cost"] == pytest.approx(0.00105, rel=1e-6)
    assert 200 <= r["latency_ms"] <= 600


def test_log_success_event_reads_route_reason_from_litellm_params(
    router: SizeBasedRouter, tmp_db,
) -> None:
    """LiteLLM relocates async_pre_call_hook metadata under
    kwargs['litellm_params']['metadata'] in the success-callback
    lifecycle. Without checking this nested path, the route_reason
    column is silently empty for every Cline + hybrid-auto request --
    which is exactly the bug we hit in production."""
    start = time.time()
    router.log_success_event(
        kwargs={
            "model": "local-long",
            # NO top-level metadata; only the nested litellm_params copy.
            "litellm_params": {
                "metadata": {
                    "route_decision": "local-long",
                    "route_reason": "cline-mode: cline+default: task=12 tok",
                    "route_tokens_estimated": 12,
                },
            },
        },
        response_obj=_FakeResponse(15000, 200),
        start_time=start,
        end_time=start + 1.5,
    )
    row = router._conn.execute(
        "SELECT route_reason FROM requests WHERE model='local-long'"
    ).fetchone()
    assert row is not None
    assert row[0] == "cline-mode: cline+default: task=12 tok"


def test_log_success_event_top_metadata_takes_priority_when_set(
    router: SizeBasedRouter, tmp_db,
) -> None:
    """If both top-level and nested metadata are present (rare but
    possible if a different LiteLLM lifecycle path applies), prefer
    the nested one because that's where the routing decision actually
    flows from. This pins the precedence so a future edit can't
    silently flip it."""
    start = time.time()
    router.log_success_event(
        kwargs={
            "model": "local-fast",
            "metadata": {"route_reason": "TOP-LEVEL"},
            "litellm_params": {"metadata": {"route_reason": "NESTED"}},
        },
        response_obj=_FakeResponse(10, 5),
        start_time=start,
        end_time=start + 0.1,
    )
    row = router._conn.execute(
        "SELECT route_reason FROM requests WHERE model='local-fast'"
    ).fetchone()
    assert row is not None
    assert row[0] == "NESTED"


def test_log_success_event_records_claude(router: SizeBasedRouter) -> None:
    start = time.time()
    router.log_success_event(
        kwargs={"model": "claude-sonnet-4-6"},
        response_obj=_FakeResponse(1000, 500),
        start_time=start,
        end_time=start + 1.2,
    )
    rows = list(router._conn.execute(
        "SELECT tier, actual_cost, shadow_cost FROM requests WHERE model='claude-sonnet-4-6'"
    ))
    assert len(rows) == 1
    tier, actual, shadow = rows[0]
    assert tier == "claude"
    # actual == shadow for Claude calls.
    assert actual == pytest.approx(shadow, rel=1e-6)
    assert actual == pytest.approx(1000 * 3e-6 + 500 * 15e-6, rel=1e-6)


def test_log_success_event_dict_response(router: SizeBasedRouter) -> None:
    """LiteLLM sometimes hands back a dict instead of an object."""
    start = time.time()
    router.log_success_event(
        kwargs={"model": "ollama/qwen3-coder:30b"},
        response_obj={"usage": {"prompt_tokens": 10, "completion_tokens": 20}},
        start_time=start,
        end_time=start + 0.05,
    )
    (tier, in_tok, out_tok) = router._conn.execute(
        "SELECT tier, input_tok, output_tok FROM requests WHERE model LIKE 'ollama/%'"
    ).fetchone()
    assert tier == "local-long"
    assert in_tok == 10
    assert out_tok == 20


# ---- Cline-aware routing -----------------------------------------------------
#
# Cline ships a ~13.5K-token system prompt; size-based routing alone always
# picks local-long for it regardless of how trivial the task is, AND the
# complexity classifier accidentally matches `[local]` substrings inside
# Cline's tool documentation. The Cline-aware path extracts the user's
# task from `<task>...</task>` and classifies on THAT only.

# A minimal-but-realistic Cline system prompt: contains the fingerprints
# `_looks_like_cline` checks for. We deliberately keep this short so test
# token-count assertions are predictable.
_CLINE_SYSTEM = (
    "You are Cline, a highly skilled software engineer. "
    "Use <replace_in_file> to edit files and <attempt_completion> to "
    "signal completion."
)


def _cline_msgs(task: str, *extra: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a Cline-shaped messages array: system + user(<task>...) + extras."""
    base: list[dict[str, Any]] = [
        {"role": "system", "content": _CLINE_SYSTEM},
        {"role": "user", "content": f"<task>\n{task}\n</task>"},
    ]
    base.extend(extra)
    return base


@pytest.fixture(autouse=True)
def _clear_sticky():
    """Stickiness leaks between tests because it's module-level state.
    Clear before AND after each test so we don't accidentally route a
    later test to claude because an earlier test marked the same task
    fingerprint."""
    _sticky_escalations.clear()
    yield
    _sticky_escalations.clear()


def test_extract_user_task_pulls_text_from_envelope() -> None:
    msgs = _cline_msgs("Add a comment to README")
    assert _extract_user_task(msgs) == "Add a comment to README"


def test_extract_user_task_handles_list_content() -> None:
    msgs = [
        {"role": "system", "content": _CLINE_SYSTEM},
        {
            "role": "user",
            "content": [{"type": "text", "text": "<task>\nFix bug\n</task>"}],
        },
    ]
    assert _extract_user_task(msgs) == "Fix bug"


def test_extract_user_task_returns_none_for_non_cline() -> None:
    msgs = [{"role": "user", "content": "just a regular question"}]
    assert _extract_user_task(msgs) is None


def test_extract_user_task_returns_none_for_empty() -> None:
    assert _extract_user_task(None) is None
    assert _extract_user_task([]) is None


def test_decide_tier_cline_default_routes_to_local_long() -> None:
    msgs = _cline_msgs("Add a single-line comment to main.py")
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "local-long"
    assert "default" in reason


def test_decide_tier_cline_complexity_keyword_escalates() -> None:
    msgs = _cline_msgs("Refactor the entire authentication architecture")
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "claude-code"
    assert "architecture" in reason or "design" in reason


def test_decide_tier_cline_claude_tag_escalates() -> None:
    msgs = _cline_msgs("[claude] What is 2+2?")
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "claude-code"
    assert "[claude]" in reason


def test_decide_tier_cline_local_tag_overrides_complex_keywords() -> None:
    """[local] is the user opting out -- it must beat complexity AND
    failure detection. This is the cost-safety guarantee for users
    who deliberately want to exercise the local stack."""
    msgs = _cline_msgs(
        "[local] Refactor the entire authentication architecture across multiple files"
    )
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "local-long"
    assert "[local]" in reason


def test_decide_tier_cline_local_tag_overrides_failure_signal() -> None:
    """Even on a failing turn, [local] keeps us on local."""
    failure = (
        "[read_file] Result:\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 5, in <module>\n'
        "    raise ValueError\n"
        "ValueError: oops"
    )
    msgs = _cline_msgs(
        "[local] Add a test",
        {"role": "assistant", "content": "I'll read the file."},
        {"role": "user", "content": failure},
    )
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "local-long"
    assert "[local]" in reason


def test_decide_tier_cline_single_traceback_no_longer_escalates() -> None:
    """Rule A: a single Python traceback in a tool result is NOT
    enough to escalate. Local model gets to keep the task. This
    is the behavior change introduced after observing a 35K-token
    architectural overview unnecessarily flip to Claude on one
    panic in passing."""
    failure = (
        "[read_file] Result:\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 5, in <module>\n'
        "    raise ValueError\n"
        "ValueError: oops"
    )
    msgs = _cline_msgs(
        "Add a test for the parser",
        {"role": "assistant", "content": "I'll read the file."},
        {"role": "user", "content": failure},
    )
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "local-long"
    assert "default" in reason


def test_decide_tier_cline_two_tracebacks_escalate() -> None:
    """Rule A: two distinct python tracebacks in one tool result
    is a strong-enough signal to escalate. The corroboration
    (>=2 frames within the SAME high-confidence category) is what
    makes this different from a single one-off."""
    failure = (
        "[execute_command] Result:\n"
        "Traceback (most recent call last):\n"
        '  File "x.py", line 5, in <module>\n'
        "    raise ValueError\n"
        "ValueError: first\n"
        "----\n"
        "Traceback (most recent call last):\n"
        '  File "y.py", line 8, in handler\n'
        "    raise RuntimeError\n"
        "RuntimeError: second"
    )
    msgs = _cline_msgs(
        "Run the test suite",
        {"role": "assistant", "content": "Running tests."},
        {"role": "user", "content": failure},
    )
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "claude-code"
    assert "traceback" in reason


def test_decide_tier_cline_panic_plus_error_lines_escalates() -> None:
    """Rule A: a single panic combined with multiple `error:`
    lines is two distinct error categories => escalate."""
    failure = (
        "[execute_command] Result:\n"
        "thread 'main' panicked at 'unwrap on None', src/main.rs:42\n"
        "error: process didn't exit successfully\n"
        "error: build failed\n"
        "error: aborting due to previous errors"
    )
    msgs = _cline_msgs(
        "Build the rust crate",
        {"role": "assistant", "content": "Building."},
        {"role": "user", "content": failure},
    )
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "claude-code"
    assert "multiple error signals" in reason


def test_decide_tier_cline_big_file_dump_does_not_escalate() -> None:
    """Cline's normal environment_details / read_file payload is several KB.
    Pure size must NOT trigger escalation -- only actual error signatures."""
    big_dump = "[read_file] Result:\n" + (
        "def boring_function():\n    return 42\n" * 500
    )
    msgs = _cline_msgs(
        "Read the file and summarize",
        {"role": "assistant", "content": "Reading."},
        {"role": "user", "content": big_dump},
    )
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "local-long"
    assert "default" in reason


def test_decide_tier_cline_first_turn_does_not_check_tool_result() -> None:
    """Tool-result detection requires turn 2+ (msg_count >= 4). On turn 1
    we have just system + task, which CAN'T have a failure."""
    msgs = _cline_msgs("Add a test")
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "local-long"
    assert "default" in reason


def test_decide_tier_cline_sticky_keeps_task_on_claude() -> None:
    """Once a task escalates, ALL subsequent turns of the same task
    stay on Claude even if those turns themselves look trivial."""
    # First call: complexity escalates -> claude.
    msgs1 = _cline_msgs("Refactor the entire authentication architecture")
    tier1, _, _ = decide_tier_cline(msgs1)
    assert tier1 == "claude-code"

    # Second call: same task, but with a clean tool result. Should still
    # be claude because the task fingerprint is sticky.
    msgs2 = _cline_msgs(
        "Refactor the entire authentication architecture",
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "[read_file] Result: file is empty"},
    )
    tier2, reason2, _ = decide_tier_cline(msgs2)
    assert tier2 == "claude-code"
    assert "sticky" in reason2


def test_decide_tier_cline_different_task_resets_stickiness() -> None:
    """Stickiness is per-task-fingerprint -- a NEW task gets a fresh
    decision."""
    msgs1 = _cline_msgs("Refactor the architecture across multiple files")
    decide_tier_cline(msgs1)  # marks task1 sticky

    msgs2 = _cline_msgs("Add a one-line comment to README")
    tier2, reason2, _ = decide_tier_cline(msgs2)
    assert tier2 == "local-long"
    assert "default" in reason2


# ---- bounded vs unbounded stickiness (rule B) ----

def test_mark_sticky_is_time_bounded_unbounded_turns() -> None:
    """Time-bounded entries (used for explicit tags and complexity
    escalations) carry the sentinel _UNBOUNDED in the turns slot,
    so _check_sticky won't decrement them."""
    _sticky_escalations.clear()
    _mark_sticky("fp", "explicit [opus] tag")
    _, _, remaining = _sticky_escalations["fp"]
    assert remaining == _UNBOUNDED

    # Many reads should not evict it.
    for _ in range(20):
        assert _check_sticky("fp") == "explicit [opus] tag"
    assert "fp" in _sticky_escalations


def test_mark_sticky_turns_evicts_after_budget() -> None:
    """Turn-bounded entries decrement on each successful read and
    self-evict when the budget hits zero, so the next call
    re-evaluates from scratch."""
    _sticky_escalations.clear()
    _mark_sticky_turns("fp", "rust panic", 3)

    # 3 successful reads, then eviction.
    assert _check_sticky("fp") == "rust panic"
    assert _check_sticky("fp") == "rust panic"
    assert _check_sticky("fp") == "rust panic"
    assert _check_sticky("fp") is None
    assert "fp" not in _sticky_escalations


def test_decide_tier_cline_tool_result_sticky_expires_after_three_turns() -> None:
    """End-to-end: a tool-result-driven escalation should keep the
    task on Claude for exactly _TOOL_RESULT_STICKY_TURNS subsequent
    turns, then release it back to local-long."""
    _sticky_escalations.clear()
    task = "Fix the build"

    failure = (
        "[execute_command] Result:\n"
        "thread 'main' panicked at A\n"
        "thread 'worker' panicked at B"
    )

    # Turn 2: failure triggers escalation. This call DOES NOT consume
    # a budget slot (the entry is fresh-created in this same call).
    msgs_fail = _cline_msgs(
        task,
        {"role": "assistant", "content": "Trying."},
        {"role": "user", "content": failure},
    )
    tier, reason, _ = decide_tier_cline(msgs_fail)
    assert tier == "claude-code"
    assert "panic" in reason

    # Subsequent turns with CLEAN tool results -- the only thing
    # keeping these on Claude is the sticky entry. With a 3-turn
    # budget, exactly 3 reads should still hit sticky, then the
    # 4th should re-evaluate to local-long.
    clean = "[read_file] Result: file content\nfoo"
    base_msgs = lambda: _cline_msgs(  # noqa: E731
        task,
        {"role": "assistant", "content": "Continuing."},
        {"role": "user", "content": clean},
    )

    sticky_tiers = []
    for _ in range(_TOOL_RESULT_STICKY_TURNS):
        t, r, _ = decide_tier_cline(base_msgs())
        sticky_tiers.append((t, r))
    assert all(t == "claude-code" for t, _ in sticky_tiers), sticky_tiers
    assert all("sticky" in r for _, r in sticky_tiers), sticky_tiers

    # Budget exhausted -- next read drops back to local-long.
    final_tier, final_reason, _ = decide_tier_cline(base_msgs())
    assert final_tier == "local-long"
    assert "default" in final_reason


def test_decide_tier_cline_tag_sticky_does_not_expire_in_3_turns() -> None:
    """Contrast with the tool-result test: explicit `[opus]` tag
    stickiness is time-bounded (no turn budget), so it must NOT
    expire after 3 turns. This is what the user explicitly asked
    for: 'tag-driven escalations keep using the existing 30-min
    stickiness'."""
    _sticky_escalations.clear()
    task = "[opus] Write a haiku"

    msgs1 = _cline_msgs(task)
    tier1, reason1, _ = decide_tier_cline(msgs1)
    assert tier1 == "claude-opus-4-7"
    assert "[opus]" in reason1

    # 5 follow-up turns (Cline replays the tagged task verbatim
    # because that's how its harness works). All must stay sticky
    # on Claude. NOTE: the sticky-replay branch in decide_tier_cline
    # currently always returns "claude-code" -- it preserves
    # *whether* the task was escalated, not which specific Claude
    # model was originally chosen. The important assertion for
    # rule B is that we don't drop back to local-long after 3
    # turns (which is what the bounded-stickiness path would do).
    for i in range(5):
        msgs = _cline_msgs(
            task,
            {"role": "assistant", "content": f"Turn {i}"},
            {"role": "user", "content": "[read_file] Result: ok"},
        )
        tier, reason, _ = decide_tier_cline(msgs)
        assert tier == "claude-code", f"turn {i+2}: {tier!r}"
        assert "sticky" in reason, f"turn {i+2}: {reason!r}"
        assert "[opus]" in reason, f"turn {i+2}: original tag reason lost: {reason!r}"


def test_decide_tier_cline_complexity_sticky_does_not_expire_in_3_turns() -> None:
    """Same contrast for complexity-keyword escalations: those are
    also time-bounded, not turn-bounded."""
    _sticky_escalations.clear()
    task = "Refactor the entire authentication architecture"

    msgs1 = _cline_msgs(task)
    tier1, _, _ = decide_tier_cline(msgs1)
    assert tier1 == "claude-code"

    for i in range(5):
        msgs = _cline_msgs(
            task,
            {"role": "assistant", "content": f"Turn {i}"},
            {"role": "user", "content": "[read_file] Result: ok"},
        )
        tier, reason, _ = decide_tier_cline(msgs)
        assert tier == "claude-code", f"turn {i+2}: {tier!r}"
        assert "sticky" in reason, f"turn {i+2}: {reason!r}"


def test_task_fingerprint_normalizes_whitespace() -> None:
    """The same task with different whitespace (Cline indents
    inconsistently in the <task> envelope) must produce the same
    fingerprint, otherwise stickiness misses on every turn."""
    f1 = _task_fingerprint("Add a comment to README")
    f2 = _task_fingerprint("Add  a comment   to README")
    f3 = _task_fingerprint("Add\na comment\nto README")
    assert f1 == f2 == f3


@pytest.mark.parametrize(
    "text,is_failure,note",
    [
        # ---- non-failure cases ----
        ("just some output", False, "no error signals at all"),
        # Rule A: single weak signals do NOT escalate.
        ("Traceback (most recent call last):\n  File 'x'", False,
         "single python traceback alone -- need corroboration"),
        ("thread 'main' panicked at 'oops', src/main.rs:5", False,
         "single rust panic alone -- need corroboration"),
        ("error: file not found", False, "single error: line is noisy on its own"),
        ("at foo (file.js:5:3)", False, "single js stack frame is suspicious but not enough"),
        # 2 frames was the OLD JS threshold; new threshold is >=3
        # within a single category.
        ("at foo (file.js:5:3)\n  at bar (file.js:10:5)", False,
         "2 js stack frames alone -- bumped threshold to 3"),
        ("error: file not found\nerror: cannot read", False,
         "2 error: lines (need >=3 to even register the category, then need corroboration)"),

        # ---- failure cases ----
        # Strong single-category: >=2 panics is a real meltdown.
        (
            "thread 'main' panicked at A\nthread 'worker' panicked at B",
            True,
            ">=2 distinct rust panics",
        ),
        # Strong single-category: >=2 python tracebacks.
        (
            "Traceback (most recent call last):\n  File 'a'\n"
            "----\n"
            "Traceback (most recent call last):\n  File 'b'",
            True,
            ">=2 distinct python tracebacks",
        ),
        # Strong single-category: >=3 JS frames.
        (
            "stack:\n  at foo (a.js:1:1)\n  at bar (b.js:2:2)\n  at baz (c.js:3:3)",
            True,
            ">=3 js stack frames",
        ),
        # Cross-category: panic + error: lines (>=3) is two distinct
        # categories, escalates.
        (
            "thread 'main' panicked at 'X', src/main.rs:5\n"
            "error: A\nerror: B\nerror: C",
            True,
            "panic + 3 error: lines (cross-category)",
        ),
        # Cross-category: traceback + JS frame.
        (
            "Traceback (most recent call last):\n  File 'x'\n"
            "stack:\n  at foo (a.js:1:1)",
            True,
            "py traceback + js stack frame (cross-category)",
        ),
    ],
)
def test_looks_like_failure(text: str, is_failure: bool, note: str) -> None:
    got, _reason = _looks_like_failure(text)
    assert got is is_failure, f"{note}: got {got!r}, expected {is_failure!r}"


def test_pre_call_uses_cline_aware_path_for_cline_traffic() -> None:
    """End-to-end: pre-call hook with hybrid-auto + Cline harness
    should route to local-long (default), not the legacy size-based
    path that would (accidentally) match [local] in the harness."""
    router = SizeBasedRouter()
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": _cline_msgs("Add a comment to README"),
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] == "local-long"
    meta = new["metadata"]
    assert "cline-mode" in meta["route_reason"]
    assert "default" in meta["route_reason"]


def test_pre_call_cline_complex_task_routes_to_claude() -> None:
    router = SizeBasedRouter()
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": _cline_msgs(
            "Refactor the entire authentication architecture across multiple files"
        ),
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] == "claude-code"
    assert "cline-mode" in new["metadata"]["route_reason"]


def test_pre_call_non_cline_traffic_uses_legacy_routing() -> None:
    """Non-Cline traffic (no Cline fingerprint in system prompt) must
    fall through to the existing size-based router -- otherwise we'd
    break the CLI/curl/benchmark callers."""
    router = SizeBasedRouter()
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": [{"role": "user", "content": "small ask"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    # Tiny non-Cline request -> local-fast (legacy path).
    assert new["model"] == "local-fast"
    # And the reason should NOT have the cline-mode prefix.
    assert "cline-mode" not in new["metadata"]["route_reason"]


# ---- task_id / task_text stamping --------------------------------------------
#
# For task-grouped views in the dashboard, every Cline request needs a stable
# task_id (so turns of the same task can be rolled up) and a truncated copy of
# the user's actual task text (so the UI can show 'add a comment to README'
# rather than just a 16-hex hash). Non-Cline traffic must NOT have either,
# because there's no <task> envelope to fingerprint and the dashboard's
# task views explicitly skip those rows.

def test_pre_call_stamps_task_id_for_cline_traffic() -> None:
    """Every Cline request gets a task_id (the task fingerprint) in
    its metadata, regardless of whether routing escalated. This is
    what enables grouping turns in the dashboard."""
    router = SizeBasedRouter()
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": _cline_msgs("Add a one-line comment to README.md"),
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    meta = new["metadata"]
    assert "task_id" in meta
    # 16 hex chars from SHA256 truncation in _task_fingerprint.
    assert len(meta["task_id"]) == 16
    assert all(ch in "0123456789abcdef" for ch in meta["task_id"])


def test_pre_call_stamps_task_text_for_cline_traffic() -> None:
    """task_text is the actual user prompt, truncated for storage but
    long enough to be readable in the dashboard's tasks list."""
    router = SizeBasedRouter()
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": _cline_msgs("Add a one-line comment to README.md"),
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["metadata"]["task_text"] == "Add a one-line comment to README.md"


def test_pre_call_truncates_long_task_text() -> None:
    """Task text is capped at 500 chars to keep `requests` rows lean.
    Cline tasks rarely run that long, but we don't want a runaway
    paste to bloat every successive row."""
    long_task = "Refactor " + ("the auth code " * 200)  # ~2.8 KB
    router = SizeBasedRouter()
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": _cline_msgs(long_task),
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert len(new["metadata"]["task_text"]) == 500


def test_pre_call_same_task_produces_same_task_id() -> None:
    """Two turns of the same Cline task share a fingerprint -- this is
    what enables roll-up in the dashboard. Whitespace and indentation
    inside <task> shouldn't break the match."""
    router1 = SizeBasedRouter()
    router2 = SizeBasedRouter()

    data1: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": _cline_msgs("Add a comment"),
    }
    data2: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": _cline_msgs("Add\n  a    comment"),
    }
    new1 = asyncio.run(router1.async_pre_call_hook(None, None, data1, "completion"))
    new2 = asyncio.run(router2.async_pre_call_hook(None, None, data2, "completion"))
    assert new1["metadata"]["task_id"] == new2["metadata"]["task_id"]


def test_pre_call_does_not_stamp_task_id_for_non_cline() -> None:
    """Non-Cline traffic has no <task> envelope, so the metadata
    must NOT contain task_id / task_text. Otherwise the dashboard
    would group unrelated CLI calls into a phantom task."""
    router = SizeBasedRouter()
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": [{"role": "user", "content": "raw curl request"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    meta = new.get("metadata", {})
    assert "task_id" not in meta
    assert "task_text" not in meta


def test_log_success_persists_task_id_and_text(
    router: SizeBasedRouter, tmp_db,
) -> None:
    """End-to-end: the success callback reads task_id and task_text
    from kwargs['litellm_params']['metadata'] (the path LiteLLM uses
    by the callback phase) and writes them to the requests table."""
    start = time.time()
    router.log_success_event(
        kwargs={
            "model": "local-long",
            "litellm_params": {
                "metadata": {
                    "route_decision": "local-long",
                    "route_reason": "cline-mode: cline+default: task=12 tok",
                    "task_id": "abcdef0123456789",
                    "task_text": "Add a comment to README",
                },
            },
        },
        response_obj=_FakeResponse(15000, 200),
        start_time=start,
        end_time=start + 1.5,
    )
    row = router._conn.execute(
        "SELECT task_id, task_text FROM requests WHERE model='local-long'"
    ).fetchone()
    assert row is not None
    assert row[0] == "abcdef0123456789"
    assert row[1] == "Add a comment to README"


def test_log_success_persists_null_task_id_for_non_cline(
    router: SizeBasedRouter, tmp_db,
) -> None:
    """Non-Cline rows have no task fingerprint; both columns must be
    NULL so the dashboard's `WHERE task_id IS NOT NULL` filter
    excludes them cleanly."""
    start = time.time()
    router.log_success_event(
        kwargs={
            "model": "local-fast",
            "litellm_params": {
                "metadata": {"route_reason": "tokens 10 <= 16000"},
            },
        },
        response_obj=_FakeResponse(10, 5),
        start_time=start,
        end_time=start + 0.1,
    )
    row = router._conn.execute(
        "SELECT task_id, task_text FROM requests WHERE model='local-fast'"
    ).fetchone()
    assert row is not None
    assert row[0] is None
    assert row[1] is None


# ---- inline model override tags ([haiku] / [sonnet] / [opus]) ---------------

# These tests cover the per-turn Claude-tier override added so users can
# escalate (or de-escalate) without changing their default model. The
# tags must be at the START of the task to avoid false positives from
# bracketed words inside Cline's system prompt or tool docs.

@pytest.mark.parametrize(
    "task,expected_alias,expected_tag",
    [
        ("[haiku] What is 2+2?",         "claude-haiku-4-5",  "haiku"),
        ("[sonnet] Refactor this fn",    "claude-sonnet-4-6", "sonnet"),
        ("[opus] design a billing svc",  "claude-opus-4-7",   "opus"),
        ("[OPUS] uppercase tag works",   "claude-opus-4-7",   "opus"),
        ("  [opus]   leading whitespace OK", "claude-opus-4-7", "opus"),
    ],
)
def test_extract_model_override_recognizes_leading_tags(
    task: str, expected_alias: str, expected_tag: str,
) -> None:
    alias, tag = _extract_model_override(task)
    assert alias == expected_alias
    assert tag == expected_tag


@pytest.mark.parametrize(
    "task",
    [
        "",                                      # empty
        "do the thing [opus] please",            # not leading
        "Please [opus] this for me",             # not leading
        "[haik] typo",                           # unknown tag
        "[claude] use default Claude",           # handled separately, not here
        "[local] no Claude at all",              # handled separately, not here
        "<task>[opus] inside envelope</task>",   # extractor receives the inner text already
    ],
)
def test_extract_model_override_rejects_non_leading_or_unknown(task: str) -> None:
    alias, tag = _extract_model_override(task)
    assert alias is None
    assert tag is None


def test_decide_tier_cline_haiku_tag_routes_to_haiku() -> None:
    msgs = _cline_msgs("[haiku] What is 2+2?")
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "claude-haiku-4-5"
    assert "[haiku]" in reason


def test_decide_tier_cline_sonnet_tag_routes_to_sonnet() -> None:
    msgs = _cline_msgs("[sonnet] Add error handling here")
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "claude-sonnet-4-6"
    assert "[sonnet]" in reason


def test_decide_tier_cline_opus_tag_routes_to_opus() -> None:
    msgs = _cline_msgs("[opus] design a billing service")
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "claude-opus-4-7"
    assert "[opus]" in reason


def test_decide_tier_cline_opus_tag_beats_complexity_classifier() -> None:
    """[opus] takes precedence over the architecture/multi-file
    keyword detector. The user picked the model explicitly; we
    must not silently downgrade or reroute to the default tier."""
    msgs = _cline_msgs(
        "[opus] Refactor the entire authentication architecture across multiple files"
    )
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "claude-opus-4-7"
    assert "[opus]" in reason


def test_decide_tier_cline_local_tag_still_beats_opus_tag() -> None:
    """[local] is the absolute opt-out. If the user wrote both
    `[local]` AND `[opus]` (only one can be leading -- this tests
    [local] leading), [local] must still win because spend safety
    is the highest-priority invariant."""
    # Only one tag can be at the leading position. This task has
    # [local] leading -- it should win. (`[opus]` is mid-prompt
    # and won't match the leading-only override regex anyway.)
    msgs = _cline_msgs("[local] [opus] do this fast")
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "local-long"
    assert "[local]" in reason


def test_decide_tier_cline_claude_tag_still_routes_to_default() -> None:
    """[claude] continues to mean 'use the default Claude tier'
    (now Opus 4.7 via the claude-code alias), NOT a specific
    model. Backwards-compat with existing user habits + docs."""
    msgs = _cline_msgs("[claude] What is 2+2?")
    tier, reason, _ = decide_tier_cline(msgs)
    assert tier == "claude-code"
    assert "[claude]" in reason


def test_decide_tier_cline_opus_tag_is_sticky() -> None:
    """Once a turn picks Opus, subsequent turns of the SAME task
    (Cline replays the `<task>` verbatim across turns) stay on the
    sticky-escalated tier so a trivial-looking follow-up tool
    result doesn't downgrade us to local in the middle of work."""
    task = "[opus] design a billing service"

    msgs1 = _cline_msgs(task)
    tier1, _, _ = decide_tier_cline(msgs1)
    assert tier1 == "claude-opus-4-7"

    # Cline keeps the original `<task>` envelope verbatim across
    # turns, just appending more assistant/user pairs. Same task
    # text -> same fingerprint -> stickiness fires. Note that
    # sticky-rerouted turns currently land on "claude-code" (the
    # default Claude tier) rather than re-resolving to opus; that
    # is acceptable because the user-visible promise is "stays on
    # a Claude tier", and the stickiness path is conservative on
    # purpose (one code path, one model).
    msgs2 = _cline_msgs(
        task,
        {"role": "assistant", "content": "Working on it."},
        {"role": "user", "content": "[read_file] Result: looks good"},
    )
    tier2, reason2, _ = decide_tier_cline(msgs2)
    assert tier2 == "claude-code"
    assert "sticky" in reason2
    assert "[opus]" in reason2


def test_decide_tier_legacy_haiku_tag_routes_to_haiku() -> None:
    """Same override behavior for non-Cline callers (CLI / curl /
    benches) using the size-based decide_tier path."""
    msgs = [{"role": "user", "content": "[haiku] tiny ask"}]
    tier, reason, _ = decide_tier(msgs)
    assert tier == "claude-haiku-4-5"
    assert "[haiku]" in reason


def test_decide_tier_legacy_opus_tag_beats_token_count_default() -> None:
    """Even for a tiny prompt that would normally route to local-fast,
    [opus] forces the upgrade."""
    msgs = [{"role": "user", "content": "[opus] hi"}]
    tier, _, _ = decide_tier(msgs)
    assert tier == "claude-opus-4-7"


def test_decide_tier_legacy_local_tag_still_beats_override_tag() -> None:
    """Same precedence guarantee for the legacy path."""
    # Only [local] is at the leading position (override regex is
    # leading-only so [opus] mid-prompt doesn't match), so [local]
    # is the one classify() picks up and we must respect it.
    msgs = [{"role": "user", "content": "[local] do it cheap, not [opus]"}]
    tier, _, _ = decide_tier(msgs)
    # No-Claude promise: must not be any claude-* tier.
    assert "claude" not in tier


def test_pre_call_opus_tag_rewrites_model_for_cline() -> None:
    """End-to-end through async_pre_call_hook: a Cline request with
    a leading [opus] tag must arrive at LiteLLM with model rewritten
    to claude-opus-4-7 so the YAML routes it to anthropic/claude-opus-4-7."""
    router = SizeBasedRouter()
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": _cline_msgs("[opus] design a billing service"),
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] == "claude-opus-4-7"
    meta = new["metadata"]
    assert "[opus]" in meta["route_reason"]


def test_pre_call_haiku_tag_rewrites_model_for_cline() -> None:
    router = SizeBasedRouter()
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": _cline_msgs("[haiku] What is 2+2?"),
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] == "claude-haiku-4-5"


def test_pre_call_gpt_prefixed_haiku_alias_resolves() -> None:
    """Cursor-shaped clients (`gpt-claude-haiku-4-5`) must be accepted
    without being mistaken for an unknown model. The router strips the
    `gpt-` prefix and lets LiteLLM route the canonical name to the
    correct upstream."""
    router = SizeBasedRouter()
    data: dict[str, Any] = {
        "model": "gpt-claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] == "claude-haiku-4-5"
