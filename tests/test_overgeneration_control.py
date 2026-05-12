"""Unit tests for router/overgeneration_control.py.

These exercise the pure-functional transforms in isolation, without
LiteLLM or any network round-trip. The same module is wired into the
LiteLLM pre-call hook in router/route_by_size.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from router.overgeneration_control import (
    CLINE_STOP_SEQUENCES,
    LOCAL_FIXUP_NUDGE,
    LOCAL_MAX_TOKENS_DEFAULT,
    LOCAL_MAX_TOKENS_FIXUP,
    LOCAL_STOP_SEQUENCES,
    LOCAL_SYSTEM_NUDGE,
    apply_all,
    apply_multi_turn_tighten,
    apply_static_guardrail,
    _is_local,
    _looks_like_cline,
    _looks_like_fixup_turn,
)


# ---------- _is_local -----------------------------------------------------

@pytest.mark.parametrize(
    "model,expected",
    [
        ("local-fast",                                  True),
        ("local-long",                                  True),
        ("ollama/qwen3-coder-next:q4_K_M",              True),
        ("openai/mlx-community/Qwen2.5-Coder-7B",       True),
        ("mlx-community/Qwen2.5",                       True),
        ("claude-code",                                 False),
        ("anthropic/claude-sonnet-4-6",                 False),
        ("hybrid-auto",                                 False),
        ("",                                            False),
        (None,                                          False),
        # Cursor-friendly `gpt-` prefixed aliases route to the same
        # upstream models, so the over-gen controls must still
        # classify them by their canonical local/remote nature even
        # if the route_by_size hook hasn't stripped the prefix yet.
        ("gpt-local-fast",                              True),
        ("gpt-local-long",                              True),
        ("gpt-local-agent",                             True),
        ("gpt-claude-code",                             False),
        ("gpt-hybrid-auto",                             False),
    ],
)
def test_is_local_classification(model: Any, expected: bool) -> None:
    assert _is_local(model) is expected


# ---------- _looks_like_fixup_turn ---------------------------------------

def test_single_user_message_is_not_fixup() -> None:
    msgs = [{"role": "user", "content": "write me a quicksort"}]
    assert _looks_like_fixup_turn(msgs) is False


def test_two_message_exchange_is_not_fixup() -> None:
    # Need at least 3 messages (user, assistant, user) for fixup shape.
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _looks_like_fixup_turn(msgs) is False


def test_three_messages_with_assistant_code_is_fixup() -> None:
    msgs = [
        {"role": "user", "content": "implement quotas"},
        {"role": "assistant", "content": "Here:\n\n```python:quotas.py\n...\n```"},
        {"role": "user", "content": "tests fail, fix it"},
    ]
    assert _looks_like_fixup_turn(msgs) is True


def test_three_messages_without_code_in_assistant_is_not_fixup() -> None:
    msgs = [
        {"role": "user", "content": "what is monad"},
        {"role": "assistant", "content": "A monad is a..."},
        {"role": "user", "content": "tell me more"},
    ]
    assert _looks_like_fixup_turn(msgs) is False


def test_trailing_assistant_message_is_not_fixup() -> None:
    msgs = [
        {"role": "user", "content": "implement quotas"},
        {"role": "assistant", "content": "```python:q.py\n...\n```"},
        # no follow-up user message yet
    ]
    assert _looks_like_fixup_turn(msgs) is False


def test_list_content_with_text_parts_is_supported() -> None:
    msgs = [
        {"role": "user",
         "content": [{"type": "text", "text": "ship it"}]},
        {"role": "assistant",
         "content": [{"type": "text", "text": "```python:x.py\n...\n```"}]},
        {"role": "user",
         "content": [{"type": "text", "text": "tests fail"}]},
    ]
    assert _looks_like_fixup_turn(msgs) is True


# ---------- apply_static_guardrail ---------------------------------------

def _basic_local_data() -> dict[str, Any]:
    return {
        "model": "local-long",
        "messages": [
            {"role": "user", "content": "implement a quota module"},
        ],
    }


def test_static_clamps_unbounded_max_tokens_for_local() -> None:
    data = _basic_local_data()
    apply_static_guardrail(data)
    assert data["max_tokens"] == LOCAL_MAX_TOKENS_DEFAULT


def test_static_does_not_raise_lower_max_tokens() -> None:
    data = _basic_local_data()
    data["max_tokens"] = 1024
    apply_static_guardrail(data)
    assert data["max_tokens"] == 1024


def test_static_lowers_higher_max_tokens() -> None:
    data = _basic_local_data()
    data["max_tokens"] = 32768
    apply_static_guardrail(data)
    assert data["max_tokens"] == LOCAL_MAX_TOKENS_DEFAULT


def test_static_extends_existing_stop_sequences_without_dup() -> None:
    data = _basic_local_data()
    data["stop"] = ["\n```python:"]  # already one of ours
    apply_static_guardrail(data)
    # No duplicates, our second sequence should still get appended.
    assert data["stop"].count("\n```python:") == 1
    assert "\n```py:" in data["stop"]


def test_static_caps_stop_at_four() -> None:
    data = _basic_local_data()
    data["stop"] = ["a", "b", "c", "d"]
    apply_static_guardrail(data)
    assert len(data["stop"]) == 4


def test_static_prepends_system_nudge_when_absent() -> None:
    data = _basic_local_data()
    apply_static_guardrail(data)
    assert data["messages"][0]["role"] == "system"
    assert data["messages"][0]["content"] == LOCAL_SYSTEM_NUDGE


def test_static_does_not_double_inject_system_nudge() -> None:
    data = _basic_local_data()
    data["messages"] = [
        {"role": "system", "content": "you are an existing system message"},
        *data["messages"],
    ]
    apply_static_guardrail(data)
    # Still only one system message, and it wasn't replaced.
    system_msgs = [m for m in data["messages"] if m.get("role") == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "you are an existing system message"


def test_static_no_op_for_claude_when_only_for_local() -> None:
    data = {"model": "claude-code", "messages": [{"role": "user", "content": "hi"}]}
    before = dict(data)
    apply_static_guardrail(data)
    assert data == before


def test_static_runs_for_claude_if_only_for_local_disabled() -> None:
    data = {"model": "claude-code", "messages": [{"role": "user", "content": "hi"}]}
    apply_static_guardrail(data, only_for_local=False)
    assert data["max_tokens"] == LOCAL_MAX_TOKENS_DEFAULT


# ---------- apply_multi_turn_tighten -------------------------------------

def _fixup_local_data() -> dict[str, Any]:
    return {
        "model": "local-long",
        "messages": [
            {"role": "user", "content": "implement quotas"},
            {"role": "assistant",
             "content": "Sure:\n\n```python:quotas.py\nbody\n```\n"},
            {"role": "user", "content": "tests fail with assert 49 == 50"},
        ],
    }


def test_multi_turn_clamps_max_tokens_on_fixup() -> None:
    data = _fixup_local_data()
    apply_multi_turn_tighten(data)
    assert data["max_tokens"] == LOCAL_MAX_TOKENS_FIXUP


def test_multi_turn_appends_nudge_to_last_user() -> None:
    data = _fixup_local_data()
    apply_multi_turn_tighten(data)
    last = data["messages"][-1]
    assert last["role"] == "user"
    assert LOCAL_FIXUP_NUDGE in last["content"]
    assert "tests fail" in last["content"]  # original preserved


def test_multi_turn_no_op_on_single_turn() -> None:
    data = _basic_local_data()
    before_msgs = list(data["messages"])
    apply_multi_turn_tighten(data)
    assert data["messages"] == before_msgs
    assert "max_tokens" not in data


def test_multi_turn_no_op_for_claude() -> None:
    data = _fixup_local_data()
    data["model"] = "claude-code"
    before_max = data.get("max_tokens")
    apply_multi_turn_tighten(data)
    assert data.get("max_tokens") == before_max
    # Last user message untouched.
    assert "tests fail with assert 49 == 50" == data["messages"][-1]["content"]


def test_multi_turn_clamps_only_when_existing_is_higher() -> None:
    data = _fixup_local_data()
    data["max_tokens"] = 512  # already tighter than the fixup bound
    apply_multi_turn_tighten(data)
    assert data["max_tokens"] == 512


# ---------- apply_all combinator -----------------------------------------

def test_apply_all_chains_both_strategies_on_fixup_turn() -> None:
    data = _fixup_local_data()
    apply_all(data)
    # Static guardrail effects:
    assert data["messages"][0]["role"] == "system"
    assert "stop" in data
    # Multi-turn tightening should have lowered max_tokens past
    # the static cap.
    assert data["max_tokens"] == LOCAL_MAX_TOKENS_FIXUP
    # And appended its nudge to the LAST user message.
    assert LOCAL_FIXUP_NUDGE in data["messages"][-1]["content"]


def test_apply_all_disable_flags_make_each_strategy_inert() -> None:
    data = _fixup_local_data()
    apply_all(data, enable_static=False, enable_multi_turn=False)
    assert "stop" not in data
    assert data["messages"][0]["role"] == "user"  # no system prepended
    assert "max_tokens" not in data
    assert LOCAL_FIXUP_NUDGE not in data["messages"][-1]["content"]


def test_apply_all_does_not_raise_on_garbage_input() -> None:
    # Empty dict, missing keys, weird content shapes -- nothing should
    # raise. The proxy must keep serving even if the request is odd.
    apply_all({})
    apply_all({"model": "local-long"})
    apply_all({"model": None, "messages": None})
    apply_all({"model": "local-long", "messages": [{"role": "user"}]})
    apply_all({"model": "local-long",
               "messages": [{"role": "user", "content": 12345}]})


# ---------- Cline-shape detection ----------------------------------------

def _cline_messages(*, with_replace_in_file: bool = True) -> list[dict[str, Any]]:
    """A minimal fixture that mimics Cline's harness shape.

    The detector keys off two stable substrings in the system prompt:
    'You are Cline,' and the XML tool-tag descriptors. We include the
    `<replace_in_file>` tag in the system prompt by default since it
    is what triggered the original bug we're guarding against.
    """
    sysprompt_parts = ["You are Cline, a highly skilled software engineer."]
    if with_replace_in_file:
        sysprompt_parts.append(
            "Use <replace_in_file>...</replace_in_file> to modify files."
        )
    return [
        {"role": "system", "content": " ".join(sysprompt_parts)},
        {"role": "user", "content": "<task>Append a line to spec.txt</task>"},
    ]


def test_looks_like_cline_detects_stable_fingerprints() -> None:
    assert _looks_like_cline(_cline_messages()) is True


def test_looks_like_cline_handles_list_content_form() -> None:
    msgs = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are Cline, expert engineer."},
                {"type": "text", "text": "<replace_in_file>...</replace_in_file>"},
            ],
        },
        {"role": "user", "content": "do a thing"},
    ]
    assert _looks_like_cline(msgs) is True


def test_looks_like_cline_returns_false_for_non_cline() -> None:
    assert _looks_like_cline(None) is False
    assert _looks_like_cline([]) is False
    assert _looks_like_cline([{"role": "user", "content": "hi"}]) is False
    assert _looks_like_cline([{"role": "system", "content": "Be helpful."}]) is False
    # No system message, just a user message that mentions Cline tags.
    assert _looks_like_cline(
        [{"role": "user", "content": "<replace_in_file>x</replace_in_file>"}]
    ) is False


# ---------- Cline-specific stop-sequence swap ----------------------------

def test_static_guardrail_uses_cline_stops_for_cline_traffic() -> None:
    data = {"model": "local-long", "messages": _cline_messages()}
    apply_static_guardrail(data)
    stops = data.get("stop") or []
    # Cline tag stops should be present, not the python-fence stops.
    assert "</replace_in_file>" in stops
    assert "</attempt_completion>" in stops
    assert "\n```python:" not in stops
    assert "\n```py:" not in stops


def test_static_guardrail_uses_python_fence_stops_for_non_cline_local() -> None:
    data = {
        "model": "local-long",
        "messages": [{"role": "user", "content": "Refactor my function"}],
    }
    apply_static_guardrail(data)
    stops = data.get("stop") or []
    assert "\n```python:" in stops
    assert "</replace_in_file>" not in stops


def test_static_guardrail_does_not_inject_system_nudge_for_cline() -> None:
    """Cline already ships a ~13K-token system prompt of its own; adding
    ours on top is at best wasted tokens and at worst confuses the
    harness's tool-parsing rules."""
    data = {"model": "local-long", "messages": _cline_messages()}
    n_msgs_before = len(data["messages"])
    apply_static_guardrail(data)
    assert len(data["messages"]) == n_msgs_before
    # The Cline system prompt must still be at index 0 untouched.
    assert "You are Cline," in data["messages"][0]["content"]
    # And our nudge must NOT have been concatenated either.
    assert LOCAL_SYSTEM_NUDGE not in data["messages"][0]["content"]


def test_static_guardrail_still_caps_max_tokens_for_cline() -> None:
    data = {"model": "local-long", "messages": _cline_messages()}
    apply_static_guardrail(data)
    assert data["max_tokens"] == LOCAL_MAX_TOKENS_DEFAULT


def test_static_guardrail_explicit_stop_sequences_override_cline_default() -> None:
    """Caller-provided stop_sequences win; the Cline auto-swap only
    applies when the caller passed None (i.e. trusted us to pick)."""
    data = {"model": "local-long", "messages": _cline_messages()}
    apply_static_guardrail(data, stop_sequences=["END"])
    assert data["stop"] == ["END"]


def test_static_guardrail_caller_extends_stops_for_cline() -> None:
    """If the caller already supplied a `stop` list (e.g. Cline itself
    via the API request), the existing entries are kept and our Cline
    stops are appended -- both up to the 4-stop cap."""
    data = {
        "model": "local-long",
        "messages": _cline_messages(),
        "stop": ["</read_file>"],
    }
    apply_static_guardrail(data)
    stops = data["stop"]
    assert stops[0] == "</read_file>"  # original first
    assert "</replace_in_file>" in stops
    assert len(stops) <= 4


def test_multi_turn_tighten_no_op_for_cline() -> None:
    """Cline traffic must not get the LOCAL_FIXUP_NUDGE appended to its
    last user message, even when the conversation looks fix-up-shaped
    by accident (multi-turn with assistant code earlier)."""
    msgs = _cline_messages()
    # Add an assistant turn that contains a python fence (would
    # normally trigger the fixup detector) and then a follow-up user
    # message.
    msgs.append({
        "role": "assistant",
        "content": "Here's the code:\n```python\nprint('hi')\n```",
    })
    msgs.append({"role": "user", "content": "Fix the import"})
    data = {"model": "local-long", "messages": msgs}
    last_user_before = msgs[-1]["content"]
    apply_multi_turn_tighten(data)
    assert msgs[-1]["content"] == last_user_before
    assert LOCAL_FIXUP_NUDGE not in msgs[-1]["content"]


def test_cline_stop_sequences_constant_shape() -> None:
    """Pin the exact stop list so accidental edits to the constant get
    caught by the test suite.

    Ordering reflects the cost of an over-generation past each tag:
      1. </attempt_completion> stops hallucinated next-turn content
         (worst case: 3-5x output token inflation)
      2. </replace_in_file> -- most-frequent edit tool
      3. </write_to_file> -- bulk file write
      4. </execute_command> -- shell tool; over-generation past this
         can produce misleading "next command" prose

    </read_file> is intentionally excluded: over-generation there is
    benign (just verbose prose) and we are hard-capped at 4 stops.
    """
    assert CLINE_STOP_SEQUENCES == [
        "</attempt_completion>",
        "</replace_in_file>",
        "</write_to_file>",
        "</execute_command>",
    ]


# ---- Plan / Act mode detection + plan max_tokens cap ---------------------

def _cline_plan_mode_messages() -> list[dict[str, Any]]:
    """A Cline system prompt that documents the plan_mode_respond tool."""
    sysprompt = (
        "You are Cline, a highly skilled software engineer. "
        "Use <replace_in_file>...</replace_in_file> to modify files. "
        "## plan_mode_respond\n"
        "Description: Respond to the user's question or message in PLAN MODE..."
    )
    return [
        {"role": "system", "content": sysprompt},
        {"role": "user", "content": "<task>Design a billing service</task>"},
    ]


def test_looks_like_cline_plan_mode_true_when_tool_in_system() -> None:
    from router.overgeneration_control import _looks_like_cline_plan_mode
    assert _looks_like_cline_plan_mode(_cline_plan_mode_messages()) is True


def test_looks_like_cline_plan_mode_false_in_act_mode() -> None:
    from router.overgeneration_control import _looks_like_cline_plan_mode
    assert _looks_like_cline_plan_mode(_cline_messages()) is False


def test_looks_like_cline_plan_mode_false_for_non_cline() -> None:
    from router.overgeneration_control import _looks_like_cline_plan_mode
    assert _looks_like_cline_plan_mode([
        {"role": "user", "content": "plan_mode_respond just a word"}
    ]) is False


def test_static_guardrail_caps_plan_mode_at_1024() -> None:
    """Plan mode requests get LOCAL_MAX_TOKENS_PLAN (1024) instead of
    the default 6144 ceiling."""
    from router.overgeneration_control import LOCAL_MAX_TOKENS_PLAN
    data = {
        "model": "local-long",
        "messages": _cline_plan_mode_messages(),
        "max_tokens": 8192,
    }
    apply_static_guardrail(data)
    assert data["max_tokens"] == LOCAL_MAX_TOKENS_PLAN


def test_static_guardrail_act_mode_keeps_default_cap() -> None:
    """Act-mode Cline traffic still uses LOCAL_MAX_TOKENS_DEFAULT."""
    from router.overgeneration_control import LOCAL_MAX_TOKENS_DEFAULT
    data = {
        "model": "local-long",
        "messages": _cline_messages(),
        "max_tokens": 8192,
    }
    apply_static_guardrail(data)
    assert data["max_tokens"] == LOCAL_MAX_TOKENS_DEFAULT


def test_static_guardrail_plan_mode_does_not_raise_lower_cap() -> None:
    """If a caller already pinned max_tokens below our plan cap, leave
    it alone."""
    data = {
        "model": "local-long",
        "messages": _cline_plan_mode_messages(),
        "max_tokens": 256,
    }
    apply_static_guardrail(data)
    assert data["max_tokens"] == 256


# ---- M6: Qwen3 /think directive injection ------------------------------------

def test_inject_qwen3_think_prepends_directive_to_string_content() -> None:
    from router.overgeneration_control import inject_qwen3_think_directive
    data = {
        "model": "local-long",
        "messages": [
            {"role": "system", "content": "You are Cline."},
            {"role": "user", "content": "<task>Fix the failing test.</task>"},
        ],
    }
    inject_qwen3_think_directive(data)
    assert data["messages"][-1]["content"].startswith("/think ")
    assert "<task>" in data["messages"][-1]["content"]


def test_inject_qwen3_think_idempotent_when_directive_present() -> None:
    """Already-prefixed messages must not be double-injected."""
    from router.overgeneration_control import inject_qwen3_think_directive
    data = {
        "model": "local-long",
        "messages": [
            {"role": "user", "content": "/think Fix the failing test."},
        ],
    }
    inject_qwen3_think_directive(data)
    assert data["messages"][-1]["content"].count("/think") == 1


def test_inject_qwen3_think_respects_no_think_opt_out() -> None:
    """If the user explicitly disabled thinking with /no_think, we
    must not flip it back on."""
    from router.overgeneration_control import inject_qwen3_think_directive
    data = {
        "model": "local-long",
        "messages": [
            {"role": "user", "content": "/no_think Fix the failing test."},
        ],
    }
    inject_qwen3_think_directive(data)
    assert data["messages"][-1]["content"].startswith("/no_think")
    assert "/think " not in data["messages"][-1]["content"]


def test_inject_qwen3_think_handles_list_content() -> None:
    """OpenAI content-parts form: prepend to the first text part."""
    from router.overgeneration_control import inject_qwen3_think_directive
    data = {
        "model": "local-long",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "<task>Fix bug.</task>"},
                    {"type": "image_url", "image_url": "data:..."},
                ],
            }
        ],
    }
    inject_qwen3_think_directive(data)
    first_text = data["messages"][-1]["content"][0]["text"]
    assert first_text.startswith("/think ")


def test_inject_qwen3_think_targets_last_user_message() -> None:
    """When there are multiple user messages we prepend to the LAST one
    (the one the model is being asked to respond to)."""
    from router.overgeneration_control import inject_qwen3_think_directive
    data = {
        "model": "local-long",
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
    }
    inject_qwen3_think_directive(data)
    assert data["messages"][0]["content"] == "first question"
    assert data["messages"][-1]["content"].startswith("/think second question")


def test_inject_qwen3_think_safe_on_empty_messages() -> None:
    from router.overgeneration_control import inject_qwen3_think_directive
    for data in (
        {"model": "local-long", "messages": []},
        {"model": "local-long"},  # missing messages key
        {},  # missing both
    ):
        # Must not raise.
        inject_qwen3_think_directive(data)
