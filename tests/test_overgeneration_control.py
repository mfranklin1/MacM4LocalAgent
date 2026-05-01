"""Unit tests for router/overgeneration_control.py.

These exercise the pure-functional transforms in isolation, without
LiteLLM or any network round-trip. The same module is wired into the
LiteLLM pre-call hook in router/route_by_size.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from router.overgeneration_control import (
    LOCAL_FIXUP_NUDGE,
    LOCAL_MAX_TOKENS_DEFAULT,
    LOCAL_MAX_TOKENS_FIXUP,
    LOCAL_STOP_SEQUENCES,
    LOCAL_SYSTEM_NUDGE,
    apply_all,
    apply_multi_turn_tighten,
    apply_static_guardrail,
    _is_local,
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
