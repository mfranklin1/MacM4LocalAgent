"""Tests for the tool-native redirect in the LiteLLM pre-call hook.

Cline executes only native OpenAI `tool_calls`; the qwen2.5-coder tiers
emit them as raw JSON text, which stalls the turn. The redirect moves any
tool-carrying request off a non-tool-native local tier onto the tool-native
tier (default `local-long`: qwen3-coder-next via ollama_chat/, tool-native
since the 2026-07-02 template fix).

The fixture pins the tier config explicitly, so these tests exercise the
redirect mechanics regardless of the shipped defaults; the defaults
themselves are asserted in test_shipped_defaults below.

These drive SizeBasedRouter.async_pre_call_hook end-to-end (resolved model,
so the hybrid-auto branch is skipped) plus unit tests of the helper.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import router.route_by_size as rbs
from router.route_by_size import SizeBasedRouter

_TOOLS = [{"type": "function", "function": {"name": "graph_stats", "parameters": {}}}]


def _router() -> SizeBasedRouter:
    r = SizeBasedRouter.__new__(SizeBasedRouter)
    r._register_active = lambda *a, **k: None  # type: ignore[attr-defined]
    return r


def _run(router: SizeBasedRouter, data: dict) -> dict:
    return asyncio.run(
        router.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )
    )


@pytest.fixture(autouse=True)
def _redirect_env(monkeypatch: Any) -> None:
    # Isolate the redirect from thinking-mode side effects and pin the tier
    # config so the tests don't depend on the process env / detected.env.
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", False)
    monkeypatch.setattr(rbs, "ENABLE_THINK_INJECTION", False)
    monkeypatch.setattr(rbs, "TOOL_NATIVE_LOCAL_TIER", "local-agent")
    monkeypatch.setattr(rbs, "_NON_TOOL_NATIVE_LOCAL", {"local-long"})
    monkeypatch.setenv("OLLAMA_TAG", "qwen3-coder-next:q4")


# --- end-to-end through the pre-call hook -----------------------------------

def test_tool_turn_on_local_long_redirects_to_local_agent() -> None:
    data = {"model": "local-long", "tools": _TOOLS,
            "messages": [{"role": "user", "content": "plan a refactor"}]}
    out = _run(_router(), data)
    assert out["model"] == "local-agent"
    assert out["metadata"]["tool_native_redirect"] is True
    assert "tool-native redirect" in out["metadata"]["route_reason"]


def test_toolless_turn_on_local_long_is_not_redirected() -> None:
    data = {"model": "local-long",
            "messages": [{"role": "user", "content": "plan a refactor"}]}
    out = _run(_router(), data)
    assert out["model"] == "local-long"
    assert "tool_native_redirect" not in out.get("metadata", {})


def test_claude_tool_turn_is_not_redirected() -> None:
    data = {"model": "claude-sonnet-5", "tools": _TOOLS,
            "messages": [{"role": "user", "content": "plan a refactor"}]}
    out = _run(_router(), data)
    assert out["model"] == "claude-sonnet-5"


def test_already_tool_native_tier_is_not_redirected() -> None:
    data = {"model": "local-agent", "tools": _TOOLS,
            "messages": [{"role": "user", "content": "plan a refactor"}]}
    out = _run(_router(), data)
    assert out["model"] == "local-agent"
    assert "tool_native_redirect" not in out.get("metadata", {})


def test_redirected_llama_does_not_get_think_injected(monkeypatch: Any) -> None:
    # Even with thinking mode ON, the post-redirect model is llama3.1, which
    # is not a Qwen3 model, so /think must NOT be injected.
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", True)
    data = {"model": "local-long", "tools": _TOOLS,
            "messages": [{"role": "user", "content": "plan a refactor"}]}
    out = _run(_router(), data)
    assert out["model"] == "local-agent"
    assert not out["messages"][-1]["content"].startswith("/think ")
    assert "qwen3_think_injected" not in out.get("metadata", {})


# --- helper unit tests ------------------------------------------------------

def test_helper_empty_tools_list_is_noop() -> None:
    data = {"model": "local-long", "tools": []}
    _router()._maybe_redirect_tools_to_native(data)
    assert data["model"] == "local-long"


def test_helper_non_list_tools_is_noop() -> None:
    data = {"model": "local-long", "tools": "not-a-list"}
    _router()._maybe_redirect_tools_to_native(data)
    assert data["model"] == "local-long"


def test_helper_disabled_when_tier_empty(monkeypatch: Any) -> None:
    monkeypatch.setattr(rbs, "TOOL_NATIVE_LOCAL_TIER", "")
    data = {"model": "local-long", "tools": _TOOLS}
    _router()._maybe_redirect_tools_to_native(data)
    assert data["model"] == "local-long"


def test_helper_strips_gpt_prefix_alias(monkeypatch: Any) -> None:
    # Defensive: if a gpt- mirror alias slips through, it still redirects.
    data = {"model": "gpt-local-long", "tools": _TOOLS}
    _router()._maybe_redirect_tools_to_native(data)
    assert data["model"] == "local-agent"


def test_helper_preserves_prior_route_reason(monkeypatch: Any) -> None:
    data = {"model": "local-long", "tools": _TOOLS,
            "metadata": {"route_reason": "cline-mode: complex"}}
    _router()._maybe_redirect_tools_to_native(data)
    assert "cline-mode: complex" in data["metadata"]["route_reason"]

