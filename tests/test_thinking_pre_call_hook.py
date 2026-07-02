"""Integration tests for thinking-mode wiring in the LiteLLM pre-call hook.

These drive SizeBasedRouter.async_pre_call_hook end-to-end (with a
resolved model, so the hybrid-auto routing branch is skipped) and assert
that the thinking transforms are applied:
  - local Qwen3 tiers get /think injected (always-on) + a max_tokens floor
  - Claude tiers get Anthropic extended thinking enabled
  - non-Qwen3 local tiers are NOT touched
  - ROUTER_THINKING=0 disables everything
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import router.route_by_size as rbs
from router.route_by_size import SizeBasedRouter


def _router() -> SizeBasedRouter:
    """A SizeBasedRouter without DB/in-flight setup (we no-op registration)."""
    r = SizeBasedRouter.__new__(SizeBasedRouter)
    r._register_active = lambda *a, **k: None  # type: ignore[attr-defined]
    return r


def _run(router: SizeBasedRouter, data: dict) -> dict:
    return asyncio.run(
        router.async_pre_call_hook(
            user_api_key_dict=None, cache=None, data=data, call_type="completion"
        )
    )


@pytest.fixture
def qwen3_env(monkeypatch: Any) -> None:
    monkeypatch.setenv("MLX_LOCAL_DIR", "/models/mlx-community_Qwen3-Coder-Next-4bit")
    monkeypatch.setenv("MLX_REPO", "mlx-community/Qwen3-Coder-Next-4bit")
    monkeypatch.setenv("OLLAMA_TAG", "qwen3-coder-next:q4")


def test_pre_call_injects_think_for_qwen3_local(monkeypatch: Any, qwen3_env: None) -> None:
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", True)
    data = {"model": "local-long", "messages": [{"role": "user", "content": "fix the bug"}]}
    out = _run(_router(), data)
    assert out["messages"][-1]["content"].startswith("/think ")
    assert out["metadata"]["qwen3_think_injected"] is True
    # Floor re-asserted above the over-generation cap so the trace fits.
    assert out["max_tokens"] >= rbs._local_think_max_tokens()


def test_pre_call_enables_claude_thinking(monkeypatch: Any) -> None:
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", True)
    data = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "design a billing service"}],
        "top_p": 0.9,
    }
    out = _run(_router(), data)
    assert out["thinking"] == {"type": "adaptive"}
    assert out["output_config"]["effort"] in {"low", "medium", "high", "xhigh", "max"}
    assert "temperature" not in out  # sampling params stripped (400 on Opus 4.7+)
    assert "top_p" not in out
    assert out["metadata"]["claude_thinking_enabled"] is True


def test_pre_call_skips_thinking_for_claude_haiku(monkeypatch: Any) -> None:
    """Haiku tiers don't support adaptive thinking — injecting it 400s, so skip."""
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", True)
    data = {
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": "hi"}],
    }
    out = _run(_router(), data)
    assert "thinking" not in out
    assert not out.get("metadata", {}).get("claude_thinking_enabled")


def test_pre_call_skips_thinking_for_claude_code_alias_on_haiku(monkeypatch: Any) -> None:
    """The claude-code alias resolves to Haiku 4.5 upstream (subscription
    policy limit, 2026-07-02) — the guard must check the RESOLVED model.
    Checking the alias string alone injected adaptive thinking and 400'd
    every escalated turn."""
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", True)
    monkeypatch.setattr(rbs, "CLAUDE_CODE_UPSTREAM_MODEL", "claude-haiku-4-5")
    data = {
        "model": "claude-code",
        "messages": [{"role": "user", "content": "design a billing service"}],
    }
    out = _run(_router(), data)
    assert "thinking" not in out
    assert not out.get("metadata", {}).get("claude_thinking_enabled")


def test_pre_call_enables_thinking_for_claude_code_alias_on_sonnet(monkeypatch: Any) -> None:
    """When the apikey setting repoints claude-code at a thinking-capable
    model (CLAUDE_CODE_MODEL), the guard re-enables thinking injection."""
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", True)
    monkeypatch.setattr(rbs, "CLAUDE_CODE_UPSTREAM_MODEL", "claude-sonnet-5")
    data = {
        "model": "claude-code",
        "messages": [{"role": "user", "content": "design a billing service"}],
    }
    out = _run(_router(), data)
    assert out["thinking"] == {"type": "adaptive"}
    assert out["metadata"]["claude_thinking_enabled"] is True


def test_resolved_claude_model(monkeypatch: Any) -> None:
    assert rbs._resolved_claude_model("claude-haiku-4-5") == "claude-haiku-4-5"
    assert rbs._resolved_claude_model("claude-sonnet-5") == "claude-sonnet-5"
    assert rbs._resolved_claude_model("claude-code") == rbs.CLAUDE_CODE_UPSTREAM_MODEL
    assert rbs._resolved_claude_model("gpt-claude-code") == rbs.CLAUDE_CODE_UPSTREAM_MODEL
    assert rbs._resolved_claude_model(None) == ""


def test_pre_call_skips_think_for_non_qwen3_local(monkeypatch: Any, qwen3_env: None) -> None:
    """local-agent is llama3.1 — must never receive /think."""
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", True)
    data = {"model": "local-agent", "messages": [{"role": "user", "content": "hi"}]}
    out = _run(_router(), data)
    assert not out["messages"][-1]["content"].startswith("/think")
    assert not out.get("metadata", {}).get("qwen3_think_injected")


def test_pre_call_legacy_local_tag_path(monkeypatch: Any, qwen3_env: None) -> None:
    """With thinking mode OFF but the legacy ROUTER_THINK_INJECTION on,
    /think still fires on the explicit [local] tag (back-compat)."""
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", False)
    monkeypatch.setattr(rbs, "ENABLE_THINK_INJECTION", True)
    data = {
        "model": "local-long",
        "messages": [{"role": "user", "content": "fix the bug"}],
        "metadata": {"route_reason": "cline+override: explicit [local] tag"},
    }
    out = _run(_router(), data)
    assert out["messages"][-1]["content"].startswith("/think ")
    assert out["metadata"]["qwen3_think_injected"] is True


def test_pre_call_kill_switch_disables_all_thinking(monkeypatch: Any, qwen3_env: None) -> None:
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", False)
    monkeypatch.setattr(rbs, "ENABLE_THINK_INJECTION", False)
    data = {"model": "local-long", "messages": [{"role": "user", "content": "fix the bug"}]}
    out = _run(_router(), data)
    assert not out["messages"][-1]["content"].startswith("/think")
    assert not out.get("metadata", {}).get("qwen3_think_injected")

    data_c = {"model": "claude-opus-4-7", "messages": [{"role": "user", "content": "hi"}]}
    out_c = _run(_router(), data_c)
    assert "thinking" not in out_c
