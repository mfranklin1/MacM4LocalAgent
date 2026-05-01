"""Unit tests for the size + complexity-based router callback."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from router.complexity_classifier import classify
from router.route_by_size import (
    SizeBasedRouter,
    decide_tier,
    _estimate_tokens,
    _flat_prompt,
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


# ---- decide_tier --------------------------------------------------------------

def test_decide_tier_routes_small_to_fast() -> None:
    msgs = [{"role": "user", "content": "what does this regex do?"}]
    model, reason, tokens = decide_tier(msgs)
    assert model == "local-fast"
    assert tokens <= ROUTE_FAST_MAX


def test_decide_tier_routes_medium_to_long() -> None:
    chars = (ROUTE_FAST_MAX + 1000) * 4   # comfortably above the fast limit
    msgs = [{"role": "user", "content": "x" * chars}]
    model, reason, tokens = decide_tier(msgs)
    assert model == "local-long"
    assert ROUTE_FAST_MAX < tokens <= ROUTE_LONG_MAX


def test_decide_tier_routes_huge_to_claude() -> None:
    chars = (ROUTE_LONG_MAX + 5000) * 4
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
