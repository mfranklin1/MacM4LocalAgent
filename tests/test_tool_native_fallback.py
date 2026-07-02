"""Tests for the tool-native fallback fixes.

Covers the failure chain found live on 2026-07-02 (cost.db requests
5031/5033/5034): a complex Cline turn routed to claude-code hit a 429,
LiteLLM's router-level fallback (which bypasses async_pre_call_hook and
therefore the tool-native redirect) landed the tool-carrying request on
qwen3-coder-next via the legacy ollama/ route, and the model emitted the
tool call as raw JSON text that Cline cannot execute.

Four fixes under test:
1. router: _is_tool_native_model / _annotate_router_fallback -- ledger
   marker + stderr warning when a router fallback lands a tool-carrying
   request on a non-tool-native model.
2. router: _record wiring -- the marker actually reaches the requests row.
3. router: _extract_user_task no-envelope fallback (native tool-calling
   Cline builds dropped the <task> envelope).
4. claude_proxy: bounded exponential 429 backoff honoring Retry-After.
"""

from __future__ import annotations

import sys
import time
import types
import unittest

import pytest

from router.route_by_size import (
    SizeBasedRouter,
    _annotate_router_fallback,
    _extract_user_task,
    _is_tool_native_model,
)

# ---- _is_tool_native_model ---------------------------------------------------


@pytest.mark.parametrize(
    "model,expected",
    [
        # Claude tiers are always tool-native.
        ("claude-code", True),
        ("anthropic/claude-sonnet-5", True),
        ("gpt-claude-code", True),
        # Legacy ollama/ route returns tool calls as raw JSON text even
        # for models that support tools.
        ("ollama/qwen3-coder-next:q4", False),
        # ollama_chat/ parses the envelope -- but only if the model emits
        # it. qwen3-coder-next does (post template fix); llama3.1 does;
        # qwen2.5-coder demonstrably does not (raw JSON, live-tested).
        ("ollama_chat/qwen3-coder-next:q4", True),
        ("ollama_chat/llama3.1:8b-instruct-q8_0", True),
        ("ollama_chat/qwen2.5-coder:32b", False),
        # Alias names: the qwen2.5-coder tiers are the configured
        # non-tool-native set; local-long is tool-native since the
        # 2026-07-02 template fix.
        ("local-long", True),
        ("gpt-local-long", True),
        ("local-agent", True),
        ("local-coder-14b", False),
        ("local-coder-32b", False),
        ("gpt-local-coder-32b", False),
    ],
)
def test_is_tool_native_model(model: str, expected: bool) -> None:
    assert _is_tool_native_model(model) is expected


# ---- _annotate_router_fallback -------------------------------------------------


def test_annotate_noop_without_decision() -> None:
    assert (
        _annotate_router_fallback("ollama/qwen3-coder-next:q4", None, True, "r")
        == "r"
    )
    assert (
        _annotate_router_fallback("ollama/qwen3-coder-next:q4", "", True, "r")
        == "r"
    )


def test_annotate_noop_when_decision_matches_served_tier() -> None:
    # Decided claude, served claude: no fallback fired.
    assert (
        _annotate_router_fallback(
            "anthropic/claude-sonnet-5", "claude-code", True, "complex"
        )
        == "complex"
    )
    # Decided local, served local: normal local turn.
    assert (
        _annotate_router_fallback(
            "ollama/qwen3-coder-next:q4", "local-long", True, "<= 128k"
        )
        == "<= 128k"
    )


def test_annotate_marks_fallback_with_tools_as_text_risk() -> None:
    # The exact production failure: claude decision, non-tool-native
    # local model served it, request carried tools.
    out = _annotate_router_fallback(
        "ollama/qwen3-coder-next:q4", "claude-code", True, "cline-mode: complex"
    )
    assert "cline-mode: complex | " in out
    assert "ROUTER-FALLBACK(claude-code->ollama/qwen3-coder-next:q4)" in out
    assert "TOOLS-AS-TEXT-RISK" in out


def test_annotate_marks_fallback_without_risk_for_tool_native_target() -> None:
    # After the config fix, fallbacks land on local-long (tool-native
    # since the template fix) -- the marker still appears (a fallback DID
    # fire) but no risk flag.
    out = _annotate_router_fallback(
        "ollama_chat/qwen3-coder-next:q4", "claude-code", True, "complex"
    )
    assert "ROUTER-FALLBACK(claude-code->ollama_chat/qwen3-coder-next:q4)" in out
    assert "TOOLS-AS-TEXT-RISK" not in out


def test_annotate_flags_qwen25_coder_even_via_ollama_chat() -> None:
    # qwen2.5-coder writes tool calls as raw JSON text despite declaring
    # the tools capability -- the risk flag must fire even though it's
    # served via ollama_chat/.
    out = _annotate_router_fallback(
        "ollama_chat/qwen2.5-coder:32b", "claude-code", True, "complex"
    )
    assert "TOOLS-AS-TEXT-RISK" in out


def test_annotate_marks_fallback_without_tools_without_risk() -> None:
    out = _annotate_router_fallback(
        "ollama/qwen3-coder-next:q4", "claude-code", False, "complex"
    )
    assert "ROUTER-FALLBACK" in out
    assert "TOOLS-AS-TEXT-RISK" not in out


def test_annotate_with_empty_reason_has_no_separator() -> None:
    out = _annotate_router_fallback(
        "ollama/qwen3-coder-next:q4", "claude-code", True, ""
    )
    assert out.startswith("ROUTER-FALLBACK(")
    assert " | " not in out


# ---- _record wiring (ledger row carries the marker) ----------------------------


def _kwargs(model: str, decision: str, tools: bool) -> dict:
    kw: dict = {
        "model": model,
        "litellm_params": {
            "metadata": {
                "route_reason": "cline-mode: complex: architecture/design language",
                "route_decision": decision,
            }
        },
    }
    if tools:
        kw["optional_params"] = {
            "tools": [{"type": "function", "function": {"name": "gortex__graph_stats"}}]
        }
    return kw


def _read_reasons(db_path) -> list[str]:
    import sqlite3

    conn = sqlite3.connect(db_path)
    try:
        return [r[0] for r in conn.execute("SELECT route_reason FROM requests")]
    finally:
        conn.close()


def test_record_annotates_fallback_row(tmp_db) -> None:
    router = SizeBasedRouter()
    router.log_success_event(
        kwargs=_kwargs("ollama/qwen3-coder-next:q4", "claude-code", tools=True),
        response_obj={"usage": {"prompt_tokens": 72092, "completion_tokens": 16}},
        start_time=time.time(),
        end_time=time.time() + 20.0,
    )
    reasons = _read_reasons(tmp_db)
    assert len(reasons) == 1
    assert "ROUTER-FALLBACK(claude-code->ollama/qwen3-coder-next:q4)" in reasons[0]
    assert "TOOLS-AS-TEXT-RISK" in reasons[0]


def test_record_leaves_normal_rows_unannotated(tmp_db) -> None:
    router = SizeBasedRouter()
    # Decision and served model agree (both local): no marker.
    router.log_success_event(
        kwargs=_kwargs("ollama/qwen3-coder-next:q4", "local-long", tools=True),
        response_obj={"usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        start_time=time.time(),
        end_time=time.time() + 1.0,
    )
    # No route_decision stamped at all (non-hybrid-auto traffic).
    router.log_success_event(
        kwargs={"model": "local-agent", "metadata": {"route_reason": "direct"}},
        response_obj={"usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        start_time=time.time(),
        end_time=time.time() + 1.0,
    )
    reasons = _read_reasons(tmp_db)
    assert len(reasons) == 2
    assert not any("ROUTER-FALLBACK" in r for r in reasons)


# ---- _extract_user_task: no-envelope Cline protocol -----------------------------


def test_extract_task_envelope_still_wins() -> None:
    msgs = [
        {"role": "user", "content": "<task>\nFix the bug\n</task>"},
    ]
    assert _extract_user_task(msgs) == "Fix the bug"


def test_extract_task_envelope_in_later_message_beats_first_plain_text() -> None:
    msgs = [
        {"role": "user", "content": "plain first message"},
        {"role": "user", "content": "<task>enveloped task</task>"},
    ]
    assert _extract_user_task(msgs) == "enveloped task"


def test_extract_task_falls_back_to_first_user_message() -> None:
    # Native tool-calling Cline builds send the task as plain text.
    msgs = [
        {"role": "system", "content": "You are Cline, ..."},
        {"role": "user", "content": "create a plan to better support the mdm updates"},
    ]
    assert (
        _extract_user_task(msgs)
        == "create a plan to better support the mdm updates"
    )


def test_extract_task_strips_environment_details() -> None:
    msgs = [
        {
            "role": "user",
            "content": "do the thing\n<environment_details># VSCode open tabs\n...</environment_details>",
        },
    ]
    assert _extract_user_task(msgs) == "do the thing"


def test_extract_task_none_for_empty_or_whitespace() -> None:
    assert _extract_user_task(None) is None
    assert _extract_user_task([]) is None
    assert _extract_user_task([{"role": "user", "content": "   "}]) is None
    assert _extract_user_task([{"role": "system", "content": "sys only"}]) is None


def test_extract_task_list_content_fallback() -> None:
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "list-shaped task"}]},
    ]
    assert _extract_user_task(msgs) == "list-shaped task"


# ---- claude_proxy: 429 backoff ---------------------------------------------------

# Stub uvicorn so the module imports without a server (mirrors
# tests/test_claude_proxy.py).
_uv = types.ModuleType("uvicorn")
_uv.run = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("uvicorn", _uv)

# Grab the REAL httpx client classes at import time: the retry tests patch
# httpx.AsyncClient (the module attribute srv reads), which would otherwise
# also hijack the ASGI test client used to drive the endpoint.
from httpx import ASGITransport as _RealASGITransport  # noqa: E402
from httpx import AsyncClient as _RealAsyncClient  # noqa: E402

import claude_proxy.server as srv  # noqa: E402


class TestRetryDelay(unittest.TestCase):
    def test_numeric_retry_after_wins(self) -> None:
        assert srv._retry_delay(0, "1.5") == 1.5

    def test_retry_after_clamped_to_cap(self) -> None:
        assert srv._retry_delay(0, "9999") == srv.RETRY_429_MAX_DELAY

    def test_negative_retry_after_floors_at_zero(self) -> None:
        assert srv._retry_delay(0, "-3") == 0.0

    def test_http_date_form_falls_back_to_exponential(self) -> None:
        assert (
            srv._retry_delay(0, "Wed, 01 Jul 2026 12:00:00 GMT")
            == srv.RETRY_429_BASE_DELAY
        )

    def test_exponential_growth_and_cap(self) -> None:
        base = srv.RETRY_429_BASE_DELAY
        assert srv._retry_delay(0, None) == base
        assert srv._retry_delay(1, None) == base * 2
        assert srv._retry_delay(2, None) == base * 4
        assert srv._retry_delay(10, None) == srv.RETRY_429_MAX_DELAY


class _FakeResp:
    def __init__(self, status_code: int, content: bytes = b"{}", headers: dict | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    async def aread(self) -> bytes:
        return self.content

    async def aiter_bytes(self):
        yield self.content


class _FakeStreamCtx:
    def __init__(self, resp: _FakeResp):
        self._resp = resp

    async def __aenter__(self) -> _FakeResp:
        return self._resp

    async def __aexit__(self, *exc) -> bool:
        return False


class _FakeClient:
    """Stands in for httpx.AsyncClient; serves canned responses in order."""

    responses: list[_FakeResp] = []
    calls: int = 0

    def __init__(self, *a, **kw) -> None:
        pass

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def post(self, *a, **kw) -> _FakeResp:
        cls = type(self)
        resp = cls.responses[cls.calls]
        cls.calls += 1
        return resp

    def stream(self, *a, **kw) -> _FakeStreamCtx:
        cls = type(self)
        resp = cls.responses[cls.calls]
        cls.calls += 1
        return _FakeStreamCtx(resp)


class TestSubscription429Retry(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _FakeClient.responses = []
        _FakeClient.calls = 0
        self._orig_client = srv.httpx.AsyncClient
        srv.httpx.AsyncClient = _FakeClient  # type: ignore[assignment]
        self._orig_sleep = srv.asyncio.sleep
        self.sleeps: list[float] = []

        async def _fake_sleep(d: float) -> None:
            self.sleeps.append(d)

        srv.asyncio.sleep = _fake_sleep  # type: ignore[assignment]
        self._orig_token = srv._get_subscription_token
        srv._get_subscription_token = lambda: "test-token"  # type: ignore[assignment]

    def tearDown(self) -> None:
        srv.httpx.AsyncClient = self._orig_client
        srv.asyncio.sleep = self._orig_sleep
        srv._get_subscription_token = self._orig_token

    async def _post(self, stream: bool = False):
        body = {"model": "claude-sonnet-5", "messages": [], "stream": stream}
        async with _RealAsyncClient(
            transport=_RealASGITransport(app=srv.app), base_url="http://test"
        ) as client:
            return await client.post("/subscription/v1/messages", json=body)

    async def test_retries_through_429_then_succeeds(self) -> None:
        _FakeClient.responses = [
            _FakeResp(429),
            _FakeResp(429),
            _FakeResp(200, b'{"ok": true}'),
        ]
        resp = await self._post()
        assert resp.status_code == 200
        assert _FakeClient.calls == 3
        # Exponential: base, base*2.
        assert self.sleeps == [
            srv.RETRY_429_BASE_DELAY,
            srv.RETRY_429_BASE_DELAY * 2,
        ]

    async def test_gives_up_after_budget_and_returns_429(self) -> None:
        _FakeClient.responses = [_FakeResp(429)] * (srv.RETRY_429_ATTEMPTS + 1)
        resp = await self._post()
        assert resp.status_code == 429
        # Initial attempt + RETRY_429_ATTEMPTS retries, then surface the
        # 429 so LiteLLM's fallback chain still engages.
        assert _FakeClient.calls == srv.RETRY_429_ATTEMPTS + 1

    async def test_honors_numeric_retry_after(self) -> None:
        _FakeClient.responses = [
            _FakeResp(429, headers={"retry-after": "1.5"}),
            _FakeResp(200, b'{"ok": true}'),
        ]
        resp = await self._post()
        assert resp.status_code == 200
        assert self.sleeps == [1.5]

    async def test_non_429_error_is_not_retried(self) -> None:
        _FakeClient.responses = [_FakeResp(500, b'{"error": "boom"}')]
        resp = await self._post()
        assert resp.status_code == 500
        assert _FakeClient.calls == 1
        assert self.sleeps == []

    async def test_streaming_non_429_error_returned_without_retry(self) -> None:
        _FakeClient.responses = [_FakeResp(500, b'{"error": "boom"}')]
        resp = await self._post(stream=True)
        assert resp.status_code == 500
        assert _FakeClient.calls == 1
        assert self.sleeps == []

    async def test_streaming_429_retried_then_streams(self) -> None:
        _FakeClient.responses = [
            _FakeResp(429),
            _FakeResp(200, b"data: {}\n\n"),
        ]
        resp = await self._post(stream=True)
        assert resp.status_code == 200
        assert _FakeClient.calls == 2
        assert self.sleeps == [srv.RETRY_429_BASE_DELAY]


if __name__ == "__main__":
    unittest.main()


# ---- shipped defaults ------------------------------------------------------


def test_shipped_defaults_route_tools_to_local_long() -> None:
    """Guard the defaults that make the primary 80B tier the tool target.

    Lives here (not test_tool_native_redirect.py) because that module's
    autouse fixture monkeypatches these very attributes. Skip the
    assertion when the process env overrides the default.
    """
    import os

    import router.route_by_size as rbs

    if not os.environ.get("CLINE_TOOL_TIER"):
        assert rbs.TOOL_NATIVE_LOCAL_TIER == "local-long"
    if not os.environ.get("CLINE_NON_TOOL_TIERS"):
        assert rbs._NON_TOOL_NATIVE_LOCAL == {"local-coder-14b", "local-coder-32b"}
