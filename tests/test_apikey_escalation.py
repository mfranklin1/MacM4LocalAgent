"""Tests for API-key escalation (Cline setting -> Sonnet/Opus/Fable).

The Team subscription OAuth token only serves Haiku on the raw API
(policy-429 for Sonnet/Opus/Fable, verified 2026-07-02). Escalation to a
bigger Claude model therefore authenticates with the org API key from the
macOS keychain (item `anthropic-api-key`, the same one
~/.claude/set-anthropic-env.sh reads). The secret never transits Cline or
LiteLLM: Cline sends only a plain `x-claude-escalation-model` header; the
router maps it to an /apikey-backed alias when claude-proxy reports a key.

Covers:
1. router: _escalation_choice header parsing
2. router: escalation rewrite / no-key downgrade in the pre-call hook
3. router: _api_key_available caching + error paths
4. router: [fable] override tag
5. claude_proxy: _get_api_key keychain/env resolution + caching
6. claude_proxy: /apikey route auth, 401-without-key, header injection
7. claude_proxy: /health api_key_available flag
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
from typing import Any

import pytest

import router.route_by_size as rbs
from router.route_by_size import SizeBasedRouter

# ---- helpers -----------------------------------------------------------------


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


def _data(model: str, choice: str | None = None) -> dict:
    d: dict = {
        "model": model,
        "messages": [{"role": "user", "content": "design a billing service"}],
    }
    if choice is not None:
        d["proxy_server_request"] = {
            "headers": {"x-claude-escalation-model": choice}
        }
    return d


async def _yes() -> bool:
    return True


async def _no() -> bool:
    return False


# ---- _escalation_choice --------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, "haiku"),
        ("", "haiku"),
        ("haiku", "haiku"),
        ("sonnet", "sonnet"),
        ("opus", "opus"),
        ("fable", "fable"),
        ("SONNET", "sonnet"),
        (" fable ", "fable"),
        ("gpt-5", "haiku"),  # unknown -> safe default
    ],
)
def test_escalation_choice_parsing(raw: str | None, expected: str) -> None:
    assert rbs._escalation_choice(_data("claude-code", raw)) == expected


def test_escalation_choice_survives_malformed_request() -> None:
    assert rbs._escalation_choice({"proxy_server_request": "garbage"}) == "haiku"
    assert rbs._escalation_choice({}) == "haiku"


# ---- escalation in the pre-call hook -------------------------------------------


@pytest.mark.parametrize(
    "choice,target",
    [
        ("sonnet", "claude-sonnet-5"),
        ("opus", "claude-opus-4-8"),
        ("fable", "claude-fable-5"),
    ],
)
def test_escalates_claude_code_when_key_available(
    monkeypatch: Any, choice: str, target: str
) -> None:
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", False)
    monkeypatch.setattr(rbs, "_api_key_available", _yes)
    out = _run(_router(), _data("claude-code", choice))
    assert out["model"] == target
    assert f"escalation[{choice}]->{target}" in out["metadata"]["route_reason"]
    assert out["metadata"]["route_decision"] == target


def test_downgrades_visibly_without_key(monkeypatch: Any) -> None:
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", False)
    monkeypatch.setattr(rbs, "_api_key_available", _no)
    out = _run(_router(), _data("claude-code", "opus"))
    assert out["model"] == "claude-code"
    assert "escalation[opus] no-apikey-downgrade->haiku" in (
        out["metadata"]["route_reason"]
    )


def test_haiku_choice_is_a_noop(monkeypatch: Any) -> None:
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", False)
    monkeypatch.setattr(rbs, "_api_key_available", _yes)
    out = _run(_router(), _data("claude-code", "haiku"))
    assert out["model"] == "claude-code"
    assert "escalation" not in (out.get("metadata", {}).get("route_reason") or "")


def test_escalation_ignored_for_non_claude_code_models(monkeypatch: Any) -> None:
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", False)
    monkeypatch.setattr(rbs, "_api_key_available", _yes)
    out = _run(_router(), _data("local-long", "opus"))
    assert out["model"] == "local-long"


def test_escalated_model_gets_thinking_params(monkeypatch: Any) -> None:
    """Escalated Sonnet/Opus/Fable are thinking-capable -- the guard must
    inject adaptive thinking for them (unlike claude-code=haiku)."""
    monkeypatch.setattr(rbs, "ENABLE_THINKING_MODE", True)
    monkeypatch.setattr(rbs, "_api_key_available", _yes)
    out = _run(_router(), _data("claude-code", "fable"))
    assert out["model"] == "claude-fable-5"
    assert out["thinking"] == {"type": "adaptive"}
    # Sampling params must be absent (400 on Fable 5 / Opus 4.7+).
    assert "temperature" not in out


# ---- _api_key_available caching -------------------------------------------------


def test_api_key_available_uses_cache(monkeypatch: Any) -> None:
    monkeypatch.setitem(rbs._apikey_health_cache, "available", True)
    monkeypatch.setitem(
        rbs._apikey_health_cache, "expires_at", rbs.time.monotonic() + 60
    )
    assert asyncio.run(rbs._api_key_available()) is True


def test_api_key_available_false_when_proxy_unreachable(monkeypatch: Any) -> None:
    monkeypatch.setitem(rbs._apikey_health_cache, "expires_at", 0.0)
    # Point at a port nothing listens on; 2s client timeout bounds the test.
    monkeypatch.setattr(rbs, "CLAUDE_PROXY_PORT", 1)
    assert asyncio.run(rbs._api_key_available()) is False


# ---- [fable] override tag --------------------------------------------------------


def test_fable_override_tag() -> None:
    alias, tag = rbs._extract_model_override("[fable] prove the collatz conjecture")
    assert alias == "claude-fable-5"
    assert tag == "fable"


# ---- claude_proxy ----------------------------------------------------------------

# Stub uvicorn so the module imports without a server (mirrors
# tests/test_claude_proxy.py).
_uv = types.ModuleType("uvicorn")
_uv.run = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("uvicorn", _uv)

# Real httpx client classes captured before any patching (the fake-client
# tests patch httpx.AsyncClient, which srv shares).
from httpx import ASGITransport as _RealASGITransport  # noqa: E402
from httpx import AsyncClient as _RealAsyncClient  # noqa: E402

import claude_proxy.server as srv  # noqa: E402


def _reset_key_cache() -> None:
    srv._api_key_cache["key"] = None
    srv._api_key_cache["expires_at"] = 0.0


class TestGetApiKey(unittest.TestCase):
    def setUp(self) -> None:
        _reset_key_cache()
        self._orig_check_output = srv.subprocess.check_output
        self._orig_env = srv.os.environ.get("ANTHROPIC_API_KEY")
        srv.os.environ.pop("ANTHROPIC_API_KEY", None)

    def tearDown(self) -> None:
        srv.subprocess.check_output = self._orig_check_output
        if self._orig_env is not None:
            srv.os.environ["ANTHROPIC_API_KEY"] = self._orig_env
        else:
            srv.os.environ.pop("ANTHROPIC_API_KEY", None)
        _reset_key_cache()

    def test_reads_key_from_keychain(self) -> None:
        srv.subprocess.check_output = lambda *a, **kw: b"sk-ant-test-key\n"
        assert srv._get_api_key() == "sk-ant-test-key"

    def test_falls_back_to_env_when_keychain_empty(self) -> None:
        def _fail(*a: object, **kw: object) -> bytes:
            raise srv.subprocess.CalledProcessError(44, "security")

        srv.subprocess.check_output = _fail
        srv.os.environ["ANTHROPIC_API_KEY"] = "sk-ant-env-key"
        assert srv._get_api_key() == "sk-ant-env-key"

    def test_none_when_no_key_anywhere(self) -> None:
        def _fail(*a: object, **kw: object) -> bytes:
            raise srv.subprocess.CalledProcessError(44, "security")

        srv.subprocess.check_output = _fail
        assert srv._get_api_key() is None

    def test_caches_hits_and_misses(self) -> None:
        calls: list[int] = []

        def _count(*a: object, **kw: object) -> bytes:
            calls.append(1)
            return b"sk-ant-cached\n"

        srv.subprocess.check_output = _count
        assert srv._get_api_key() == "sk-ant-cached"
        assert srv._get_api_key() == "sk-ant-cached"
        assert len(calls) == 1  # second read served from cache


class _FakeResp:
    def __init__(self, status_code: int, content: bytes = b"{}", headers: dict | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    async def aread(self) -> bytes:
        return self.content

    async def aiter_bytes(self):
        yield self.content


class _FakeClient:
    """Stands in for httpx.AsyncClient; records the headers of each POST."""

    responses: list[_FakeResp] = []
    calls: int = 0
    seen_headers: list[dict] = []

    def __init__(self, *a, **kw) -> None:
        pass

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def post(self, *a, **kw) -> _FakeResp:
        cls = type(self)
        cls.seen_headers.append(dict(kw.get("headers") or {}))
        resp = cls.responses[cls.calls]
        cls.calls += 1
        return resp


class TestApikeyRoute(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        _reset_key_cache()
        _FakeClient.responses = []
        _FakeClient.calls = 0
        _FakeClient.seen_headers = []
        self._orig_client = srv.httpx.AsyncClient
        srv.httpx.AsyncClient = _FakeClient  # type: ignore[assignment]
        self._orig_sleep = srv.asyncio.sleep

        async def _fake_sleep(d: float) -> None:
            return None

        srv.asyncio.sleep = _fake_sleep  # type: ignore[assignment]
        self._orig_get_key = srv._get_api_key

    def tearDown(self) -> None:
        srv.httpx.AsyncClient = self._orig_client
        srv.asyncio.sleep = self._orig_sleep
        srv._get_api_key = self._orig_get_key
        _reset_key_cache()

    async def _post(self, path: str = "/apikey/v1/messages", stream: bool = False):
        body = {"model": "claude-opus-4-8", "messages": [], "stream": stream}
        async with _RealAsyncClient(
            transport=_RealASGITransport(app=srv.app), base_url="http://test"
        ) as client:
            return await client.post(path, json=body)

    async def test_401_without_key(self) -> None:
        srv._get_api_key = lambda: None  # type: ignore[assignment]
        resp = await self._post()
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "authentication_error"
        assert _FakeClient.calls == 0  # never reached Anthropic

    async def test_forwards_with_x_api_key_header(self) -> None:
        srv._get_api_key = lambda: "sk-ant-test-123"  # type: ignore[assignment]
        _FakeClient.responses = [_FakeResp(200, b'{"ok": true}')]
        resp = await self._post()
        assert resp.status_code == 200
        assert _FakeClient.seen_headers[0].get("x-api-key") == "sk-ant-test-123"
        # No Bearer auth on the apikey path.
        assert "authorization" not in _FakeClient.seen_headers[0]

    async def test_apikey_route_retries_429(self) -> None:
        srv._get_api_key = lambda: "sk-ant-test-123"  # type: ignore[assignment]
        _FakeClient.responses = [_FakeResp(429), _FakeResp(200, b'{"ok": true}')]
        resp = await self._post()
        assert resp.status_code == 200
        assert _FakeClient.calls == 2

    async def test_health_reports_api_key_available(self) -> None:
        srv._get_api_key = lambda: "sk-ant-test-123"  # type: ignore[assignment]
        async with _RealAsyncClient(
            transport=_RealASGITransport(app=srv.app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
        assert resp.json()["api_key_available"] is True

    async def test_health_reports_missing_key(self) -> None:
        srv._get_api_key = lambda: None  # type: ignore[assignment]
        async with _RealAsyncClient(
            transport=_RealASGITransport(app=srv.app), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
        assert resp.json()["api_key_available"] is False


if __name__ == "__main__":
    unittest.main()
