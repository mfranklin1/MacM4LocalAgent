"""Tests for claude_proxy/server.py.

Covers:
- Token estimation
- Anthropic→OpenAI message translation
- OpenAI→Anthropic response translation
- Routing decision: local vs upstream based on token count
- Passthrough header building (OAuth token forwarded, no API key used)
- Apikey header building (API key substituted)
- /health endpoint response shape
"""
from __future__ import annotations

import json
import os
import sys
import types
import unittest

# ── stub uvicorn so the module can be imported without running a server ──────
_uv = types.ModuleType("uvicorn")
_uv.run = lambda *a, **kw: None  # type: ignore[attr-defined]
sys.modules.setdefault("uvicorn", _uv)

# Ensure repo root is on sys.path
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import claude_proxy.server as srv  # noqa: E402


class TestEstimateTokens(unittest.TestCase):
    def test_simple_string(self) -> None:
        msgs = [{"role": "user", "content": "Hello world"}]
        # len("Hello world") == 11 → 11/3.6 ≈ 3
        assert srv._estimate_tokens(msgs) == 3

    def test_list_content(self) -> None:
        msgs = [{"role": "user", "content": [{"type": "text", "text": "abc" * 100}]}]
        est = srv._estimate_tokens(msgs)
        assert est > 0

    def test_empty(self) -> None:
        assert srv._estimate_tokens([]) == 0

    def test_multi_message(self) -> None:
        msgs = [
            {"role": "user", "content": "a" * 360},
            {"role": "assistant", "content": "b" * 360},
        ]
        # 720 chars / 3.6 = 200
        assert srv._estimate_tokens(msgs) == 200


class TestAnthropicToOpenAI(unittest.TestCase):
    def test_basic(self) -> None:
        body = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 100,
        }
        result = srv._anthropic_to_openai(body)
        assert result["model"] == "hybrid-auto"
        assert result["messages"][-1]["content"] == "Say hi"
        assert result["max_tokens"] == 100

    def test_system_prompt(self) -> None:
        body = {
            "model": "claude-opus-4-7",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = srv._anthropic_to_openai(body)
        assert result["messages"][0]["role"] == "system"
        assert "helpful assistant" in result["messages"][0]["content"]
        assert result["messages"][1]["role"] == "user"

    def test_human_role_mapped(self) -> None:
        body = {
            "model": "claude-opus-4-7",
            "messages": [{"role": "human", "content": "Hi"}],
        }
        result = srv._anthropic_to_openai(body)
        assert result["messages"][0]["role"] == "user"

    def test_stream_forwarded(self) -> None:
        body = {"model": "claude-opus-4-7", "messages": [], "stream": True}
        result = srv._anthropic_to_openai(body)
        assert result.get("stream") is True


class TestOpenAIToAnthropic(unittest.TestCase):
    def test_basic(self) -> None:
        oai = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = srv._openai_to_anthropic(oai, "claude-opus-4-7")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-opus-4-7"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_length_stop_reason(self) -> None:
        oai = {"id": "x", "choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {}}
        result = srv._openai_to_anthropic(oai, "claude-opus-4-7")
        assert result["stop_reason"] == "max_tokens"


class TestBuildUpstreamHeaders(unittest.TestCase):
    """Test that passthrough never uses ANTHROPIC_API_KEY and apikey always does."""

    def _make_request(self, auth_header: str | None = None) -> object:
        """Build a minimal fake Request-like object."""
        hdrs: dict[str, str] = {"content-type": "application/json", "anthropic-version": "2023-06-01"}
        if auth_header:
            hdrs["authorization"] = auth_header

        class FakeHeaders:
            def __init__(self, d: dict[str, str]) -> None:
                self._d = {k.lower(): v for k, v in d.items()}

            def get(self, key: str) -> str | None:
                return self._d.get(key.lower())

        class FakeRequest:
            headers = FakeHeaders(hdrs)

        return FakeRequest()

    def test_passthrough_forwards_oauth(self) -> None:
        req = self._make_request("Bearer oauth-team-token-xyz")
        headers = srv._build_upstream_headers(req, "passthrough")  # type: ignore[arg-type]
        assert headers.get("authorization") == "Bearer oauth-team-token-xyz"
        # Must NOT contain a plain x-api-key derived from ANTHROPIC_API_KEY
        assert "sk-ant" not in headers.get("x-api-key", "")

    def test_passthrough_does_not_use_api_key(self) -> None:
        """Even if ANTHROPIC_API_KEY is set, passthrough must not inject it."""
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"
        try:
            req = self._make_request("Bearer oauth-token")
            headers = srv._build_upstream_headers(req, "passthrough")  # type: ignore[arg-type]
            assert "sk-ant-test-key" not in json.dumps(headers)
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_apikey_mode_uses_env_key(self) -> None:
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-api-key-123"
        try:
            req = self._make_request("Bearer oauth-token")
            headers = srv._build_upstream_headers(req, "apikey")  # type: ignore[arg-type]
            assert headers.get("x-api-key") == "sk-ant-api-key-123"
            # OAuth token must NOT be forwarded in apikey mode
            assert "oauth-token" not in headers.get("authorization", "")
        finally:
            del os.environ["ANTHROPIC_API_KEY"]

    def test_apikey_mode_missing_key_raises(self) -> None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        req = self._make_request("Bearer oauth-token")
        with self.assertRaises(ValueError, msg="ANTHROPIC_API_KEY is not set"):
            srv._build_upstream_headers(req, "apikey")  # type: ignore[arg-type]

    def test_x_api_key_header_forwarded_in_passthrough(self) -> None:
        """Some Claude Code versions send a raw token in x-api-key (no Bearer prefix).
        The proxy should forward it as x-api-key unchanged."""

        class FakeHeaders:
            def get(self, key: str) -> str | None:
                return {"x-api-key": "oauth-raw-token-no-prefix"}.get(key.lower())

        class FakeReq:
            headers = FakeHeaders()

        headers = srv._build_upstream_headers(FakeReq(), "passthrough")  # type: ignore[arg-type]
        assert headers.get("x-api-key") == "oauth-raw-token-no-prefix"


class TestHealthEndpoint(unittest.IsolatedAsyncioTestCase):
    async def test_health(self) -> None:
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=srv.app), base_url="http://test") as client:
            resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "route_long_max" in data
        assert "large_ctx_mode" in data
        assert data["large_ctx_mode"] in ("passthrough", "apikey")


if __name__ == "__main__":
    unittest.main()
