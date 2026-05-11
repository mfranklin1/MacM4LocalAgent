"""Unit tests for the A/B comparator."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from compare import ab
from cost import ingest


# ---- judge score --------------------------------------------------------------

def test_judge_identical_outputs_high_score() -> None:
    s = ab._judge("hello world", "hello world")
    assert s >= 0.95


def test_judge_empty_returns_zero() -> None:
    assert ab._judge("", "x") == 0.0
    assert ab._judge("x", "") == 0.0


def test_judge_length_mismatch_drops_score() -> None:
    long  = "x" * 1000
    short = "x" * 10
    s = ab._judge(long, short)
    # 0.5 base + 0.25 * (10/1000) ratio + 0.25 fence = 0.7525
    assert s < 0.76


def test_judge_fence_count_matters() -> None:
    a = "```python\nx=1\n```"
    b = "x=1"
    s = ab._judge(a, b)
    # Same length-ish, but no shared fences => score not maximal
    assert s < 1.0


# ---- _call (with mocked transport) -------------------------------------------

def _resp(content: str, in_tok: int = 5, out_tok: int = 7) -> httpx.Response:
    body = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
    }
    return httpx.Response(200, json=body)


def _patch_httpx_with_handler(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    """Replace httpx.Client with a subclass that always uses our MockTransport."""
    real_client = httpx.Client

    class _MockClient(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _MockClient)


def test_call_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"]  = str(request.url)
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers.get("Authorization")
        return _resp("hi there", 12, 8)

    _patch_httpx_with_handler(monkeypatch, handler)

    res = ab._call("local-fast", "hello?")
    assert res["ok"] is True
    assert res["output"] == "hi there"
    assert res["input_tokens"] == 12
    assert res["output_tokens"] == 8
    assert "/v1/chat/completions" in captured["url"]
    assert captured["body"]["model"] == "local-fast"
    # The proxy is loopback-only with no master_key, so ab._call sends no
    # Authorization header. If we ever re-enable auth, swap this back to
    # `assert captured["auth"].startswith("Bearer ")`.
    assert captured["auth"] is None


def test_call_handles_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    _patch_httpx_with_handler(monkeypatch, handler)

    res = ab._call("local-fast", "hello?")
    assert res["ok"] is False
    assert res["output"] == ""
    assert res["input_tokens"] == 0


# ---- ab.run() end-to-end (mocked HTTP, real DB) ------------------------------

def test_run_persists_comparison(tmp_db, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body["model"]
        if "claude" in model:
            return _resp("Claude response.\n```\ncode\n```", 100, 200)
        return _resp("Local response.\n```\ncode\n```", 80, 150)

    _patch_httpx_with_handler(monkeypatch, handler)

    out = ab.run("Compare these.")
    assert out["id"] >= 1
    assert 0.0 <= out["score"] <= 1.0

    c = ingest.connect()
    r = c.execute("SELECT * FROM comparisons WHERE id = ?", (out["id"],)).fetchone()
    assert r["prompt"] == "Compare these."
    assert r["local_in_tok"] == 80
    assert r["local_out_tok"] == 150
    assert r["claude_in_tok"] == 100
    assert r["claude_out_tok"] == 200
    # Claude cost = 100*3e-6 + 200*15e-6 = 0.0033
    assert r["claude_cost"] == pytest.approx(0.0033, rel=1e-6)
    assert r["local_cost"]  == 0.0
