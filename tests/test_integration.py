"""End-to-end integration test.

Simulates the full flow:
  router pre-call rewrites hybrid-auto -> chooses tier -> log_success_event
  records the request -> savings.summarize observes it -> dashboard renders it.

No external services involved.
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from compare import ab
from cost import ingest, savings
from dashboard import app as dash_app
from router.route_by_size import SizeBasedRouter


@pytest.fixture
def client(tmp_db) -> TestClient:                                              # noqa: ARG001
    return TestClient(dash_app.app)


def _resp(content: str, in_tok: int = 1, out_tok: int = 1) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
    })


def test_full_flow(tmp_db, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    router = SizeBasedRouter()

    # ---- 1) hybrid-auto rewrite for a small prompt -> local-fast
    data: dict = {
        "model": "hybrid-auto",
        "messages": [{"role": "user", "content": "tiny ask"}],
    }
    out = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert out["model"] == "local-fast"

    # ---- 2) record three different-tier requests
    now = int(time.time())
    router.log_success_event(
        kwargs={"model": "mlx-community/Qwen3-Coder-Next-4bit", "metadata": {"route_reason": "<= 16k"}},
        response_obj={"usage": {"prompt_tokens": 1500, "completion_tokens": 600}},
        start_time=time.time(),
        end_time=time.time() + 0.2,
    )
    router.log_success_event(
        kwargs={"model": "ollama/qwen3-coder:30b", "metadata": {"route_reason": "16k-128k"}},
        response_obj={"usage": {"prompt_tokens": 30_000, "completion_tokens": 1500}},
        start_time=time.time(),
        end_time=time.time() + 1.0,
    )
    router.log_success_event(
        kwargs={"model": "claude-sonnet-4-6", "metadata": {"route_reason": "complex"}},
        response_obj={"usage": {"prompt_tokens": 5000, "completion_tokens": 2000}},
        start_time=time.time(),
        end_time=time.time() + 2.0,
    )

    # ---- 3) savings rolls up correctly
    s = savings.summarize(7)
    assert s["total_requests"] == 3
    assert {"local-fast", "local-long", "claude"}.issubset(s["by_tier"].keys())
    # Claude actual = 5000*3e-6 + 2000*15e-6 = 0.015 + 0.030 = 0.045
    assert s["actual_spend_usd"] == pytest.approx(0.045, rel=1e-6)
    assert s["savings_usd"] > 0

    # ---- 4) /api/stats serves the same numbers
    api = client.get("/api/stats").json()
    assert api["week"]["total_requests"] == 3
    assert api["week"]["actual_spend_usd"] == pytest.approx(0.045, rel=1e-6)

    # ---- 5) Dashboard fragment shows the most recent request
    page = client.get("/stats").text
    assert "claude-sonnet-4-6" in page
    assert "ollama/qwen3-coder:30b" in page
    assert "mlx-community/Qwen3-Coder-Next-4bit" in page

    # ---- 6) A/B comparator persists and the dashboard surfaces it
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        return _resp(f"answer({body['model']})", 50, 25)

    real_client = httpx.Client

    class _MockClient(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _MockClient)

    cmp = ab.run("Hello there. Write a haiku about KV cache.")
    detail = client.get(f"/compare/{cmp['id']}")
    assert detail.status_code == 200
    assert "haiku about KV cache" in detail.text
    assert "answer(local-long)"   in detail.text
    assert "answer(claude-code)"  in detail.text
