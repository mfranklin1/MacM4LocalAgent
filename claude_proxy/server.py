"""claude_proxy — thin reverse-proxy on :4002 for the Claude Code VS Code extension.

Routing logic
-------------
Claude Code sends Anthropic-format POST /v1/messages requests.  The proxy
inspects the token count and decides:

  ≤ ROUTE_LONG_MAX tokens  →  translate to OpenAI format, forward to LiteLLM
                               on :4000 (free local Ollama/MLX stack).

  > ROUTE_LONG_MAX tokens  →  large-context path, controlled by
                               CLAUDE_PROXY_LARGE_CTX_MODE:

    passthrough (default)  →  stream-proxy the original request to
                               api.anthropic.com, forwarding the ORIGINAL
                               Authorization header that Claude Code supplied
                               (its Team subscription OAuth token).
                               ANTHROPIC_API_KEY is NEVER read on this path.

    apikey                 →  same upstream call but the Authorization header
                               is replaced with ANTHROPIC_API_KEY (pay-per-token
                               billing against a platform.claude.com account).

Key guarantees
--------------
- In passthrough mode the Anthropic API key is never touched, which means
  large-context escalations are billed to the Claude Team subscription, not
  to a separate API account.
- The two auth paths are isolated: Cline only reaches :4000 (API-key only);
  Claude Code only reaches :4002 (this proxy).  There is no cross-contamination.
- Streaming (text/event-stream) is transparently proxied in both the local and
  the passthrough/apikey paths.

Usage
-----
  python claude_proxy/server.py [--port 4002] [--host 127.0.0.1]

The server reads config/detected.env on startup to resolve ROUTE_LONG_MAX,
CLAUDE_PROXY_PORT, CLAUDE_PROXY_LARGE_CTX_MODE, and LITELLM_PORT.  Live env
vars always win over detected.env values.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

# ── repo-root bootstrap ──────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── detected.env loader (mirrors scripts/run_litellm.py) ────────────────────
def _load_detected_env() -> None:
    env_path = REPO_ROOT / "config" / "detected.env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_detected_env()

# ── configuration ────────────────────────────────────────────────────────────
ROUTE_LONG_MAX: int = int(os.environ.get("ROUTE_LONG_MAX", "128000"))
LITELLM_PORT: int = int(os.environ.get("LITELLM_PORT", "4000"))
LARGE_CTX_MODE: str = os.environ.get("CLAUDE_PROXY_LARGE_CTX_MODE", "passthrough").lower()

LITELLM_BASE = f"http://127.0.0.1:{LITELLM_PORT}"
ANTHROPIC_BASE = "https://api.anthropic.com"

# Headers forwarded verbatim when proxying to Anthropic.
_ANTHROPIC_FORWARD_HEADERS = {
    "content-type",
    "anthropic-version",
    "anthropic-beta",
    "x-api-key",
}

log = logging.getLogger("claude_proxy")

app = FastAPI(title="claude_proxy", docs_url=None, redoc_url=None)


# ── token estimation (reuse router heuristic) ────────────────────────────────
def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """chars / 3.6 heuristic — same approach as router/route_by_size.py.

    We intentionally avoid importing route_by_size here so the proxy can
    start independently of LiteLLM being installed.  The heuristic is
    accurate enough for a routing threshold check (not a billing number).
    """
    if not messages:
        return 0
    total_chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    total_chars += len(part.get("text") or "")
                    # tool_use / tool_result blocks also carry text
                    total_chars += len(part.get("input") or "")
    return max(1, int(total_chars / 3.6))


# ── Anthropic → OpenAI message translation ──────────────────────────────────

def _role(role: str) -> str:
    """Anthropic uses 'human'/'assistant'; OpenAI uses 'user'/'assistant'."""
    return "user" if role == "human" else role


def _content_to_str(content: Any) -> str:
    """Flatten Anthropic content (str or list of blocks) to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    # Represent tool calls as JSON text so local models see them
                    parts.append(json.dumps(block))
                elif btype == "tool_result":
                    c = block.get("content", "")
                    if isinstance(c, list):
                        parts.append(" ".join(p.get("text", "") for p in c if isinstance(p, dict)))
                    else:
                        parts.append(str(c))
                else:
                    parts.append(str(block.get("text") or block.get("content") or ""))
        return "\n".join(parts)
    return str(content)


def _anthropic_to_openai(body: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic /v1/messages request body to OpenAI chat/completions."""
    messages: list[dict[str, str]] = []

    # Anthropic puts the system prompt as a top-level key, not in messages[].
    system = body.get("system")
    if system:
        messages.append({"role": "system", "content": _content_to_str(system)})

    for m in body.get("messages", []):
        messages.append({"role": _role(m.get("role", "user")), "content": _content_to_str(m.get("content", ""))})

    openai_body: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": messages,
    }
    if body.get("max_tokens"):
        openai_body["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        openai_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        openai_body["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        openai_body["stop"] = body["stop_sequences"]
    if body.get("stream"):
        openai_body["stream"] = True
    return openai_body


def _openai_to_anthropic(oai: dict[str, Any], original_model: str) -> dict[str, Any]:
    """Translate an OpenAI chat/completions response to Anthropic /v1/messages format."""
    choice = (oai.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "stop_sequence",
    }
    finish = choice.get("finish_reason") or "end_turn"
    usage = oai.get("usage") or {}
    return {
        "id": oai.get("id", f"msg_{int(time.time())}"),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason_map.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ── SSE / streaming helpers ──────────────────────────────────────────────────

async def _stream_openai_to_anthropic(
    oai_stream: AsyncIterator[bytes],
    original_model: str,
) -> AsyncIterator[bytes]:
    """Convert OpenAI SSE stream to Anthropic SSE stream.

    Anthropic streaming events:
      message_start  → metadata
      content_block_start → index 0, type text
      content_block_delta → text_delta chunks
      content_block_stop
      message_delta  → stop_reason / usage
      message_stop
    """
    sent_start = False
    sent_block_start = False
    msg_id = f"msg_{int(time.time())}"

    def _sse(event: str, data: dict[str, Any]) -> bytes:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    accumulated_input_tokens = 0
    accumulated_output_tokens = 0
    stop_reason = "end_turn"

    async for raw in oai_stream:
        line = raw.decode(errors="replace").strip()
        if not line or line == "data: [DONE]":
            continue
        if line.startswith("data: "):
            payload_str = line[6:]
            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError:
                continue

            if not sent_start:
                usage_so_far = chunk.get("usage") or {}
                accumulated_input_tokens = usage_so_far.get("prompt_tokens", 0)
                yield _sse("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": msg_id,
                        "type": "message",
                        "role": "assistant",
                        "model": original_model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": accumulated_input_tokens, "output_tokens": 0},
                    },
                })
                sent_start = True

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            text_chunk = delta.get("content") or ""
            finish = choice.get("finish_reason")

            if text_chunk and not sent_block_start:
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                })
                sent_block_start = True

            if text_chunk:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text_chunk},
                })

            if finish:
                stop_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                }
                stop_reason = stop_reason_map.get(finish, "end_turn")
                usage_final = chunk.get("usage") or {}
                accumulated_output_tokens = usage_final.get("completion_tokens", 0)

    if sent_block_start:
        yield _sse("content_block_stop", {"type": "content_block_stop", "index": 0})

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"output_tokens": accumulated_output_tokens},
    })
    yield _sse("message_stop", {"type": "message_stop"})


async def _passthrough_stream(
    response: httpx.Response,
) -> AsyncIterator[bytes]:
    """Stream raw SSE bytes from an httpx response."""
    async for chunk in response.aiter_bytes():
        yield chunk


# ── local routing path ───────────────────────────────────────────────────────

async def _route_local(request: Request, body: dict[str, Any]) -> Response:
    """Translate to OpenAI format and forward to LiteLLM on :LITELLM_PORT."""
    original_model = body.get("model", "claude-sonnet-4-6")
    is_stream = body.get("stream", False)
    oai_body = _anthropic_to_openai(body)

    log.info(
        "route=local model=%s tokens_est=%d stream=%s",
        original_model,
        _estimate_tokens(body.get("messages", [])),
        is_stream,
    )

    timeout = httpx.Timeout(360.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if is_stream:
            async with client.stream(
                "POST",
                f"{LITELLM_BASE}/v1/chat/completions",
                json=oai_body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status_code != 200:
                    body_bytes = await resp.aread()
                    return Response(content=body_bytes, status_code=resp.status_code,
                                    media_type="application/json")

                async def _gen() -> AsyncIterator[bytes]:
                    async for chunk in _stream_openai_to_anthropic(resp.aiter_lines(), original_model):
                        yield chunk

                return StreamingResponse(_gen(), media_type="text/event-stream")
        else:
            resp = await client.post(
                f"{LITELLM_BASE}/v1/chat/completions",
                json=oai_body,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                return Response(content=resp.content, status_code=resp.status_code,
                                media_type="application/json")
            oai_json = resp.json()
            return JSONResponse(_openai_to_anthropic(oai_json, original_model))


# ── upstream (passthrough / apikey) routing path ─────────────────────────────

def _build_upstream_headers(request: Request, mode: str) -> dict[str, str]:
    """Build the headers to send upstream to api.anthropic.com.

    passthrough: forward the original Authorization/x-api-key header from
                 Claude Code (its Team OAuth token).  ANTHROPIC_API_KEY is
                 never read.

    apikey: replace the auth header with ANTHROPIC_API_KEY.
    """
    headers: dict[str, str] = {}

    # Always forward Anthropic-specific headers.
    for hname in ("anthropic-version", "anthropic-beta", "content-type"):
        val = request.headers.get(hname)
        if val:
            headers[hname] = val

    if mode == "passthrough":
        # Forward whatever auth Claude Code sent — its own Team OAuth token.
        auth = request.headers.get("authorization") or request.headers.get("x-api-key")
        if auth:
            if auth.lower().startswith("bearer "):
                headers["authorization"] = auth
            else:
                headers["x-api-key"] = auth
        # No else: if somehow no auth was sent, let Anthropic return a 401 naturally.
    else:
        # apikey mode: use ANTHROPIC_API_KEY regardless of what Claude Code sent.
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("CLAUDE_PROXY_LARGE_CTX_MODE=apikey but ANTHROPIC_API_KEY is not set")
        headers["x-api-key"] = api_key

    return headers


async def _route_upstream(request: Request, body: dict[str, Any], mode: str) -> Response:
    """Stream-proxy to api.anthropic.com with the appropriate auth."""
    model = body.get("model", "claude-sonnet-4-6")
    is_stream = body.get("stream", False)
    tokens = _estimate_tokens(body.get("messages", []))

    log.info(
        "route=upstream mode=%s model=%s tokens_est=%d stream=%s",
        mode, model, tokens, is_stream,
    )

    try:
        upstream_headers = _build_upstream_headers(request, mode)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    upstream_headers.setdefault("content-type", "application/json")
    upstream_headers.setdefault("anthropic-version", "2023-06-01")

    raw_body = json.dumps(body).encode()
    timeout = httpx.Timeout(360.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        if is_stream:
            async with client.stream(
                "POST",
                f"{ANTHROPIC_BASE}/v1/messages",
                content=raw_body,
                headers=upstream_headers,
            ) as resp:
                upstream_status = resp.status_code
                if upstream_status != 200:
                    body_bytes = await resp.aread()
                    return Response(content=body_bytes, status_code=upstream_status,
                                    media_type="application/json")

                async def _gen() -> AsyncIterator[bytes]:
                    async for chunk in _passthrough_stream(resp):
                        yield chunk

                return StreamingResponse(_gen(), media_type="text/event-stream",
                                         status_code=upstream_status)
        else:
            resp = await client.post(
                f"{ANTHROPIC_BASE}/v1/messages",
                content=raw_body,
                headers=upstream_headers,
            )
            return Response(content=resp.content, status_code=resp.status_code,
                            media_type="application/json")


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "route_long_max": ROUTE_LONG_MAX,
        "large_ctx_mode": LARGE_CTX_MODE,
        "litellm_port": LITELLM_PORT,
    })


@app.post("/v1/messages")
async def messages(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    tokens = _estimate_tokens(body.get("messages", []))

    if tokens <= ROUTE_LONG_MAX:
        return await _route_local(request, body)
    else:
        return await _route_upstream(request, body, LARGE_CTX_MODE)


# Anthropic SDK also sends version-probe / ping requests — return 200.
@app.get("/v1/messages")
async def messages_ping() -> JSONResponse:
    return JSONResponse({"type": "ping"})


# ── CLI entry point ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="claude_proxy — Claude Code local routing proxy")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.environ.get("CLAUDE_PROXY_PORT", "4002")))
    p.add_argument("--log-level", default="info")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    log.info(
        "claude_proxy starting host=%s port=%d large_ctx_mode=%s route_long_max=%d litellm_port=%d",
        args.host, args.port, LARGE_CTX_MODE, ROUTE_LONG_MAX, LITELLM_PORT,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
