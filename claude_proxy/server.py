"""claude_proxy — thin reverse-proxy on :4002 for the Claude Code VS Code extension
and a subscription-auth injector for LiteLLM Claude escalations.

Routing logic (Claude Code path — /v1/messages)
------------------------------------------------
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

LiteLLM subscription path  (/subscription/v1/messages)
-------------------------------------------------------
LiteLLM Claude model entries point api_base at http://127.0.0.1:4002/subscription.
This endpoint ignores the incoming API key and injects the claude.ai Team OAuth
token from the macOS keychain (read fresh with a 30-second TTL cache).

  CLAUDE_AUTH_MODE=subscription (default) — use keychain OAuth token
  CLAUDE_AUTH_MODE=apikey                 — use ANTHROPIC_API_KEY instead

Key guarantees
--------------
- In passthrough/subscription mode the Anthropic API key is never touched.
- The two auth paths are isolated: Cline reaches :4000 (LiteLLM) which escalates
  via /subscription/v1/messages; Claude Code reaches /v1/messages directly.
- Streaming (text/event-stream) is transparently proxied on all paths.

Usage
-----
  python claude_proxy/server.py [--port 4002] [--host 127.0.0.1]

The server reads config/detected.env on startup to resolve ROUTE_LONG_MAX,
CLAUDE_PROXY_PORT, CLAUDE_PROXY_LARGE_CTX_MODE, CLAUDE_AUTH_MODE, and
LITELLM_PORT.  Live env vars always win over detected.env values.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import subprocess
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
CLAUDE_AUTH_MODE: str = os.environ.get("CLAUDE_AUTH_MODE", "subscription").lower()

LITELLM_BASE = f"http://127.0.0.1:{LITELLM_PORT}"
ANTHROPIC_BASE = "https://api.anthropic.com"

# ── subscription token cache ─────────────────────────────────────────────────
_token_cache: dict[str, Any] = {"token": None, "expires_at": 0.0}
_TOKEN_TTL = 30.0  # seconds


def _get_subscription_token() -> str:
    """Read the claude.ai OAuth access token from the macOS keychain.

    Caches the result for _TOKEN_TTL seconds to avoid a subprocess call on
    every request while still picking up rotated tokens promptly.

    Returns the token string, or raises RuntimeError if unavailable.
    """
    now = time.monotonic()
    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]  # type: ignore[return-value]

    try:
        raw = subprocess.check_output(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
        data = json.loads(raw)
        oauth = data.get("claudeAiOauth") or {}
        token = oauth.get("accessToken", "")
        if not token:
            raise RuntimeError("claudeAiOauth.accessToken missing from keychain entry")
        _token_cache["token"] = token
        _token_cache["expires_at"] = now + _TOKEN_TTL
        return token
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("keychain read timed out") from exc
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(f"failed to read subscription token from keychain: {exc}") from exc


# ── Anthropic API key (keychain) ─────────────────────────────────────────────
# The subscription OAuth token is policy-limited to Haiku on the raw API
# (verified 2026-07-02) -- Sonnet/Opus/Fable escalation authenticates with
# the org API key instead. Single source of truth is the macOS keychain
# item `anthropic-api-key` (account $USER): the same item read by
# ~/.claude/set-anthropic-env.sh and writable via
# scripts/setup-anthropic-key.sh or Cline's escalation setting. The env
# var is a fallback for headless/test environments.
_api_key_cache: dict[str, Any] = {"key": None, "expires_at": 0.0}
_API_KEY_SERVICE = "anthropic-api-key"


def _get_api_key() -> str | None:
    """Anthropic API key from the keychain (env fallback), or None.

    Same TTL-cache pattern as _get_subscription_token. Never raises --
    callers branch on None to produce a clear 'no API key configured'
    error instead of a stack trace.
    """
    now = time.monotonic()
    if _api_key_cache["key"] is not None and now < _api_key_cache["expires_at"]:
        return _api_key_cache["key"] or None
    key = ""
    try:
        key = subprocess.check_output(
            ["security", "find-generic-password",
             "-a", os.environ.get("USER", ""), "-s", _API_KEY_SERVICE, "-w"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode().strip()
    except Exception:
        key = ""
    if not key:
        key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    # Cache the miss too (empty string) so an unconfigured key doesn't cost
    # a subprocess call per request.
    _api_key_cache["key"] = key
    _api_key_cache["expires_at"] = now + _TOKEN_TTL
    return key or None


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
        "claude_auth_mode": CLAUDE_AUTH_MODE,
        "litellm_port": LITELLM_PORT,
        # Whether the keychain (or env) holds an Anthropic API key -- the
        # router reads this to decide if Sonnet/Opus/Fable escalation is
        # available or the choice must downgrade to haiku (subscription).
        "api_key_available": _get_api_key() is not None,
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


# ── LiteLLM subscription endpoint ────────────────────────────────────────────
# LiteLLM Claude model entries use api_base=http://127.0.0.1:4002/subscription,
# so LiteLLM POSTs to /subscription/v1/messages.  This endpoint ignores the
# incoming API key and injects the appropriate auth:
#   CLAUDE_AUTH_MODE=subscription  →  OAuth token from macOS keychain (default)
#   CLAUDE_AUTH_MODE=apikey        →  ANTHROPIC_API_KEY env var

# ── 429 retry policy ─────────────────────────────────────────────────────────
# The subscription token shares a per-hour quota with every other Claude
# consumer on the account, so transient 429s are routine. Before this
# backoff, a single 429 (after LiteLLM's one immediate retry) tripped
# LiteLLM's claude->local fallback and dumped 70K+-token complex tasks
# onto a local model. Bounded exponential backoff -- honoring a numeric
# Retry-After when Anthropic sends one -- rides out the quota window
# instead. Worst-case added latency with defaults: 2+4+8 = 14s.
RETRY_429_ATTEMPTS = int(os.environ.get("CLAUDE_PROXY_429_RETRIES", "3"))
RETRY_429_BASE_DELAY = float(os.environ.get("CLAUDE_PROXY_429_BASE_DELAY", "2.0"))
RETRY_429_MAX_DELAY = float(os.environ.get("CLAUDE_PROXY_429_MAX_DELAY", "30.0"))


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Seconds to wait before retry number `attempt` (0-based).

    A parseable numeric Retry-After header wins (clamped to the cap);
    the HTTP-date form falls through to exponential backoff.
    """
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), RETRY_429_MAX_DELAY)
        except ValueError:
            pass
    return min(RETRY_429_BASE_DELAY * (2 ** attempt), RETRY_429_MAX_DELAY)


def _build_subscription_headers(request: Request) -> dict[str, str]:
    """Build auth headers for the LiteLLM → Anthropic path."""
    headers: dict[str, str] = {}
    for hname in ("anthropic-version", "anthropic-beta", "content-type"):
        val = request.headers.get(hname)
        if val:
            headers[hname] = val
    headers.setdefault("content-type", "application/json")
    headers.setdefault("anthropic-version", "2023-06-01")

    if CLAUDE_AUTH_MODE == "apikey":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise ValueError("CLAUDE_AUTH_MODE=apikey but ANTHROPIC_API_KEY is not set")
        headers["x-api-key"] = api_key
    else:
        token = _get_subscription_token()
        headers["authorization"] = f"Bearer {token}"

    return headers


async def _forward_messages(
    body: bytes,
    is_stream: bool,
    upstream_headers: dict[str, str],
    log_tag: str,
) -> Response:
    """Forward a /v1/messages body to Anthropic with the 429 retry loop.

    Shared by the subscription and apikey routes -- only the auth headers
    (and the log tag) differ. Retry loop: only 429s are retried (with
    backoff); every other status -- success or failure -- returns on the
    first pass. The final attempt returns the 429 to LiteLLM so its
    fallback chain still engages once the backoff budget is exhausted.
    """
    timeout = httpx.Timeout(360.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(RETRY_429_ATTEMPTS + 1):
            delay = 0.0
            if is_stream:
                async with client.stream(
                    "POST",
                    f"{ANTHROPIC_BASE}/v1/messages",
                    content=body,
                    headers=upstream_headers,
                ) as resp:
                    if resp.status_code == 429 and attempt < RETRY_429_ATTEMPTS:
                        delay = _retry_delay(attempt, resp.headers.get("retry-after"))
                        await resp.aread()  # drain before closing
                        log.warning(
                            "%s 429 (stream); retry %d/%d in %.1fs",
                            log_tag, attempt + 1, RETRY_429_ATTEMPTS, delay,
                        )
                    elif resp.status_code != 200:
                        body_bytes = await resp.aread()
                        return Response(content=body_bytes, status_code=resp.status_code,
                                        media_type="application/json")
                    else:
                        # Bind resp explicitly: _gen is defined inside the
                        # retry loop, and a closure over the loop variable
                        # would see whatever resp is when iterated (B023).
                        async def _gen(resp: httpx.Response = resp) -> AsyncIterator[bytes]:
                            async for chunk in _passthrough_stream(resp):
                                yield chunk

                        return StreamingResponse(_gen(), media_type="text/event-stream",
                                                 status_code=resp.status_code)
                await asyncio.sleep(delay)
            else:
                resp = await client.post(
                    f"{ANTHROPIC_BASE}/v1/messages",
                    content=body,
                    headers=upstream_headers,
                )
                if resp.status_code == 429 and attempt < RETRY_429_ATTEMPTS:
                    delay = _retry_delay(attempt, resp.headers.get("retry-after"))
                    log.warning(
                        "%s 429; retry %d/%d in %.1fs",
                        log_tag, attempt + 1, RETRY_429_ATTEMPTS, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                return Response(content=resp.content, status_code=resp.status_code,
                                media_type="application/json")
        # Unreachable: the loop always returns on its final attempt
        # (attempt == RETRY_429_ATTEMPTS disables the retry branch).
        raise AssertionError("retry loop exited without returning")


@app.post("/subscription/v1/messages")
async def subscription_messages(request: Request) -> Response:
    """Forward to api.anthropic.com using the claude.ai subscription token (or API key)."""
    try:
        body = await request.body()
        body_json = json.loads(body)
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    is_stream = body_json.get("stream", False)
    model = body_json.get("model", "unknown")

    try:
        upstream_headers = _build_subscription_headers(request)
    except (ValueError, RuntimeError) as exc:
        log.error("subscription auth error: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    log.info(
        "route=subscription auth=%s model=%s stream=%s",
        CLAUDE_AUTH_MODE, model, is_stream,
    )
    return await _forward_messages(body, is_stream, upstream_headers, "subscription")


@app.get("/subscription/v1/messages")
async def subscription_ping() -> JSONResponse:
    return JSONResponse({"type": "ping"})


# ── LiteLLM apikey endpoint ──────────────────────────────────────────────────
# Escalation models the subscription OAuth token cannot serve (Sonnet 5,
# Opus 4.8, Fable 5 -- all policy-429 on subscription, verified 2026-07-02)
# route here instead: LiteLLM entries use
# api_base=http://127.0.0.1:4002/apikey. Auth is the org API key from the
# macOS keychain (`anthropic-api-key`), env fallback -- see _get_api_key().


@app.post("/apikey/v1/messages")
async def apikey_messages(request: Request) -> Response:
    """Forward to api.anthropic.com using the Anthropic API key (keychain)."""
    try:
        body = await request.body()
        body_json = json.loads(body)
    except Exception:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)

    is_stream = body_json.get("stream", False)
    model = body_json.get("model", "unknown")

    api_key = _get_api_key()
    if not api_key:
        # 401 (not 500): this is a configuration state the router checks
        # via /health before routing here, so reaching this means the key
        # was removed mid-flight. LiteLLM does not retry 401s.
        log.error("apikey route called but no API key in keychain/env")
        return JSONResponse(
            {"type": "error",
             "error": {"type": "authentication_error",
                       "message": "no Anthropic API key configured: add one to the "
                                  f"macOS keychain (service '{_API_KEY_SERVICE}') via "
                                  "scripts/setup-anthropic-key.sh or the Cline "
                                  "escalation setting"}},
            status_code=401,
        )

    headers: dict[str, str] = {}
    for hname in ("anthropic-version", "anthropic-beta", "content-type"):
        val = request.headers.get(hname)
        if val:
            headers[hname] = val
    headers.setdefault("content-type", "application/json")
    headers.setdefault("anthropic-version", "2023-06-01")
    headers["x-api-key"] = api_key

    log.info("route=apikey model=%s stream=%s", model, is_stream)
    return await _forward_messages(body, is_stream, headers, "apikey")


@app.get("/apikey/v1/messages")
async def apikey_ping() -> JSONResponse:
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
        "claude_proxy starting host=%s port=%d large_ctx_mode=%s claude_auth_mode=%s "
        "route_long_max=%d litellm_port=%d",
        args.host, args.port, LARGE_CTX_MODE, CLAUDE_AUTH_MODE, ROUTE_LONG_MAX, LITELLM_PORT,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
