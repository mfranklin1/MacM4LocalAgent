"""Tests for the Qwen3 `<think>` -> reasoning_content streaming transform.

Two layers:
  - `_split_think_stream`: the pure incremental parser (no deps).
  - `SizeBasedRouter.async_post_call_streaming_iterator_hook`: the LiteLLM
    streaming hook that drives the parser over ModelResponseStream chunks
    (guarded by importorskip on litellm types).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from router.route_by_size import SizeBasedRouter, _split_think_stream


# ---- pure splitter ----------------------------------------------------------

def _feed(chunks: list[str]) -> tuple[str, str]:
    """Run the splitter chunk-by-chunk like the hook does; return the
    concatenated (content, reasoning) including a final flush."""
    buf = ""
    in_think = False
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for piece in chunks:
        buf += piece
        c, r, buf, in_think = _split_think_stream(buf, in_think)
        content_parts.append(c)
        reasoning_parts.append(r)
    # flush
    if in_think:
        reasoning_parts.append(buf)
    else:
        content_parts.append(buf)
    return "".join(content_parts), "".join(reasoning_parts)


def test_split_basic_single_chunk() -> None:
    content, reasoning = _feed(["<think>reason here</think>answer"])
    assert reasoning == "reason here"
    assert content == "answer"


def test_split_tag_across_chunk_boundary() -> None:
    # The opening tag is split across three chunks; the closing tag too.
    content, reasoning = _feed(["<th", "ink>deep ", "thought</thi", "nk>final"])
    assert reasoning == "deep thought"
    assert content == "final"


def test_split_no_think_is_all_content() -> None:
    content, reasoning = _feed(["just ", "a normal ", "answer"])
    assert content == "just a normal answer"
    assert reasoning == ""


def test_split_leading_content_before_think() -> None:
    content, reasoning = _feed(["pre <think>mid</think> post"])
    assert content == "pre  post"
    assert reasoning == "mid"


def test_split_unclosed_think_flushes_as_reasoning() -> None:
    # Model opened <think> but the stream ended before </think>.
    content, reasoning = _feed(["<think>still thinking"])
    assert reasoning == "still thinking"
    assert content == ""


# ---- streaming hook ---------------------------------------------------------

def _make_chunk(text: str) -> Any:
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    return ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(content=text))]
    )


async def _drive(router: SizeBasedRouter, chunks: list[Any], request_data: dict) -> list[Any]:
    async def _src():
        for c in chunks:
            yield c

    out: list[Any] = []
    async for c in router.async_post_call_streaming_iterator_hook(
        user_api_key_dict=None, response=_src(), request_data=request_data
    ):
        out.append(c)
    return out


def _collect(out: list[Any]) -> tuple[str, str]:
    content = "".join(
        (getattr(c.choices[0].delta, "content", None) or "") for c in out if c.choices
    )
    reasoning = "".join(
        (getattr(c.choices[0].delta, "reasoning_content", None) or "")
        for c in out
        if c.choices
    )
    return content, reasoning


def test_hook_routes_think_to_reasoning_content() -> None:
    pytest.importorskip("litellm")
    router = SizeBasedRouter.__new__(SizeBasedRouter)  # skip __init__/db setup
    chunks = [_make_chunk("<think>because"), _make_chunk(" reasons</think>"), _make_chunk("done")]
    out = asyncio.run(
        _drive(router, chunks, {"metadata": {"qwen3_think_injected": True}})
    )
    content, reasoning = _collect(out)
    assert reasoning == "because reasons"
    assert content == "done"


def test_hook_unclosed_think_streams_all_as_reasoning() -> None:
    pytest.importorskip("litellm")
    router = SizeBasedRouter.__new__(SizeBasedRouter)
    chunks = [_make_chunk("<think>reason A"), _make_chunk(" reason B")]
    out = asyncio.run(
        _drive(router, chunks, {"metadata": {"qwen3_think_injected": True}})
    )
    content, reasoning = _collect(out)
    assert reasoning == "reason A reason B"
    assert content == ""


def test_hook_flushes_partial_trailing_tag() -> None:
    """Stream truncated mid-`</think>`: the held-back tail must still be
    flushed (exercises the end-of-stream flush branch)."""
    pytest.importorskip("litellm")
    router = SizeBasedRouter.__new__(SizeBasedRouter)
    chunks = [_make_chunk("<think>deep</thi")]
    out = asyncio.run(
        _drive(router, chunks, {"metadata": {"qwen3_think_injected": True}})
    )
    content, reasoning = _collect(out)
    assert reasoning == "deep</thi"
    assert content == ""


def test_hook_passthrough_when_not_a_think_turn() -> None:
    pytest.importorskip("litellm")
    router = SizeBasedRouter.__new__(SizeBasedRouter)
    chunks = [_make_chunk("<think>x</think>y")]
    out = asyncio.run(_drive(router, chunks, {"metadata": {}}))
    content, reasoning = _collect(out)
    # Untouched: the <think> text stays in content, nothing on reasoning.
    assert content == "<think>x</think>y"
    assert reasoning == ""


def test_hook_tolerates_empty_and_choiceless_chunks() -> None:
    """Role-only deltas, empty content, and choiceless chunks pass through
    untouched (defensive guards), and a trailing partial open tag is
    flushed as content."""
    pytest.importorskip("litellm")
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    router = SizeBasedRouter.__new__(SizeBasedRouter)
    role_chunk = ModelResponseStream(
        choices=[StreamingChoices(index=0, delta=Delta(role="assistant"))]
    )
    choiceless = ModelResponseStream(choices=[])
    chunks = [role_chunk, choiceless, _make_chunk("hi <thi")]
    out = asyncio.run(
        _drive(router, chunks, {"metadata": {"qwen3_think_injected": True}})
    )
    content, reasoning = _collect(out)
    assert content == "hi <thi"
    assert reasoning == ""
