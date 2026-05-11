"""Shared driver for the two automated arms (local-only, claude-only).

Both arms are just `POST /v1/chat/completions` against the local LiteLLM proxy
at :4000. They differ only in the `model` field they request and in the arm
label written to `bench_runs`. Keeping the logic in one place means we measure
generate_ms, ttft_ms, token counts, and actual_cost identically for both.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
from typing import Any

import httpx

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench import db, grader  # noqa: E402
from cost.pricing import (  # noqa: E402
    actual_claude_cost,
    shadow_cost as _shadow_cost_fn,
    sonnet_rate,
)

# Backwards-compat constants for callers (and bench/runners/cursor_loop.py)
# that imported these directly. Mirrors Sonnet 4.6 rates so the shadow
# cost computed below stays the historical baseline.
CLAUDE_INPUT_PER_TOKEN = sonnet_rate().input
CLAUDE_OUTPUT_PER_TOKEN = sonnet_rate().output


def _read_env(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


_ENV = _read_env(REPO_ROOT / "config" / "detected.env")
LITELLM_BASE = f"http://127.0.0.1:{_ENV.get('LITELLM_PORT', '4000')}"


REPO_ROOT_FOR_PROGRESS = REPO_ROOT
PROGRESS_LOG = REPO_ROOT_FOR_PROGRESS / ".logs" / "bench-progress.log"


def _progress_emit(line: str) -> None:
    """Write a single progress line to .logs/bench-progress.log AND stderr.
    Stderr is for foreground users; the file is for `tail -f` from another
    shell or for the Cursor agent to poll."""
    try:
        PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(PROGRESS_LOG, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
    except Exception:
        pass
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def call_streaming(
    model: str,
    prompt: str | None = None,
    *,
    messages: list[dict[str, Any]] | None = None,
    base_url: str = LITELLM_BASE,
    timeout: float = 900.0,
    progress: bool = True,
    progress_every: float = 10.0,
) -> dict[str, Any]:
    """Streaming chat completion. Captures TTFT, full text, usage.

    Pass either `prompt` (single user turn, the legacy shape) or
    `messages` (full chat history including prior assistant turns).
    `messages` takes precedence; if neither is supplied this raises.

    Progress is written to BOTH stderr and `.logs/bench-progress.log` every
    `progress_every` seconds (default 10s). The file-based output exists so
    a parent process or `tail -f` can monitor a long-running call without
    waiting for the streaming pipe to flush.
    """
    if messages is None:
        if prompt is None:
            raise ValueError("call_streaming requires either prompt or messages")
        messages = [{"role": "user", "content": prompt}]
    started = time.time()
    ttft_ms = 0
    chunks: list[str] = []
    total_chars = 0
    usage: dict[str, Any] = {}
    model_reported = model
    last_tick = started

    def _tick(force: bool = False) -> None:
        nonlocal last_tick
        if not progress:
            return
        now = time.time()
        if not force and (now - last_tick) < progress_every:
            return
        last_tick = now
        elapsed = now - started
        ts = time.strftime("%H:%M:%S")
        _progress_emit(
            f"[{ts}] [{model}] streaming  "
            f"chunks={len(chunks):<5} chars={total_chars:<7} "
            f"elapsed={elapsed:>6.1f}s"
        )

    # Claude Sonnet 4.6's "extended thinking" can silently burn the
    # max_tokens budget on internal reasoning before emitting any
    # user-visible content. Anthropic also rejects `temperature != 1`
    # when extended thinking is enabled, so we cannot just pass
    # `reasoning_effort=low` alongside our preferred temperature for
    # graders. Cleanest workaround: keep temperature low and bump
    # max_tokens high enough that thinking AND the multi-file output
    # both fit. Claude Sonnet 4.6 supports up to 64k output tokens;
    # 32k gives plenty of headroom for ~10k of code with reasonable
    # internal reasoning on top.
    is_claude = "claude" in model.lower()
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 32768 if is_claude else 16384,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                headers={
                    "Content-Type":  "application/json",
                },
                json=payload,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line:
                        continue
                    s = line[6:] if line.startswith("data: ") else line
                    if s == "[DONE]":
                        break
                    try:
                        evt = json.loads(s)
                    except Exception:
                        continue
                    if not chunks and ttft_ms == 0:
                        ttft_ms = int((time.time() - started) * 1000)
                        if progress:
                            _progress_emit(
                                f"[{time.strftime('%H:%M:%S')}] [{model}] "
                                f"first token after {ttft_ms} ms"
                            )
                    if evt.get("model"):
                        model_reported = evt["model"]
                    for ch in evt.get("choices") or []:
                        delta = ch.get("delta") or {}
                        if "content" in delta and delta["content"]:
                            chunks.append(delta["content"])
                            total_chars += len(delta["content"])
                    if evt.get("usage"):
                        usage = evt["usage"]
                    _tick()
    except Exception as exc:
        if progress:
            _progress_emit(
                f"[{time.strftime('%H:%M:%S')}] [{model}] ERROR: {exc}"
            )
        return {
            "ok": False, "error": str(exc),
            "model": model, "model_reported": model_reported,
            "output": "",
            "input_tokens": 0, "output_tokens": 0,
            "wall_ms": int((time.time() - started) * 1000),
            "ttft_ms": ttft_ms,
        }

    if progress:
        _tick(force=True)
        _progress_emit(
            f"[{time.strftime('%H:%M:%S')}] [{model}] DONE  "
            f"chunks={len(chunks)} chars={total_chars} "
            f"wall={int((time.time() - started) * 1000)}ms"
        )

    out = "".join(chunks)
    wall_ms = int((time.time() - started) * 1000)
    return {
        "ok": True,
        "model": model,
        "model_reported": model_reported,
        "output": out,
        "input_tokens":  int((usage or {}).get("prompt_tokens", 0) or 0),
        "output_tokens": int((usage or {}).get("completion_tokens", 0) or 0),
        "wall_ms": wall_ms,
        "ttft_ms": ttft_ms,
    }


def run_arm(
    *,
    arm: str,
    model: str,
    task: dict[str, Any],
    work_dir: pathlib.Path,
    attempt: int = 1,
    base_url: str = LITELLM_BASE,
    pytest_timeout: float = 120.0,
) -> dict[str, Any]:
    """End-to-end: call model, grade, persist `bench_runs` row, return summary."""
    started = time.time()
    prompt = task["prompt"]

    _progress_emit(
        f"[{time.strftime('%H:%M:%S')}] [{model}] arm={arm} "
        f"calling LiteLLM at {base_url}"
    )
    res = call_streaming(model, prompt, base_url=base_url)
    response_text = res.get("output", "") if res.get("ok") else ""

    # Always save the raw response for offline inspection, even if grading
    # is going to short-circuit because the response was empty or the call
    # errored. This makes debugging extended-thinking timeouts and stream
    # truncation much easier later.
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "raw_response.txt").write_text(response_text or "")
    if not res.get("ok"):
        (work_dir / "call_error.txt").write_text(
            res.get("error") or "unknown_error",
        )

    if response_text:
        _progress_emit(
            f"[{time.strftime('%H:%M:%S')}] [{model}] "
            f"grading {len(response_text)} chars of output..."
        )
        gr = grader.grade_task(task, response_text, work_dir=work_dir)
        grade_row = gr.as_db_row()
        _progress_emit(
            f"[{time.strftime('%H:%M:%S')}] [{model}] graded: "
            f"matched {gr.pytest_passed}/{gr.pytest_total} "
            f"score={gr.composite_score:.3f}"
        )
    else:
        grade_row = {
            "syntactic_ok": 0, "has_docstring": 0, "has_type_hints": 0,
            "no_thirdparty": 0,
            "pytest_passed": 0, "pytest_failed": 0,
            "pytest_errors": 1, "pytest_total": 0,
            "passes_tests": 0.0, "grade_ms": 0,
            "composite_score": 0.0,
            "output_path": "",
        }

    # Cost: only Claude calls hit real $. actual rate comes from the
    # model id we got back (covers Opus/Haiku/Sonnet correctly);
    # shadow stays pinned to Sonnet 4.6 for stable benchmarking.
    reported = res.get("model_reported", "") or model
    is_claude = "claude" in reported.lower()
    in_tok = res.get("input_tokens", 0)
    out_tok = res.get("output_tokens", 0)
    actual = actual_claude_cost(reported, in_tok, out_tok) if is_claude else 0.0
    shadow = _shadow_cost_fn(in_tok, out_tok)

    row = {
        "ts": int(started),
        "task_id": task["id"],
        "arm": arm,
        "model": res.get("model_reported", model),
        "attempt": attempt,
        "input_tok":  in_tok,
        "output_tok": out_tok,
        "actual_cost": actual,
        "shadow_cost": shadow,
        "wall_ms":     int((time.time() - started) * 1000),
        "generate_ms": res.get("wall_ms", 0),
        "ttft_ms":     res.get("ttft_ms", 0),
        "output_chars": len(res.get("output", "")),
        "notes": "" if res.get("ok") else f"call_error: {res.get('error', '')}"[:500],
        "raw_metadata": {
            "ok": res.get("ok"),
            "base_url": base_url,
            "prompt_chars": len(prompt),
        },
        **grade_row,
    }
    rid = db.record_run(row)
    row["id"] = rid
    return row
