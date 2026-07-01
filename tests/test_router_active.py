"""Tests for the in-flight (active) request registry on SizeBasedRouter.

The registry is a tiny in-memory dict the router maintains so the
dashboard can show a green "Active" panel for currently-running calls.
It's populated in async_pre_call_hook, drained in the success/failure
hooks, and mirrored to .logs/active.json after every mutation so the
dashboard process (a separate launchd plist) can read it.

Each test isolates the on-disk mirror by pointing ACTIVE_PATH at a
tmp file. We intentionally do NOT use the `tmp_db` fixture because
none of these tests need the SQLite layer -- they exercise the
in-memory dict and JSON serialization only.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import time
from typing import Iterator

import pytest

import router.route_by_size as rrs
from router.route_by_size import ACTIVE_TTL_SEC, SizeBasedRouter


@pytest.fixture
def tmp_active(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[pathlib.Path]:
    """Redirect the in-flight mirror file at ACTIVE_PATH to a tmp file
    so tests don't stomp the real .logs/active.json from a running
    proxy. The router only reads ACTIVE_PATH from the module global,
    so a single setattr is enough."""
    path = tmp_path / "active.json"
    monkeypatch.setattr(rrs, "ACTIVE_PATH", path, raising=True)
    yield path


def _make_pre_call_data(
    *,
    call_id: str,
    model: str = "ollama/qwen3-coder-next:q4_K_M",
    tokens: int = 1234,
    task_id: str | None = None,
    task_text: str | None = None,
    route_reason: str = "cline-mode: cline+default: task=8 tok",
) -> dict:
    """Shape the dict that LiteLLM hands to `async_pre_call_hook` *after*
    routing has run. async_pre_call_hook itself rewrites `data["model"]`
    and stamps metadata; the registration happens last so we can call
    `_register_active(data)` directly with this fixture and skip the
    rewrite path."""
    meta = {
        "route_decision": model,
        "route_reason": route_reason,
        "route_tokens_estimated": tokens,
    }
    if task_id is not None:
        meta["task_id"] = task_id
    if task_text is not None:
        meta["task_text"] = task_text
    return {
        "model": model,
        "litellm_call_id": call_id,
        "metadata": meta,
    }


# ---- registration ------------------------------------------------------------

def test_register_active_records_basic_fields(tmp_active: pathlib.Path) -> None:
    """A registration after pre-call routing should appear in
    snapshot_active() with the resolved model + tier and the
    estimated input tokens stamped by the router."""
    router = SizeBasedRouter()
    data = _make_pre_call_data(call_id="cid-1", tokens=4242)

    router._register_active(data)

    snap = router.snapshot_active()
    assert len(snap) == 1
    row = snap[0]
    assert row["call_id"] == "cid-1"
    assert row["model"] == "ollama/qwen3-coder-next:q4_K_M"
    assert row["tier"] == "local-long"           # _model_to_tier classification
    assert row["in_tok_est"] == 4242
    # elapsed_sec is computed at snapshot time and must be non-negative;
    # we don't pin a tighter bound to avoid flakiness on slow CI.
    assert row["elapsed_sec"] >= 0.0
    assert row["route_reason"].startswith("cline-mode:")


def test_register_active_picks_correct_tier_per_model(tmp_active: pathlib.Path) -> None:
    """_register_active must classify into the same tier set as
    _record. Hitting one example per tier guards against drift."""
    router = SizeBasedRouter()

    cases = [
        ("anthropic/claude-opus-4-7",                       "claude"),
        ("ollama/qwen3-coder-next:q4_K_M",                   "local-long"),
        ("local-long",                                       "local-long"),
        ("local-agent",                                       "local-long"),
    ]
    for i, (model, _expected) in enumerate(cases):
        router._register_active(_make_pre_call_data(call_id=f"cid-{i}", model=model))

    snap = {row["call_id"]: row for row in router.snapshot_active()}
    for i, (_model, expected_tier) in enumerate(cases):
        assert snap[f"cid-{i}"]["tier"] == expected_tier


def test_register_active_propagates_task_metadata(tmp_active: pathlib.Path) -> None:
    """Cline traffic stamps task_id + task_text in metadata. Both
    should survive registration so the dashboard can deep-link to
    /tasks/<id> from the Active row."""
    router = SizeBasedRouter()
    data = _make_pre_call_data(
        call_id="cid-cline",
        task_id="abc123def4567890",
        task_text="Add a one-line comment to README.md",
    )

    router._register_active(data)

    [row] = router.snapshot_active()
    assert row["task_id"] == "abc123def4567890"
    assert row["task_text_short"].startswith("Add a one-line comment")


def test_register_active_uses_synthetic_id_when_missing(tmp_active: pathlib.Path) -> None:
    """Older LiteLLM versions don't stamp litellm_call_id pre-call.
    The registration must still appear (with a synthetic id) so the
    dashboard never silently drops rows."""
    router = SizeBasedRouter()
    data = _make_pre_call_data(call_id="cid-x")
    data.pop("litellm_call_id")

    router._register_active(data)

    [row] = router.snapshot_active()
    assert row["call_id"].startswith("anon-")


def test_register_active_writes_json_mirror(tmp_active: pathlib.Path) -> None:
    """The on-disk mirror is the cross-process IPC channel to the
    dashboard. Every registration must flush so the next 5s poll
    sees the new entry."""
    router = SizeBasedRouter()
    router._register_active(_make_pre_call_data(call_id="cid-1"))

    rows = json.loads(tmp_active.read_text())
    assert isinstance(rows, list) and len(rows) == 1
    assert rows[0]["call_id"] == "cid-1"


# ---- success drain -----------------------------------------------------------

def test_log_success_drops_active(tmp_active: pathlib.Path) -> None:
    """When LiteLLM fires `log_success_event`, the matching entry must
    disappear from the in-flight registry (and from the JSON mirror).
    We bypass _record's SQLite write by passing a kwargs dict that
    drops the row first, then short-circuits via a missing model id."""
    router = SizeBasedRouter()
    router._register_active(_make_pre_call_data(call_id="cid-1"))
    assert len(router.snapshot_active()) == 1

    # _drop_active runs at the very top of _record; we can call it
    # directly to test the drain in isolation without exercising the
    # SQLite write path (which is covered elsewhere).
    router._drop_active({"litellm_call_id": "cid-1"})

    assert router.snapshot_active() == []
    assert json.loads(tmp_active.read_text()) == []


def test_log_success_drops_via_litellm_params(tmp_active: pathlib.Path) -> None:
    """LiteLLM moves call_id around between lifecycle paths. The drop
    path must look in litellm_params too, not just the top-level
    kwargs."""
    router = SizeBasedRouter()
    router._register_active(_make_pre_call_data(call_id="cid-2"))

    router._drop_active({"litellm_params": {"litellm_call_id": "cid-2"}})

    assert router.snapshot_active() == []


# ---- failure drain -----------------------------------------------------------

def test_log_failure_drops_active(tmp_active: pathlib.Path) -> None:
    """A failure (4xx/5xx, network error, timeout) must also drain
    the registry. Without this, a stuck Cline session would leave a
    ghost row in the dashboard's Active panel until ACTIVE_TTL_SEC
    swept it 10 minutes later."""
    router = SizeBasedRouter()
    router._register_active(_make_pre_call_data(call_id="cid-fail"))

    router.log_failure_event(
        kwargs={"litellm_call_id": "cid-fail"},
        response_obj=None, start_time=0.0, end_time=0.0,
    )

    assert router.snapshot_active() == []


def test_async_log_failure_drops_active(tmp_active: pathlib.Path) -> None:
    """async_log_failure_event mirrors log_failure_event and must also
    drain. LiteLLM picks one or the other depending on the deployment."""
    router = SizeBasedRouter()
    router._register_active(_make_pre_call_data(call_id="cid-async-fail"))

    asyncio.run(router.async_log_failure_event(
        kwargs={"litellm_call_id": "cid-async-fail"},
        response_obj=None, start_time=0.0, end_time=0.0,
    ))

    assert router.snapshot_active() == []


def test_drop_active_tolerates_missing_call_id(tmp_active: pathlib.Path) -> None:
    """Some upstream errors fire failure hooks with empty kwargs. The
    drop path must be a silent no-op rather than raising and breaking
    the proxy's error path."""
    router = SizeBasedRouter()
    router._register_active(_make_pre_call_data(call_id="cid-1"))

    # No litellm_call_id, no metadata, no litellm_params -- nothing to do.
    router._drop_active({})

    # The original entry must still be there; we only drop on a match.
    assert len(router.snapshot_active()) == 1


# ---- TTL sweeper -------------------------------------------------------------

def test_ttl_sweep_drops_stale_entries(tmp_active: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """If LiteLLM ever drops a call without firing either hook (e.g.
    a crashed worker), the in-flight row would leak forever without
    a guardrail. snapshot_active() sweeps anything older than
    ACTIVE_TTL_SEC."""
    router = SizeBasedRouter()
    router._register_active(_make_pre_call_data(call_id="cid-stale"))

    # Forward time past the TTL. We monkeypatch time.time on the rrs
    # module so both the sweeper's "now" and elapsed_sec see it.
    real_now = time.time()
    monkeypatch.setattr(rrs.time, "time", lambda: real_now + ACTIVE_TTL_SEC + 1)

    snap = router.snapshot_active()
    assert snap == []
    # Mirror file should reflect the post-sweep state.
    assert json.loads(tmp_active.read_text()) == []


def test_ttl_sweep_keeps_fresh_entries(tmp_active: pathlib.Path) -> None:
    """A row that's only a few seconds old must NOT be swept; that
    would cause a blinking dashboard during normal operation."""
    router = SizeBasedRouter()
    router._register_active(_make_pre_call_data(call_id="cid-fresh"))

    # No time monkeypatching: the entry was just created.
    snap = router.snapshot_active()
    assert len(snap) == 1
    assert snap[0]["call_id"] == "cid-fresh"


# ---- end-to-end via async_pre_call_hook --------------------------------------

def test_pre_call_hook_registers_for_explicit_local_long(tmp_active: pathlib.Path) -> None:
    """End-to-end check: a request that explicitly picks `local-long`
    (no hybrid-auto rewrite needed) should still register in the
    active dict. This exercises the same code path Cline traffic hits
    when it sends `local-long` directly."""
    router = SizeBasedRouter()
    data = {
        "model": "local-long",
        "litellm_call_id": "cid-e2e",
        "messages": [{"role": "user", "content": "hello"}],
    }

    asyncio.run(router.async_pre_call_hook(
        user_api_key_dict=None, cache=None, data=data, call_type="completion",
    ))

    snap = router.snapshot_active()
    assert len(snap) == 1
    assert snap[0]["call_id"] == "cid-e2e"
    assert snap[0]["tier"] == "local-long"
