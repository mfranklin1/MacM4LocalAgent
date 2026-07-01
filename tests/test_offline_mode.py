"""Unit tests for the offline-mode guard.

Covers (1) the env-flag precedence + network-probe state machine in
`router/offline_mode.py`, (2) the routing-decision short-circuits in
`decide_tier` / `decide_tier_cline`, and (3) the chokepoint that
catches direct `claude-*` requests inside the LiteLLM pre-call hook.

We never hit the real network: every test stubs the TCP-probe helper
to a deterministic value via monkeypatch.
"""

from __future__ import annotations

import asyncio
import pathlib
from typing import Any

import pytest

from router import offline_mode
from router.offline_mode import (
    DEFAULT_OFFLINE_FALLBACK,
    OfflineStrictReject,
    is_claude_model,
    is_offline,
    is_online,
    maybe_downgrade,
    reset_state_for_tests,
)
from router.route_by_size import (
    SizeBasedRouter,
    decide_tier,
    decide_tier_cline,
)


# ---- Fixtures ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_offline_state() -> None:
    """Clear module-local caches (probe result, one-time-warned) so
    each test starts from a clean slate. Autouse because every test
    in this file depends on it -- forgetting to clear leaks state
    between tests and produces order-dependent failures."""
    reset_state_for_tests()
    yield
    reset_state_for_tests()


@pytest.fixture
def force_online(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the network is reachable. Stubs the TCP probe to True
    and pins OFFLINE=auto so the precedence rules still go through
    the probe path (not the OFFLINE=0 short-circuit)."""
    monkeypatch.setenv("OFFLINE", "auto")
    monkeypatch.delenv("OFFLINE_STRICT", raising=False)
    monkeypatch.setattr(offline_mode, "_probe_anthropic", lambda: True)


@pytest.fixture
def force_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend the network is unreachable. Stubs the TCP probe to
    False. Distinct from setting OFFLINE=1 because some tests need
    to verify the probe-driven path specifically (e.g. cache TTLs)."""
    monkeypatch.setenv("OFFLINE", "auto")
    monkeypatch.delenv("OFFLINE_STRICT", raising=False)
    monkeypatch.setattr(offline_mode, "_probe_anthropic", lambda: False)


@pytest.fixture
def _tmp_detected_env(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> pathlib.Path:
    """Redirect offline_mode's env-file reader at an empty tmp file
    so detected.env on the dev machine cannot influence the test."""
    fake_repo = tmp_path / "repo"
    (fake_repo / "config").mkdir(parents=True)
    (fake_repo / "config" / "detected.env").write_text("")
    monkeypatch.setattr(offline_mode, "REPO_ROOT", fake_repo)
    return fake_repo / "config" / "detected.env"


# ---- env-flag precedence + probe state machine ------------------------------


def test_is_online_forced_online_skips_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OFFLINE=0 in real env means "trust me, I'm online" -- the
    probe should NOT run at all. We assert by making the probe
    raise; if it gets called, the test fails."""
    def _exploding_probe() -> bool:
        raise AssertionError("probe must not run when OFFLINE=0")

    monkeypatch.setenv("OFFLINE", "0")
    monkeypatch.setattr(offline_mode, "_probe_anthropic", _exploding_probe)
    assert is_online() is True
    assert is_offline() is False


def test_is_offline_forced_offline_skips_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _exploding_probe() -> bool:
        raise AssertionError("probe must not run when OFFLINE=1")

    monkeypatch.setenv("OFFLINE", "1")
    monkeypatch.setattr(offline_mode, "_probe_anthropic", _exploding_probe)
    assert is_offline() is True
    assert is_online() is False


def test_is_online_auto_probes_when_unset(force_online: None) -> None:
    """Default (OFFLINE absent / 'auto') goes through the probe."""
    assert is_online() is True


def test_is_offline_auto_when_probe_fails(force_offline: None) -> None:
    assert is_offline() is True


def test_probe_result_is_cached_within_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second call within _ONLINE_TTL_SEC must reuse the cached
    result, not re-probe. We track call count via a counter."""
    monkeypatch.setenv("OFFLINE", "auto")
    monkeypatch.delenv("OFFLINE_STRICT", raising=False)
    calls = {"n": 0}

    def _counting_probe() -> bool:
        calls["n"] += 1
        return True

    monkeypatch.setattr(offline_mode, "_probe_anthropic", _counting_probe)
    assert is_online() is True
    assert is_online() is True
    assert is_online() is True
    assert calls["n"] == 1


def test_force_refresh_bypasses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OFFLINE", "auto")
    monkeypatch.delenv("OFFLINE_STRICT", raising=False)
    calls = {"n": 0}

    def _counting_probe() -> bool:
        calls["n"] += 1
        return True

    monkeypatch.setattr(offline_mode, "_probe_anthropic", _counting_probe)
    is_online()
    is_online(force_refresh=True)
    is_online(force_refresh=True)
    assert calls["n"] == 3


def test_env_file_used_when_real_env_unset(
    _tmp_detected_env: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When OFFLINE is unset in the real env, the helper falls back
    to config/detected.env. This is how `make offline` takes effect
    without a proxy restart."""
    monkeypatch.delenv("OFFLINE", raising=False)
    monkeypatch.delenv("OFFLINE_STRICT", raising=False)
    _tmp_detected_env.write_text("OFFLINE=1\n")

    def _exploding_probe() -> bool:
        raise AssertionError("OFFLINE=1 in env-file should skip probe")

    monkeypatch.setattr(offline_mode, "_probe_anthropic", _exploding_probe)
    assert is_offline() is True


def test_real_env_wins_over_env_file(
    _tmp_detected_env: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If detected.env says OFFLINE=1 but the real env says OFFLINE=0,
    the real env wins so `OFFLINE=0 make verify` still works."""
    _tmp_detected_env.write_text("OFFLINE=1\n")
    monkeypatch.setenv("OFFLINE", "0")
    monkeypatch.setattr(offline_mode, "_probe_anthropic", lambda: True)
    assert is_online() is True


# ---- is_claude_model classification ----------------------------------------


@pytest.mark.parametrize("model,expected", [
    ("claude-code",           True),
    ("claude-opus-4-7",       True),
    ("claude-sonnet-4-6",     True),
    ("claude-haiku-4-5",      True),
    ("gpt-claude-code",       True),
    ("gpt-claude-opus-4-7",   True),
    ("anthropic/claude-opus-4-7", True),
    ("hybrid-auto",           False),
    ("local-fast",            False),
    ("local-long",            False),
    ("local-agent",           False),
    ("ollama/qwen3-coder-next:q4_K_M", False),
    ("openai/mlx-community/Qwen2.5-Coder-7B-Instruct-4bit", False),
    ("",                      False),
    (None,                    False),
])
def test_is_claude_model(model: str | None, expected: bool) -> None:
    assert is_claude_model(model) is expected


# ---- maybe_downgrade silent path -------------------------------------------


def test_maybe_downgrade_noop_when_online(force_online: None) -> None:
    data: dict[str, Any] = {"model": "claude-code"}
    downgraded, err = maybe_downgrade(
        data, requested_alias="claude-code", explicit_claude=True,
    )
    assert downgraded is False
    assert err is None
    assert data["model"] == "claude-code"


def test_maybe_downgrade_noop_when_local_request(force_offline: None) -> None:
    """Local-model requests must pass through untouched even while
    offline. Otherwise a perfectly happy local-long call would get
    re-pointed to local-long for no reason and metadata would lie."""
    data: dict[str, Any] = {"model": "local-long"}
    downgraded, err = maybe_downgrade(
        data, requested_alias="local-long", explicit_claude=False,
    )
    assert downgraded is False
    assert err is None
    assert data["model"] == "local-long"
    assert "offline_downgrade" not in data.get("metadata", {})


def test_maybe_downgrade_silent_rewrites_model(force_offline: None) -> None:
    data: dict[str, Any] = {"model": "claude-code"}
    downgraded, err = maybe_downgrade(
        data, requested_alias="hybrid-auto", explicit_claude=False,
    )
    assert downgraded is True
    assert err is None
    assert data["model"] == DEFAULT_OFFLINE_FALLBACK
    meta = data["metadata"]
    assert meta["offline_downgrade"] is True
    assert meta["offline_orig_model"] == "hybrid-auto"
    assert "offline_user_notice" in meta
    assert "Cline" in meta["offline_user_notice"]  # context-clear hint
    assert "offline-downgrade" in meta["route_reason"]


def test_maybe_downgrade_preserves_prior_route_reason(
    force_offline: None,
) -> None:
    """A hybrid-auto turn that already wrote a cline-mode reason
    should keep that reason in the trail when we downgrade -- the
    audit log still needs to show WHY the router would have gone
    to Claude."""
    data: dict[str, Any] = {
        "model": "claude-code",
        "metadata": {"route_reason": "cline-mode: cline+task(abc): complex"},
    }
    downgraded, _ = maybe_downgrade(
        data, requested_alias="hybrid-auto", explicit_claude=False,
    )
    assert downgraded is True
    reason = data["metadata"]["route_reason"]
    assert "offline-downgrade" in reason
    assert "cline-mode" in reason


# ---- maybe_downgrade strict path -------------------------------------------


def test_maybe_downgrade_strict_rejects_explicit_claude(
    force_offline: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OFFLINE_STRICT", "1")
    data: dict[str, Any] = {"model": "claude-opus-4-7"}
    downgraded, err = maybe_downgrade(
        data, requested_alias="claude-opus-4-7", explicit_claude=True,
    )
    assert downgraded is True
    assert err is not None
    assert "OFFLINE_STRICT" in err
    # data["model"] is NOT rewritten in strict-reject -- the caller
    # is expected to raise and never dispatch upstream.
    assert data["model"] == "claude-opus-4-7"
    assert data["metadata"]["offline_downgrade"] == "rejected-strict"


def test_maybe_downgrade_strict_still_silent_on_hybrid_auto(
    force_offline: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode only rejects EXPLICIT Claude pins. A hybrid-auto
    request that the router happened to route to Claude should still
    silently downgrade -- the user didn't ask for Claude by name."""
    monkeypatch.setenv("OFFLINE_STRICT", "1")
    data: dict[str, Any] = {"model": "claude-code"}
    downgraded, err = maybe_downgrade(
        data, requested_alias="hybrid-auto", explicit_claude=False,
    )
    assert downgraded is True
    assert err is None
    assert data["model"] == DEFAULT_OFFLINE_FALLBACK


# ---- decide_tier short-circuit ---------------------------------------------


def test_decide_tier_complex_downgrades_when_offline(
    force_offline: None,
) -> None:
    msgs = [{"role": "user", "content": "Refactor the architecture across multiple files"}]
    model, reason, _ = decide_tier(msgs)
    assert model == DEFAULT_OFFLINE_FALLBACK
    assert "offline-downgrade" in reason
    assert "complex" in reason  # original reason preserved in trail


def test_decide_tier_huge_downgrades_when_offline(
    force_offline: None,
) -> None:
    # 5000 tokens past ROUTE_LONG_MAX; would normally go to claude-code.
    from router.route_by_size import ROUTE_LONG_MAX
    chars = (ROUTE_LONG_MAX + 5000) * 4
    msgs = [{"role": "user", "content": "x" * chars}]
    model, reason, _ = decide_tier(msgs)
    assert model == DEFAULT_OFFLINE_FALLBACK
    assert "offline-downgrade" in reason


def test_decide_tier_explicit_opus_tag_downgrades_when_offline(
    force_offline: None,
) -> None:
    """The user typed [opus] but the network is down. The router
    must NOT honor the tag silently -- it has to downgrade AND
    record that the tag was ignored."""
    msgs = [{"role": "user", "content": "[opus] design a billing service"}]
    model, reason, _ = decide_tier(msgs)
    assert model == DEFAULT_OFFLINE_FALLBACK
    assert "offline-downgrade" in reason
    assert "[opus]" in reason


def test_decide_tier_local_path_unchanged_when_offline(
    force_offline: None,
) -> None:
    """Small + non-complex prompts route to local-long regardless
    of offline state -- the network has no bearing on a local call."""
    msgs = [{"role": "user", "content": "rename foo to bar"}]
    model, _, _ = decide_tier(msgs)
    assert model == "local-long"


# ---- decide_tier_cline short-circuit ---------------------------------------


def _cline_msgs(
    task: str,
    *,
    extra: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build a Cline-shaped conversation. The system prompt has to
    contain enough Cline fingerprints (`You are Cline,`,
    `<replace_in_file>`, `<attempt_completion>`) for
    `_looks_like_cline` to fire. We don't actually call _looks_like_cline
    in this test, but decide_tier_cline expects the <task> envelope."""
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content":
            "You are Cline, a software engineer. Use <replace_in_file> "
            "and <attempt_completion> tool tags."},
        {"role": "user", "content": f"<task>{task}</task>"},
    ]
    if extra:
        msgs.extend(extra)
    return msgs


def test_decide_tier_cline_complex_downgrades_when_offline(
    force_offline: None,
) -> None:
    msgs = _cline_msgs("Refactor the architecture across multiple files")
    model, reason, _ = decide_tier_cline(msgs)
    assert model == DEFAULT_OFFLINE_FALLBACK
    assert "cline+offline-downgrade" in reason


def test_decide_tier_cline_opus_tag_downgrades_when_offline(
    force_offline: None,
) -> None:
    msgs = _cline_msgs("[opus] design a billing service")
    model, reason, _ = decide_tier_cline(msgs)
    assert model == DEFAULT_OFFLINE_FALLBACK
    assert "cline+offline-downgrade" in reason
    assert "[opus]" in reason


def test_decide_tier_cline_failure_signal_downgrades_when_offline(
    force_offline: None,
) -> None:
    """Tool-result-driven Claude rescues must also be suppressed
    while offline. Two python tracebacks would normally escalate."""
    tb = (
        "Traceback (most recent call last):\n"
        "  File 'a.py', line 1, in <module>\n"
        "ValueError: bad\n"
        "Traceback (most recent call last):\n"
        "  File 'b.py', line 1, in <module>\n"
        "ValueError: also bad\n"
    )
    msgs = _cline_msgs(
        "fix the bug",
        extra=[
            {"role": "assistant", "content": "<read_file>a.py</read_file>"},
            {"role": "user",      "content": tb},
        ],
    )
    model, reason, _ = decide_tier_cline(msgs)
    assert model == DEFAULT_OFFLINE_FALLBACK
    assert "cline+offline-downgrade" in reason


def test_decide_tier_cline_default_local_unchanged(
    force_offline: None,
) -> None:
    """A trivial Cline turn already routes to local-long, so the
    offline state should produce no audible change (no
    'offline-downgrade' tag) -- otherwise the audit log would be
    full of noise from harmless turns."""
    msgs = _cline_msgs("rename foo to bar")
    model, reason, _ = decide_tier_cline(msgs)
    assert model == "local-long"
    assert "offline-downgrade" not in reason


# ---- async_pre_call_hook chokepoint ----------------------------------------


@pytest.fixture
def router(tmp_db) -> SizeBasedRouter:                                       # noqa: ARG001
    return SizeBasedRouter()


def test_pre_call_downgrades_explicit_claude_when_offline(
    router: SizeBasedRouter, force_offline: None,
) -> None:
    """A client that asked for `claude-code` by name (not hybrid-auto)
    must also get downgraded -- otherwise the chokepoint is just
    the route-decision function and direct claude-* requests would
    leak through on an airplane."""
    data: dict[str, Any] = {
        "model": "claude-code",
        "messages": [{"role": "user", "content": "hello"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] == DEFAULT_OFFLINE_FALLBACK
    assert new["metadata"]["offline_downgrade"] is True


def test_pre_call_downgrades_gpt_claude_alias_when_offline(
    router: SizeBasedRouter, force_offline: None,
) -> None:
    """The Cursor-shaped `gpt-claude-*` alias must downgrade too --
    the canonicalization step strips `gpt-` so by the time we hit
    the chokepoint the model name is `claude-*`."""
    data: dict[str, Any] = {
        "model": "gpt-claude-opus-4-8",
        "messages": [{"role": "user", "content": "hello"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] == DEFAULT_OFFLINE_FALLBACK
    assert new["metadata"]["offline_orig_model"] == "claude-opus-4-8"


def test_pre_call_strict_mode_raises(
    router: SizeBasedRouter,
    force_offline: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OFFLINE_STRICT=1 + explicit Claude request must raise the
    sentinel exception so LiteLLM returns a 503 to the client. The
    test asserts the specific subclass, NOT bare RuntimeError, so
    a future refactor that swallows it would break this test."""
    monkeypatch.setenv("OFFLINE_STRICT", "1")
    data: dict[str, Any] = {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "hello"}],
    }
    with pytest.raises(OfflineStrictReject):
        asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))


def test_pre_call_passes_through_local_when_offline(
    router: SizeBasedRouter, force_offline: None,
) -> None:
    """Direct local-* requests must NOT be re-stamped with offline
    metadata. The downgrade is exclusively a Claude-tier concern."""
    data: dict[str, Any] = {
        "model": "local-long",
        "messages": [{"role": "user", "content": "hello"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert new["model"] == "local-long"
    assert "offline_downgrade" not in (new.get("metadata") or {})


def test_pre_call_hybrid_auto_silent_when_online(
    router: SizeBasedRouter, force_online: None,
) -> None:
    """Sanity: online + hybrid-auto + small prompt must NOT stamp
    any offline metadata. Catches the regression where the chokepoint
    fires unconditionally."""
    data: dict[str, Any] = {
        "model": "hybrid-auto",
        "messages": [{"role": "user", "content": "tiny"}],
    }
    new = asyncio.run(router.async_pre_call_hook(None, None, data, "completion"))
    assert new is not None
    assert "offline_downgrade" not in (new.get("metadata") or {})
