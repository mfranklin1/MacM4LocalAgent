"""Tests for the Cline Session Monitor data module and FastAPI routes.

Organisation:
  1. dashboard.monitor functions — tested in isolation with monkeypatched deps
  2. /monitor, /monitor/_live, /api/monitor — FastAPI routes via TestClient
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from cost import ingest
from dashboard import app as dash_app
from dashboard import monitor as mon


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_db) -> TestClient:  # noqa: ARG001
    return TestClient(dash_app.app)


@pytest.fixture
def fake_janitor(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Redirect JANITOR_BASE to a fresh temp directory."""
    base = tmp_path / "janitor"
    base.mkdir()
    monkeypatch.setattr(mon, "JANITOR_BASE", base, raising=True)
    return base


def _minimal_monitor_data() -> dict[str, Any]:
    """Return a minimal but structurally valid monitor_data dict for route stubs."""
    return {
        "health": [
            {"key": "ollama",  "name": "Ollama",  "port": 11434, "ok": True,  "status_code": 200, "status": "UP"},
            {"key": "litellm", "name": "LiteLLM", "port": 4000,  "ok": True,  "status_code": 200, "status": "UP"},
            {"key": "claude_proxy", "name": "Claude proxy", "port": 4002, "ok": False, "status_code": 429, "status": "429"},
            {"key": "mlx",     "name": "MLX",     "port": 8081,  "ok": False, "status_code": None, "status": "TIMEOUT"},
        ],
        "gortex": {
            "running": True, "pid": 1, "uptime": "1h", "state": "ready",
            "mcp_sessions": 2, "cline_sessions": 1, "detail_ok": True, "raw": "",
        },
        "janitor": {
            "enabled": True, "headroom_enabled": True,
            "active_task_id": "1782834473054", "entry_count": 3,
            "pack_exists": True, "task_dir_mtime": 1700000000.0,
        },
        "calls": [],
        "summary": {
            "total_requests": 8, "total_input_tokens": 100000,
            "total_output_tokens": 5000, "actual_spend_usd": 0.0,
            "shadow_spend_usd": 1.11, "savings_usd": 1.11, "savings_pct": 100.0,
            "by_tier": {"local-long": {"requests": 8}},
        },
        "generated_at": 1000,
    }


# ---------------------------------------------------------------------------
# health_checks()
# ---------------------------------------------------------------------------

def test_health_checks_returns_one_row_per_endpoint() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mon, "_probe_http", lambda url, timeout=0.5: (True, 200))
        rows = mon.health_checks()
    assert len(rows) == len(mon.ENDPOINTS)
    for r in rows:
        assert {"key", "name", "port", "ok", "status_code", "status"} <= r.keys()


def test_health_checks_all_up() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mon, "_probe_http", lambda url, timeout=0.5: (True, 200))
        rows = mon.health_checks()
    assert all(r["ok"] for r in rows)
    assert all(r["status"] == "UP" for r in rows)


def test_health_checks_all_timeout() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mon, "_probe_http", lambda url, timeout=0.5: (False, None))
        rows = mon.health_checks()
    assert all(not r["ok"] for r in rows)
    assert all(r["status"] == "TIMEOUT" for r in rows)


def test_health_checks_http_error_surfaced() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mon, "_probe_http", lambda url, timeout=0.5: (False, 429))
        rows = mon.health_checks()
    assert all(r["status"] == "429" for r in rows)


def test_health_checks_keys_match_endpoints() -> None:
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mon, "_probe_http", lambda url, timeout=0.5: (True, 200))
        rows = mon.health_checks()
    assert {r["key"] for r in rows} == {ep["key"] for ep in mon.ENDPOINTS}


# ---------------------------------------------------------------------------
# _probe_http() — low-level probe, never raises
# ---------------------------------------------------------------------------

def test_probe_http_returns_false_on_connection_error() -> None:
    # Port 19999 is almost certainly not listening.
    ok, code = mon._probe_http("http://127.0.0.1:19999/", timeout=0.1)
    assert ok is False


def test_probe_http_never_raises_on_garbage_url() -> None:
    ok, code = mon._probe_http("http://[invalid]:999/", timeout=0.1)
    assert ok is False
    assert code is None


# ---------------------------------------------------------------------------
# gortex_mcp_status()
#
# `running` comes solely from the pidfile + kill(pid, 0) liveness check
# (mon.GORTEX_PID_FILE); the `daemon status` subprocess only ever supplies
# the supplementary uptime/state/session fields, cached and rate-limited to
# GORTEX_DETAIL_TTL_SEC. The `_reset_gortex_cache` fixture forces that cache
# to always miss so each test observes its own monkeypatched subprocess call.
# ---------------------------------------------------------------------------

# Real `gortex daemon status` output: a column-aligned summary block (no
# colons) plus a separate box-drawn "MCP sessions:" table with a `client`
# column -- there's no --json flag to fall back on.
_SAMPLE_STATUS = (
    " daemon    v0.47.0+775d8bb3\n"
    " pid       34646\n"
    " socket    /Users/martinfr/.gortex/cache/daemon.sock\n"
    " uptime    14h22m\n"
    " state     ready (warmup 4h2m ago)\n"
    " sessions  3\n"
    "\n"
    "MCP sessions:\n"
    "┌───────────────────────┬────────┬─────────┬───────────┐\n"
    "│ id                    │ client │ version │ connected │\n"
    "├───────────────────────┼────────┼─────────┼───────────┤\n"
    "│ sess_aaaaaaaaaaaaaaaa │ cli    │         │    14h22m │\n"
    "│ sess_bbbbbbbbbbbbbbbb │ cli    │         │     8h27m │\n"
    "│ sess_cccccccccccccccc │ Cline  │ 3.83.0  │      36m  │\n"
    "└───────────────────────┴────────┴─────────┴───────────┘\n"
)


def _make_completed(returncode: int, stdout: str, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture(autouse=True)
def _reset_gortex_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mon, "_gortex_detail_cache", dict(mon._GORTEX_DETAIL_DEFAULT))
    monkeypatch.setattr(mon, "_gortex_detail_cached_at", 0.0)


def _live_pidfile(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, pid: int) -> None:
    p = tmp_path / "daemon.pid"
    p.write_text(str(pid))
    monkeypatch.setattr(mon, "GORTEX_PID_FILE", p)


def _missing_pidfile(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mon, "GORTEX_PID_FILE", tmp_path / "no-such.pid")


def test_pid_alive_true_for_own_process() -> None:
    assert mon._pid_alive(os.getpid()) is True


def test_pid_alive_false_for_reaped_process() -> None:
    proc = subprocess.Popen(["true"])
    proc.wait()
    assert mon._pid_alive(proc.pid) is False


def test_gortex_liveness_true_when_pidfile_names_live_process(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _live_pidfile(tmp_path, monkeypatch, os.getpid())
    live = mon.gortex_liveness()
    assert live == {"running": True, "pid": os.getpid()}


def test_gortex_liveness_false_when_pidfile_missing(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _missing_pidfile(tmp_path, monkeypatch)
    assert mon.gortex_liveness() == {"running": False, "pid": None}


def test_gortex_status_parses_all_fields(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _live_pidfile(tmp_path, monkeypatch, os.getpid())
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_completed(0, _SAMPLE_STATUS))
    g = mon.gortex_mcp_status()
    assert g["running"] is True
    assert g["pid"] == os.getpid()
    assert g["uptime"] == "14h22m"
    assert g["state"] == "ready"
    assert g["mcp_sessions"] == 3
    assert g["cline_sessions"] == 1
    assert g["detail_ok"] is True


def test_gortex_status_no_cline_sessions(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _live_pidfile(tmp_path, monkeypatch, os.getpid())
    raw = (
        " pid       1\n"
        " uptime    1m\n"
        " state     ready\n"
        " sessions  2\n"
        "\n"
        "MCP sessions:\n"
        "┌───────────────────────┬────────┬─────────┬───────────┐\n"
        "│ id                    │ client │ version │ connected │\n"
        "├───────────────────────┼────────┼─────────┼───────────┤\n"
        "│ sess_aaaaaaaaaaaaaaaa │ cli    │         │        1m │\n"
        "│ sess_bbbbbbbbbbbbbbbb │ cli    │         │        2m │\n"
        "└───────────────────────┴────────┴─────────┴───────────┘\n"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_completed(0, raw))
    g = mon.gortex_mcp_status()
    assert g["cline_sessions"] == 0
    assert g["mcp_sessions"] == 2


def test_gortex_status_repo_literally_named_cline_is_not_counted_as_a_session(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One of the tracked repos in this workspace is itself named "cline".
    Its row in the "tracked repos" table (`│ cline    │ github  │ ...`) must
    not be mistaken for a Cline MCP session -- this was a real false
    positive caught against live daemon output."""
    _live_pidfile(tmp_path, monkeypatch, os.getpid())
    raw = (
        " pid       1\n"
        " uptime    1m\n"
        " state     ready\n"
        " sessions  1\n"
        "\n"
        "tracked repos:\n"
        "┌──────────┬─────────┐\n"
        "│ repo     │ workspace │\n"
        "├──────────┼─────────┤\n"
        "│ cline                                                                    │ github  │\n"
        "└──────────┴─────────┘\n"
        "\n"
        "MCP sessions:\n"
        "┌───────────────────────┬────────┬─────────┬───────────┐\n"
        "│ id                    │ client │ version │ connected │\n"
        "├───────────────────────┼────────┼─────────┼───────────┤\n"
        "│ sess_aaaaaaaaaaaaaaaa │ cli    │         │        1m │\n"
        "└───────────────────────┴────────┴─────────┴───────────┘\n"
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_completed(0, raw))
    g = mon.gortex_mcp_status()
    assert g["cline_sessions"] == 0
    assert g["mcp_sessions"] == 1


def test_gortex_status_daemon_not_running(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No pidfile => not running, regardless of what the CLI would say."""
    _missing_pidfile(tmp_path, monkeypatch)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: _make_completed(1, "", "daemon not running"))
    g = mon.gortex_mcp_status()
    assert g["running"] is False
    assert g["pid"] is None
    assert g["mcp_sessions"] == 0


def test_gortex_status_never_raises_on_missing_binary(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _live_pidfile(tmp_path, monkeypatch, os.getpid())

    def _boom(*a: Any, **kw: Any) -> None:
        raise FileNotFoundError("gortex not found")
    monkeypatch.setattr(subprocess, "run", _boom)
    g = mon.gortex_mcp_status()
    assert g["running"] is True  # liveness is unaffected by the CLI being unusable
    assert isinstance(g["raw"], str)


def test_gortex_status_running_survives_detail_call_timeout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A live-but-backlogged daemon whose `daemon status` RPC times out must
    still report running=True — this was the actual production bug: a slow
    detail call was mistaken for the daemon being down. detail_ok must be
    False so callers know the (absent) session counts aren't confirmed."""
    _live_pidfile(tmp_path, monkeypatch, os.getpid())

    def _boom(*a: Any, **kw: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="gortex", timeout=mon.GORTEX_DETAIL_TIMEOUT_SEC)
    monkeypatch.setattr(subprocess, "run", _boom)
    g = mon.gortex_mcp_status()
    assert g["running"] is True
    assert g["pid"] == os.getpid()
    assert g["state"] is None
    assert g["detail_ok"] is False


def test_gortex_status_empty_stdout_leaves_detail_unavailable(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _live_pidfile(tmp_path, monkeypatch, os.getpid())
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_completed(0, ""))
    g = mon.gortex_mcp_status()
    assert g["running"] is True
    assert g["state"] is None
    assert g["detail_ok"] is False


def test_gortex_status_timeout_preserves_last_known_good_sessions(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient timeout must not wipe out a previously confirmed session
    count -- this was the second production bug: one slow poll flashed
    "0 Cline sessions" (reading as a disconnect) even though Cline's actual
    MCP connection never dropped."""
    _live_pidfile(tmp_path, monkeypatch, os.getpid())
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_completed(0, _SAMPLE_STATUS))
    good = mon.gortex_mcp_status()
    assert good["cline_sessions"] == 1
    assert good["detail_ok"] is True

    monkeypatch.setattr(mon, "_gortex_detail_cached_at", 0.0)  # force the next call to re-fetch

    def _boom(*a: Any, **kw: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="gortex", timeout=mon.GORTEX_DETAIL_TIMEOUT_SEC)
    monkeypatch.setattr(subprocess, "run", _boom)
    stale = mon.gortex_mcp_status()
    assert stale["running"] is True
    assert stale["detail_ok"] is False
    assert stale["cline_sessions"] == 1  # last known-good value, not reset to 0
    assert stale["mcp_sessions"] == 3


def test_gortex_status_detail_cached_within_ttl(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `daemon status` subprocess must not be re-invoked on every call
    -- only once per GORTEX_DETAIL_TTL_SEC. This is the fix for the
    dashboard hammering the daemon every 3 s."""
    _live_pidfile(tmp_path, monkeypatch, os.getpid())
    monkeypatch.setattr(mon, "_gortex_detail_cached_at", time.time())
    calls: list[Any] = []

    def _spy(*a: Any, **kw: Any) -> subprocess.CompletedProcess:
        calls.append(a)
        return _make_completed(0, _SAMPLE_STATUS)
    monkeypatch.setattr(subprocess, "run", _spy)
    mon.gortex_mcp_status()
    mon.gortex_mcp_status()
    assert len(calls) == 0  # cache is fresh; subprocess never runs


def test_gortex_status_not_running_never_calls_subprocess(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _missing_pidfile(tmp_path, monkeypatch)

    def _boom(*a: Any, **kw: Any) -> None:
        raise AssertionError("subprocess.run should not be called when the daemon is down")
    monkeypatch.setattr(subprocess, "run", _boom)
    g = mon.gortex_mcp_status()
    assert g["running"] is False


# ---------------------------------------------------------------------------
# janitor_state()
# ---------------------------------------------------------------------------

def test_janitor_state_no_base_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mon, "JANITOR_BASE", tmp_path / "nonexistent", raising=True)
    j = mon.janitor_state()
    assert j["enabled"] is False
    assert j["active_task_id"] is None


def test_janitor_state_empty_base(fake_janitor: pathlib.Path) -> None:
    j = mon.janitor_state()
    assert j["enabled"] is False
    assert j["active_task_id"] is None


def test_janitor_state_with_entry_files(fake_janitor: pathlib.Path) -> None:
    task_dir = fake_janitor / "1782834473054"
    task_dir.mkdir()
    (task_dir / "entry_1.json").write_text("{}")
    (task_dir / "entry_2.json").write_text("{}")
    j = mon.janitor_state()
    assert j["enabled"] is True
    assert j["active_task_id"] == "1782834473054"
    assert j["entry_count"] == 2
    assert j["pack_exists"] is False


def test_janitor_state_with_context_pack(fake_janitor: pathlib.Path) -> None:
    task_dir = fake_janitor / "task_abc"
    task_dir.mkdir()
    (task_dir / "active-context-pack.json").write_text('{"key":"value"}')
    j = mon.janitor_state()
    assert j["pack_exists"] is True
    assert j["enabled"] is True
    assert j["entry_count"] == 0  # pack is excluded from entry count


def test_janitor_state_pack_not_counted_as_entry(fake_janitor: pathlib.Path) -> None:
    task_dir = fake_janitor / "task_abc"
    task_dir.mkdir()
    (task_dir / "active-context-pack.json").write_text("{}")
    (task_dir / "entry_1.json").write_text("{}")
    j = mon.janitor_state()
    assert j["entry_count"] == 1  # only the real entry, not the pack


def test_janitor_state_picks_most_recently_modified_dir(fake_janitor: pathlib.Path) -> None:
    old_dir = fake_janitor / "task_old"
    old_dir.mkdir()
    (old_dir / "entry_1.json").write_text("{}")
    time.sleep(0.02)
    new_dir = fake_janitor / "task_new"
    new_dir.mkdir()
    (new_dir / "entry_1.json").write_text("{}")
    j = mon.janitor_state()
    assert j["active_task_id"] == "task_new"


def test_janitor_state_headroom_always_enabled(fake_janitor: pathlib.Path) -> None:
    j = mon.janitor_state()
    assert j["headroom_enabled"] is True


# ---------------------------------------------------------------------------
# recent_calls()
# ---------------------------------------------------------------------------

def test_recent_calls_empty_db(tmp_db: pathlib.Path) -> None:
    rows = mon.recent_calls()
    assert rows == []


def test_recent_calls_returns_seeded_rows(tmp_db: pathlib.Path) -> None:
    ingest.record_request(
        model="ollama/qwen3", tier="local-long",
        in_tok=5000, out_tok=200, actual_cost=0.0, latency_ms=1000,
        ts=int(time.time()), route_reason="test",
    )
    rows = mon.recent_calls(today_only=False)
    assert len(rows) == 1
    assert rows[0]["input_tok"] == 5000
    assert rows[0]["output_maxed"] is False
    assert "ts_human" in rows[0]


def test_recent_calls_flags_maxed_output(tmp_db: pathlib.Path) -> None:
    ingest.record_request(
        model="ollama/qwen3", tier="local-long",
        in_tok=100, out_tok=8192, actual_cost=0.0, latency_ms=100,
        ts=int(time.time()),
    )
    rows = mon.recent_calls(today_only=False)
    assert rows[0]["output_maxed"] is True


def test_recent_calls_output_below_ceiling_not_flagged(tmp_db: pathlib.Path) -> None:
    ingest.record_request(
        model="ollama/qwen3", tier="local-long",
        in_tok=100, out_tok=8191, actual_cost=0.0, latency_ms=100,
        ts=int(time.time()),
    )
    rows = mon.recent_calls(today_only=False)
    assert rows[0]["output_maxed"] is False


def test_recent_calls_respects_limit(tmp_db: pathlib.Path) -> None:
    now = int(time.time())
    for i in range(20):
        ingest.record_request(
            model="ollama/qwen3", tier="local-long",
            in_tok=100, out_tok=10, actual_cost=0.0, latency_ms=100,
            ts=now + i,
        )
    rows = mon.recent_calls(limit=5, today_only=False)
    assert len(rows) == 5


def test_recent_calls_today_only_filters_old_rows(tmp_db: pathlib.Path) -> None:
    now = int(time.time())
    # Row from 2 days ago — should be excluded.
    ingest.record_request(
        model="ollama/qwen3", tier="local-long",
        in_tok=100, out_tok=10, actual_cost=0.0, latency_ms=100,
        ts=now - 172800,  # 2 days ago
    )
    # Row from now — should be included.
    ingest.record_request(
        model="ollama/qwen3", tier="local-long",
        in_tok=200, out_tok=20, actual_cost=0.0, latency_ms=100,
        ts=now,
    )
    rows = mon.recent_calls(today_only=True)
    assert len(rows) == 1
    assert rows[0]["input_tok"] == 200


def test_recent_calls_never_raises_on_bad_db(monkeypatch: pytest.MonkeyPatch) -> None:
    from cost import ingest as _ingest
    monkeypatch.setattr(_ingest, "DB_PATH", pathlib.Path("/nonexistent/path/db.db"), raising=True)
    rows = mon.recent_calls()
    assert rows == []


# ---------------------------------------------------------------------------
# today_summary()
# ---------------------------------------------------------------------------

def test_today_summary_empty_db_returns_zeros(tmp_db: pathlib.Path) -> None:
    s = mon.today_summary()
    assert s["total_requests"] == 0
    assert s["actual_spend_usd"] == 0.0


def test_today_summary_has_all_keys(tmp_db: pathlib.Path) -> None:
    s = mon.today_summary()
    required = {
        "total_requests", "total_input_tokens", "total_output_tokens",
        "actual_spend_usd", "shadow_spend_usd", "savings_usd", "savings_pct", "by_tier",
    }
    assert required <= s.keys()


# ---------------------------------------------------------------------------
# monitor_data()
# ---------------------------------------------------------------------------

def test_monitor_data_has_all_top_level_keys(tmp_db: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
                                              fake_janitor: pathlib.Path) -> None:
    monkeypatch.setattr(mon, "_probe_http", lambda url, timeout=0.5: (True, 200))
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: _make_completed(0, _SAMPLE_STATUS))
    data = mon.monitor_data()
    assert {"health", "gortex", "janitor", "calls", "summary", "generated_at"} <= data.keys()


def test_monitor_data_generated_at_is_recent(tmp_db: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
                                              fake_janitor: pathlib.Path) -> None:
    monkeypatch.setattr(mon, "_probe_http", lambda url, timeout=0.5: (True, 200))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _make_completed(1, "", "down"))
    data = mon.monitor_data()
    assert abs(data["generated_at"] - int(time.time())) <= 5


def test_monitor_data_never_raises_when_all_deps_fail(monkeypatch: pytest.MonkeyPatch,
                                                       tmp_path: pathlib.Path) -> None:
    """Even with every service unreachable, monitor_data() must return a dict, not raise.

    _probe_http never raises (it returns (False, None) on any error), so we
    simulate all probes failing by returning that sentinel.  subprocess.run and
    the filesystem are broken via their own monkeypatches.
    """
    monkeypatch.setattr(mon, "_probe_http", lambda url, timeout=0.5: (False, None))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(OSError("no bin")))
    monkeypatch.setattr(mon, "GORTEX_PID_FILE", tmp_path / "no-such.pid")
    monkeypatch.setattr(mon, "JANITOR_BASE", tmp_path / "nonexistent", raising=True)
    from cost import ingest as _ingest
    monkeypatch.setattr(_ingest, "DB_PATH", pathlib.Path("/nonexistent/db.db"), raising=True)
    data = mon.monitor_data()
    assert isinstance(data, dict)
    assert "generated_at" in data
    # All health probes report failure, gortex reports not running.
    assert all(not ep["ok"] for ep in data["health"])
    assert data["gortex"]["running"] is False


# ---------------------------------------------------------------------------
# FastAPI routes
# ---------------------------------------------------------------------------

def test_monitor_page_renders(client: TestClient) -> None:
    r = client.get("/monitor")
    assert r.status_code == 200
    assert "Cline Monitor" in r.text
    assert 'hx-get="/monitor/_live"' in r.text


def test_monitor_page_has_nav_link(client: TestClient) -> None:
    r = client.get("/monitor")
    assert r.status_code == 200
    assert 'href="/monitor"' in r.text


def test_monitor_live_renders_health_grid(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mon, "monitor_data", _minimal_monitor_data)
    r = client.get("/monitor/_live")
    assert r.status_code == 200
    assert "Ollama" in r.text
    assert "LiteLLM" in r.text


def test_monitor_live_shows_gortex_connected(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mon, "monitor_data", _minimal_monitor_data)
    r = client.get("/monitor/_live")
    assert r.status_code == 200
    assert "connected" in r.text


def test_monitor_live_shows_gortex_disconnected_anomaly(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _disconnected() -> dict[str, Any]:
        d = _minimal_monitor_data()
        d["gortex"] = {**d["gortex"], "cline_sessions": 0}
        return d
    monkeypatch.setattr(mon, "monitor_data", _disconnected)
    r = client.get("/monitor/_live")
    assert r.status_code == 200
    assert "GORTEX MCP DISCONNECTED" in r.text


def test_monitor_live_shows_gortex_unknown_not_disconnected_when_detail_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon that's up but hasn't answered `daemon status` yet (detail_ok=False,
    so cline_sessions defaults to 0) must render as "status unknown", never as the
    "disconnected" anomaly -- that was the actual production bug."""
    def _unknown() -> dict[str, Any]:
        d = _minimal_monitor_data()
        d["gortex"] = {**d["gortex"], "cline_sessions": 0, "detail_ok": False}
        return d
    monkeypatch.setattr(mon, "monitor_data", _unknown)
    r = client.get("/monitor/_live")
    assert r.status_code == 200
    assert "GORTEX SESSION STATUS UNKNOWN" in r.text
    assert "GORTEX MCP DISCONNECTED" not in r.text


def test_monitor_live_shows_maxed_output_anomaly(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def _with_maxed() -> dict[str, Any]:
        d = _minimal_monitor_data()
        d["calls"] = [{
            "ts_human": "09:00:00", "tier": "local-long", "model": "ollama/qwen3",
            "input_tok": 30000, "output_tok": 8192, "actual_cost": 0.0,
            "shadow_cost": 0.30, "latency_ms": 5000, "route_reason": "test",
            "task_id": None, "output_maxed": True,
        }]
        return d
    monkeypatch.setattr(mon, "monitor_data", _with_maxed)
    r = client.get("/monitor/_live")
    assert r.status_code == 200
    assert "8192" in r.text
    assert "CEILING" in r.text


def test_monitor_live_shows_savings_stats(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mon, "monitor_data", _minimal_monitor_data)
    r = client.get("/monitor/_live")
    assert r.status_code == 200
    assert "1.11" in r.text   # savings_usd
    assert "100" in r.text    # savings_pct


def test_monitor_live_shows_janitor_status(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mon, "monitor_data", _minimal_monitor_data)
    r = client.get("/monitor/_live")
    assert r.status_code == 200
    assert "HeadroomAdapter" in r.text
    assert "Context Janitor" in r.text
    assert "1782834473054" in r.text


def test_api_monitor_returns_json(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mon, "monitor_data", _minimal_monitor_data)
    r = client.get("/api/monitor")
    assert r.status_code == 200
    body = r.json()
    assert "health" in body
    assert "gortex" in body
    assert "janitor" in body
    assert "calls" in body
    assert "summary" in body
    assert "generated_at" in body


def test_api_monitor_health_rows_have_required_shape(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mon, "monitor_data", _minimal_monitor_data)
    r = client.get("/api/monitor")
    assert r.status_code == 200
    for ep in r.json()["health"]:
        assert {"key", "name", "port", "ok", "status"} <= ep.keys()


def test_api_monitor_still_200_when_gortex_binary_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_db: pathlib.Path, tmp_path: pathlib.Path,
) -> None:
    """Route must return 200 even when `gortex` binary is absent."""
    monkeypatch.setattr(mon, "GORTEX_PID_FILE", tmp_path / "no-such.pid")

    def _boom(*a: Any, **kw: Any) -> None:
        raise FileNotFoundError("gortex not found")
    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(mon, "_probe_http", lambda url, timeout=0.5: (False, None))
    r = client.get("/api/monitor")
    assert r.status_code == 200
    assert r.json()["gortex"]["running"] is False


def test_index_nav_includes_monitor_link(client: TestClient) -> None:
    """The global nav must have a link to /monitor on every page."""
    r = client.get("/")
    assert r.status_code == 200
    assert 'href="/monitor"' in r.text


def test_tasks_nav_includes_monitor_link(client: TestClient) -> None:
    r = client.get("/tasks")
    assert r.status_code == 200
    assert 'href="/monitor"' in r.text
