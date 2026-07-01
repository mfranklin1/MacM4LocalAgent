"""Live data aggregation for the Cline Monitor tab (/monitor).

All public functions return safe defaults on any error — the dashboard
must never crash because an external service is down or a file is missing.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import subprocess
import time
from typing import Any

try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    import urllib.request as _urllib_request  # type: ignore[assignment]
    import urllib.error as _urllib_error  # type: ignore[assignment]
    _HAS_HTTPX = False

# ---------------------------------------------------------------------------
# Configurable paths — module-level so tests can monkeypatch them.
# ---------------------------------------------------------------------------

JANITOR_BASE: pathlib.Path = (
    pathlib.Path.home()
    / "Library/Application Support/Code/User/globalStorage"
    / "saoudrizwan.claude-dev/janitor"
)
GORTEX_BIN: pathlib.Path = pathlib.Path("/opt/homebrew/bin/gortex")
GORTEX_PID_FILE: pathlib.Path = pathlib.Path.home() / ".gortex" / "cache" / "daemon.pid"

# Endpoints to health-check. The dashboard itself (:4001) is omitted —
# it's always reachable if we're generating this response.
ENDPOINTS: list[dict[str, Any]] = [
    {"key": "ollama",       "name": "Ollama",        "port": 11434, "path": "/api/version"},
    # LiteLLM's plain `/health` fires a live test completion against every
    # configured model deployment (all local models AND all Claude aliases)
    # on every single call. Polled every 3s by /monitor/_live, that turns an
    # idle browser tab into a continuous real-money Claude-spend generator
    # and saturates the single-threaded Ollama backend. `/health/liveliness`
    # is the lightweight liveness probe with no backend fan-out.
    {"key": "litellm",      "name": "LiteLLM proxy", "port": 4000,  "path": "/health/liveliness"},
    {"key": "claude_proxy", "name": "Claude proxy",  "port": 4002,  "path": "/health"},
]


# ---------------------------------------------------------------------------
# HTTP probe
# ---------------------------------------------------------------------------

def _probe_http(url: str, timeout: float = 0.5) -> tuple[bool, int | None]:
    """Return (reachable, http_status). Never raises."""
    try:
        if _HAS_HTTPX:
            r = _httpx.get(url, timeout=timeout)
            return r.status_code < 500, r.status_code
        req = _urllib_request.Request(url)  # type: ignore[name-defined]
        with _urllib_request.urlopen(req, timeout=timeout) as r:  # type: ignore[name-defined]
            return r.status < 500, r.status
    except Exception:
        return False, None


def health_checks() -> list[dict[str, Any]]:
    """Probe each service endpoint and return one status row per entry."""
    rows: list[dict[str, Any]] = []
    for ep in ENDPOINTS:
        url = f"http://127.0.0.1:{ep['port']}{ep['path']}"
        ok, code = _probe_http(url)
        rows.append({
            "key":         ep["key"],
            "name":        ep["name"],
            "port":        ep["port"],
            "ok":          ok,
            "status_code": code,
            "status":      "UP" if ok else ("TIMEOUT" if code is None else str(code)),
        })
    return rows


# ---------------------------------------------------------------------------
# Gortex daemon status
# ---------------------------------------------------------------------------
#
# `running` must never depend on shelling out to `gortex daemon status`:
# that call round-trips the daemon's control socket, and when the daemon's
# indexer is backlogged (e.g. watching a large vendored directory) it can
# take minutes to answer. Polling that on every 3s HTMX tick both (a) misreports
# a live-but-slow daemon as "not running" and (b) piles up connections on the
# daemon faster than a backlogged daemon can clean them up. Liveness is
# instead a pidfile + kill(pid, 0) check — no subprocess, no socket I/O.
# The richer fields (uptime, state, session counts) still need the CLI
# round-trip, so that call is capped to once per GORTEX_DETAIL_TTL_SEC and
# its result cached; a slow daemon degrades that cache's freshness without
# ever blocking a poll.

GORTEX_DETAIL_TTL_SEC = 30.0
GORTEX_DETAIL_TIMEOUT_SEC = 3.0

# detail_ok distinguishes "we asked and got a real answer" from "we don't
# know" — a busy daemon that just times out on `daemon status` must not be
# rendered the same as a daemon that positively reported 0 sessions. Without
# this, a transient slow response gets misread as "Cline disconnected."
_GORTEX_DETAIL_DEFAULT: dict[str, Any] = {
    "uptime": None, "state": None, "mcp_sessions": 0, "cline_sessions": 0,
    "raw": "", "detail_ok": False,
}
_gortex_detail_cache: dict[str, Any] = dict(_GORTEX_DETAIL_DEFAULT)
_gortex_detail_cached_at: float = 0.0


def _pid_alive(pid: int) -> bool:
    """True if `pid` names a live process. Never raises."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else
    except Exception:
        return False
    return True


def gortex_liveness() -> dict[str, Any]:
    """Instant "is the daemon process alive" check via its pidfile.

    Returns {"running": bool, "pid": int | None}. No subprocess, no
    socket I/O — safe to call on every poll. Never raises.
    """
    try:
        pid = int(GORTEX_PID_FILE.read_text().strip())
    except Exception:
        return {"running": False, "pid": None}
    if not _pid_alive(pid):
        return {"running": False, "pid": None}
    return {"running": True, "pid": pid}


def _parse_gortex_status(raw: str) -> dict[str, Any]:
    """Parse the structured fields out of `gortex daemon status` stdout.

    The CLI prints a column-aligned summary block (`label<whitespace>value`,
    no colons) followed by a separate "MCP sessions:" box-drawn table with a
    `client` column -- there is no `--json` output to parse instead. This
    format is not a stable contract; if a `gortex` upgrade reformats it,
    these fields silently go back to their defaults (caught by detail_ok
    remaining True while mcp_sessions/cline_sessions read 0 for a daemon
    that's actually busy).
    """
    info: dict[str, Any] = {**_GORTEX_DETAIL_DEFAULT, "detail_ok": True}

    m = re.search(r"^\s*uptime\s+(\S+)", raw, re.MULTILINE)
    if m:
        info["uptime"] = m.group(1)

    m = re.search(r"^\s*state\s+(\S+)", raw, re.MULTILINE)
    if m:
        info["state"] = m.group(1).rstrip(",)")

    m = re.search(r"^\s*sessions\s+(\d+)", raw, re.MULTILINE)
    if m:
        info["mcp_sessions"] = int(m.group(1))

    # Rows in the "MCP sessions:" table look like:
    #   │ sess_xxxxxxxxxxxxxxxx │ Cline  │ 3.83.0  │    12m23s │ /path │
    # Count rows whose `client` column is "Cline" (case-insensitive). Must
    # search only the text *after* the "MCP sessions:" header -- one of the
    # tracked repos is itself named "cline", and its row in the "tracked
    # repos" table (`│ cline    │ github  │ ...`) otherwise also matches.
    _, _, sessions_table = raw.partition("MCP sessions:")
    info["cline_sessions"] = len(re.findall(r"│\s*Cline\s*│", sessions_table, flags=re.IGNORECASE))
    return info


def _fetch_gortex_detail(timeout: float = GORTEX_DETAIL_TIMEOUT_SEC) -> dict[str, Any]:
    """Run `gortex daemon status` once and parse it. Only ever called from
    the rate-limited cache refresh below — never on every poll. `detail_ok`
    is False on any failure so the caller can tell "confirmed" from
    "unknown" instead of reading the zeroed-out fields as a real answer."""
    try:
        result = subprocess.run(
            [str(GORTEX_BIN), "daemon", "status"],
            capture_output=True, text=True, timeout=timeout,
        )
        raw = result.stdout.strip()
        if result.returncode != 0 or not raw:
            return {"detail_ok": False, "raw": result.stderr.strip() or raw}
        return _parse_gortex_status(raw)
    except Exception as exc:
        return {"detail_ok": False, "raw": str(exc)}


def _refresh_gortex_detail_cache(force: bool = False) -> dict[str, Any]:
    """Refresh the cached detail fields at most once per GORTEX_DETAIL_TTL_SEC.

    A failed/timed-out refresh only updates `detail_ok`/`raw` — it must not
    clobber the last known-good uptime/state/session counts, otherwise one
    slow poll flashes "0 sessions" even though nothing about Cline's actual
    connection changed.
    """
    global _gortex_detail_cache, _gortex_detail_cached_at
    now = time.time()
    if force or (now - _gortex_detail_cached_at) >= GORTEX_DETAIL_TTL_SEC:
        result = _fetch_gortex_detail()
        _gortex_detail_cached_at = now
        if result.get("detail_ok"):
            _gortex_detail_cache = result
        else:
            _gortex_detail_cache = {**_gortex_detail_cache, "detail_ok": False, "raw": result.get("raw", "")}
    return _gortex_detail_cache


def gortex_mcp_status() -> dict[str, Any]:
    """Combined gortex status for the dashboard.

    Returns a dict with keys:
      running       bool          from the pidfile liveness check (always fresh)
      pid           int | None    ditto
      uptime        str | None    from the cached `daemon status` detail
      state         str | None    ("ready", "warmup", …), cached
      mcp_sessions  int           total MCP sessions in daemon, cached
      cline_sessions int          sessions whose description contains "cline", cached
      detail_ok     bool          True iff mcp_sessions/cline_sessions are a confirmed
                                   answer from `daemon status`, not a stale/default 0 —
                                   callers must not treat detail_ok=False + cline_sessions=0
                                   as "confirmed disconnected"
      raw           str           raw stdout/error for debugging, cached
    Never raises.
    """
    live = gortex_liveness()
    detail = _refresh_gortex_detail_cache() if live["running"] else dict(_GORTEX_DETAIL_DEFAULT)
    return {**detail, **live}


# ---------------------------------------------------------------------------
# Context Janitor state
# ---------------------------------------------------------------------------

def janitor_state() -> dict[str, Any]:
    """Inspect the most-recently-modified Cline task directory under JANITOR_BASE.

    Returns a dict with keys:
      enabled         bool   True if any entry files or context pack exist
      headroom_enabled bool  Always True — HeadroomAdapter is default-on
      active_task_id  str | None
      entry_count     int    JSON entry files (excluding active-context-pack.json)
      pack_exists     bool   True if active-context-pack.json is present
      task_dir_mtime  float | None  mtime of the task directory
    Never raises.
    """
    result: dict[str, Any] = {
        "enabled": False,
        "headroom_enabled": True,
        "active_task_id": None,
        "entry_count": 0,
        "pack_exists": False,
        "task_dir_mtime": None,
    }
    try:
        base = JANITOR_BASE
        if not base.is_dir():
            return result
        task_dirs = [d for d in base.iterdir() if d.is_dir()]
        if not task_dirs:
            return result

        task_dir = max(task_dirs, key=lambda d: d.stat().st_mtime)
        result["active_task_id"] = task_dir.name
        result["task_dir_mtime"] = task_dir.stat().st_mtime

        pack_path = task_dir / "active-context-pack.json"
        entry_files = [
            f for f in task_dir.glob("*.json")
            if f.name != "active-context-pack.json"
        ]
        result["entry_count"] = len(entry_files)
        result["pack_exists"] = pack_path.exists()
        result["enabled"] = result["entry_count"] > 0 or result["pack_exists"]
    except Exception:
        pass
    return result


# ---------------------------------------------------------------------------
# Call log + summary
# ---------------------------------------------------------------------------

def recent_calls(limit: int = 15, today_only: bool = True) -> list[dict[str, Any]]:
    """Return recent API calls from cost.db, newest first.

    Imports cost.ingest lazily so DB_PATH monkeypatching in tests works
    correctly (the monkeypatch must be applied before the first connect()).
    """
    try:
        from cost.ingest import connect  # noqa: PLC0415  (lazy import intentional)
        conn = connect()
        if today_only:
            cutoff = int(time.time()) - 86400
            sql = (
                "SELECT id, ts, model, tier, input_tok, output_tok, actual_cost, "
                "shadow_cost, latency_ms, route_reason, task_id "
                "FROM requests WHERE ts >= ? ORDER BY id DESC LIMIT ?"
            )
            params: tuple[Any, ...] = (cutoff, limit)
        else:
            sql = (
                "SELECT id, ts, model, tier, input_tok, output_tok, actual_cost, "
                "shadow_cost, latency_ms, route_reason, task_id "
                "FROM requests ORDER BY id DESC LIMIT ?"
            )
            params = (limit,)
        rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
        conn.close()
        for r in rows:
            r["ts_human"] = dt.datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S")
            r["output_maxed"] = r["output_tok"] >= 8192
        return rows
    except Exception:
        return []


def today_summary() -> dict[str, Any]:
    """Aggregate stats for the last 24 h from cost.db."""
    try:
        from cost.savings import summarize  # noqa: PLC0415
        s = summarize(1)
        return {
            "total_requests":    s.total_requests,
            "total_input_tokens": s.total_input_tokens,
            "total_output_tokens": s.total_output_tokens,
            "actual_spend_usd":  float(s.actual_spend_usd),
            "shadow_spend_usd":  float(s.shadow_spend_usd),
            "savings_usd":       float(s.savings_usd),
            "savings_pct":       float(s.savings_pct),
            "by_tier": {
                k: {"requests": v.requests} for k, v in s.by_tier.items()
            },
        }
    except Exception:
        return {
            "total_requests": 0, "total_input_tokens": 0, "total_output_tokens": 0,
            "actual_spend_usd": 0.0, "shadow_spend_usd": 0.0,
            "savings_usd": 0.0, "savings_pct": 0.0, "by_tier": {},
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def monitor_data() -> dict[str, Any]:
    """Collect all live data in one call.

    Used by both the HTMX-polled fragment and the JSON endpoint so both
    surfaces always show identical data from a single round of queries.
    """
    return {
        "health":       health_checks(),
        "gortex":       gortex_mcp_status(),
        "janitor":      janitor_state(),
        "calls":        recent_calls(),
        "summary":      today_summary(),
        "generated_at": int(time.time()),
    }
