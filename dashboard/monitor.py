"""Live data aggregation for the Cline Monitor tab (/monitor).

All public functions return safe defaults on any error — the dashboard
must never crash because an external service is down or a file is missing.
"""

from __future__ import annotations

import datetime as dt
import json
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

# Endpoints to health-check. The dashboard itself (:4001) is omitted —
# it's always reachable if we're generating this response.
ENDPOINTS: list[dict[str, Any]] = [
    {"key": "ollama",       "name": "Ollama",        "port": 11434, "path": "/api/version"},
    {"key": "litellm",      "name": "LiteLLM proxy", "port": 4000,  "path": "/health"},
    {"key": "claude_proxy", "name": "Claude proxy",  "port": 4002,  "path": "/health"},
    {"key": "mlx",          "name": "MLX server",    "port": 8081,  "path": "/health"},
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

def gortex_mcp_status() -> dict[str, Any]:
    """Run ``gortex daemon status`` and parse structured fields.

    Returns a dict with keys:
      running       bool
      pid           int | None
      uptime        str | None
      state         str | None  ("ready", "warmup", …)
      mcp_sessions  int          total MCP sessions in daemon
      cline_sessions int         sessions whose description contains "cline"
      raw           str          raw stdout for debugging
    Never raises.
    """
    default: dict[str, Any] = {
        "running": False, "pid": None, "uptime": None, "state": None,
        "mcp_sessions": 0, "cline_sessions": 0, "raw": "",
    }
    try:
        result = subprocess.run(
            [str(GORTEX_BIN), "daemon", "status"],
            capture_output=True, text=True, timeout=3.0,
        )
        raw = result.stdout.strip()
        if result.returncode != 0 or not raw:
            return {**default, "raw": result.stderr.strip() or raw}

        info: dict[str, Any] = {**default, "running": True, "raw": raw}

        m = re.search(r"pid:\s*(\d+)", raw)
        if m:
            info["pid"] = int(m.group(1))

        m = re.search(r"uptime:\s*([^,\n]+)", raw)
        if m:
            info["uptime"] = m.group(1).strip()

        m = re.search(r"state:\s*(\S+)", raw)
        if m:
            info["state"] = m.group(1).rstrip(",)")

        m = re.search(r"(\d+)\s+MCP\s+sessions?", raw)
        if m:
            info["mcp_sessions"] = int(m.group(1))

        # Count session descriptors that contain "cline" (case-insensitive).
        info["cline_sessions"] = len(re.findall(r"cli:cline", raw, flags=re.IGNORECASE))
        return info
    except Exception as exc:
        return {**default, "raw": str(exc)}


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
