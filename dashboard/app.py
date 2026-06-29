"""FastAPI + HTMX dashboard at http://127.0.0.1:4001.

Pages:
  /           live stats (HTMX polling), routing pie, recent requests
  /compare    A/B compare form; stored history
  /compare/{id}  side-by-side view of one comparison

The dashboard reads cost/cost.db (populated by the LiteLLM router callback)
and also drives the A/B comparator via compare/ab.py.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import time
from typing import Any

# httpx is bundled in the dashboard venv; fall back to urllib so the
# module can be imported in a plain stdlib environment (e.g. tests).
try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False
    import urllib.request as _urllib_request
    import urllib.error as _urllib_error

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cost.ingest import connect       # noqa: E402
from cost.pricing import sonnet_rate  # noqa: E402
from cost.savings import summarize    # noqa: E402
from compare.ab import run as ab_run  # noqa: E402

# Mirror of the in-flight registry maintained by the LiteLLM-side
# router callback. Written by router/route_by_size.py:_flush_active(),
# read here on every /stats poll. The two processes never share
# memory, so this file is the IPC channel.
ACTIVE_PATH = REPO_ROOT / ".logs" / "active.json"

app = FastAPI(title="MacM4LocalAgent Dashboard")
templates = Jinja2Templates(directory=str(REPO_ROOT / "dashboard" / "templates"))
# Disable Jinja2's template cache: avoids an upstream LRUCache hash bug seen on
# Python 3.14, and the perf hit is negligible for a tiny local dashboard.
templates.env.cache = None
app.mount("/static", StaticFiles(directory=str(REPO_ROOT / "dashboard" / "static")), name="static")


def _fmt_ts(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _estimate_active_cost(entry: dict[str, Any]) -> float:
    """Lower-bound cost estimate for an in-flight call.

    Local tiers are free, so always return 0.0. For Claude calls we
    have the input-token estimate the router stamped at routing time
    (`route_tokens_estimated`), but no output tokens yet -- they
    only land on success. We use the canonical Sonnet rate as the
    shadow baseline; the real bill might be Haiku/Opus, but this
    column is explicitly labelled "(est)" in the UI to make the
    approximation visible. The post-completion `actual_cost` in
    `requests` is always authoritative."""
    if entry.get("tier") != "claude":
        return 0.0
    in_tok = int(entry.get("in_tok_est", 0) or 0)
    if in_tok <= 0:
        return 0.0
    rate = sonnet_rate()
    return in_tok * rate.input


def _load_active() -> list[dict[str, Any]]:
    """Read the router's in-flight snapshot. Returns [] on any failure
    -- the dashboard must never crash because the proxy hasn't
    written this file yet (cold-start ordering) or because the file
    is being rewritten mid-poll."""
    try:
        raw = ACTIVE_PATH.read_text()
    except FileNotFoundError:
        return []
    except Exception:
        return []
    try:
        rows = json.loads(raw)
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    now = time.time()
    out: list[dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        started = float(r.get("started", now))
        r["elapsed_sec"] = max(0.0, now - started)
        r["est_cost"] = _estimate_active_cost(r)
        out.append(r)
    # Newest first matches the recent-requests table.
    out.sort(key=lambda r: r.get("started", 0), reverse=True)
    return out


_BACKEND_STATUS_DEFAULT: dict[str, Any] = {
    "active_backend": "unknown",
    "state": "UNKNOWN",
    "pending_requests": 0,
    "memory_pressure": "unknown",
    "last_switch_duration_ms": None,
    "turbo_enabled": False,
    "context_compression_enabled": False,
}


def _load_backend_status() -> dict[str, Any]:
    """GET http://127.0.0.1:4000/backend/status and return its JSON.

    Returns a safe default dict on any failure (connection refused,
    timeout, unexpected response shape) so the dashboard card is always
    renderable even when the proxy is not running.

    The 300 ms timeout is deliberate: the dashboard poll loop runs
    every 5 seconds; a slow proxy must not stall the render cycle.
    """
    url = "http://127.0.0.1:4000/backend/status"
    try:
        if _HAS_HTTPX:
            resp = _httpx.get(url, timeout=0.3)
            if resp.status_code != 200:
                return dict(_BACKEND_STATUS_DEFAULT)
            data = resp.json()
        else:
            req = _urllib_request.Request(url)
            with _urllib_request.urlopen(req, timeout=0.3) as r:
                if r.status != 200:
                    return dict(_BACKEND_STATUS_DEFAULT)
                data = json.loads(r.read().decode())
    except Exception:
        return dict(_BACKEND_STATUS_DEFAULT)

    if not isinstance(data, dict):
        return dict(_BACKEND_STATUS_DEFAULT)

    # Merge into defaults so the template never sees a missing key.
    merged = dict(_BACKEND_STATUS_DEFAULT)
    merged.update(data)
    return merged


@app.get("/backend-status-card", response_class=HTMLResponse)
def backend_status_card(request: Request) -> Any:
    """HTMX-polled fragment that renders the backend state card.

    Polled every 5 seconds by the placeholder div in index.html via
    ``hx-trigger="load, every 5s"``.  Returns a self-contained HTML
    partial — no page chrome, no layout wrapper.
    """
    status = _load_backend_status()
    return templates.TemplateResponse(
        request, "backend_status_card.html",
        {"status": status},
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    s7 = summarize(7)
    return templates.TemplateResponse(
        request, "index.html", {"s7": s7},
    )


@app.get("/stats", response_class=HTMLResponse)
def stats_fragment(request: Request) -> Any:
    """HTMX-polled live tickers."""
    today  = summarize(1)
    week   = summarize(7)
    alltime = summarize(None)

    conn = connect()
    recent = [dict(r) for r in conn.execute(
        "SELECT id, ts, model, tier, input_tok, output_tok, actual_cost, "
        "latency_ms, route_reason, task_id "
        "FROM requests ORDER BY id DESC LIMIT 25"
    ).fetchall()]
    conn.close()
    for r in recent:
        r["ts_human"] = _fmt_ts(r["ts"])

    active = _load_active()

    return templates.TemplateResponse(
        request, "_stats.html",
        {"today": today, "week": week, "alltime": alltime,
         "recent": recent, "active": active},
    )


@app.get("/api/stats")
def api_stats() -> JSONResponse:
    return JSONResponse({
        "today": summarize(1),
        "week":  summarize(7),
        "month": summarize(30),
        "all":   summarize(None),
        "active": _load_active(),
    })


# ----- Cline tasks ----------------------------------------------------------
#
# Cline tasks are turns-of-the-same-task grouped by `task_id`. The router
# stamps a 16-hex SHA256 fingerprint of the user's <task> envelope into
# every request that came through Cline; this UI rolls those rows up.

def _task_summary_rows() -> list[dict[str, Any]]:
    """Build the per-task rollup. One row per distinct task_id, newest
    first. Groups every Cline-tagged request and aggregates token,
    cost, latency, and tier breakdowns."""
    conn = connect()
    rows = conn.execute(
        """
        SELECT
            task_id,
            -- A task's text is the same on every turn, but COALESCE
            -- defends against the (impossible-but-cheap-to-handle)
            -- case of NULL on some turns. MAX picks the longest non-null.
            COALESCE(MAX(task_text), '(no task text)') AS task_text,
            MIN(ts)             AS started_ts,
            MAX(ts)             AS ended_ts,
            COUNT(*)            AS turns,
            SUM(input_tok)      AS input_tok,
            SUM(output_tok)     AS output_tok,
            SUM(actual_cost)    AS actual_cost,
            SUM(shadow_cost)    AS shadow_cost,
            SUM(latency_ms)     AS total_latency_ms
        FROM requests
        WHERE task_id IS NOT NULL
        GROUP BY task_id
        ORDER BY MAX(ts) DESC
        LIMIT 50
        """
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        # Tier counts are computed in a follow-up query because SQLite
        # doesn't have GROUP_CONCAT-with-counts as a clean primitive.
        # 50 rows * 1 query = 50 queries; acceptable for a local
        # dashboard. If this becomes a bottleneck we can switch to
        # a single window-function query.
        tier_rows = conn.execute(
            "SELECT tier, COUNT(*) FROM requests WHERE task_id = ? GROUP BY tier",
            (d["task_id"],),
        ).fetchall()
        d["tier_counts"] = {tier: n for tier, n in tier_rows}
        d["started_human"] = _fmt_ts(d["started_ts"])
        d["wall_seconds"] = max(0, d["ended_ts"] - d["started_ts"])
        # Truncate displayed task text. The DB-side cap is 500 chars
        # already, but the table cell looks bad with anything > ~100.
        text = d["task_text"]
        d["task_text_short"] = (text[:100] + "...") if len(text) > 100 else text
        out.append(d)
    conn.close()
    return out


@app.get("/tasks", response_class=HTMLResponse)
def tasks_index(request: Request) -> Any:
    """Full page; the actual table is loaded via HTMX into a placeholder."""
    return templates.TemplateResponse(request, "tasks_index.html", {})


@app.get("/tasks/_list", response_class=HTMLResponse)
def tasks_list_fragment(request: Request) -> Any:
    """HTMX-polled fragment with the live task list. Returns just the
    table HTML so the parent page can swap it without re-rendering
    the chrome."""
    return templates.TemplateResponse(
        request, "_tasks_list.html",
        {"tasks": _task_summary_rows()},
    )


@app.get("/tasks/{task_id}", response_class=HTMLResponse)
def tasks_one(request: Request, task_id: str) -> Any:
    """Drill-down: static header + HTMX-polled live turns table.
    The page shell loads instantly; the fragment at /tasks/{id}/_live
    fills in the turns and keeps them up-to-date every 3 s."""
    conn = connect()
    # Pull just the static header fields — task text and start time.
    # Everything that changes (turns, tokens, latency) lives in the
    # polled _live fragment so we don't duplicate query logic here.
    text_row = conn.execute(
        "SELECT task_text, MIN(ts) FROM requests "
        "WHERE task_id = ? AND task_text IS NOT NULL LIMIT 1",
        (task_id,),
    ).fetchone()
    conn.close()

    # This is an aggregate query (MIN), so it ALWAYS returns exactly one
    # row -- `(None, None)` when the task_id matches nothing. Detect
    # "not found" via a NULL task_text, not row presence, otherwise the
    # 404 branch never fires and `int(started_ts)` below blows up on None.
    has_db_row = text_row is not None and text_row[0] is not None

    # Allow opening the page for an in-flight task that has no DB rows yet.
    active_for_task = [a for a in _load_active() if a.get("task_id") == task_id]
    if not has_db_row and not active_for_task:
        return HTMLResponse(f"<p>task {task_id} not found</p>", status_code=404)

    task_text = (text_row[0] if has_db_row else None) or (
        active_for_task[0].get("task_text_short") if active_for_task else "(no task text)"
    )
    started_ts = (text_row[1] if has_db_row else None) or (
        active_for_task[0]["started"] if active_for_task else time.time()
    )
    summary = {
        "task_text": task_text,
        "started_human": _fmt_ts(int(started_ts)),
    }
    return templates.TemplateResponse(
        request, "tasks_one.html",
        {"task_id": task_id, "summary": summary},
    )


@app.get("/tasks/{task_id}/_live", response_class=HTMLResponse)
def tasks_one_live(request: Request, task_id: str) -> Any:
    """HTMX-polled fragment: turns table + in-flight row for one task.
    Replaces the full #live-region on every poll so the user sees new
    turns appear and the in-flight elapsed counter tick up."""
    conn = connect()
    turns_rows = conn.execute(
        "SELECT id, ts, model, tier, input_tok, output_tok, actual_cost, "
        "shadow_cost, latency_ms, route_reason "
        "FROM requests WHERE task_id = ? ORDER BY id DESC",
        (task_id,),
    ).fetchall()
    conn.close()

    # Rows are newest-first from the DB; compute the logical turn number
    # (1-based, ascending by time) before reversing so #1 is the oldest.
    total = len(turns_rows)
    turns = []
    for i, r in enumerate(turns_rows):
        d = dict(r)
        d["turn_num"] = total - i   # newest row gets the highest number
        d["ts_human"] = _fmt_ts(d["ts"])
        turns.append(d)

    # Find an active request belonging to this task.
    active_for_task = [a for a in _load_active() if a.get("task_id") == task_id]
    in_flight = active_for_task[0] if active_for_task else None

    summary: dict[str, Any] = {
        "turns": len(turns),
        "wall_seconds": max(0, turns[-1]["ts"] - turns[0]["ts"]) if turns else 0,
        "input_tok": sum(r["input_tok"] for r in turns),
        "output_tok": sum(r["output_tok"] for r in turns),
        "actual_cost": sum(r["actual_cost"] for r in turns),
        "shadow_cost": sum(r["shadow_cost"] for r in turns),
        "tier_counts": {},
    }
    for r in turns:
        summary["tier_counts"][r["tier"]] = summary["tier_counts"].get(r["tier"], 0) + 1

    return templates.TemplateResponse(
        request, "_task_live.html",
        {"task_id": task_id, "turns": turns, "in_flight": in_flight,
         "summary": summary},
    )


@app.get("/compare", response_class=HTMLResponse)
def compare_index(request: Request) -> Any:
    conn = connect()
    rows = [dict(r) for r in conn.execute(
        "SELECT id, ts, prompt, judge_score, local_ms, claude_ms, local_cost, claude_cost "
        "FROM comparisons ORDER BY id DESC LIMIT 50"
    ).fetchall()]
    conn.close()
    for r in rows:
        r["ts_human"] = _fmt_ts(r["ts"])
        r["prompt_short"] = (r["prompt"][:140] + "...") if len(r["prompt"]) > 140 else r["prompt"]
    return templates.TemplateResponse(request, "compare_index.html", {"rows": rows})


@app.post("/compare/run")
def compare_run(prompt: str = Form(...)) -> RedirectResponse:
    res = ab_run(prompt)
    return RedirectResponse(url=f"/compare/{res['id']}", status_code=303)


@app.get("/compare/{cmp_id}", response_class=HTMLResponse)
def compare_one(request: Request, cmp_id: int) -> Any:
    conn = connect()
    row = conn.execute("SELECT * FROM comparisons WHERE id = ?", (cmp_id,)).fetchone()
    conn.close()
    if not row:
        return HTMLResponse(f"<p>comparison {cmp_id} not found</p>", status_code=404)
    r = dict(row)
    r["ts_human"] = _fmt_ts(r["ts"])
    return templates.TemplateResponse(request, "compare_one.html", {"r": r})


# ----- M7: Rich model metadata for Cline (and other clients) --------------
#
# LiteLLM already exposes /v1/models, but that endpoint:
#   - Doesn't include local vs cloud tier classification
#   - Reports no context window
#   - Has no warm / loaded signal
#   - Returns Anthropic/MLX/Ollama pricing inconsistently
#
# /api/macm4-models is a thin JSON shape designed for Cline's
# routing-tier badge (C7) and cost-savings sidebar (C6). Polled
# at most once per session by the Cline fork; cheap to compute on
# every call so we don't bother caching.

# Pinned tier metadata. Values match config/litellm-config.yaml and
# the Ollama / MLX backend Modelfiles. Update both in lockstep if
# either side changes.
_MACM4_TIERS = [
    {
        "id": "local-fast",
        "tier": "local",
        "backend": "mlx",
        "backend_url": "http://127.0.0.1:8081",
        "model_repo_env": "MLX_REPO",
        "context_window": 16384,
        "max_output_tokens": 6144,
        "tokens_per_second_est": 70,
        "pricing": {
            "input_per_million_usd": 0.0,
            "output_per_million_usd": 0.0,
        },
        "capabilities": {
            "streaming": True,
            "tool_use_native": False,
            "vision": False,
        },
    },
    {
        "id": "local-long",
        "tier": "local",
        "backend": "ollama",
        "backend_url": "http://127.0.0.1:11434",
        "model_tag_env": "OLLAMA_TAG",
        "context_window": 131072,
        "max_output_tokens": 6144,
        "tokens_per_second_est": 12,
        "pricing": {
            "input_per_million_usd": 0.0,
            "output_per_million_usd": 0.0,
        },
        "capabilities": {
            "streaming": True,
            "tool_use_native": False,
            "vision": False,
        },
    },
    {
        "id": "claude-haiku-4-5",
        "tier": "cloud",
        "backend": "anthropic",
        "model_id": "claude-haiku-4-5",
        "context_window": 200000,
        "max_output_tokens": 8192,
        "tokens_per_second_est": 80,
        "pricing": {
            "input_per_million_usd": 1.0,
            "output_per_million_usd": 5.0,
        },
        "capabilities": {
            "streaming": True,
            "tool_use_native": True,
            "vision": True,
        },
    },
    {
        "id": "claude-sonnet-4-6",
        "tier": "cloud",
        "backend": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "context_window": 200000,
        "max_output_tokens": 8192,
        "tokens_per_second_est": 60,
        "pricing": {
            "input_per_million_usd": 3.0,
            "output_per_million_usd": 15.0,
        },
        "capabilities": {
            "streaming": True,
            "tool_use_native": True,
            "vision": True,
        },
    },
    {
        "id": "claude-opus-4-7",
        "tier": "cloud",
        "backend": "anthropic",
        "model_id": "claude-opus-4-7",
        "context_window": 1000000,
        "max_output_tokens": 8192,
        "tokens_per_second_est": 30,
        "pricing": {
            "input_per_million_usd": 5.0,
            "output_per_million_usd": 25.0,
        },
        "capabilities": {
            "streaming": True,
            "tool_use_native": True,
            "vision": True,
        },
    },
    {
        "id": "claude-code",
        "tier": "cloud",
        "backend": "anthropic",
        "context_window": 1000000,
        "max_output_tokens": 8192,
        "tokens_per_second_est": 30,
        "pricing": {
            "input_per_million_usd": 5.0,
            "output_per_million_usd": 25.0,
        },
        "capabilities": {
            "streaming": True,
            "tool_use_native": True,
            "vision": True,
        },
        "note": "alias for the default Claude tier; currently Opus 4.7",
    },
    {
        "id": "hybrid-auto",
        "tier": "router",
        "backend": "litellm-proxy",
        "backend_url": "http://127.0.0.1:4000",
        "context_window": 1000000,
        "max_output_tokens": 8192,
        "pricing": {
            "input_per_million_usd": None,
            "output_per_million_usd": None,
        },
        "capabilities": {
            "streaming": True,
            "tool_use_native": True,
            "vision": True,
        },
        "note": "size + complexity router; picks the cheapest tier that fits",
    },
]


def _probe_ollama_loaded_models() -> set[str]:
    """Return the set of model tags currently loaded in Ollama. Uses a
    300ms timeout so a hung Ollama can't slow down the metadata
    endpoint -- on timeout we just report nothing loaded."""
    try:
        import httpx  # local import: httpx is in the dashboard venv

        resp = httpx.get("http://127.0.0.1:11434/api/ps", timeout=0.3)
        if resp.status_code != 200:
            return set()
        data = resp.json()
        return {m.get("name", "") for m in data.get("models", []) if m.get("name")}
    except Exception:
        return set()


def _resolve_ollama_tag() -> str:
    """Return the tag the operator expects Ollama to be running.

    Prefers an explicit OLLAMA_TAG env var (matches the launchd plist
    contract), but falls back to config/detected.env so the warm probe
    still works when the dashboard process inherits a stripped
    environment (e.g. manual invocation, login-shell quirks)."""
    import os
    explicit = os.environ.get("OLLAMA_TAG", "").strip()
    if explicit:
        return explicit
    try:
        env_path = pathlib.Path(__file__).resolve().parent.parent / "config" / "detected.env"
        if not env_path.is_file():
            return ""
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if key.strip() == "OLLAMA_TAG":
                val = val.strip().strip('"').strip("'")
                return val
    except Exception:
        pass
    return ""


def _macm4_models_payload() -> dict[str, Any]:
    """Build the response body. Centralised so tests can call it
    without spinning up the ASGI app."""
    try:
        loaded_ollama = _probe_ollama_loaded_models()
    except Exception:
        loaded_ollama = set()
    expected_ollama = _resolve_ollama_tag()

    models: list[dict[str, Any]] = []
    for tier in _MACM4_TIERS:
        entry = dict(tier)
        # Add a "warm" flag for local tiers based on Ollama /api/ps
        # output. MLX doesn't have a comparable endpoint so we trust
        # the launchd plist (KeepAlive=true) and report warm=true
        # unconditionally for local-fast.
        if tier["tier"] == "local":
            if tier["backend"] == "ollama":
                entry["warm"] = bool(expected_ollama) and expected_ollama in loaded_ollama
            elif tier["backend"] == "mlx":
                # The MLX server stays warm for the process lifetime
                # of the launchd-managed binary. Best-effort port
                # check would add latency for little gain.
                entry["warm"] = True
            else:
                entry["warm"] = False
        models.append(entry)
    return {
        "data": models,
        "object": "list",
        "_meta": {
            "schema_version": 1,
            "generated_at": int(time.time()),
            "dashboard_url": "http://127.0.0.1:4001",
            "proxy_url": "http://127.0.0.1:4000",
        },
    }


@app.get("/api/macm4-models")
def macm4_models() -> JSONResponse:
    """Rich model metadata for Cline (and other clients) to drive UI
    badges, cost-savings widgets, and warm-up indicators."""
    return JSONResponse(_macm4_models_payload())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host="127.0.0.1", port=4001, reload=False)
