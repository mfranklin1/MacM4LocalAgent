"""Integration tests for the FastAPI dashboard."""

from __future__ import annotations

import json
import pathlib
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from compare import ab
from cost import ingest
from dashboard import app as dash_app


@pytest.fixture
def client(tmp_db) -> TestClient:                                              # noqa: ARG001
    return TestClient(dash_app.app)


@pytest.fixture
def tmp_active(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    """Redirect the dashboard's view of `.logs/active.json` to a tmp file.
    Lets each test seed its own fixture without racing the real proxy
    or other tests."""
    path = tmp_path / "active.json"
    monkeypatch.setattr(dash_app, "ACTIVE_PATH", path, raising=True)
    return path


def _seed(now: int) -> None:
    ingest.record_request(model="local-agent",       tier="local-long",  in_tok=1000, out_tok=500, actual_cost=0.0,    latency_ms=100, ts=now-100, route_reason="<= 128k")
    ingest.record_request(model="ollama/qwen3-30b",  tier="local-long",  in_tok=2000, out_tok=800, actual_cost=0.0,    latency_ms=400, ts=now-200, route_reason="16k-128k")
    ingest.record_request(model="claude-sonnet-4-6", tier="claude",      in_tok=500,  out_tok=200, actual_cost=0.0045, latency_ms=900, ts=now-300, route_reason="complex")


def test_index_renders(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "Cost" in r.text
    assert "stats" in r.text   # the htmx target id


def test_stats_fragment(client: TestClient) -> None:
    _seed(int(time.time()))
    r = client.get("/stats")
    assert r.status_code == 200
    assert "Today"        in r.text
    assert "Last 7 days"  in r.text
    assert "claude"       in r.text
    assert "local-long"   in r.text
    # request rows
    assert "ollama/qwen3-30b" in r.text


def test_api_stats_json(client: TestClient) -> None:
    _seed(int(time.time()))
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    for window in ("today", "week", "month", "all"):
        assert window in body
    assert body["week"]["total_requests"] == 3


def test_compare_index_empty(client: TestClient) -> None:
    r = client.get("/compare")
    assert r.status_code == 200
    assert "No comparisons yet" in r.text


def test_compare_one_404(client: TestClient) -> None:
    r = client.get("/compare/9999")
    assert r.status_code == 404


def _resp(content: str, in_tok: int = 1, out_tok: int = 1) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok},
    })


def test_compare_run_creates_row(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        return _resp(f"answer for {body['model']}", 5, 7)

    real_client = httpx.Client

    class _MockClient(real_client):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _MockClient)

    r = client.post("/compare/run", data={"prompt": "say hi"}, follow_redirects=False)
    assert r.status_code in (302, 303)
    location = r.headers["location"]
    assert location.startswith("/compare/")

    one = client.get(location)
    assert one.status_code == 200
    assert "say hi" in one.text
    assert "answer for local-long"  in one.text
    assert "answer for claude-code" in one.text


# ---- /tasks pages ------------------------------------------------------------
#
# The Tasks views aggregate Cline traffic by task_id. Every test here seeds
# a representative shape (one Cline task with multiple turns + one non-Cline
# row that should be EXCLUDED from the task views).

def _seed_cline_task(now: int) -> None:
    """Seed one Cline task with three turns + one unrelated non-Cline row.
    Picked to exercise: tier rollup (local-long + claude), latency totals,
    NULL task_id exclusion, and ordering."""
    ingest.record_request(
        model="ollama/qwen3-coder-next:q4_K_M",
        tier="local-long",
        in_tok=13000, out_tok=120, actual_cost=0.0, latency_ms=2000,
        ts=now - 30,
        route_reason="cline-mode: cline+default: task=8 tok",
        task_id="abc123def4567890",
        task_text="Add a one-line comment to README.md",
    )
    ingest.record_request(
        model="ollama/qwen3-coder-next:q4_K_M",
        tier="local-long",
        in_tok=14000, out_tok=200, actual_cost=0.0, latency_ms=2200,
        ts=now - 20,
        route_reason="cline-mode: cline+default: task=8 tok",
        task_id="abc123def4567890",
        task_text="Add a one-line comment to README.md",
    )
    ingest.record_request(
        model="claude-sonnet-4-6",
        tier="claude",
        in_tok=15000, out_tok=400, actual_cost=0.0510, latency_ms=3500,
        ts=now - 10,
        route_reason="cline-mode: cline+sticky(abc123def4567890): traceback",
        task_id="abc123def4567890",
        task_text="Add a one-line comment to README.md",
    )
    # Non-Cline traffic: must NOT appear on /tasks because task_id IS NULL.
    ingest.record_request(
        model="local-agent",
        tier="local-long",
        in_tok=50, out_tok=10, actual_cost=0.0, latency_ms=200,
        ts=now - 5,
        route_reason="tokens 50 <= 128000",
    )


def test_tasks_index_renders(client: TestClient) -> None:
    r = client.get("/tasks")
    assert r.status_code == 200
    # Page includes the HTMX target div that polls /tasks/_list.
    assert 'hx-get="/tasks/_list"' in r.text


def test_tasks_list_empty(client: TestClient) -> None:
    """Fresh DB: no rows, the fragment shows the friendly empty-state."""
    r = client.get("/tasks/_list")
    assert r.status_code == 200
    assert "No Cline tasks yet" in r.text


def test_tasks_list_groups_by_task_id(client: TestClient) -> None:
    """Three turns of one task collapse to ONE row in the list, with
    aggregate counts and tier breakdown."""
    _seed_cline_task(int(time.time()))
    r = client.get("/tasks/_list")
    assert r.status_code == 200
    # Task text is rendered once (the rollup row).
    assert r.text.count("Add a one-line comment to README.md") == 1
    # Tier breakdown shows both local-long and claude with their counts.
    assert "local-long: 2" in r.text
    assert "claude: 1" in r.text
    # Task fingerprint is shown as a code block under the text.
    assert "abc123def4567890" in r.text


def test_tasks_list_excludes_non_cline_rows(client: TestClient) -> None:
    """Non-Cline rows have task_id=NULL and must not appear on /tasks --
    they aren't part of any agent task."""
    _seed_cline_task(int(time.time()))
    r = client.get("/tasks/_list")
    assert r.status_code == 200
    # The local-agent model from the non-Cline seed row must NOT appear.
    assert "local-agent" not in r.text


def test_tasks_one_renders_per_turn_breakdown(client: TestClient) -> None:
    """Drill-down: every turn for a task in chronological order with
    full route_reason, tier, and cost columns. The static page renders
    the header; the per-turn breakdown lives in the HTMX-polled
    /tasks/{id}/_live fragment."""
    _seed_cline_task(int(time.time()))
    page = client.get("/tasks/abc123def4567890")
    assert page.status_code == 200
    assert "Add a one-line comment to README.md" in page.text
    # The per-turn breakdown + counts live in the live fragment.
    r = client.get("/tasks/abc123def4567890/_live")
    assert r.status_code == 200
    # All three turns are rendered (turn count badge in the summary card).
    assert "<b>3</b>" in r.text or ">3<" in r.text
    # Each turn's route_reason should be present, including the sticky one.
    assert "cline+sticky" in r.text
    assert "cline+default" in r.text


def test_tasks_one_404_for_unknown_id(client: TestClient) -> None:
    r = client.get("/tasks/nonexistent")
    assert r.status_code == 404
    assert "not found" in r.text


def test_tasks_one_aggregates_cost_correctly(client: TestClient) -> None:
    """Summary card totals must equal the sum of the turn rows. Pinning
    this prevents future refactors from quietly breaking cost rollup.
    The summary card is rendered by the polled /tasks/{id}/_live
    fragment."""
    _seed_cline_task(int(time.time()))
    r = client.get("/tasks/abc123def4567890/_live")
    assert r.status_code == 200
    # The seeded data: 13000 + 14000 + 15000 = 42000 input, 120 + 200 + 400 = 720 out
    # Actual: 0 + 0 + 0.051 = $0.0510
    assert "42,000" in r.text
    assert "720" in r.text
    assert "0.0510" in r.text


def test_stats_recent_table_links_to_task(client: TestClient) -> None:
    """Cline rows in the /stats recent-requests table should link to
    /tasks/<id>; non-Cline rows show a dash."""
    _seed_cline_task(int(time.time()))
    r = client.get("/stats")
    assert r.status_code == 200
    # Cline rows: linkified short-id.
    assert 'href="/tasks/abc123def4567890"' in r.text
    # Non-Cline row: dash placeholder.
    assert "<span class=\"muted small\">-</span>" in r.text


# ---- /stats Active panel -----------------------------------------------------
#
# The router process writes .logs/active.json after every in-flight
# mutation; the dashboard reads it on every poll. The tests below
# stub that file directly to avoid spinning up the LiteLLM proxy.

def _seed_active_file(path: pathlib.Path, *, started_offset: float = -3.0) -> None:
    """Write a single in-flight Cline row that started `started_offset`
    seconds ago (negative = in the past). Picked to look like a real
    Cline turn routed to local-long."""
    payload = [{
        "call_id": "abc-call-id-9999",
        "started": time.time() + started_offset,
        "model": "ollama/qwen3-coder-next:q4_K_M",
        "tier": "local-long",
        "in_tok_est": 12345,
        "task_id": "abc123def4567890",
        "task_text_short": "Add a one-line comment to README.md",
        "route_reason": "cline-mode: cline+default: task=8 tok",
    }]
    path.write_text(json.dumps(payload))


def test_stats_active_panel_hidden_when_no_active(client: TestClient, tmp_active: pathlib.Path) -> None:
    """No file -> no Active heading. The whole panel is wrapped in
    `{% if active %}` so the recent-requests table stays at the top
    when nothing is in flight."""
    # tmp_active fixture pinned the path; we deliberately don't seed it.
    r = client.get("/stats")
    assert r.status_code == 200
    assert "Active" not in r.text or "active-heading" not in r.text


def test_stats_active_panel_renders_in_flight_row(client: TestClient, tmp_active: pathlib.Path) -> None:
    """A seeded active row should render with the model, tier badge,
    elapsed-time column, and a link back to the task page."""
    _seed_active_file(tmp_active, started_offset=-7.0)

    r = client.get("/stats")
    assert r.status_code == 200
    # Heading + count badge.
    assert "active-heading" in r.text
    assert "1 in flight" in r.text
    # Row content: model, tier, deep-link to /tasks/<id>.
    assert "ollama/qwen3-coder-next" in r.text
    assert "tier-local-long" in r.text
    assert 'href="/tasks/abc123def4567890"' in r.text
    # Token estimate is rendered with thousands-separator formatting.
    assert "12,345" in r.text


def test_stats_active_cost_zero_for_local(client: TestClient, tmp_active: pathlib.Path) -> None:
    """Local tiers cost nothing in flight or out. The estimated cost
    column must show $0.0000 to avoid scaring the user."""
    _seed_active_file(tmp_active)
    r = client.get("/stats")
    assert r.status_code == 200
    # The Active row must contain a $0.0000 cell. Other rows' costs are
    # also formatted to 4dp so we just assert presence here.
    assert "$0.0000" in r.text


def test_stats_active_cost_nonzero_for_claude(client: TestClient, tmp_active: pathlib.Path) -> None:
    """Claude calls have a lower-bound cost based on the input-token
    estimate at the canonical Sonnet rate. 1,000,000 in_tok_est at
    $3 / 1M = $3.00."""
    payload = [{
        "call_id": "claude-cid",
        "started": time.time() - 2.0,
        "model": "anthropic/claude-sonnet-4-6",
        "tier": "claude",
        "in_tok_est": 1_000_000,
        "task_id": None,
        "task_text_short": "",
        "route_reason": "complex; tokens 5",
    }]
    tmp_active.write_text(json.dumps(payload))

    r = client.get("/stats")
    assert r.status_code == 200
    # $3.0000 = 1M tokens * $3/1M (Sonnet input rate).
    assert "$3.0000" in r.text


def test_api_stats_includes_active(client: TestClient, tmp_active: pathlib.Path) -> None:
    """/api/stats JSON must surface the same active list the HTML
    fragment does, so external tools (statusline scripts, etc.) can
    poll it without scraping HTML."""
    _seed_active_file(tmp_active)

    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()
    assert "active" in body
    assert len(body["active"]) == 1
    assert body["active"][0]["call_id"] == "abc-call-id-9999"
    assert body["active"][0]["elapsed_sec"] >= 0


def test_api_stats_active_empty_when_file_missing(client: TestClient, tmp_active: pathlib.Path) -> None:
    """The dashboard must boot even if the proxy hasn't written the
    file yet (cold-start ordering). Empty list, not a 500."""
    # tmp_active path doesn't exist on disk yet.
    assert not tmp_active.exists()

    r = client.get("/api/stats")
    assert r.status_code == 200
    assert r.json()["active"] == []


def test_api_stats_active_empty_on_garbage_file(client: TestClient, tmp_active: pathlib.Path) -> None:
    """A half-written file (race between write and read) must not crash
    the dashboard. We swallow JSON errors and return an empty list."""
    tmp_active.write_text("{not valid json")

    r = client.get("/api/stats")
    assert r.status_code == 200
    assert r.json()["active"] == []


# ---- M7: rich model metadata endpoint ---------------------------------------

def test_macm4_models_endpoint_returns_all_tiers(client: TestClient, monkeypatch) -> None:
    """The /api/macm4-models endpoint must list every tier the proxy
    exposes, with the canonical id matching what shows up in
    litellm-config.yaml."""
    monkeypatch.setattr(dash_app, "_probe_ollama_loaded_models", lambda: set())
    r = client.get("/api/macm4-models")
    assert r.status_code == 200
    body = r.json()
    ids = {m["id"] for m in body["data"]}
    expected = {
        "local-long",
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-code",
        "hybrid-auto",
    }
    assert expected.issubset(ids), f"missing tiers: {expected - ids}"


def test_macm4_models_context_windows_match_litellm_config(client: TestClient, monkeypatch) -> None:
    """Context windows must agree with the proxy config so Cline's
    ContextManager doesn't truncate at the wrong threshold."""
    monkeypatch.setattr(dash_app, "_probe_ollama_loaded_models", lambda: set())
    r = client.get("/api/macm4-models")
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert by_id["local-long"]["context_window"] == 131_072
    # Anthropic 1M-context Opus is the cloud baseline.
    assert by_id["claude-opus-4-7"]["context_window"] == 1_000_000


def test_macm4_models_local_pricing_is_zero(client: TestClient, monkeypatch) -> None:
    """Local tiers must report zero pricing so cost-savings widgets
    don't double-count them as cloud."""
    monkeypatch.setattr(dash_app, "_probe_ollama_loaded_models", lambda: set())
    r = client.get("/api/macm4-models")
    for m in r.json()["data"]:
        if m["tier"] == "local":
            assert m["pricing"]["input_per_million_usd"] == 0.0
            assert m["pricing"]["output_per_million_usd"] == 0.0


def test_macm4_models_warm_flag_reflects_ollama_ps(client: TestClient, monkeypatch) -> None:
    """The `warm` flag for local-long must come from Ollama's
    /api/ps endpoint -- if our expected OLLAMA_TAG is in the loaded
    set, warm=true."""
    monkeypatch.setenv("OLLAMA_TAG", "qwen3-coder-next:q4_K_M")
    monkeypatch.setattr(
        dash_app, "_probe_ollama_loaded_models",
        lambda: {"qwen3-coder-next:q4_K_M"},
    )
    r = client.get("/api/macm4-models")
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert by_id["local-long"]["warm"] is True


def test_macm4_models_warm_false_when_ollama_unloaded(client: TestClient, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_TAG", "qwen3-coder-next:q4_K_M")
    monkeypatch.setattr(dash_app, "_probe_ollama_loaded_models", lambda: set())
    r = client.get("/api/macm4-models")
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert by_id["local-long"]["warm"] is False


def test_macm4_models_meta_fields_present(client: TestClient, monkeypatch) -> None:
    """Clients need schema_version + timestamps to invalidate caches."""
    monkeypatch.setattr(dash_app, "_probe_ollama_loaded_models", lambda: set())
    r = client.get("/api/macm4-models")
    meta = r.json()["_meta"]
    assert meta["schema_version"] == 1
    assert "generated_at" in meta
    assert meta["dashboard_url"] == "http://127.0.0.1:4001"
    assert meta["proxy_url"] == "http://127.0.0.1:4000"


def test_macm4_models_endpoint_does_not_crash_on_ollama_timeout(client: TestClient, monkeypatch) -> None:
    """If /api/ps times out (Ollama down) we should still serve the
    metadata, just with warm=false for the local-long tier."""
    def _boom() -> set[str]:
        raise RuntimeError("simulated Ollama outage")
    monkeypatch.setattr(dash_app, "_probe_ollama_loaded_models", _boom)
    r = client.get("/api/macm4-models")
    assert r.status_code == 200
    by_id = {m["id"]: m for m in r.json()["data"]}
    assert by_id["local-long"]["warm"] is False
