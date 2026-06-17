# Testing

The test suite has two halves:

- **Python**: unit + integration tests under `tests/test_*.py`, run with
  `pytest`.
- **Shell**: bash assertions under `tests/test_*.sh`, run directly.

`tests/run.sh` runs both halves and prints a summary. The
top-level `make test` is just a wrapper.

## Layout

```
tests/
├── conftest.py              # shared fixtures: tmp_db (incl. bench), repo_root
├── test_router.py           # SizeBasedRouter, classifier, decide_tier
├── test_cost.py             # ingest, schema, savings windows, JSON CLI
├── test_compare.py          # judge score, mocked HTTP, DB persistence
├── test_dashboard.py        # FastAPI TestClient on every route
├── test_integration.py      # end-to-end: router -> DB -> savings -> dashboard
├── test_overgeneration_control.py  # static/multi-turn guardrails + Qwen3 /think + thinking gating
├── test_think_stream_transform.py  # <think> -> reasoning_content splitter + streaming hook
├── test_thinking_pre_call_hook.py  # thinking-mode wiring in async_pre_call_hook (both tiers)
├── bench/
│   ├── test_db.py           # bench_runs + provider_spend schema + writes
│   ├── test_grader.py       # code extraction, AST checks, pytest grading
│   ├── test_collectors.py   # Anthropic + Cursor collectors, mocked HTTP
│   ├── test_runners.py      # litellm_arm + cursor_session, mocked streaming
│   └── test_report.py       # 3-arm summary + provider-billed overlay
├── test_detect.sh           # runs scripts/00-detect.sh and asserts env keys
├── test_scripts.sh          # bash -n + plist lint + Makefile target presence
└── run.sh                   # orchestrator: pytest + shell suites + summary
```

## Running

```bash
make test           # full suite (Python + shell)
make test-py        # Python only
make test-sh        # shell only
PYTHONPATH=. pytest -q tests/test_router.py::test_classify  # narrow
```

The first `make test` provisions `.venvs/test/` via `uv` and pip-installs
pytest, FastAPI, httpx, jinja2 and python-multipart. Subsequent runs reuse
the venv.

## Coverage by file

| Subject                  | Tested in                                | Notes                                                  |
| ------------------------ | ---------------------------------------- | ------------------------------------------------------ |
| Token estimator          | `test_router.py`                         | Empty / string / list-of-parts content                 |
| Complexity classifier    | `test_router.py`                         | Architectural language, `[claude]` / `[local]` tags    |
| `decide_tier`            | `test_router.py`                         | Small → fast, medium → long, huge → claude, complex → claude |
| `async_pre_call_hook`    | `test_router.py`                         | Hybrid-auto rewrite + metadata, no-op for explicit     |
| `log_success_event`      | `test_router.py`, `test_integration.py`  | Cost math, latency, dict + Pydantic usage              |
| Thinking mode            | `test_overgeneration_control.py`, `test_think_stream_transform.py`, `test_thinking_pre_call_hook.py` | Qwen3 model gating, Claude `thinking` params, `<think>`→`reasoning_content` stream, pre-call wiring + kill switch |
| Cost schema + ingest     | `test_cost.py`                           | Idempotent connect, record + retrieve                  |
| `savings.summarize`      | `test_cost.py`                           | 7-day window, exclude-old, empty case                  |
| `savings.main` CLI       | `test_cost.py`                           | JSON, single window, three-block default               |
| A/B `_judge`             | `test_compare.py`                        | Identical / empty / length-skew / fence count          |
| A/B `_call`              | `test_compare.py`                        | Mocked transport, happy and error paths                |
| A/B `run` end-to-end     | `test_compare.py`, `test_integration.py` | Both calls + DB write                                  |
| Dashboard `/`            | `test_dashboard.py`                      | Renders                                                |
| Dashboard `/stats`       | `test_dashboard.py`                      | Today / week / claude / local-fast badges              |
| Dashboard `/api/stats`   | `test_dashboard.py`                      | JSON shape                                             |
| Dashboard `/compare`     | `test_dashboard.py`                      | Empty + with rows                                      |
| Dashboard `/compare/run` | `test_dashboard.py`                      | Form post → 303 redirect                               |
| Dashboard `/compare/{id}`| `test_dashboard.py`                      | 200 + 404 paths                                        |
| End-to-end flow          | `test_integration.py`                    | Router → ingest → CLI → dashboard, all in one test     |
| Bench DB schema          | `bench/test_db.py`                       | `bench_runs` + `provider_spend` writes, window query   |
| Bench grader             | `bench/test_grader.py`                   | Fence extraction, AST checks, full pytest grade        |
| Provider collectors      | `bench/test_collectors.py`               | Anthropic + Cursor wire shape via httpx.MockTransport  |
| Bench runners            | `bench/test_runners.py`                  | litellm_arm streaming + cursor_session ingest          |
| 3-arm report             | `bench/test_report.py`                   | Deltas, provider-billed overlay, source priority       |
| `00-detect.sh`           | `test_detect.sh`                         | All required env keys, KV cache, threshold ordering    |
| All shell scripts        | `test_scripts.sh`                        | `bash -n`, strict mode, plist lint, Make targets       |

## Fixtures

`tmp_db` is the workhorse. It points both `cost.ingest.DB_PATH` and
`router.route_by_size.DB_PATH` at a fresh tmp file, so each test gets an
isolated SQLite DB. No test ever touches `cost/cost.db`.

```python
def test_thing(tmp_db):
    ingest.record_request(...)
    assert ingest.connect().execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 1
```

## Mocking HTTP

We use `httpx.MockTransport` plus a small `_MockClient` subclass to
intercept `httpx.Client(...)` construction. See
`tests/test_compare.py::_patch_httpx_with_handler`. No real network in
tests.

## CI hint

A minimal GitHub Actions runner would look like:

```yaml
- uses: actions/checkout@v4
- uses: astral-sh/setup-uv@v3
- run: make test
```

The shell suites assume `make`, `bash`, `plutil`, and `sysctl` (Apple
Silicon). On non-macOS hosts the detect test will skip with a clear failure
message; the rest still runs.

## Adding a test

1. Add a function whose name starts with `test_` to the right `tests/test_*.py`.
2. If you need DB access, take the `tmp_db` fixture.
3. If you need HTTP, copy `_patch_httpx_with_handler` from `test_compare.py`.
4. Run `make test-py` until green.

For a new shell test, drop `tests/test_<thing>.sh`, give it a `set -euo
pipefail`, and call it from `tests/run.sh`.
