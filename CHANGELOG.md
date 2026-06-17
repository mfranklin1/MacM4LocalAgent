# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Thinking mode** — the router now streams a live reasoning trace into
  the harness (Cline) for both tiers instead of an empty "thinking" tag.
  Local Qwen3 turns get `/think` injected and their inline `<think>…</think>`
  trace is re-routed onto the `reasoning_content` delta channel by a new
  `async_post_call_streaming_iterator_hook`; Claude turns get Anthropic
  extended thinking enabled (`thinking` block + `temperature=1`,
  `max_tokens` floor). `/think` is gated on the resolved model actually
  being Qwen3 (`_model_supports_think()` reads `MLX_LOCAL_DIR`/`MLX_REPO`
  and `OLLAMA_TAG`), so `local-agent` (llama3.1) and `local-coder-*`
  (qwen2.5) are never mis-injected. New env vars: `ROUTER_THINKING`
  (default on), `ROUTER_THINKING_BUDGET` (2048), `ROUTER_THINKING_LOCAL_MAX`
  (12288). Wired for the Cline → LiteLLM path; see
  [docs/routing.md](docs/routing.md#thinking-mode-live-reasoning-trace).
- `docs/RUNBOOK-cursor-setup.md` — first-time install runbook with the
  concrete values from the install (master key, ports, model status,
  troubleshooting).
- `scripts/run_litellm.py` — thin Python launcher that imports and runs
  `litellm.run_server`. Replaces the venv `litellm` console script in the
  launchd plist; the wrapper script triggers a TCC `pyvenv.cfg` permission
  error on macOS due to `com.apple.provenance` xattrs.
- `make finalize` target — re-renders the launchd plists with the
  post-download `MLX_LOCAL_DIR` value and (re)starts all services. Use
  this once both background model downloads finish.
- `scripts/40-litellm.sh` now drops `config/router` and `config/cost`
  symlinks so LiteLLM's `get_instance_fn` (which resolves modules
  relative to the config file's directory when `config_file_path` is
  set) can import the router callback.
- `scripts/30-mlx.sh` learned `MLX_PREFER_NEXT=0` to skip the unstable
  Qwen3-Coder-Next Xet-CAS repos, plus a watchdog that aborts a download
  after `STALL_WINDOW` seconds of no progress.
- `scripts/20-ollama.sh` retries each candidate up to `PULL_RETRIES`
  times before falling through, so a single TCP reset doesn't disqualify
  a model tag.

### Changed
- `scripts/00-detect.sh` now picks `qwen3-coder-next:q8_0` /
  `qwen3-coder-next:q4_K_M` (the actual Ollama tag names) instead of
  the made-up `:80b-q8` / `:80b-q4`.
- `scripts/30-mlx.sh` now enables `hf_transfer` for parallel-chunked
  downloads and falls back through Qwen3-Coder-30B-A3B and
  Qwen2.5-Coder-32B repos when the 80B repo is rate-limited.
- LiteLLM YAML now references `proxy_handler_instance` (a module-level
  instance) for both `callbacks` and `success_callback`, instead of the
  bare class. LiteLLM 1.50+ requires an instance for the post-call hooks
  introduced in `CustomLogger`.
- `launchd/com.local.litellm.plist` invokes
  `python scripts/run_litellm.py` instead of the venv console script,
  to side-step macOS provenance / TCC errors.
- `router/route_by_size.py::SizeBasedRouter` now subclasses
  `litellm.integrations.custom_logger.CustomLogger`, inheriting no-op
  implementations of `async_post_call_success_hook` etc.
- `tests/test_scripts.sh` skips `*.rendered.plist` files when checking
  for `@@REPO_ROOT@@` placeholders.

### Fixed
- LiteLLM service crashing on launchd start with
  `PermissionError: ... pyvenv.cfg`. Caused by `com.apple.provenance`
  xattrs that can't be removed without disabling SIP. Worked around by
  invoking the resolved `python` symlink directly.
- LiteLLM proxy returning HTTP 500
  `CustomLogger.async_post_call_success_hook() missing 1 required
  positional argument: 'self'` when `hybrid-auto` was used. Caused by
  YAML registering the class instead of an instance.
- Misleading `OLLAMA_TAG` in `config/detected.env` for high-RAM tier.

---

## [0.1.0] — 2026-04-27

The first usable release. Hybrid local + Claude coding setup with cost
tracking, an A/B comparator and a local dashboard.

### Added

#### Installer & orchestration
- `Makefile` with 16 targets: `help`, `detect`, `install`, `start`, `stop`,
  `restart`, `status`, `dashboard`, `verify`, `report`, `compare`, `clean`,
  `nuke`, `test`, `test-py`, `test-sh`, `lint`.
- `install.sh` thin wrapper for `make install`.
- `scripts/00-detect.sh` — chip / RAM / GPU / disk scan, picks quant tier,
  emits `config/detected.env`.
- `scripts/10-brew.sh` — idempotent brew installs (ollama, jq, sqlite, uv,
  llama.cpp).
- `scripts/20-ollama.sh` — Ollama + TurboQuant `tq3` env, optional
  `OLLAMA_FROM_SOURCE=1` rebuild, model pull with fallbacks.
- `scripts/30-mlx.sh` — `uv venv`, mlx-lm install, MLX model download.
- `scripts/40-litellm.sh` — LiteLLM proxy install + config render.
- `scripts/50-cursor.sh` — Cursor IDE setup checklist + `.cursor/rules/`
  drop.
- `scripts/60-dashboard.sh` — dashboard deps + launchd plist rendering.
- `scripts/90-verify.sh` — health probes + smoke matrix + KV cache
  assertion.

#### Routing & proxy
- `config/litellm-config.yaml` template with three tiers
  (`local-fast`, `local-long`, `claude-code`) and the `hybrid-auto` alias.
- `router/route_by_size.py` — LiteLLM `CustomLogger`/router with
  `async_pre_call_hook` that estimates tokens, classifies complexity, and
  rewrites `hybrid-auto` to a concrete tier. `log_success_event` records
  every call (including shadow Claude cost) into SQLite.
- `router/complexity_classifier.py` — regex heuristics for
  architectural / multi-file / deep-reasoning prompts plus `[claude]`
  and `[local]` overrides.

#### Cost tracking
- `cost/schema.sql` — `requests` and `comparisons` tables.
- `cost/ingest.py` — connect, schema bootstrap, `shadow_cost`,
  `record_request`, `record_comparison`.
- `cost/savings.py` — `summarize(days)` and CLI (`make report`,
  `--json`).

#### A/B comparator
- `compare/ab.py` — fan-out the same prompt to `local-long` and
  `claude-code`, persist outputs, latency, token counts and a simple
  judge score.

#### Dashboard
- `dashboard/app.py` — FastAPI app on port 4001 with HTMX-polled stats,
  routing pie chart, A/B compare form and side-by-side compare detail.
- Templates: `_layout.html`, `index.html`, `_stats.html`,
  `compare_index.html`, `compare_one.html`.
- `dashboard/static/style.css`.

#### Service management
- `launchd/com.local.{ollama,mlx,litellm,dashboard}.plist` templates.
- Renderer in `scripts/60-dashboard.sh` substitutes `@@REPO_ROOT@@` and
  `@@MLX_LOCAL_DIR@@` into `*.rendered.plist`.

#### Tests
- `tests/conftest.py` — `tmp_db` fixture isolating each test from the
  real `cost/cost.db`.
- `tests/test_router.py` — token estimator, complexity classifier,
  `decide_tier`, `async_pre_call_hook`, `log_success_event`.
- `tests/test_cost.py` — schema bootstrap, ingest, `summarize`
  windows, CLI JSON + human output.
- `tests/test_compare.py` — judge score edge cases, `_call` happy/error
  paths, `run` end-to-end with mocked HTTP.
- `tests/test_dashboard.py` — `TestClient` over every route.
- `tests/test_integration.py` — end-to-end: router → DB → savings →
  dashboard.
- `tests/test_detect.sh` — runs `00-detect.sh` and asserts every
  required env key.
- `tests/test_scripts.sh` — `bash -n`, strict-mode check, plist
  validation, Makefile target presence.
- `tests/run.sh` — orchestrator. `make test` is a thin wrapper.

#### Documentation
- Top-level `README.md` with badges, common-task table, architecture
  overview and pointers into `docs/`.
- `docs/architecture.md`, `docs/routing.md`, `docs/cost-model.md`,
  `docs/operations.md`, `docs/cursor-integration.md`, `docs/testing.md`,
  `docs/troubleshooting.md`, `docs/faq.md`, `docs/contributing.md`,
  `docs/security.md`.
- `.cursor/rules/hybrid-routing.mdc` describing the tier strategy for
  Cursor's in-IDE agent.

### Notes
- Targets Apple Silicon (M4 / M5 Max). Other platforms are not supported
  by the installer; the Python pieces are portable.
- Cursor's Agent mode does not yet honour custom OpenAI providers for all
  features. Chat / inline edits work fully through the proxy.

[Unreleased]: https://example.com/compare/v0.1.0...HEAD
[0.1.0]:      https://example.com/releases/tag/v0.1.0
