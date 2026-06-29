SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

REPO_ROOT := $(shell pwd)
SCRIPTS := $(REPO_ROOT)/scripts
LAUNCHD_DIR := $(REPO_ROOT)/launchd
LAUNCH_AGENTS := $(HOME)/Library/LaunchAgents
PLISTS := com.local.ollama com.local.mlx com.local.litellm com.local.dashboard com.local.ollama-warm com.local.watchdog com.local.claude-proxy

.PHONY: help detect install reconfigure start stop restart status dashboard verify watchdog report compare clean nuke test test-py test-sh lint finalize downloads downloads-watch wait-and-finalize resume-ollama bench bench-local bench-claude bench-cursor bench-report bench-pull-spend turboquant-status turboquant-upgrade turboquant-watch turboquant-experimental-build turboquant-experimental-serve turboquant-experimental-stop turboquant-experimental-status turboquant-experimental-ab turboquant-experimental-nuke perf perf-short perf-stress perf-prefix perf-prefix-cold check-pricing cline warm offline online offline-status worktree worktree-rm worktree-sync worktree-list backend-status backend-stop-large turbo-install turbo-enable turbo-disable turbo-status turbo-start-256 turbo-stop turbo-bench janitor-enable janitor-disable janitor-status janitor-show-ledger janitor-show-active-context janitor-reset upgrade-to-q8 TURBO_ENABLED CONTEXT_JANITOR_ENABLED TURBO_MODEL_LOCAL_DIR

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

detect: ## Scan hardware and write config/detected.env
	@bash $(SCRIPTS)/00-detect.sh

reconfigure: ## Re-render LiteLLM config + restart LiteLLM (no model downloads)
	@bash $(SCRIPTS)/40-litellm.sh
	@launchctl unload $(LAUNCH_AGENTS)/com.local.litellm.plist 2>/dev/null || true
	@cp $(LAUNCHD_DIR)/com.local.litellm.rendered.plist $(LAUNCH_AGENTS)/com.local.litellm.plist
	@launchctl load -w $(LAUNCH_AGENTS)/com.local.litellm.plist
	@echo "LiteLLM restarted with updated config."

install: detect ## Install all components (idempotent)
	@bash $(SCRIPTS)/10-brew.sh
	@bash $(SCRIPTS)/20-ollama.sh
	@bash $(SCRIPTS)/30-mlx.sh
	@bash $(SCRIPTS)/40-litellm.sh
	@bash $(SCRIPTS)/60-dashboard.sh
	@bash $(SCRIPTS)/50-cursor.sh
	@echo ""
	@echo "Install complete. Next: \`make start\` then \`make verify\`."

downloads: ## One-shot: are the Ollama + MLX background pulls done?
	@bash $(SCRIPTS)/check-downloads.sh || true

resume-ollama: ## Foreground: keep retrying `ollama pull $$OLLAMA_TAG` past TCP resets
	@bash $(SCRIPTS)/resume-ollama.sh

downloads-watch: ## Live status loop (refreshes every 30s, Ctrl-C to exit)
	@while true; do \
	  clear; \
	  bash $(SCRIPTS)/check-downloads.sh && exit 0 || true; \
	  echo "  refresh in 30s (Ctrl-C to stop)"; \
	  sleep 30; \
	done

wait-and-finalize: ## Block until both downloads finish, then run finalize automatically
	@echo "Polling every 60s; status printed each cycle. Will run \`make finalize\` once both downloads are done."
	@echo ""
	@while true; do \
	  if bash $(SCRIPTS)/check-downloads.sh; then \
	    echo ""; \
	    echo ">>> Both downloads complete - running \`make finalize\`"; \
	    echo ""; \
	    $(MAKE) finalize; \
	    break; \
	  fi; \
	  sleep 60; \
	done

finalize: ## Re-render plists with the post-download values, then start everything
	@bash $(SCRIPTS)/60-dashboard.sh
	@$(MAKE) start
	@echo ""
	@echo "Finalized. Run \`make verify\` once services have warmed up."

start: ## Load launchd plists (Ollama, MLX, LiteLLM, dashboard)
	@mkdir -p $(LAUNCH_AGENTS)
	@for p in $(PLISTS); do \
	  src="$(LAUNCHD_DIR)/$$p.rendered.plist"; \
	  dst="$(LAUNCH_AGENTS)/$$p.plist"; \
	  if [ ! -f "$$src" ]; then echo "Missing $$src - run \`make install\` first (renders plists)"; exit 1; fi; \
	  cp "$$src" "$$dst"; \
	  launchctl unload "$$dst" 2>/dev/null || true; \
	  launchctl load -w "$$dst"; \
	  echo "loaded $$p"; \
	done
	@echo ""
	@echo "Services starting in the background. Check with \`make status\`."

stop: ## Unload launchd plists
	@for p in $(PLISTS); do \
	  dst="$(LAUNCH_AGENTS)/$$p.plist"; \
	  if [ -f "$$dst" ]; then launchctl unload "$$dst" 2>/dev/null || true; echo "stopped $$p"; fi; \
	done

restart: stop start ## Restart all services

status: ## Show running ports
	@echo "Port  Service        Status"
	@echo "----  -------------  ------"
	@for entry in "11434 ollama" "8081 mlx" "4000 litellm" "4001 dashboard" "4002 claude-proxy"; do \
	  port="$${entry%% *}"; name="$${entry##* }"; \
	  if lsof -nP -iTCP:$$port -sTCP:LISTEN >/dev/null 2>&1; then \
	    echo "$$port  $$name        UP"; \
	  else \
	    echo "$$port  $$name        DOWN"; \
	  fi; \
	done

# ---------- backend lifecycle ----------

backend-status: ## Show active backend, state, and pending requests
	GET http://127.0.0.1:4000/backend/status | python3 -m json.tool

backend-stop-large: ## Stop the current large (non-fast) backend safely
	@python3 -c "import asyncio; from router.backend_lifecycle import lifecycle_manager; asyncio.run(lifecycle_manager.stop_backend(lifecycle_manager.active_backend))"

dashboard: ## Open the cost/savings dashboard
	@open http://127.0.0.1:4001

verify: ## Run endpoint health + smoke matrix
	@bash $(SCRIPTS)/90-verify.sh

watchdog: ## Run orphaned-task cleanup once (use --status for dry-run)
	@bash $(SCRIPTS)/95-watchdog.sh $(ARGS)

cline: ## Install the Cline extension into Cursor (or VS Code as fallback)
	@bash $(SCRIPTS)/install-cline.sh

warm: ## Pre-load the long-context Ollama model (eliminates first-turn cold start)
	@bash $(SCRIPTS)/warm-ollama.sh

report: ## Print savings report (today / 7d / MTD)
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	python3 $(REPO_ROOT)/cost/savings.py

check-pricing: ## Diff cost/pricing.py against Anthropic's published rates (no auto-write)
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	python3 $(REPO_ROOT)/scripts/check_claude_pricing.py

# ---------- offline mode (airplane / no-network sessions) ----------
#
# Toggles the OFFLINE flag in config/detected.env. The router reads
# this on every call (no proxy restart required) and downgrades any
# Claude tier to local-long. See docs/offline-mode.md.

offline: ## Force offline mode (no Claude calls) - persists across restarts
	@bash $(SCRIPTS)/offline-mode.sh on

online: ## Disable forced offline mode (back to auto-detect)
	@bash $(SCRIPTS)/offline-mode.sh off

offline-status: ## Show current offline-mode state + live probe result
	@bash $(SCRIPTS)/offline-mode.sh status

compare: ## Run an A/B comparison: make compare PROMPT="..."
	@if [ -z "$${PROMPT:-}" ]; then echo 'Usage: make compare PROMPT="..."'; exit 1; fi
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	python3 $(REPO_ROOT)/compare/ab.py "$${PROMPT}"

# ---------- worktree management (auto-syncs MacM4LocalAgent.code-workspace) ----------
#
# Always use these targets instead of raw `git worktree add/remove` so the
# .code-workspace file stays in step and Cursor's Cmd+P sees every tree.
#
# Usage:
#   make worktree BRANCH=feat/my-thing   # create branch + worktree + sync workspace
#   make worktree-rm BRANCH=feat/my-thing # remove worktree + sync workspace
#   make worktree-sync                    # repair workspace after a manual git op
#   make worktree-list                    # show all worktrees

BRANCH ?=

worktree: ## Create a branch + worktree and add to .code-workspace  [BRANCH=name]
	@if [ -z "$(BRANCH)" ]; then echo "Usage: make worktree BRANCH=<branch-name>"; exit 1; fi
	@SLUG=$$(echo "$(BRANCH)" | sed 's|.*/||'); \
	 DIR="$(REPO_ROOT)/.worktrees/$$SLUG"; \
	 git worktree add -b "$(BRANCH)" "$$DIR" 2>/dev/null || git worktree add "$$DIR" "$(BRANCH)"; \
	 python3 $(SCRIPTS)/workspace-sync.py; \
	 echo ""; \
	 echo "Worktree ready: $$DIR"; \
	 echo "Open MacM4LocalAgent.code-workspace in Cursor for Cmd+P across both trees."

worktree-rm: ## Remove a worktree and update .code-workspace  [BRANCH=name]
	@if [ -z "$(BRANCH)" ]; then echo "Usage: make worktree-rm BRANCH=<branch-name>"; exit 1; fi
	@SLUG=$$(echo "$(BRANCH)" | sed 's|.*/||'); \
	 DIR="$(REPO_ROOT)/.worktrees/$$SLUG"; \
	 git worktree remove "$$DIR" --force 2>/dev/null || true; \
	 git worktree prune; \
	 python3 $(SCRIPTS)/workspace-sync.py

worktree-sync: ## Sync .code-workspace to match current git worktrees (repair after manual ops)
	@python3 $(SCRIPTS)/workspace-sync.py

worktree-list: ## List all active worktrees
	@git worktree list

clean: stop ## Stop services and remove venvs (keeps models)
	@rm -rf $(REPO_ROOT)/.venvs
	@for p in $(PLISTS); do rm -f $(LAUNCH_AGENTS)/$$p.plist; done
	@echo "Cleaned venvs and launchd plists. Models on disk are kept."

nuke: clean ## clean + remove Ollama models and MLX cache
	@ollama list 2>/dev/null | awk 'NR>1 {print $$1}' | xargs -I{} ollama rm {} || true
	@rm -rf $(REPO_ROOT)/models
	@rm -f  $(REPO_ROOT)/cost/cost.db
	@echo "Wiped models and cost.db."

test: ## Run the full test suite (Python + shell)
	@bash $(REPO_ROOT)/tests/run.sh

test-py: ## Run only the Python test suite
	@if [ ! -d $(REPO_ROOT)/.venvs/test ]; then uv venv --python 3.12 $(REPO_ROOT)/.venvs/test; fi
	@. $(REPO_ROOT)/.venvs/test/bin/activate && \
	  uv pip install --quiet --upgrade pytest pytest-asyncio pytest-cov fastapi httpx jinja2 python-multipart && \
	  PYTHONPATH=$(REPO_ROOT) pytest -q tests

test-sh: ## Run only the shell test suites
	@bash $(REPO_ROOT)/tests/test_scripts.sh
	@bash $(REPO_ROOT)/tests/test_detect.sh

lint: ## ruff + bash -n on all files
	@. $(REPO_ROOT)/.venvs/test/bin/activate 2>/dev/null && uv pip install --quiet ruff || true; \
	. $(REPO_ROOT)/.venvs/test/bin/activate 2>/dev/null && ruff check . || true
	@for f in scripts/*.sh tests/*.sh install.sh; do bash -n "$$f" && echo "ok: $$f"; done

# ---------- benchmark suite (local vs. claude vs. cursor-no-proxy) ----------

# TASK defaults to lru_ttl_cache; override with `make bench TASK=<id>`.
TASK ?= lru_ttl_cache
ATTEMPTS ?= 1

bench: ## Run automated arms (local-only + claude-only) on TASK; prints follow-up commands
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	PYTHONPATH=$(REPO_ROOT) python3 -m bench.runner --task $(TASK) --attempts $(ATTEMPTS)

bench-local: ## Run only the local-only arm: `make bench-local TASK=...`
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	PYTHONPATH=$(REPO_ROOT) python3 -m bench.runners.local_only --task $(TASK) --attempts $(ATTEMPTS)

bench-claude: ## Run only the claude-only arm: `make bench-claude TASK=...`
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	PYTHONPATH=$(REPO_ROOT) python3 -m bench.runners.claude_only --task $(TASK) --attempts $(ATTEMPTS)

bench-cursor: ## Ingest a recorded Cursor session: `make bench-cursor SESSION=path/to/session.json OUTPUT=path/to/output.txt`
	@if [ -z "$${SESSION:-}" ] || [ -z "$${OUTPUT:-}" ]; then \
	  echo 'Usage: make bench-cursor SESSION=bench/results/<dir>/session.json OUTPUT=bench/results/<dir>/output.txt'; \
	  exit 1; \
	fi
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	PYTHONPATH=$(REPO_ROOT) python3 -m bench.runners.cursor_session --session "$${SESSION}" --output "$${OUTPUT}"

bench-pull-spend: ## Snapshot Anthropic + Cursor spend for an arm/window:
                  ## `make bench-pull-spend ARM=claude-only START=... END=... PROVIDERS=anthropic,cursor`
	@if [ -z "$${ARM:-}" ] || [ -z "$${START:-}" ] || [ -z "$${END:-}" ]; then \
	  echo 'Usage: make bench-pull-spend ARM=<claude-only|cursor-no-proxy|cursor-hybrid|local-only> START=<unix> END=<unix> [PROVIDERS=anthropic,cursor] [TASK=...]'; \
	  exit 1; \
	fi
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	PYTHONPATH=$(REPO_ROOT) python3 -m bench.pull_spend \
	  --arm "$${ARM}" --task-id "$(TASK)" \
	  --window-start "$${START}" --window-end "$${END}" \
	  --providers "$${PROVIDERS:-anthropic,cursor}"

bench-report: ## Print 3-arm comparison report for TASK
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	PYTHONPATH=$(REPO_ROOT) python3 -m bench.report --task $(TASK)

# ---------- TurboQuant: live (option b) + experimental (option c) ----------

turboquant-status: ## Report what KV cache types this Ollama supports + upstream PR status
	@bash $(SCRIPTS)/turboquant-upgrade.sh

turboquant-upgrade: ## If Ollama now supports tq3, flip config + bounce daemon
	@bash $(SCRIPTS)/turboquant-upgrade.sh --apply

turboquant-watch: ## Daily poll loop; auto-applies tq3 the moment it lands upstream (Ctrl-C to stop)
	@bash $(SCRIPTS)/turboquant-upgrade.sh --watch

turboquant-experimental-build: ## OPT-IN: build llama.cpp + PR #21131 (--turbo-kv) into .experimental/
	@bash $(SCRIPTS)/turboquant-experimental.sh build

turboquant-experimental-serve: ## Start the experimental server on :8082 (does NOT touch live Ollama)
	@bash $(SCRIPTS)/turboquant-experimental.sh serve

turboquant-experimental-stop: ## Stop the experimental server
	@bash $(SCRIPTS)/turboquant-experimental.sh stop

turboquant-experimental-status: ## Is the experimental server up?
	@bash $(SCRIPTS)/turboquant-experimental.sh status

turboquant-experimental-ab: ## A/B same prompt against live Ollama vs experimental: PROMPT="..."
	@bash $(SCRIPTS)/turboquant-experimental.sh ab "$${PROMPT:-Write a one-line Python function that returns x+1.}"

turboquant-experimental-nuke: ## Stop + remove the experimental worktree
	@bash $(SCRIPTS)/turboquant-experimental.sh nuke

# ---------- turbo backends ----------

turbo-install: ## Download Qwen2.5-Coder-32B-Instruct-4bit for turbo tiers (~20 GB MLX)
	@bash $(SCRIPTS)/turbo-install.sh

turbo-enable: ## Enable turbo backend escalation (set TURBO_ENABLED=1 in detected.env)
	@sed -i '' 's/^TURBO_ENABLED=.*/TURBO_ENABLED=1/' config/detected.env || echo 'TURBO_ENABLED=1' >> config/detected.env

turbo-disable: ## Disable turbo backend escalation (set TURBO_ENABLED=0 in detected.env)
	@sed -i '' 's/^TURBO_ENABLED=.*/TURBO_ENABLED=0/' config/detected.env || echo 'TURBO_ENABLED=0' >> config/detected.env

turbo-status: ## Check turbo backend readiness (model downloaded? mlx-lm turbo_kv_bits available?)
	@bash $(SCRIPTS)/turbo-status.sh

turbo-start-256: ## Manually start the 256k turbo backend (lifecycle manager handles this automatically)
	@launchctl bootstrap gui/$$(id -u) $(LAUNCHD)/com.local.turbo-256k.plist

turbo-stop: ## Stop all turbo backends
	@launchctl bootout gui/$$(id -u) $(LAUNCHD)/com.local.turbo-256k.plist 2>/dev/null || true
	@launchctl bootout gui/$$(id -u) $(LAUNCHD)/com.local.turbo-512k.plist 2>/dev/null || true

turbo-bench: ## Benchmark turbo backends vs 128k (requires TURBO_ENABLED=1)
	@python3 bench/runner.py --backends local-long-128k,local-turbo-256k,local-turbo-512k

# ---------- perf suite ----------

perf: ## End-to-end perf pass: cold + 500/5k/18k tok runs + router boundary
	@bash $(SCRIPTS)/perf-suite.sh

perf-short: ## Quick perf pass: skips the 18k-token long run
	@bash $(SCRIPTS)/perf-suite.sh --short

perf-stress: ## Full perf pass + ~110k local stress + ~140k over-ceiling claude routing test
	@bash $(SCRIPTS)/perf-suite.sh --stress

perf-prefix: ## Cursor-style prefix cache probe: 80k shared prefix + 4 follow-ups
	@bash $(SCRIPTS)/perf-prefix-cache.sh

perf-prefix-cold: ## Same probe but bounce Ollama first (true cold turn 1)
	@bash $(SCRIPTS)/perf-prefix-cache.sh --evict

# ---------- context janitor ----------

janitor-enable: ## Enable context janitor (sets CONTEXT_JANITOR_ENABLED=1)
	@sed -i '' 's/^CONTEXT_JANITOR_ENABLED=.*/CONTEXT_JANITOR_ENABLED=1/' config/detected.env || echo 'CONTEXT_JANITOR_ENABLED=1' >> config/detected.env

janitor-disable: ## Disable context janitor
	@sed -i '' 's/^CONTEXT_JANITOR_ENABLED=.*/CONTEXT_JANITOR_ENABLED=0/' config/detected.env || echo 'CONTEXT_JANITOR_ENABLED=0' >> config/detected.env

janitor-status: ## Show context janitor state (enabled, last run, token stats)
	@python3 -c "import json,pathlib; p=pathlib.Path('.runtime/context_janitor/status.json'); print(json.dumps(json.loads(p.read_text()),indent=2) if p.exists() else 'janitor has not run yet')"

janitor-show-ledger: ## Print the project ledger
	@cat .runtime/context_janitor/project-ledger.json 2>/dev/null | python3 -m json.tool || echo 'no ledger yet'

janitor-show-active-context: ## Print the active context pack
	@cat .runtime/context_janitor/active-context-pack.md 2>/dev/null || echo 'no active context pack yet'

janitor-reset: ## Clear all janitor state (ledger, active context pack, manifests)
	@rm -rf .runtime/context_janitor && echo 'janitor state cleared'

# ---------- q8 upgrade ----------

upgrade-to-q8: ## Upgrade local Ollama backend from q4 to q8_0 (requires 128 GB+; ~85 GB download)
	@bash $(SCRIPTS)/upgrade-to-q8.sh
