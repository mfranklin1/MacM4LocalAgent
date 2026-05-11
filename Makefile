SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

REPO_ROOT := $(shell pwd)
SCRIPTS := $(REPO_ROOT)/scripts
LAUNCHD_DIR := $(REPO_ROOT)/launchd
LAUNCH_AGENTS := $(HOME)/Library/LaunchAgents
PLISTS := com.local.ollama com.local.mlx com.local.litellm com.local.dashboard

.PHONY: help detect install start stop restart status dashboard verify report compare clean nuke test test-py test-sh lint finalize downloads downloads-watch wait-and-finalize resume-ollama bench bench-local bench-claude bench-cursor bench-report bench-pull-spend turboquant-status turboquant-upgrade turboquant-watch turboquant-experimental-build turboquant-experimental-serve turboquant-experimental-stop turboquant-experimental-status turboquant-experimental-ab turboquant-experimental-nuke perf perf-short perf-stress perf-prefix perf-prefix-cold check-pricing cline

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

detect: ## Scan hardware and write config/detected.env
	@bash $(SCRIPTS)/00-detect.sh

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
	@for entry in "11434 ollama" "8081 mlx" "4000 litellm" "4001 dashboard"; do \
	  port="$${entry%% *}"; name="$${entry##* }"; \
	  if lsof -nP -iTCP:$$port -sTCP:LISTEN >/dev/null 2>&1; then \
	    echo "$$port  $$name        UP"; \
	  else \
	    echo "$$port  $$name        DOWN"; \
	  fi; \
	done

dashboard: ## Open the cost/savings dashboard
	@open http://127.0.0.1:4001

verify: ## Run endpoint health + smoke matrix
	@bash $(SCRIPTS)/90-verify.sh

cline: ## Install the Cline extension into Cursor (or VS Code as fallback)
	@bash $(SCRIPTS)/install-cline.sh

report: ## Print savings report (today / 7d / MTD)
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	python3 $(REPO_ROOT)/cost/savings.py

check-pricing: ## Diff cost/pricing.py against Anthropic's published rates (no auto-write)
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	python3 $(REPO_ROOT)/scripts/check_claude_pricing.py

compare: ## Run an A/B comparison: make compare PROMPT="..."
	@if [ -z "$${PROMPT:-}" ]; then echo 'Usage: make compare PROMPT="..."'; exit 1; fi
	@. $(REPO_ROOT)/.venvs/litellm/bin/activate 2>/dev/null || true; \
	python3 $(REPO_ROOT)/compare/ab.py "$${PROMPT}"

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
