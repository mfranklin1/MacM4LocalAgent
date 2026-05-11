# TurboQuant — when this changes

> **Status as of April 2026: NOT in effect.** The repo is wired for
> TurboQuant but stable Ollama doesn't ship it yet. We currently run on
> `OLLAMA_KV_CACHE_TYPE=q4_0` — standard 4-bit block quantization that
> compresses the KV cache ~4×. This is fine for everyday use; the
> document below explains what changes when upstream lands `tq3` support.

## Why we care

TurboQuant `tq3` compresses the attention KV cache ~5–6× with minimal
quality loss. For a 128 GB Mac running Qwen3-Coder-Next, that's the
difference between **~64 k** tokens of comfortable local context (today)
and **~128 k** tokens (post-TurboQuant). For the 80B model on q8, it's
the difference between "barely fits at 32k" and "easy at 64k".

## How we pick the KV cache type

`scripts/00-detect.sh` probes the installed `ollama` binary for
recognized KV-cache strings via `strings $(which ollama) | grep -Eo
'\b(tq3|tq4|q4_0|q8_0|f16)\b'` and picks the strongest one available:

```
tq3  (TurboQuant 3-bit, ~5-6x)   ← target, not yet shipping
tq4  (TurboQuant 4-bit, ~4-5x)   ← target, not yet shipping
q4_0 (block quant, ~4x)          ← what we use today
q8_0 (block quant, ~2x)
f16  (no compression)
```

The result is pinned in `config/detected.env` as `KV_CACHE_TYPE` and gets
re-applied to `OLLAMA_KV_CACHE_TYPE` whenever the Ollama plist is
re-rendered. The detection avoids the silent-fallback footgun where
asking for an unsupported value reverts to `f16` without warning.

## Upstream tracking

| Project    | PR / Issue                                                                                                          | Status                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Ollama     | [#15090](https://github.com/ollama/ollama/pull/15090)                                                               | **CLOSED Apr 2026** — maintainers waiting for MLX upstream    |
| Ollama     | [#15125](https://github.com/ollama/ollama/pull/15125), [#15505](https://github.com/ollama/ollama/pull/15505)        | engine-wiring follow-ups                                      |
| MLX        | [#3328](https://github.com/ml-explore/mlx/pull/3328)                                                                | TurboQuant inside MLX core (the gating dependency)            |
| llama.cpp  | [#21131](https://github.com/ggml-org/llama.cpp/pull/21131)                                                          | `--turbo-kv` flag working on CPU / CUDA / HIP                 |
| Paper      | [arXiv 2504.19874](https://arxiv.org/abs/2504.19874)                                                                | TurboQuant, ICLR 2026                                         |

## Helper targets

```bash
make turboquant-status     # one-shot: what does my Ollama support today?
make turboquant-upgrade    # if tq3 is supported now, flip detected.env + bounce daemon
make turboquant-watch      # poll daily, auto-apply when ready (Ctrl-C to stop)
```

`make turboquant-upgrade` will:

1. Rewrite `config/detected.env` so `KV_CACHE_TYPE="tq3"`.
2. Re-render the Ollama launchd plist via `scripts/60-dashboard.sh`.
3. `launchctl bootout` + `bootstrap` the Ollama service to pick up the
   new env.
4. Print "applied tq3" and leave you to `make verify`.

It's a no-op if `KV_CACHE_TYPE` is already `tq3` or if the installed
Ollama binary doesn't expose `tq3` in its symbol table.

## Experimental track (opt-in, today)

If you want to A/B-test TurboQuant **today** without touching the live
Ollama, there's a separate llama.cpp-based path:

```bash
make turboquant-experimental-build   # builds llama.cpp PR #21131 into .experimental/
make turboquant-experimental-serve   # starts it on :8082, leaves live stack alone
make turboquant-experimental-ab PROMPT="…"   # diffs output vs live Ollama
make turboquant-experimental-status
make turboquant-experimental-stop
make turboquant-experimental-nuke    # remove the worktree
```

This path is for evaluating quality + speed under `--turbo-kv` before
upstream Ollama lands it. The :8082 server is **not** wired into LiteLLM
routing — it's purely a manual probe.

## What gets bigger when tq3 lands

| Variable               | Today (q4_0) | After tq3 flip                |
| ---------------------- | ------------ | ----------------------------- |
| `KV_CACHE_TYPE`        | `q4_0`       | `tq3`                         |
| `LOCAL_LONG_CTX`       | 65 536 (64k) | up to 131 072 (128k)          |
| `ROUTE_LONG_MAX`       | 128 000      | unchanged (already permissive) |
| Effective KV compression | ~4×        | ~5–6×                         |

When that day comes, the router's `local-long` ceiling will move
automatically because `LOCAL_LONG_CTX` is sourced from `detected.env`.
You will not need to touch any router code.
