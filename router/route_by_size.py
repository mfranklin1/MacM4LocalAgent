"""LiteLLM custom callback that routes requests by prompt size + complexity
and records every call (with shadow Claude cost) into cost/cost.db.

Wired in as both `callbacks` (for pre-call routing) and `success_callback`
(for post-call cost ingestion) in config/litellm-config.yaml.

Routing rules (thresholds come from config/detected.env):
  - <ROUTE_FAST_MAX tokens, not complex   -> local-fast (MLX)
  - ROUTE_FAST_MAX..ROUTE_LONG_MAX tokens -> local-long (Ollama + TurboQuant)
  - >ROUTE_LONG_MAX tokens OR complex     -> claude-code

If the user explicitly picks a real model name (local-fast, local-long, or
claude-code) we never override. We only intercept the magical `hybrid-auto`
alias defined in litellm-config.yaml.
"""

from __future__ import annotations

import json
import os
import pathlib
import sqlite3
import sys
import time
from typing import Any, Iterable

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from router.complexity_classifier import classify  # noqa: E402
from router.overgeneration_control import (  # noqa: E402
    apply_multi_turn_tighten,
    apply_static_guardrail,
)

try:
    from litellm.integrations.custom_logger import CustomLogger as _LiteLLMCustomLogger
except Exception:  # tests / standalone runs don't need LiteLLM installed
    class _LiteLLMCustomLogger:  # type: ignore[no-redef]
        pass

# Anthropic claude-sonnet-4-6 published pricing (per token), used for shadow cost.
CLAUDE_INPUT_PER_TOKEN = 3.0 / 1_000_000
CLAUDE_OUTPUT_PER_TOKEN = 15.0 / 1_000_000

DB_PATH = REPO_ROOT / "cost" / "cost.db"


def _read_env_file(path: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


_ENV = _read_env_file(REPO_ROOT / "config" / "detected.env")
ROUTE_FAST_MAX = int(_ENV.get("ROUTE_FAST_MAX", "16000"))
ROUTE_LONG_MAX = int(_ENV.get("ROUTE_LONG_MAX", "128000"))


def _control_flag(name: str, default: str = "1") -> bool:
    """Read a bool flag from env or detected.env. We let real env vars
    win over the file so a quick `OVERGEN_STATIC=0 make restart`
    flips the strategy without editing detected.env."""
    raw = os.environ.get(name)
    if raw is None:
        raw = _ENV.get(name, default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# Strategy A (static guardrail) is on by default; Strategy C is on by
# default too. Disable with OVERGEN_STATIC=0 / OVERGEN_MULTI_TURN=0.
ENABLE_STATIC_GUARDRAIL = _control_flag("OVERGEN_STATIC", "1")
ENABLE_MULTI_TURN_TIGHTEN = _control_flag("OVERGEN_MULTI_TURN", "1")

# When OVERGEN_TRACE=1, append a one-line summary to .logs/overgen-trace.log
# every time a control fires. Used to verify Cursor traffic is being
# constrained correctly. Off by default so we don't pollute disk in normal
# use.
OVERGEN_TRACE = _control_flag("OVERGEN_TRACE", "0")
_TRACE_PATH = REPO_ROOT / ".logs" / "overgen-trace.log"

# When CLINE_TRACE=1, dump every inbound /v1/chat/completions request body
# (BEFORE any routing or control mutation) to a per-request JSON file under
# .logs/cline-dumps/. Bounded by CLINE_TRACE_MAX so an idle Cline session
# can't fill the disk. Used to investigate why Cline-style agent requests
# misbehave on small local models (e.g. tool-call confusion).
#
# REMOVE OR DISABLE AFTER INVESTIGATION -- these dumps contain the full
# system prompt + messages + tool definitions, which can be sensitive.
CLINE_TRACE = _control_flag("CLINE_TRACE", "0")
# Read from real env first, then detected.env (same precedence as _control_flag).
CLINE_TRACE_MAX = int(
    os.environ.get("CLINE_TRACE_MAX")
    or _ENV.get("CLINE_TRACE_MAX")
    or "20"
)
_CLINE_DUMP_DIR = REPO_ROOT / ".logs" / "cline-dumps"
_cline_dump_count = 0  # process-local; resets on proxy restart


# Keys we know LiteLLM injects that contain circular refs or arbitrary
# Python objects (like internal callback handles). We project the request
# down to only the OpenAI-API-shaped fields that matter for analysis.
_DUMP_KEEP_KEYS = (
    "model",
    "messages",
    "tools",
    "tool_choice",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "stream",
    "response_format",
    "user",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "seed",
)


def _dump_full_request(data: dict[str, Any]) -> None:
    """Best-effort dump of the inbound request to .logs/cline-dumps/.

    Bounded by CLINE_TRACE_MAX. Never raises -- a dumper that crashes
    requests is worse than no dumper at all.

    We project `data` down to OpenAI-API-shaped fields only; LiteLLM
    augments the dict with internal handles (callbacks, deployment
    objects) that contain circular references and aren't JSON-serializable.
    """
    global _cline_dump_count
    try:
        if _cline_dump_count >= CLINE_TRACE_MAX:
            return
        _CLINE_DUMP_DIR.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        path = _CLINE_DUMP_DIR / f"req-{ts}-{_cline_dump_count:03d}.json"
        projected = {k: data[k] for k in _DUMP_KEEP_KEYS if k in data}
        # default=str catches any stragglers (e.g. enum values).
        path.write_text(json.dumps(projected, indent=2, default=str))
        _cline_dump_count += 1
    except Exception as e:  # pragma: no cover -- diagnostic during investigation
        print(f"[router] cline-dumper failed: {type(e).__name__}: {e}", file=sys.stderr)


def _trace_overgen(**kw: Any) -> None:
    """Best-effort one-line append for live observation. Never raises."""
    try:
        _TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        line = (
            f"{int(time.time())} "
            f"req={kw.get('requested')} "
            f"resolved={kw.get('resolved')} "
            f"max_tokens={kw.get('pre_max')}->{kw.get('post_max')} "
            f"stop_set={bool(kw.get('post_stop'))} "
            f"n_msgs={kw.get('pre_n_msgs')}->{kw.get('post_n_msgs')} "
            f"first_role={kw.get('pre_first_role')}->{kw.get('post_first_role')}\n"
        )
        with _TRACE_PATH.open("a") as f:
            f.write(line)
    except Exception:
        pass


def _ensure_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    schema = (REPO_ROOT / "cost" / "schema.sql").read_text()
    conn.executescript(schema)
    return conn


def _estimate_tokens(messages: Iterable[dict[str, Any]] | None) -> int:
    """Cheap, fast token estimate. We do not need exact tokenization here -
    the goal is a routing decision, not billing."""
    if not messages:
        return 0
    total_chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total_chars += len(part["text"])
    # Rough: 1 token ~ 3.6 chars for English+code.
    return max(1, int(total_chars / 3.6))


def _flat_prompt(messages: Iterable[dict[str, Any]] | None) -> str:
    if not messages:
        return ""
    parts: list[str] = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    parts.append(p["text"])
    return "\n".join(parts)


def decide_tier(messages: Iterable[dict[str, Any]] | None) -> tuple[str, str, int]:
    """Return (model_name, reason, estimated_tokens)."""
    tokens = _estimate_tokens(messages)
    prompt = _flat_prompt(messages)
    is_complex, why = classify(prompt)

    if is_complex:
        return ("claude-code", f"complex: {why}", tokens)
    if tokens > ROUTE_LONG_MAX:
        return ("claude-code", f"tokens {tokens} > {ROUTE_LONG_MAX}", tokens)
    if tokens > ROUTE_FAST_MAX:
        return ("local-long", f"tokens {tokens} in [{ROUTE_FAST_MAX},{ROUTE_LONG_MAX}]", tokens)
    return ("local-fast", f"tokens {tokens} <= {ROUTE_FAST_MAX}", tokens)


# ----- LiteLLM callback shim ------------------------------------------------
#
# LiteLLM looks for a class with `async_pre_call_hook` and/or `log_success_event`.
# We implement both. The class is referenced by dotted path in the YAML config.

class SizeBasedRouter(_LiteLLMCustomLogger):
    """Subclass LiteLLM's CustomLogger so all post-call/streaming/failure hooks
    are no-ops by default; we only override the two we care about
    (pre-call routing and success logging)."""

    user_api_key_cache: dict[str, Any] = {}

    def __init__(self) -> None:
        try:
            super().__init__()
        except Exception:
            pass
        self._conn = _ensure_db()

    # ------------------ pre-call: rewrite hybrid-auto -> tier ---------------
    async def async_pre_call_hook(  # type: ignore[override]
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any] | None:
        try:
            # FIRST: optional full-request dump for investigations. Runs
            # before any of our routing or control mutation so the dump
            # is exactly what the client sent.
            if CLINE_TRACE:
                _dump_full_request(data)

            requested = data.get("model", "")
            # Cursor's "Verify" button rejects non-OpenAI-shaped names, so the
            # YAML config exposes mirror aliases prefixed with `gpt-`. Strip
            # that prefix here so downstream routing/decision logic sees the
            # canonical name (`hybrid-auto`, `local-long`, etc.) and behaves
            # identically regardless of which alias the client used.
            if requested.startswith("gpt-") and requested[len("gpt-"):] in (
                "hybrid-auto",
                "local-fast",
                "local-long",
                "local-agent",
                "claude-code",
            ):
                canonical = requested[len("gpt-"):]
                data["model"] = canonical
                requested = canonical

            if requested == "hybrid-auto":
                model, reason, tokens = decide_tier(data.get("messages"))
                data["model"] = model
                # Stash routing metadata for the success callback.
                meta = data.setdefault("metadata", {})
                meta["route_decision"] = model
                meta["route_reason"] = reason
                meta["route_tokens_estimated"] = tokens

            # Over-generation controls run AFTER the hybrid-auto rewrite
            # so the controls can see the resolved model name. Both
            # strategies are no-ops for non-local models and for
            # request shapes they don't recognize, so it's safe to call
            # them unconditionally even when the user picked Claude.
            #
            # Snapshot the inbound request so we can log a delta if any
            # control fires. This is cheap (the request dict is small)
            # and only matters when the trace log is enabled.
            pre_max = data.get("max_tokens")
            pre_stop = data.get("stop")
            pre_n_msgs = len(data.get("messages") or [])
            pre_first_role = ((data.get("messages") or [{}])[0] or {}).get("role")

            if ENABLE_STATIC_GUARDRAIL:
                apply_static_guardrail(data)
                meta = data.setdefault("metadata", {})
                meta["overgen_static_applied"] = True
            if ENABLE_MULTI_TURN_TIGHTEN:
                pre_max_mt = data.get("max_tokens")
                apply_multi_turn_tighten(data)
                post_max_mt = data.get("max_tokens")
                if pre_max_mt != post_max_mt:
                    meta = data.setdefault("metadata", {})
                    meta["overgen_multi_turn_applied"] = True

            if OVERGEN_TRACE and (ENABLE_STATIC_GUARDRAIL or ENABLE_MULTI_TURN_TIGHTEN):
                _trace_overgen(
                    requested=requested,
                    resolved=data.get("model"),
                    pre_max=pre_max,
                    post_max=data.get("max_tokens"),
                    pre_stop=pre_stop,
                    post_stop=data.get("stop"),
                    pre_n_msgs=pre_n_msgs,
                    post_n_msgs=len(data.get("messages") or []),
                    pre_first_role=pre_first_role,
                    post_first_role=((data.get("messages") or [{}])[0] or {}).get("role"),
                )
        except Exception as e:  # never break user requests
            print(f"[router] pre-call hook error: {e}", file=sys.stderr)
        return data

    # ------------------ post-call: record actual + shadow cost --------------
    def log_success_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        try:
            self._record(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            print(f"[router] log_success_event error: {e}", file=sys.stderr)

    async def async_log_success_event(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        try:
            self._record(kwargs, response_obj, start_time, end_time)
        except Exception as e:
            print(f"[router] async_log_success_event error: {e}", file=sys.stderr)

    def _record(self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
        model = kwargs.get("model") or kwargs.get("litellm_params", {}).get("model") or "unknown"

        usage = {}
        if hasattr(response_obj, "usage") and response_obj.usage:
            usage = (
                response_obj.usage.model_dump()
                if hasattr(response_obj.usage, "model_dump")
                else dict(response_obj.usage)
            )
        elif isinstance(response_obj, dict):
            usage = response_obj.get("usage", {}) or {}

        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)

        # Determine "tier": claude vs local-fast vs local-long. The `model`
        # string LiteLLM hands us depends on where in the lifecycle we are:
        #   - From async_pre_call_hook we already rewrote `data["model"]`
        #     to the alias (local-fast / local-long / claude-code), and
        #     LiteLLM sometimes echoes that back as `model` to the callback.
        #   - For direct alias calls (no hybrid-auto) the model can be the
        #     alias itself.
        #   - In other paths LiteLLM passes the upstream id (ollama/...,
        #     openai/mlx-community/..., anthropic/claude-...).
        # We need to handle all of those.
        m_lower = model.lower()
        if "claude" in m_lower or "anthropic" in m_lower:
            tier = "claude"
        elif (
            model.startswith("ollama/")
            or model == "local-long"
            or "qwen3-coder-next" in m_lower
            or m_lower.endswith((":q4_k_m", ":q8_0", ":q4_0"))
        ):
            tier = "local-long"
        elif (
            model == "local-fast"
            or model.startswith(("openai/", "mlx-"))
            or "mlx-community" in m_lower
            or "/" in model and "mlx" in m_lower
        ):
            tier = "local-fast"
        else:
            tier = "local-fast"  # default for unknown local routes

        # Actual cost: only Claude calls cost real money in this setup.
        actual_cost = 0.0
        if tier == "claude":
            actual_cost = in_tok * CLAUDE_INPUT_PER_TOKEN + out_tok * CLAUDE_OUTPUT_PER_TOKEN

        # Shadow cost: what Claude *would* have charged for this token volume.
        shadow_cost = in_tok * CLAUDE_INPUT_PER_TOKEN + out_tok * CLAUDE_OUTPUT_PER_TOKEN

        try:
            latency_ms = int((float(end_time) - float(start_time)) * 1000)
        except Exception:
            try:
                latency_ms = int((end_time - start_time).total_seconds() * 1000)
            except Exception:
                latency_ms = 0

        meta = kwargs.get("metadata", {}) or {}
        reason = meta.get("route_reason", "")

        self._conn.execute(
            """
            INSERT INTO requests
              (ts, model, tier, input_tok, output_tok, actual_cost, shadow_cost, latency_ms, route_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(time.time()),
                model,
                tier,
                in_tok,
                out_tok,
                actual_cost,
                shadow_cost,
                latency_ms,
                reason,
            ),
        )
        self._conn.commit()


# Module-level instance so LiteLLM can import either the class or this object.
proxy_handler_instance = SizeBasedRouter()


if __name__ == "__main__":
    # Tiny self-test.
    msgs = [{"role": "user", "content": "Refactor this function across multiple files please."}]
    print(decide_tier(msgs))
    msgs2 = [{"role": "user", "content": "x = 1\n" * 5000}]
    print(decide_tier(msgs2))
