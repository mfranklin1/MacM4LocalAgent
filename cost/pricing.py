"""Per-model Claude pricing.

Single source of truth for what each Claude model costs. Used by:
  - cost/ingest.py to compute shadow_cost (always Sonnet 4.6 rates,
    so historical comparisons stay stable -- see CLAUDE_SHADOW_MODEL).
  - router/route_by_size.py to compute actual_cost based on which
    Claude model was actually called.
  - bench/runners/litellm_arm.py and bench/runners/cursor_loop.py
    for benchmark cost estimates.

The table is hand-maintained. To check for upstream drift, run
`make check-pricing` (which fetches Anthropic's published pricing
page and diffs against this file). The proxy ALSO does an async
freshness check on startup -- it's strictly informational and
never blocks requests; see _maybe_run_pricing_check().

Rates are stored as USD-per-token, derived from Anthropic's
USD-per-million-tokens published rates. See
https://docs.anthropic.com/en/docs/about-claude/pricing.
"""

from __future__ import annotations

import os
import pathlib
import sys
import threading
import time
from dataclasses import dataclass

# When the table below was last reconciled with Anthropic's docs.
# scripts/check_claude_pricing.py compares against this and the
# startup nag fires when this is more than PRICING_STALE_DAYS old.
PRICING_LAST_UPDATED = "2026-07-01"
PRICING_STALE_DAYS = 30

# The model whose rates define `shadow_cost`. shadow_cost is the
# canonical "what would Claude have charged?" benchmark, used to
# compute savings. Pinning it to a single model keeps savings
# numbers comparable month-over-month, even if the router starts
# using cheaper Haiku or pricier Opus models for some requests.
CLAUDE_SHADOW_MODEL = "claude-sonnet-4-6"


@dataclass(frozen=True)
class ClaudeRate:
    """Per-token rates for one Claude model (USD per token).

    The fields named `input` / `output` are the standard rates.
    `cache_write_5m`, `cache_write_1h`, and `cache_read` are the
    prompt-caching rates from the docs. They aren't currently used
    for cost recording (the proxy doesn't have a way to attribute
    cache-hit tokens separately from base input tokens) but they
    live here so a future pass can pick them up without touching
    the table.
    """
    input: float
    output: float
    cache_write_5m: float
    cache_write_1h: float
    cache_read: float


def _per_token(per_mtok: float) -> float:
    return per_mtok / 1_000_000.0


# Order matches the Anthropic docs table for easier diffing.
# Source of truth: https://docs.anthropic.com/en/docs/about-claude/pricing
# Last reconciled: see PRICING_LAST_UPDATED above.
CLAUDE_PRICES: dict[str, ClaudeRate] = {
    "claude-sonnet-5": ClaudeRate(
        # Introductory pricing through 2026-08-31; standard after is $3/$15/1M.
        input=_per_token(2.0),
        output=_per_token(10.0),
        cache_write_5m=_per_token(2.50),
        cache_write_1h=_per_token(4.0),
        cache_read=_per_token(0.20),
    ),
    "claude-opus-4-8": ClaudeRate(
        input=_per_token(5.0),
        output=_per_token(25.0),
        cache_write_5m=_per_token(6.25),
        cache_write_1h=_per_token(10.0),
        cache_read=_per_token(0.50),
    ),
    "claude-opus-4-7": ClaudeRate(
        input=_per_token(5.0),
        output=_per_token(25.0),
        cache_write_5m=_per_token(6.25),
        cache_write_1h=_per_token(10.0),
        cache_read=_per_token(0.50),
    ),
    "claude-opus-4-6": ClaudeRate(
        input=_per_token(5.0),
        output=_per_token(25.0),
        cache_write_5m=_per_token(6.25),
        cache_write_1h=_per_token(10.0),
        cache_read=_per_token(0.50),
    ),
    "claude-opus-4-5": ClaudeRate(
        input=_per_token(5.0),
        output=_per_token(25.0),
        cache_write_5m=_per_token(6.25),
        cache_write_1h=_per_token(10.0),
        cache_read=_per_token(0.50),
    ),
    "claude-opus-4-1": ClaudeRate(
        input=_per_token(15.0),
        output=_per_token(75.0),
        cache_write_5m=_per_token(18.75),
        cache_write_1h=_per_token(30.0),
        cache_read=_per_token(1.50),
    ),
    "claude-opus-4": ClaudeRate(
        input=_per_token(15.0),
        output=_per_token(75.0),
        cache_write_5m=_per_token(18.75),
        cache_write_1h=_per_token(30.0),
        cache_read=_per_token(1.50),
    ),
    "claude-sonnet-4-6": ClaudeRate(
        input=_per_token(3.0),
        output=_per_token(15.0),
        cache_write_5m=_per_token(3.75),
        cache_write_1h=_per_token(6.0),
        cache_read=_per_token(0.30),
    ),
    "claude-sonnet-4-5": ClaudeRate(
        input=_per_token(3.0),
        output=_per_token(15.0),
        cache_write_5m=_per_token(3.75),
        cache_write_1h=_per_token(6.0),
        cache_read=_per_token(0.30),
    ),
    "claude-sonnet-4": ClaudeRate(
        input=_per_token(3.0),
        output=_per_token(15.0),
        cache_write_5m=_per_token(3.75),
        cache_write_1h=_per_token(6.0),
        cache_read=_per_token(0.30),
    ),
    "claude-sonnet-3-7": ClaudeRate(  # deprecated, kept for historical rows
        input=_per_token(3.0),
        output=_per_token(15.0),
        cache_write_5m=_per_token(3.75),
        cache_write_1h=_per_token(6.0),
        cache_read=_per_token(0.30),
    ),
    "claude-haiku-4-5": ClaudeRate(
        input=_per_token(1.0),
        output=_per_token(5.0),
        cache_write_5m=_per_token(1.25),
        cache_write_1h=_per_token(2.0),
        cache_read=_per_token(0.10),
    ),
    "claude-haiku-3-5": ClaudeRate(
        input=_per_token(0.80),
        output=_per_token(4.0),
        cache_write_5m=_per_token(1.0),
        cache_write_1h=_per_token(1.6),
        cache_read=_per_token(0.08),
    ),
    "claude-opus-3": ClaudeRate(  # deprecated
        input=_per_token(15.0),
        output=_per_token(75.0),
        cache_write_5m=_per_token(18.75),
        cache_write_1h=_per_token(30.0),
        cache_read=_per_token(1.50),
    ),
    "claude-haiku-3": ClaudeRate(
        input=_per_token(0.25),
        output=_per_token(1.25),
        cache_write_5m=_per_token(0.30),
        cache_write_1h=_per_token(0.50),
        cache_read=_per_token(0.03),
    ),
}

# A few common aliases LiteLLM / Anthropic docs use that don't match
# the canonical key. Map them here so callers don't have to know.
_ALIASES: dict[str, str] = {
    # LiteLLM/Anthropic shorthand sometimes uses underscores or no hyphens.
    "claude-sonnet-5.0": "claude-sonnet-5",
    "claude-opus-4.8":   "claude-opus-4-8",
    "claude-sonnet-4.6": "claude-sonnet-4-6",
    "claude-opus-4.7":   "claude-opus-4-7",
    "claude-opus-4.6":   "claude-opus-4-6",
    "claude-haiku-4.5":  "claude-haiku-4-5",
    # The "claude-code" tier alias used internally by this proxy. Must
    # match the upstream `model` field for `claude-code` in
    # config/litellm-config.yaml. If you change one, change both.
    "claude-code":       "claude-sonnet-5",
}

# Track which unknown models we've already warned about so we only
# nag stderr once per process per unknown id.
_warned_unknown: set[str] = set()
_warned_lock = threading.Lock()


def _normalize(model_id: str) -> str:
    """Strip provider prefixes and lowercase. Returns the canonical
    table key on a best-effort basis.

    Examples:
        anthropic/claude-sonnet-4-6 -> claude-sonnet-4-6
        Claude-Opus-4-7             -> claude-opus-4-7
        claude-opus-4-7-20260416    -> claude-opus-4-7
        claude-3-5-sonnet-...       -> claude-sonnet-3-5  (best effort)
    """
    if not model_id:
        return ""
    m = model_id.strip().lower()
    # Strip provider prefix(es): anthropic/, vertex_ai/, bedrock/, etc.
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    # Resolve known aliases first.
    if m in _ALIASES:
        return _ALIASES[m]
    if m in CLAUDE_PRICES:
        return m
    # Some Anthropic ids include a date suffix: claude-opus-4-7-20260416.
    # Try progressively shorter prefixes.
    parts = m.split("-")
    for cut in range(len(parts), 1, -1):
        candidate = "-".join(parts[:cut])
        if candidate in CLAUDE_PRICES:
            return candidate
        if candidate in _ALIASES:
            return _ALIASES[candidate]
    return m  # caller will treat as unknown


def claude_rate(model_id: str) -> ClaudeRate:
    """Return the per-token rate for a Claude model id.

    Falls back to Sonnet 4.6 rates for unknown models, and prints a
    one-time stderr warning so you notice you need to update the
    table. Never raises; the proxy must always be able to compute
    a number.
    """
    canonical = _normalize(model_id)
    if canonical in CLAUDE_PRICES:
        return CLAUDE_PRICES[canonical]
    _warn_once_unknown(model_id, canonical)
    return CLAUDE_PRICES[CLAUDE_SHADOW_MODEL]


def sonnet_rate() -> ClaudeRate:
    """Shorthand for the canonical shadow-cost rate (Sonnet 4.6)."""
    return CLAUDE_PRICES[CLAUDE_SHADOW_MODEL]


def shadow_cost(in_tok: int, out_tok: int) -> float:
    """Stable shadow benchmark: 'what Sonnet 4.6 would have charged'.

    Pinning this to one model means savings comparisons are
    apples-to-apples across time. If we switch the actual router to
    Haiku for some traffic tomorrow, savings.py will still subtract
    the same Sonnet baseline, so trends don't have a phantom step.
    """
    r = sonnet_rate()
    return in_tok * r.input + out_tok * r.output


def actual_claude_cost(model_id: str, in_tok: int, out_tok: int) -> float:
    """What we actually paid for this Claude call."""
    r = claude_rate(model_id)
    return in_tok * r.input + out_tok * r.output


def _warn_once_unknown(model_id: str, canonical: str) -> None:
    key = canonical or model_id
    with _warned_lock:
        if key in _warned_unknown:
            return
        _warned_unknown.add(key)
    print(
        f"[pricing] WARNING: unknown Claude model '{model_id}' "
        f"(normalized='{canonical}'). Falling back to "
        f"{CLAUDE_SHADOW_MODEL} rates. Update cost/pricing.py "
        f"and run `make check-pricing` to refresh.",
        file=sys.stderr,
        flush=True,
    )


# ---- freshness ---------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
LAST_CHECK_PATH = REPO_ROOT / ".logs" / "pricing-last-check"


def _table_age_days(now: float | None = None) -> float:
    """How many days has it been since the table was reconciled?"""
    now = now if now is not None else time.time()
    try:
        last = time.mktime(time.strptime(PRICING_LAST_UPDATED, "%Y-%m-%d"))
    except ValueError:
        return 0.0
    return max(0.0, (now - last) / 86400.0)


def is_table_stale() -> bool:
    return _table_age_days() > PRICING_STALE_DAYS


def maybe_run_pricing_check(*, force: bool = False) -> None:
    """Best-effort startup nag: if the table is stale or we haven't
    done a network check in 24h, run scripts/check_claude_pricing.py
    in a background thread. Never blocks the caller; never crashes
    on errors. The check itself is read-only (it never writes back
    to cost/pricing.py); on drift it just logs a one-line warning.

    Disable with PRICING_STARTUP_CHECK=0 (e.g. for tests).
    """
    if os.environ.get("PRICING_STARTUP_CHECK", "1").strip().lower() in {"0", "false", "no", "off"}:
        return

    LAST_CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not force and LAST_CHECK_PATH.exists():
        try:
            age_s = time.time() - LAST_CHECK_PATH.stat().st_mtime
            if age_s < 24 * 3600 and not is_table_stale():
                return
        except OSError:
            pass

    def _run() -> None:
        try:
            import subprocess
            checker = REPO_ROOT / "scripts" / "check_claude_pricing.py"
            if not checker.exists():
                return
            proc = subprocess.run(
                [sys.executable, str(checker), "--quiet"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            LAST_CHECK_PATH.touch()
            if proc.returncode != 0 and proc.stdout.strip():
                # Only the first line: avoid spamming stderr on big diffs.
                first = proc.stdout.strip().splitlines()[0]
                print(
                    f"[pricing] startup check flagged drift: {first}. "
                    f"Run `make check-pricing` for the full diff.",
                    file=sys.stderr,
                    flush=True,
                )
            elif is_table_stale():
                print(
                    f"[pricing] table last reconciled {PRICING_LAST_UPDATED} "
                    f"({_table_age_days():.0f}d ago). Run `make check-pricing`.",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception:
            return

    threading.Thread(target=_run, daemon=True, name="pricing-check").start()
