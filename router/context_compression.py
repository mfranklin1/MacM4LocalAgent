"""Context compression for long agent conversations.

When a conversation's estimated token count approaches the model's context
window, this module prunes redundant or oversized content from the message
list before forwarding the request upstream.

Two engines are available:

  RuleBasedEngine  -- pure-Python, always available.  Applies three rules:
    1. Deduplicates consecutive identical tool results (same role+content hash).
    2. Truncates any single tool-result message whose content exceeds
       TOOL_RESULT_MAX_CHARS (default 50 000 chars).
    3. Preserves the system message and the most-recent KEEP_RECENT_MSGS
       (default 4) messages unconditionally, plus any message that contains
       an error keyword.

  HeadroomEngine   -- thin wrapper around the optional `headroom` library.
    Raises NotImplementedError on import; callers must catch it and fall
    back to RuleBasedEngine.

ContextCompressor is the public entry point.  It:
  - Returns the original messages unchanged when
    CONTEXT_COMPRESSION_ENABLED=0 or the estimated token count is below
    MIN_TOKENS_TO_TRIGGER (default 32 000).
  - Otherwise delegates to whichever engine is configured.

Environment variables
---------------------
CONTEXT_COMPRESSION_ENABLED  -- set to "0" to disable entirely (default "1")
CONTEXT_COMPRESSION_ENGINE   -- "rule_based" (default) or "headroom"
MIN_TOKENS_TO_TRIGGER         -- int, skip compression below this threshold
TOOL_RESULT_MAX_CHARS         -- int, truncation threshold per tool-result
KEEP_RECENT_MSGS              -- int, unconditional tail to preserve
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any

# ---- tuneable defaults -------------------------------------------------------

_DEFAULT_MIN_TOKENS_TO_TRIGGER = 32_000
_DEFAULT_TOOL_RESULT_MAX_CHARS = 50_000
_DEFAULT_KEEP_RECENT_MSGS = 4

_ERROR_KEYWORDS = ("error", "traceback", "exception", "failed", "failure")

# Conservative estimated compression ratio (token count after / before).
# 0.65 ≈ 35% reduction; used by the module-level `compress()` shim when
# only the pre-compression token count is known (no message content available).
_COMPRESSION_RATIO_ESTIMATE = float(
    os.environ.get("CONTEXT_COMPRESSION_RATIO_ESTIMATE", "0.65")
)


# ---- module-level shim -------------------------------------------------------

def compress(token_count: int) -> int:
    """Estimate the compressed token count for a given raw token count.

    Called by route_by_size._try_turbo_escalation when it knows the token
    count but not the message content (messages are not available at routing
    decision time in that path).  Returns a conservative estimate using
    CONTEXT_COMPRESSION_RATIO_ESTIMATE (default 0.65 = 35% reduction).

    This is deliberately an estimation; the real compression happens inside
    ContextCompressor.maybe_compress() when the request is forwarded.
    """
    if token_count <= 0:
        return 0
    return max(1, int(token_count * _COMPRESSION_RATIO_ESTIMATE))


# ---- helpers -----------------------------------------------------------------

def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """Cheap char / 3.6 heuristic.  Fast; no external deps."""
    total_chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            total_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total_chars += len(part["text"])
    return max(1, int(total_chars / 3.6)) if total_chars else 0


def _message_text(msg: dict[str, Any]) -> str:
    """Return a flat string representation of a message's content."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(
            part.get("text", "") for part in c if isinstance(part, dict)
        )
    return ""


def _content_hash(msg: dict[str, Any]) -> str:
    text = _message_text(msg)
    role = msg.get("role", "")
    return hashlib.md5(f"{role}:{text}".encode()).hexdigest()


def _contains_error(msg: dict[str, Any]) -> bool:
    text = _message_text(msg).lower()
    return any(kw in text for kw in _ERROR_KEYWORDS)


def _is_system(msg: dict[str, Any]) -> bool:
    return msg.get("role") == "system"


# ---- result dataclass --------------------------------------------------------

@dataclass
class CompressionResult:
    messages: list[dict[str, Any]]
    original_count: int
    dropped_items_count: int
    original_estimated_tokens: int
    compressed_estimated_tokens: int
    compression_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        if self.original_estimated_tokens > 0:
            self.compression_ratio = (
                self.compressed_estimated_tokens / self.original_estimated_tokens
            )
        else:
            self.compression_ratio = 1.0


# ---- engines -----------------------------------------------------------------

class RuleBasedEngine:
    """Pure-Python context compressor.  No external dependencies."""

    def __init__(
        self,
        tool_result_max_chars: int = _DEFAULT_TOOL_RESULT_MAX_CHARS,
        keep_recent_msgs: int = _DEFAULT_KEEP_RECENT_MSGS,
    ) -> None:
        self.tool_result_max_chars = tool_result_max_chars
        self.keep_recent_msgs = keep_recent_msgs

    def compress(self, messages: list[dict[str, Any]]) -> CompressionResult:
        """Apply the three compression rules and return a CompressionResult."""
        original_count = len(messages)
        original_tokens = estimate_tokens(messages)

        # Partition into system messages, protected tail, and body.
        system_msgs = [m for m in messages if _is_system(m)]
        non_system = [m for m in messages if not _is_system(m)]

        # Protected tail: the most-recent N non-system messages.
        tail_size = min(self.keep_recent_msgs, len(non_system))
        body = non_system[: len(non_system) - tail_size]
        tail = non_system[len(non_system) - tail_size :]

        # Rule 1 + Rule 3 applied to the body only.
        seen_hashes: set[str] = set()
        kept_body: list[dict[str, Any]] = []
        for msg in body:
            # Rule 3: always preserve error-containing messages.
            if _contains_error(msg):
                kept_body.append(msg)
                seen_hashes.add(_content_hash(msg))
                continue

            # Rule 1: deduplicate consecutive identical content.
            h = _content_hash(msg)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            kept_body.append(msg)

        # Rule 2: truncate oversized content in all non-system messages.
        max_chars = self.tool_result_max_chars

        def _truncate(msg: dict[str, Any]) -> dict[str, Any]:
            c = msg.get("content")
            if isinstance(c, str) and len(c) > max_chars:
                truncated = c[:max_chars]
                note = f"\n[...truncated {len(c) - max_chars} chars...]"
                return {**msg, "content": truncated + note}
            if isinstance(c, list):
                new_parts: list[dict[str, Any]] = []
                changed = False
                for part in c:
                    if isinstance(part, dict) and isinstance(part.get("text"), str):
                        t = part["text"]
                        if len(t) > max_chars:
                            note = f"\n[...truncated {len(t) - max_chars} chars...]"
                            new_parts.append({**part, "text": t[:max_chars] + note})
                            changed = True
                            continue
                    new_parts.append(part)
                if changed:
                    return {**msg, "content": new_parts}
            return msg

        kept_body = [_truncate(m) for m in kept_body]
        tail = [_truncate(m) for m in tail]

        compressed = system_msgs + kept_body + tail
        dropped = original_count - len(compressed)
        compressed_tokens = estimate_tokens(compressed)

        return CompressionResult(
            messages=compressed,
            original_count=original_count,
            dropped_items_count=dropped,
            original_estimated_tokens=original_tokens,
            compressed_estimated_tokens=compressed_tokens,
        )


class HeadroomEngine:
    """Wrapper around the optional `headroom` library.

    Raises NotImplementedError immediately: the library is not yet installed
    in this environment.  Callers should catch the error and fall back to
    RuleBasedEngine.
    """

    def __init__(self, **_: Any) -> None:
        raise NotImplementedError(
            "HeadroomEngine requires the 'headroom' package, which is not "
            "installed.  Install it with: pip install headroom"
        )

    def compress(self, messages: list[dict[str, Any]]) -> CompressionResult:  # pragma: no cover
        raise NotImplementedError("HeadroomEngine is not available.")


# ---- public compressor -------------------------------------------------------

class ContextCompressor:
    """Public entry point for context compression.

    Checks the CONTEXT_COMPRESSION_ENABLED env var and the estimated token
    count before delegating to an engine.
    """

    def __init__(
        self,
        engine: str | None = None,
        min_tokens_to_trigger: int | None = None,
        tool_result_max_chars: int | None = None,
        keep_recent_msgs: int | None = None,
    ) -> None:
        self._engine_name = engine or os.environ.get(
            "CONTEXT_COMPRESSION_ENGINE", "rule_based"
        )
        self._min_tokens = min_tokens_to_trigger if min_tokens_to_trigger is not None else int(
            os.environ.get("MIN_TOKENS_TO_TRIGGER", str(_DEFAULT_MIN_TOKENS_TO_TRIGGER))
        )
        self._max_chars = tool_result_max_chars if tool_result_max_chars is not None else int(
            os.environ.get("TOOL_RESULT_MAX_CHARS", str(_DEFAULT_TOOL_RESULT_MAX_CHARS))
        )
        self._keep_recent = keep_recent_msgs if keep_recent_msgs is not None else int(
            os.environ.get("KEEP_RECENT_MSGS", str(_DEFAULT_KEEP_RECENT_MSGS))
        )

    def _is_enabled(self) -> bool:
        raw = os.environ.get("CONTEXT_COMPRESSION_ENABLED", "1")
        return str(raw).strip() not in {"0", "false", "no", "off"}

    def _build_engine(self) -> RuleBasedEngine:
        if self._engine_name == "headroom":
            # Raises NotImplementedError -- callers can catch.
            return HeadroomEngine()  # type: ignore[return-value]
        return RuleBasedEngine(
            tool_result_max_chars=self._max_chars,
            keep_recent_msgs=self._keep_recent,
        )

    def maybe_compress(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Return (possibly compressed) messages.

        Skips compression if disabled or if the estimated token count is
        below the trigger threshold.
        """
        if not self._is_enabled():
            return messages

        estimated = estimate_tokens(messages)
        if estimated < self._min_tokens:
            return messages

        engine = self._build_engine()
        result = engine.compress(messages)
        return result.messages
