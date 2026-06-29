"""Unit tests for router/context_compression.py."""

from __future__ import annotations

import pytest

from router.context_compression import (
    CompressionResult,
    ContextCompressor,
    HeadroomEngine,
    RuleBasedEngine,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content}


# ---------------------------------------------------------------------------
# 1. RuleBasedEngine.compress() — duplicate tool results → deduplicated
# ---------------------------------------------------------------------------

class TestRuleBasedDeduplication:
    def test_duplicate_body_messages_are_deduplicated(self) -> None:
        """Two consecutive identical messages in the body should collapse to one."""
        engine = RuleBasedEngine(keep_recent_msgs=1)
        duplicate_msg = _tool("SELECT * FROM users;")
        messages = [
            duplicate_msg,
            duplicate_msg,  # exact copy
            _user("done"),  # recent tail — preserved unconditionally
        ]
        result = engine.compress(messages)
        # The tail ("done") is kept; of the two identical tool results only
        # one should survive.
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

    def test_different_messages_are_not_deduplicated(self) -> None:
        """Different messages must not be collapsed."""
        engine = RuleBasedEngine(keep_recent_msgs=1)
        messages = [
            _tool("result A"),
            _tool("result B"),
            _user("tail"),
        ]
        result = engine.compress(messages)
        tool_msgs = [m for m in result.messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2

    def test_dropped_count_reflects_deduplication(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=1)
        dup = _tool("dup")
        messages = [dup, dup, dup, _user("tail")]
        result = engine.compress(messages)
        # Three dups → one kept; two dropped.
        assert result.dropped_items_count == 2


# ---------------------------------------------------------------------------
# 2. RuleBasedEngine.compress() — very large tool result → truncated
# ---------------------------------------------------------------------------

class TestRuleBasedTruncation:
    def test_large_string_content_is_truncated(self) -> None:
        """A tool result exceeding TOOL_RESULT_MAX_CHARS must be cut."""
        max_chars = 1000
        engine = RuleBasedEngine(tool_result_max_chars=max_chars, keep_recent_msgs=0)
        big = "x" * (max_chars + 5000)
        messages = [_tool(big)]
        result = engine.compress(messages)
        assert len(result.messages) == 1
        content = result.messages[0]["content"]
        assert len(content) <= max_chars + 60  # small truncation note appended
        assert "truncated" in content

    def test_large_list_content_is_truncated(self) -> None:
        """A tool result with list-of-parts content is also truncated."""
        max_chars = 500
        engine = RuleBasedEngine(tool_result_max_chars=max_chars, keep_recent_msgs=0)
        big_text = "y" * (max_chars + 2000)
        messages = [{"role": "tool", "content": [{"type": "text", "text": big_text}]}]
        result = engine.compress(messages)
        part = result.messages[0]["content"][0]
        assert len(part["text"]) <= max_chars + 60
        assert "truncated" in part["text"]

    def test_small_content_is_not_truncated(self) -> None:
        engine = RuleBasedEngine(tool_result_max_chars=1000, keep_recent_msgs=0)
        messages = [_tool("small")]
        result = engine.compress(messages)
        assert result.messages[0]["content"] == "small"


# ---------------------------------------------------------------------------
# 3. RuleBasedEngine.compress() — system + recent tail always preserved
# ---------------------------------------------------------------------------

class TestRuleBasedPreservation:
    def test_system_message_always_preserved(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=4)
        messages = [
            _system("You are a helpful assistant."),
            _tool("old result 1"),
            _tool("old result 2"),
            _tool("old result 3"),
            _user("turn 1"),
            _assistant("reply 1"),
            _user("turn 2"),
            _assistant("reply 2"),
        ]
        result = engine.compress(messages)
        system_msgs = [m for m in result.messages if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are a helpful assistant."

    def test_recent_four_messages_always_preserved(self) -> None:
        """The last 4 non-system messages must survive verbatim."""
        engine = RuleBasedEngine(keep_recent_msgs=4)
        tail_msgs = [
            _user("tail-1"),
            _assistant("tail-2"),
            _user("tail-3"),
            _assistant("tail-4"),
        ]
        messages = [
            _system("sys"),
            _tool("old"),
            _tool("old"),  # would be deduped from body
        ] + tail_msgs
        result = engine.compress(messages)
        result_roles_content = [(m["role"], m["content"]) for m in result.messages]
        for tm in tail_msgs:
            assert (tm["role"], tm["content"]) in result_roles_content

    def test_empty_message_list_handled(self) -> None:
        engine = RuleBasedEngine()
        result = engine.compress([])
        assert result.messages == []
        assert result.dropped_items_count == 0


# ---------------------------------------------------------------------------
# 4. RuleBasedEngine.compress() — error-containing messages preserved
# ---------------------------------------------------------------------------

class TestRuleBasedErrorPreservation:
    def test_error_message_not_dropped_even_when_duplicate(self) -> None:
        """A message with 'error' in its text must never be dropped."""
        engine = RuleBasedEngine(keep_recent_msgs=1)
        error_msg = _tool("Traceback (most recent call last): ...")
        messages = [
            error_msg,
            error_msg,  # duplicate, but contains error keyword
            _user("tail"),
        ]
        result = engine.compress(messages)
        error_msgs = [
            m for m in result.messages
            if "traceback" in m.get("content", "").lower()
        ]
        # Both copies survive (error preservation beats dedup).
        assert len(error_msgs) == 2

    def test_message_with_failure_keyword_preserved(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=1)
        fail_msg = _tool("Build failure: missing semicolon")
        messages = [fail_msg, _user("tail")]
        result = engine.compress(messages)
        assert any("failure" in m.get("content", "") for m in result.messages)

    def test_message_with_exception_keyword_preserved(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=1)
        exc_msg = _tool("RuntimeException: null pointer")
        messages = [exc_msg, _user("tail")]
        result = engine.compress(messages)
        assert any("exception" in m.get("content", "").lower() for m in result.messages)


# ---------------------------------------------------------------------------
# 5. CompressionResult fields are accurate
# ---------------------------------------------------------------------------

class TestCompressionResultFields:
    def test_dropped_items_count_matches_actual_drop(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=1)
        dup = _tool("same")
        messages = [dup, dup, _user("tail")]
        result = engine.compress(messages)
        assert result.original_count == 3
        assert result.dropped_items_count == 1
        assert len(result.messages) == result.original_count - result.dropped_items_count

    def test_compression_ratio_is_lte_one_when_tokens_reduced(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=1)
        big = "z" * 3600  # ~1000 tokens
        dup = _tool(big)
        messages = [dup, dup, _user("ok")]
        result = engine.compress(messages)
        assert result.compression_ratio <= 1.0

    def test_compression_ratio_is_one_when_nothing_dropped(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=10)
        messages = [_user("a"), _assistant("b"), _user("c")]
        result = engine.compress(messages)
        # Nothing is dropped (all fit in tail or are unique).
        assert result.dropped_items_count == 0
        assert result.compression_ratio == pytest.approx(1.0, abs=0.05)

    def test_original_estimated_tokens_matches_estimate_tokens(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=0)
        messages = [_user("hello" * 36)]  # 180 chars → 50 tokens
        result = engine.compress(messages)
        expected = estimate_tokens(messages)
        assert result.original_estimated_tokens == expected

    def test_compression_ratio_is_one_for_empty_input(self) -> None:
        engine = RuleBasedEngine()
        result = engine.compress([])
        assert result.compression_ratio == pytest.approx(1.0)

    def test_compressed_estimated_tokens_gte_zero(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=2)
        messages = [_tool("x" * 1000), _tool("x" * 1000), _user("hi")]
        result = engine.compress(messages)
        assert result.compressed_estimated_tokens >= 0


# ---------------------------------------------------------------------------
# 6. ContextCompressor skips compression when estimated tokens < min_tokens
# ---------------------------------------------------------------------------

class TestContextCompressorSkipsLowTokens:
    def test_skips_when_below_threshold(self) -> None:
        """With a very high min_tokens_to_trigger the messages pass through unchanged."""
        compressor = ContextCompressor(min_tokens_to_trigger=10_000_000)
        messages = [_user("short message"), _assistant("reply")]
        result = compressor.maybe_compress(messages)
        assert result is messages or result == messages

    def test_applies_when_above_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With a low threshold, compression should be attempted."""
        # Use keep_recent_msgs=1 so the dups fall in the body and get deduped.
        dup = _tool("same text")
        # 5 dups → body has 4 dups, tail has 1; body dups collapse to 1.
        messages = [dup] * 5 + [_user("tail")]
        compressor = ContextCompressor(
            min_tokens_to_trigger=1,
            keep_recent_msgs=1,
        )
        result = compressor.maybe_compress(messages)
        # Deduplication should have collapsed the body dups to 1.
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

    def test_threshold_boundary_exact_tokens(self) -> None:
        """When estimated tokens exactly equal the threshold, skip compression."""
        # 360 chars / 3.6 = 100 tokens exactly.
        messages = [_user("a" * 360)]
        compressor = ContextCompressor(min_tokens_to_trigger=100)
        result = compressor.maybe_compress(messages)
        # Exactly at threshold → skip (strict less-than).
        assert result == messages


# ---------------------------------------------------------------------------
# 7. ContextCompressor skips compression when CONTEXT_COMPRESSION_ENABLED=0
# ---------------------------------------------------------------------------

class TestContextCompressorEnvDisable:
    def test_disabled_via_env_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTEXT_COMPRESSION_ENABLED", "0")
        compressor = ContextCompressor(min_tokens_to_trigger=1)
        dup = _tool("same")
        messages = [dup, dup, _user("tail")]
        result = compressor.maybe_compress(messages)
        # When disabled, messages returned as-is.
        assert result == messages

    def test_disabled_via_env_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONTEXT_COMPRESSION_ENABLED", "false")
        compressor = ContextCompressor(min_tokens_to_trigger=1)
        messages = [_user("a"), _user("a"), _user("tail")]
        result = compressor.maybe_compress(messages)
        assert result == messages

    def test_enabled_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CONTEXT_COMPRESSION_ENABLED", raising=False)
        # Keep only 1 recent message so the dups land in the body.
        dup = _tool("dup")
        messages = [dup] * 5 + [_user("tail")]
        compressor = ContextCompressor(min_tokens_to_trigger=1, keep_recent_msgs=1)
        result = compressor.maybe_compress(messages)
        # Enabled by default → compression runs → body dups collapsed to 1.
        tool_msgs = [m for m in result if m.get("role") == "tool"]
        assert len(tool_msgs) == 1

    def test_env_overrides_checked_at_call_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_is_enabled() reads the env at call time, not at __init__ time."""
        compressor = ContextCompressor(min_tokens_to_trigger=1)
        monkeypatch.setenv("CONTEXT_COMPRESSION_ENABLED", "0")
        dup = _tool("same")
        messages = [dup, dup, _user("tail")]
        result = compressor.maybe_compress(messages)
        assert result == messages


# ---------------------------------------------------------------------------
# 8. HeadroomEngine.compress() raises NotImplementedError
# ---------------------------------------------------------------------------

class TestHeadroomEngine:
    def test_instantiation_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="headroom"):
            HeadroomEngine()

    def test_context_compressor_headroom_engine_raises(self) -> None:
        """When the headroom engine is selected, ContextCompressor must propagate
        the NotImplementedError from HeadroomEngine.__init__."""
        compressor = ContextCompressor(
            engine="headroom", min_tokens_to_trigger=1
        )
        # Need enough tokens to pass the threshold check.
        messages = [_user("x" * 3600)]  # 1000 estimated tokens
        with pytest.raises(NotImplementedError):
            compressor.maybe_compress(messages)


# ---------------------------------------------------------------------------
# 9. estimate_tokens() heuristic matches chars/3.6 formula
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    def test_string_content(self) -> None:
        messages = [_user("a" * 360)]
        assert estimate_tokens(messages) == 100  # 360 / 3.6 = 100

    def test_list_content(self) -> None:
        messages = [{"role": "user", "content": [
            {"type": "text", "text": "a" * 180},
            {"type": "image", "data": "..."},  # non-text ignored
            {"type": "text", "text": "b" * 180},
        ]}]
        assert estimate_tokens(messages) == 100  # (180+180) / 3.6 = 100

    def test_empty_messages(self) -> None:
        assert estimate_tokens([]) == 0

    def test_multiple_messages_summed(self) -> None:
        messages = [
            _user("a" * 180),   # 50 tokens
            _assistant("b" * 180),  # 50 tokens
        ]
        assert estimate_tokens(messages) == 100

    def test_minimum_is_one_for_nonempty_content(self) -> None:
        messages = [_user("x")]  # 1 char → max(1, int(1/3.6)) = max(1, 0) = 1
        assert estimate_tokens(messages) >= 1

    def test_formula_consistency_across_sizes(self) -> None:
        for n_chars in (36, 360, 3600, 36000):
            messages = [_user("z" * n_chars)]
            expected = int(n_chars / 3.6)
            assert estimate_tokens(messages) == expected

    def test_non_text_parts_ignored(self) -> None:
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": "http://example.com/img.png"},
        ]}]
        assert estimate_tokens(messages) == 0


# ---------------------------------------------------------------------------
# 10. Compression never removes system messages
# ---------------------------------------------------------------------------

class TestSystemMessageNeverRemoved:
    def test_single_system_message_survives_heavy_compression(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=2)
        sys_msg = _system("SYSTEM PROMPT")
        dup = _tool("dup result")
        messages = [sys_msg] + [dup] * 20 + [_user("a"), _assistant("b")]
        result = engine.compress(messages)
        assert any(m.get("role") == "system" for m in result.messages)

    def test_multiple_system_messages_all_survive(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=1)
        sys1 = _system("System 1")
        sys2 = _system("System 2")
        messages = [sys1, sys2, _tool("dup"), _tool("dup"), _user("tail")]
        result = engine.compress(messages)
        system_msgs = [m for m in result.messages if m.get("role") == "system"]
        assert len(system_msgs) == 2

    def test_system_message_position_is_first(self) -> None:
        engine = RuleBasedEngine(keep_recent_msgs=2)
        sys_msg = _system("sys")
        messages = [sys_msg, _tool("old"), _user("a"), _assistant("b")]
        result = engine.compress(messages)
        assert result.messages[0]["role"] == "system"

    def test_context_compressor_preserves_system_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CONTEXT_COMPRESSION_ENABLED", raising=False)
        compressor = ContextCompressor(min_tokens_to_trigger=1)
        sys_msg = _system("You are an expert assistant.")
        dup = _tool("dup")
        messages = [sys_msg] + [dup] * 5 + [_user("tail")]
        result = compressor.maybe_compress(messages)
        assert any(m.get("role") == "system" for m in result)
        assert any(
            m.get("content") == "You are an expert assistant."
            for m in result
            if m.get("role") == "system"
        )
