"""Cheap heuristic classifier for whether a request looks 'complex' enough
to bypass the local tiers and go straight to Claude.

Triggers on multi-file debugging, architecture/design language, deep reasoning
chains, and explicit user opt-in via the [claude] tag. Returns a tuple
(is_complex, reason).
"""

from __future__ import annotations

import re

_REFACTOR_SCOPE = re.compile(
    # "refactor" immediately followed by a scope word (entire codebase, whole service, etc.)
    r"\brefactor\s+(?:entire|whole|all|everything|codebase|repo|subsystem|module|service|layer|stack)"
    # OR "refactor [the/this/our/your] [adj?] <named-system-component>"
    r"|\brefactor(?:\s+\w+){0,3}\s+(?:subsystem|service|layer|stack|codebase|repo)",
    re.IGNORECASE,
)

ARCH_PATTERNS = re.compile(
    r"\b(design|architect|architecture|system\s+design|trade-?offs?|"
    r"migrate|migration|migration\s+plan|rewrite|restructure|root\s*cause)\b",
    re.IGNORECASE,
)

MULTIFILE_PATTERNS = re.compile(
    r"(across\s+(?:multiple|several|many)\s+files|"
    r"(?:more\s+than\s+\d+|several|multiple)\s+files|"
    r"whole\s+repo|entire\s+codebase|monorepo)",
    re.IGNORECASE,
)

REASONING_PATTERNS = re.compile(
    r"\b(prove|theorem|formal\s+proof|why\s+does|step\s*[- ]*by\s*[- ]*step\s+reasoning|"
    r"think\s+(?:deeply|carefully|step\s+by\s+step)|chain\s+of\s+thought)\b",
    re.IGNORECASE,
)

EXPLICIT_TAG = re.compile(r"\[(?:claude|big|cloud|remote)\]", re.IGNORECASE)
LOCAL_TAG    = re.compile(r"\[(?:local|fast|cheap)\]",        re.IGNORECASE)


def classify(prompt: str) -> tuple[bool, str]:
    """Return (is_complex, reason). Empty reason means 'not complex'."""
    if not prompt:
        return (False, "")

    if LOCAL_TAG.search(prompt):
        return (False, "explicit [local] tag")

    if EXPLICIT_TAG.search(prompt):
        return (True, "explicit [claude] tag")

    if _REFACTOR_SCOPE.search(prompt) or ARCH_PATTERNS.search(prompt):
        return (True, "architecture/design language")

    if MULTIFILE_PATTERNS.search(prompt):
        return (True, "multi-file change")

    if REASONING_PATTERNS.search(prompt):
        return (True, "deep reasoning chain")

    return (False, "")
