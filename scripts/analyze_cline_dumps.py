#!/usr/bin/env python3
"""Summarize the captured Cline request dumps.

Reads every JSON file in .logs/cline-dumps/ and prints a turn-by-turn
report: model, message-count, system prompt size, tool definitions,
each user/assistant message preview.

Usage:
    .venvs/litellm/bin/python scripts/analyze_cline_dumps.py
    .venvs/litellm/bin/python scripts/analyze_cline_dumps.py --full   # full content
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DUMP_DIR = REPO_ROOT / ".logs" / "cline-dumps"


def _preview(text: str, n: int = 200) -> str:
    if len(text) <= n:
        return text
    return text[:n] + f" ... [+{len(text) - n} chars]"


def _summarize(path: pathlib.Path, full: bool) -> None:
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        print(f"  ERROR reading {path.name}: {e}")
        return

    msgs = d.get("messages", []) or []
    tools = d.get("tools", []) or []

    print(f"\n=== {path.name} ===")
    print(f"  model:           {d.get('model')}")
    print(f"  messages:        {len(msgs)}")
    print(f"  tools defined:   {len(tools)}")
    print(f"  max_tokens:      {d.get('max_tokens')}")
    print(f"  stream:          {d.get('stream')}")
    print(f"  tool_choice:     {d.get('tool_choice')}")

    if tools:
        print(f"  tool names:      {[t.get('function', {}).get('name', '?') for t in tools[:20]]}")

    sys_msgs = [m for m in msgs if m.get("role") == "system"]
    if sys_msgs:
        sys_text = sys_msgs[0].get("content", "") or ""
        if isinstance(sys_text, list):
            sys_text = "\n".join(p.get("text", "") for p in sys_text if isinstance(p, dict))
        print(f"  system prompt:   {len(sys_text):,} chars (~{len(sys_text)//4:,} tokens)")
        if full:
            print("  --- system prompt content ---")
            print(sys_text)
            print("  --- end system prompt ---")
        else:
            print(f"  system preview:  {_preview(sys_text, 300)!r}")

    for i, m in enumerate(msgs):
        role = m.get("role", "?")
        if role == "system":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not isinstance(content, str):
            content = str(content)
        tc = m.get("tool_calls") or []
        if tc:
            tc_names = [c.get("function", {}).get("name", "?") for c in tc]
            print(f"  msg[{i}] {role}: tool_calls={tc_names}")
            for c in tc:
                args = c.get("function", {}).get("arguments", "")
                print(f"    -> {c['function']['name']}({_preview(args, 150)})")
        else:
            print(f"  msg[{i}] {role}: ({len(content):,} chars) {_preview(content, 200)!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="print full system prompt content")
    args = ap.parse_args()

    if not DUMP_DIR.exists():
        print(f"  no dump directory at {DUMP_DIR}", file=sys.stderr)
        return 1
    files = sorted(DUMP_DIR.glob("*.json"))
    if not files:
        print(f"  no dumps in {DUMP_DIR}", file=sys.stderr)
        return 1

    print(f"  found {len(files)} dump(s) in {DUMP_DIR}")
    for path in files:
        _summarize(path, args.full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
