#!/usr/bin/env python3
"""Replay a captured Cline conversation against multiple models.

Loads a captured `.logs/cline-dumps/req-*.json` dump, takes its
`messages` array as-is (system + task + assistant + tool-result + ...),
and asks each candidate model to generate the next assistant turn.
This lets us compare how different models handle the exact same
Cline-harness state without having to click through Cline's UI N
times.

Outputs a table: model | wall_time | output_tokens | snippet | tool_call_detected.

Usage:
    .venvs/litellm/bin/python scripts/replay_cline_turn.py \\
        --dump .logs/cline-dumps/req-1777660153436-004.json \\
        --models gpt-local-agent,gpt-local-long,gpt-claude-code

Set --truncate-after-msg to cut the conversation short (e.g. replay
just turn 2 by truncating after msg index 3, the first tool result).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
from typing import Any

import httpx

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _detect_tool_call(text: str) -> tuple[str | None, str | None]:
    """Return (tool_name, snippet_of_args) if the assistant text
    contains a Cline-style XML tool use, else (None, None)."""
    m = re.search(
        r"<(read_file|write_to_file|replace_in_file|list_files|"
        r"search_files|execute_command|attempt_completion|"
        r"ask_followup_question|new_task|use_mcp_tool|access_mcp_resource|"
        r"plan_mode_respond|load_mcp_documentation)\b",
        text,
    )
    if not m:
        return None, None
    name = m.group(1)
    snippet = text[m.start() : m.start() + 200]
    return name, snippet


def _projected_messages(messages: list[dict[str, Any]], cut_after: int | None) -> list[dict[str, Any]]:
    """Cline's tool-result messages have content as a list of {type:text,text:...}
    parts. The OpenAI API accepts that, but for cleaner replay we flatten
    each list-content into a single string so all backends behave the same.
    """
    out = []
    for m in messages[: (cut_after + 1 if cut_after is not None else None)]:
        c = m.get("content")
        if isinstance(c, list):
            joined = "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
            m = dict(m)
            m["content"] = joined
        out.append(m)
    return out


def _replay_one(
    model: str,
    messages: list[dict[str, Any]],
    base_url: str,
    timeout: float,
    stop: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        # Mirror Cline's defaults: no tools field (it uses XML in system),
        # no max_tokens (let the over-gen control clamp it), low temp for repeatability.
        "temperature": 0.2,
    }
    if stop:
        payload["stop"] = stop
    headers = {
        "Content-Type": "application/json",
    }
    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout) as cx:
            r = cx.post(f"{base_url}/v1/chat/completions", json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return {
            "model": model,
            "wall_seconds": time.time() - t0,
            "error": f"{type(e).__name__}: {e}",
            "content": "",
            "tool_call": None,
        }
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    usage = data.get("usage", {}) or {}
    tool_name, snippet = _detect_tool_call(content)
    return {
        "model": model,
        "wall_seconds": round(time.time() - t0, 2),
        "model_echoed": data.get("model"),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "tool_call": tool_name,
        "tool_snippet": snippet,
        "content_chars": len(content),
        "content_head": content[:400],
        "content_tail": content[-200:] if len(content) > 600 else "",
        "content_full": content,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", required=True, help="path to a captured .logs/cline-dumps/req-*.json")
    ap.add_argument(
        "--models",
        default="gpt-local-agent,gpt-local-long,gpt-claude-code",
        help="comma-separated model aliases to replay against",
    )
    ap.add_argument(
        "--cut-after-msg",
        type=int,
        default=None,
        help="truncate the conversation after this message index (0=just system, 1=system+task, 3=after first tool result)",
    )
    ap.add_argument("--base-url", default="http://127.0.0.1:4000")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument(
        "--stop",
        default=None,
        help=(
            "comma-separated list of stop strings to pass to the model. "
            "Useful for verifying that adding `stop=['</read_file>', "
            "'</replace_in_file>', '</attempt_completion>', ...]` makes "
            "the model emit exactly one tool tag per response. "
            "Pass the literal string 'cline' as a shortcut for the full "
            "Cline tool-close-tag list."
        ),
    )
    args = ap.parse_args()

    stop_list: list[str] | None = None
    if args.stop:
        if args.stop.strip().lower() == "cline":
            stop_list = [
                "</read_file>",
                "</write_to_file>",
                "</replace_in_file>",
                "</list_files>",
                "</search_files>",
                "</execute_command>",
                "</attempt_completion>",
                "</ask_followup_question>",
                "</new_task>",
                "</use_mcp_tool>",
                "</access_mcp_resource>",
                "</plan_mode_respond>",
                "</load_mcp_documentation>",
            ]
        else:
            stop_list = [s for s in args.stop.split(",") if s]

    dump_path = pathlib.Path(args.dump)
    if not dump_path.is_absolute():
        dump_path = REPO_ROOT / dump_path
    d = json.loads(dump_path.read_text())
    messages = _projected_messages(d.get("messages", []), args.cut_after_msg)

    print(f"  dump:        {dump_path.name}")
    print(f"  base_url:    {args.base_url}")
    print(f"  messages:    {len(messages)}")
    if messages:
        last = messages[-1]
        last_role = last.get("role")
        last_len = len(last.get("content", "") or "")
        print(f"  last msg:    role={last_role}  len={last_len}")
    print()

    if stop_list:
        print(f"  stop:        {len(stop_list)} sequence(s) passed (e.g. {stop_list[0]!r})")
    rows: list[dict[str, Any]] = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"  >>> replaying with model={model} ...", flush=True)
        row = _replay_one(model, messages, args.base_url, args.timeout, stop=stop_list)
        rows.append(row)
        if "error" in row:
            print(f"      ERROR after {row['wall_seconds']:.2f}s: {row['error']}")
        else:
            tc = row["tool_call"] or "(none — likely Cline error trigger)"
            print(
                f"      done in {row['wall_seconds']}s  "
                f"in_tok={row['input_tokens']}  out_tok={row['output_tokens']}  "
                f"tool_call={tc}"
            )

    print()
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(
        f"  {'model':<20} {'wall_s':>7} {'in_tok':>7} {'out_tok':>7} {'tool_call':<22} {'content_chars':>13}"
    )
    for r in rows:
        if "error" in r:
            print(f"  {r['model']:<20} {r['wall_seconds']:>7.2f}  ERROR: {r['error'][:60]}")
            continue
        print(
            f"  {r['model']:<20} {r['wall_seconds']:>7.2f} "
            f"{r['input_tokens'] or 0:>7} {r['output_tokens'] or 0:>7} "
            f"{(r['tool_call'] or '(none)'):<22} {r['content_chars']:>13}"
        )
    print()
    print("=" * 80)
    print("  RESPONSE PREVIEWS")
    print("=" * 80)
    for r in rows:
        if "error" in r:
            continue
        print(f"\n  --- {r['model']} ---")
        print("  " + r["content_head"].replace("\n", "\n  ")[:1200])
        if r["content_tail"]:
            print("  ... [truncated] ...")
            print("  " + r["content_tail"].replace("\n", "\n  "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
