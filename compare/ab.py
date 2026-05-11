"""Fan-out the same prompt to local-long and claude-code through the
LiteLLM proxy. Persist both outputs (with token counts, cost, latency, and
a simple judge score) into cost/cost.db so the dashboard can render them.

Usage:
  python3 compare/ab.py "Your prompt here"
  python3 compare/ab.py - < prompt.txt
"""

from __future__ import annotations

import os
import pathlib
import sys
import time
from typing import Any

import httpx

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from cost.ingest import record_comparison, shadow_cost  # noqa: E402


def _read_env(path: pathlib.Path) -> dict[str, str]:
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


_ENV = _read_env(REPO_ROOT / "config" / "detected.env")
LITELLM_BASE = f"http://127.0.0.1:{_ENV.get('LITELLM_PORT', '4000')}"


def _call(model: str, prompt: str, *, timeout: float = 600.0) -> dict[str, Any]:
    """Send prompt to LiteLLM and return a normalized result dict."""
    started = time.time()
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(
                f"{LITELLM_BASE}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
            )
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "model": model,
            "output": "",
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": int((time.time() - started) * 1000),
        }

    msg = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    return {
        "ok": True,
        "model": model,
        "output": msg,
        "input_tokens":  int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "latency_ms": int((time.time() - started) * 1000),
    }


def _judge(local_out: str, claude_out: str) -> float:
    """Lightweight judge score in [0, 1]. Cheaper proxy for quality:

      - 0.5 base
      - + up to 0.25 for similar length (within 30%)
      - + up to 0.25 for shared code-fence count + structure overlap
    """
    if not local_out or not claude_out:
        return 0.0
    base = 0.5

    a, b = len(local_out), len(claude_out)
    ratio = min(a, b) / max(a, b) if max(a, b) else 0.0
    base += 0.25 * ratio

    a_fences = local_out.count("```")
    b_fences = claude_out.count("```")
    fence_match = 1.0 - (abs(a_fences - b_fences) / max(1, a_fences + b_fences))
    base += 0.25 * fence_match

    return round(min(1.0, base), 3)


def run(prompt: str) -> dict[str, Any]:
    print(f"[compare] -> local-long  ({len(prompt)} chars)")
    local = _call("local-long", prompt)
    print(f"[compare] -> claude-code ({len(prompt)} chars)")
    claude = _call("claude-code", prompt)

    local_cost  = 0.0  # local has no actual $ cost
    claude_cost = shadow_cost(claude["input_tokens"], claude["output_tokens"]) if claude["ok"] else 0.0
    score = _judge(local.get("output", ""), claude.get("output", ""))

    rid = record_comparison({
        "prompt": prompt,
        "local_model":  local["model"],
        "claude_model": claude["model"],
        "local_output":  local.get("output", "")[:200_000],
        "claude_output": claude.get("output", "")[:200_000],
        "local_in_tok":   local.get("input_tokens", 0),
        "local_out_tok":  local.get("output_tokens", 0),
        "claude_in_tok":  claude.get("input_tokens", 0),
        "claude_out_tok": claude.get("output_tokens", 0),
        "local_cost":  local_cost,
        "claude_cost": claude_cost,
        "local_ms":    local.get("latency_ms", 0),
        "claude_ms":   claude.get("latency_ms", 0),
        "judge_score": score,
    })

    print()
    print(f"  local-long:  {local.get('latency_ms', 0):>5d} ms  "
          f"in={local.get('input_tokens', 0):>5}  out={local.get('output_tokens', 0):>5}  "
          f"cost=$0.0000")
    print(f"  claude-code: {claude.get('latency_ms', 0):>5d} ms  "
          f"in={claude.get('input_tokens', 0):>5}  out={claude.get('output_tokens', 0):>5}  "
          f"cost=${claude_cost:.4f}")
    print(f"  judge_score (length+structure overlap, 0-1): {score}")
    print(f"  saved to comparisons.id={rid}; view at http://127.0.0.1:4001/compare/{rid}")
    return {"id": rid, "local": local, "claude": claude, "score": score}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 1
    if argv[0] == "-":
        prompt = sys.stdin.read()
    else:
        prompt = argv[0]
    run(prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
