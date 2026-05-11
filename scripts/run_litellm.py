"""Tiny launcher that imports and runs the LiteLLM proxy.

Equivalent to the `litellm` console-script wrapper but invoked directly via
`python` so launchd / TCC don't choke on the venv wrapper's pyvenv.cfg
provenance check on macOS.

Also force-inserts the repo root onto sys.path so the YAML's
`router.route_by_size.SizeBasedRouter` callback can be imported even if
PYTHONPATH gets stripped by the proxy startup hooks.

Sources `config/detected.env` so any per-host pins defined there
(e.g. OLLAMA_TAG, MLX_REPO) are present in os.environ before LiteLLM
resolves `os.environ/...` references in the YAML config. The launchd
plist only injects PATH and PYTHONPATH; this is how everything else
gets in. Auth is intentionally disabled (the proxy is loopback-only),
so there is no master-key resolution to fail on.
"""
from __future__ import annotations

import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))


def _load_detected_env() -> None:
    """Best-effort load of config/detected.env into os.environ.

    Format is `KEY="value"` or `KEY=value`, one per line, with `#` comments
    and blank lines. Idempotent: existing env vars are NOT overwritten so a
    caller that explicitly sets something via the shell still wins.

    Silently ignores a missing detected.env -- a fresh checkout that hasn't
    run `make detect` yet should still be able to import this launcher
    (e.g. from a unit test that just wants to pull in router.route_by_size).
    """
    env_path = REPO_ROOT / "config" / "detected.env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        os.environ.setdefault(key, val)


_load_detected_env()

# Touch the import so any early ImportError surfaces here, not deep inside
# the LiteLLM startup sequence.
import router.route_by_size  # noqa: F401,E402

from litellm import run_server  # noqa: E402

if __name__ == "__main__":
    sys.exit(run_server())
