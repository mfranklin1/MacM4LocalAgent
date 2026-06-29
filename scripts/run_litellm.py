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

# Serve backend status endpoints on a dedicated sidecar port (4010).
# LiteLLM instruments and freezes its ASGI stack at import time, so neither
# include_router(), add_api_route(), nor add_middleware() survive. Running a
# second lightweight uvicorn server in a daemon thread sidesteps all of that.
try:
    import threading as _threading
    import uvicorn as _uvicorn
    from fastapi import FastAPI as _FastAPI
    from router.status_api import lifecycle_manager as _lm
    from starlette.responses import JSONResponse as _JSONResponse

    _status_app = _FastAPI(title="MacM4 Backend Status API", docs_url=None, redoc_url=None)

    @_status_app.get("/backend/status")
    def _status():
        return _lm.get_status()

    @_status_app.get("/backend/pending")
    def _pending():
        return {"pending_request_ids": _lm.list_pending()}

    @_status_app.post("/backend/reset-failure")
    def _reset():
        previous = _lm.reset_failure()
        return {"ok": previous == "FAILED", "previous_state": previous,
                "current_state": _lm.get_status()["state"]}

    def _run_status_server():
        _uvicorn.run(_status_app, host="127.0.0.1", port=4010, log_level="warning")

    _t = _threading.Thread(target=_run_status_server, daemon=True, name="status-api")
    _t.start()
    print("[run_litellm] backend status API sidecar started on :4010", file=sys.stderr)
except Exception as _e:
    print(f"[run_litellm] could not start backend status sidecar: {_e}", file=sys.stderr)

if __name__ == "__main__":
    # Pass argv explicitly through Click's main() to defeat a regression in
    # newer click/uvicorn pairings where `run_server()`'s implicit __call__
    # resolves --port to an ephemeral instead of the value on the command
    # line. Without this, launchd would bring the proxy up on a random
    # high port and `make status` would report it DOWN. See:
    # https://click.palletsprojects.com/en/stable/api/#click.BaseCommand.main
    sys.exit(run_server.main(args=sys.argv[1:], prog_name="run_litellm"))
