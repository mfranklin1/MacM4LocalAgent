"""Backend lifecycle manager for MacM4LocalAgent.

Handles dynamic loading/unloading of local model backends across context tiers
(128k -> 256k -> 512k) as described in docs/macm4-lifecycle-and-context-plan.md.

Key responsibilities:
  - memory_pressure(): probe macOS memory state before starting large backends
  - health_check(): async HTTP ping to verify a backend is responsive
  - _backend_start_command(): build the subprocess argv for each backend kind
  - BackendLifecycleManager.switch_to(): coordinate a backend switch with
    single-switch locking so concurrent requests coalesce correctly
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from enum import Enum
from typing import Any, Optional

# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class BackendState(str, Enum):
    IDLE = "IDLE"
    ACTIVE_FAST = "ACTIVE_FAST"
    ACTIVE_LONG_128K = "ACTIVE_LONG_128K"
    SWITCHING_TO_TURBO_256K = "SWITCHING_TO_TURBO_256K"
    ACTIVE_TURBO_256K = "ACTIVE_TURBO_256K"
    SWITCHING_TO_TURBO_512K = "SWITCHING_TO_TURBO_512K"
    ACTIVE_TURBO_512K = "ACTIVE_TURBO_512K"
    FALLING_BACK_EXTERNAL = "FALLING_BACK_EXTERNAL"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Memory pressure
# ---------------------------------------------------------------------------

def memory_pressure() -> str:
    """Return the macOS system memory pressure level as a string.

    Runs the ``memory_pressure`` CLI tool and parses its output.

    Returns:
        ``"low"`` when the system reports NORMAL,
        ``"medium"`` when WARN,
        ``"high"`` when CRITICAL.
        Falls back to ``"unknown"`` on any parse or subprocess error.
    """
    try:
        result = subprocess.run(
            ["memory_pressure"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = result.stdout + result.stderr
        # The macOS `memory_pressure` command prints a line like:
        #   System Memory Pressure: NORMAL
        # We scan for the canonical keywords in order of severity.
        output_upper = output.upper()
        if "CRITICAL" in output_upper:
            return "high"
        if "WARN" in output_upper:
            return "medium"
        if "NORMAL" in output_upper:
            return "low"
        return "unknown"
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def health_check(url: str, timeout: float = 5.0) -> bool:
    """Check whether a backend HTTP endpoint is responsive.

    Attempts to open ``url`` and returns ``True`` on a 2xx response.
    Returns ``False`` on connection error, timeout, or non-2xx status.

    Args:
        url: The URL to probe (e.g. ``"http://127.0.0.1:11434/api/tags"``).
        timeout: Socket connect + read timeout in seconds.

    Uses ``urllib`` from the standard library so the module stays importable
    without optional dependencies.  An ``aiohttp`` fast-path is attempted first
    when the package is available.
    """
    # Fast path: aiohttp
    try:
        import aiohttp  # type: ignore[import]

        connector = aiohttp.TCPConnector()
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(
            connector=connector, timeout=client_timeout
        ) as session:
            async with session.get(url) as resp:
                return resp.status < 300
    except ImportError:
        pass
    except Exception:
        return False

    # Fallback: asyncio + urllib (stdlib only)
    try:
        loop = asyncio.get_event_loop()
        import urllib.request
        import urllib.error

        def _blocking_get() -> bool:
            try:
                req = urllib.request.urlopen(url, timeout=timeout)
                return req.status < 300
            except Exception:
                return False

        return await loop.run_in_executor(None, _blocking_get)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backend start command
# ---------------------------------------------------------------------------

def _backend_start_command(
    kind: str,
    *,
    port: int = 11434,
    model: str = "",
    model_path: str = "",
    max_context: int = 131072,
    kv_bits: int = 3,
    fp16_layers: int = 2,
    extra: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Return the subprocess argv required to start a backend of the given kind.

    Args:
        kind: One of ``"mlx"``, ``"ollama"``, or ``"mlx-turbo"``.
        port: Port number the backend should listen on.
        model: Model tag / name (used by ollama).
        model_path: Filesystem path to the model (used by mlx / mlx-turbo).
        max_context: Context length to configure.
        kv_bits: KV-cache quantisation bits for mlx-turbo.
        fp16_layers: Number of FP16 layers for mlx-turbo.
        extra: Additional keyword args (ignored; reserved for future use).

    Returns:
        A list of strings suitable for passing to :func:`subprocess.Popen`.

    Raises:
        ValueError: When ``kind`` is not one of the known backend kinds.
    """
    if kind == "mlx":
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            model_path or model,
            "--port",
            str(port),
            "--max-tokens",
            str(max_context),
        ]
        return cmd

    if kind == "ollama":
        return [
            "ollama",
            "serve",
        ]

    if kind == "mlx-turbo":
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.server",
            "--model",
            model_path or model,
            "--port",
            str(port),
            "--max-tokens",
            str(max_context),
            "--kv-bits",
            str(kv_bits),
            "--fp16-layers",
            str(fp16_layers),
        ]
        return cmd

    raise ValueError(f"Unknown backend kind: {kind!r}")


# ---------------------------------------------------------------------------
# Lifecycle manager
# ---------------------------------------------------------------------------

class BackendLifecycleManager:
    """Coordinates single-backend-at-a-time lifecycle switching.

    The manager holds the current state plus a single async lock that prevents
    two concurrent callers from both attempting a backend switch.  It does NOT
    start or stop real processes by itself — callers are expected to implement
    the actual subprocess management and call the state-transition helpers
    (``begin_switch``, ``complete_switch``, etc.) on this instance.

    Designed to be used as a module-level singleton; ``asyncio.Lock`` is
    created lazily on first access so the object can be instantiated before
    an event loop is running (e.g. at import time in a test harness).
    """

    def __init__(self) -> None:
        self._state: BackendState = BackendState.IDLE
        self._active_backend: str = "local-long-128k"
        self._target_backend: Optional[str] = None
        self._switch_started_at: Optional[str] = None
        self._memory_pressure: str = "low"
        self._last_status_message: str = ""
        self._pending_ids: list[str] = []
        self._turbo_enabled: bool = False
        # The switch lock is created lazily so we don't require a running loop
        # at construction time.
        self.__lock: Optional[asyncio.Lock] = None

    @property
    def _lock(self) -> asyncio.Lock:
        if self.__lock is None:
            self.__lock = asyncio.Lock()
        return self.__lock

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a plain dict describing the current lifecycle state.

        The dict is serialisable and matches the ``BackendStatusResponse``
        pydantic model defined in ``router/status_api.py``.
        """
        return {
            "active_backend": self._active_backend,
            "state": self._state.value,
            "switch_started_at": self._switch_started_at,
            "target_backend": self._target_backend,
            "pending_requests": len(self._pending_ids),
            "memory_pressure": self._memory_pressure,
            "last_status_message": self._last_status_message,
            "turbo_enabled": self._turbo_enabled,
        }

    # ------------------------------------------------------------------
    # State writes
    # ------------------------------------------------------------------

    def set_state(self, state: BackendState) -> None:
        self._state = state

    def set_active_backend(self, name: str) -> None:
        self._active_backend = name

    def set_target_backend(self, name: Optional[str]) -> None:
        self._target_backend = name

    def begin_switch(self, target: str, message: str = "") -> None:
        self._target_backend = target
        self._switch_started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if message:
            self._last_status_message = message

    def complete_switch(self, new_backend: str, message: str = "") -> None:
        self._active_backend = new_backend
        self._target_backend = None
        self._switch_started_at = None
        if message:
            self._last_status_message = message

    def set_memory_pressure(self, level: str) -> None:
        self._memory_pressure = level

    def set_turbo_enabled(self, enabled: bool) -> None:
        self._turbo_enabled = enabled

    def add_pending(self, request_id: str) -> None:
        if request_id not in self._pending_ids:
            self._pending_ids.append(request_id)

    def remove_pending(self, request_id: str) -> None:
        try:
            self._pending_ids.remove(request_id)
        except ValueError:
            pass

    def reset_failure(self) -> str:
        """Clear FAILED -> IDLE.  Returns the previous state name."""
        previous = self._state.value
        if self._state == BackendState.FAILED:
            self._state = BackendState.IDLE
            self._target_backend = None
            self._switch_started_at = None
            self._last_status_message = "Failure cleared; reset to IDLE."
        return previous

    # ------------------------------------------------------------------
    # Switch coordination
    # ------------------------------------------------------------------

    async def switch_to(
        self,
        target_backend: str,
        *,
        transition_state: Optional[BackendState] = None,
        active_state: Optional[BackendState] = None,
        message: str = "",
    ) -> bool:
        """Coordinate a backend switch with a single-switch lock.

        Acquires the switch lock so that concurrent callers queue behind the
        first switch rather than triggering a second simultaneous switch.

        This method only performs the state-machine bookkeeping.  The caller
        is responsible for the actual subprocess work (stop old backend, start
        new one, wait for health).

        Args:
            target_backend: Registry name of the backend to switch to.
            transition_state: ``BackendState`` to set while switching is in
                progress.  Defaults to ``SWITCHING_TO_TURBO_256K`` if the
                target name contains "256", ``SWITCHING_TO_TURBO_512K`` if
                it contains "512", otherwise the current state is left as-is.
            active_state: ``BackendState`` to set once the switch is complete.
                Inferred from ``target_backend`` when ``None``.
            message: User-visible status message.

        Returns:
            ``True`` if the state was changed (i.e. the target differs from
            the current active backend), ``False`` if the backend was already
            active and no switch was needed.
        """
        async with self._lock:
            if self._active_backend == target_backend:
                return False

            # Determine transition state from target name when not supplied.
            if transition_state is None:
                if "256" in target_backend:
                    transition_state = BackendState.SWITCHING_TO_TURBO_256K
                elif "512" in target_backend:
                    transition_state = BackendState.SWITCHING_TO_TURBO_512K

            if transition_state is not None:
                self._state = transition_state

            self.begin_switch(target_backend, message=message)

            # Determine active state from target name when not supplied.
            if active_state is None:
                if "256" in target_backend:
                    active_state = BackendState.ACTIVE_TURBO_256K
                elif "512" in target_backend:
                    active_state = BackendState.ACTIVE_TURBO_512K
                elif "fast" in target_backend:
                    active_state = BackendState.ACTIVE_FAST
                elif "128" in target_backend:
                    active_state = BackendState.ACTIVE_LONG_128K

            if active_state is not None:
                self._state = active_state
            self.complete_switch(target_backend, message=message)
            return True


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

lifecycle_manager = BackendLifecycleManager()
