"""FastAPI router for backend lifecycle status endpoints.

Exposes three endpoints consumed by the dashboard and external monitors:

    GET  /backend/status        — full lifecycle snapshot
    GET  /backend/pending       — list of pending request IDs
    POST /backend/reset-failure — clear a FAILED state back to IDLE

All endpoints read from / write to ``lifecycle_manager`` — the module-level
singleton that owns the state machine defined in this file.

States (mirrors the plan in docs/macm4-lifecycle-and-context-plan.md):
    IDLE, ACTIVE_FAST, ACTIVE_LONG_128K,
    SWITCHING_TO_TURBO_256K, ACTIVE_TURBO_256K,
    SWITCHING_TO_TURBO_512K, ACTIVE_TURBO_512K,
    FALLING_BACK_EXTERNAL, FAILED
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel


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
# Response models
# ---------------------------------------------------------------------------

class BackendStatusResponse(BaseModel):
    """Payload returned by GET /backend/status."""

    active_backend: str
    state: str
    switch_started_at: Optional[str]
    target_backend: Optional[str]
    pending_requests: int
    memory_pressure: str
    last_status_message: str
    turbo_enabled: bool
    last_switch_duration_ms: Optional[int] = None


class PendingResponse(BaseModel):
    """Payload returned by GET /backend/pending."""

    pending_request_ids: list[str]


class ResetFailureResponse(BaseModel):
    """Payload returned by POST /backend/reset-failure."""

    ok: bool
    previous_state: str
    current_state: str


# ---------------------------------------------------------------------------
# Lifecycle manager
# ---------------------------------------------------------------------------

class LifecycleManager:
    """Thread-safe state machine for the backend lifecycle.

    The manager is deliberately kept simple: it stores the current state
    plus the metadata the status endpoint needs.  The actual subprocess
    start/stop logic lives elsewhere; this class only tracks what happened.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: BackendState = BackendState.IDLE
        self._active_backend: str = "local-long-128k"
        self._target_backend: Optional[str] = None
        self._switch_started_at: Optional[str] = None
        self._switch_start_epoch: Optional[float] = None
        self._last_switch_duration_ms: Optional[int] = None
        self._memory_pressure: str = "low"
        self._last_status_message: str = ""
        self._turbo_enabled: bool = False
        # Pending request IDs are tracked here so the API can surface them.
        # In production these are also written to disk by PendingRequestStore;
        # the manager keeps an in-memory copy so /backend/pending is fast.
        self._pending_ids: list[str] = []

    # ------------------------------------------------------------------
    # State reads
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Return a dict matching ``BackendStatusResponse``."""
        with self._lock:
            return {
                "active_backend": self._active_backend,
                "state": self._state.value,
                "switch_started_at": self._switch_started_at,
                "target_backend": self._target_backend,
                "pending_requests": len(self._pending_ids),
                "memory_pressure": self._memory_pressure,
                "last_status_message": self._last_status_message,
                "turbo_enabled": self._turbo_enabled,
                "last_switch_duration_ms": self._last_switch_duration_ms,
            }

    def list_pending(self) -> list[str]:
        """Return the current list of pending request IDs."""
        with self._lock:
            return list(self._pending_ids)

    # ------------------------------------------------------------------
    # State writes
    # ------------------------------------------------------------------

    def set_state(self, state: BackendState) -> None:
        with self._lock:
            self._state = state

    def set_active_backend(self, name: str) -> None:
        with self._lock:
            self._active_backend = name

    def set_target_backend(self, name: Optional[str]) -> None:
        with self._lock:
            self._target_backend = name

    def begin_switch(self, target: str, message: str = "") -> None:
        """Record the start of a backend switch."""
        with self._lock:
            self._target_backend = target
            self._switch_started_at = time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            )
            self._switch_start_epoch = time.monotonic()
            if message:
                self._last_status_message = message

    def complete_switch(self, new_backend: str, message: str = "") -> None:
        """Record a completed backend switch."""
        with self._lock:
            if self._switch_start_epoch is not None:
                self._last_switch_duration_ms = int(
                    (time.monotonic() - self._switch_start_epoch) * 1000
                )
                self._switch_start_epoch = None
            self._active_backend = new_backend
            self._target_backend = None
            self._switch_started_at = None
            if message:
                self._last_status_message = message

    def set_memory_pressure(self, level: str) -> None:
        with self._lock:
            self._memory_pressure = level

    def set_turbo_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._turbo_enabled = enabled

    def set_last_status_message(self, message: str) -> None:
        with self._lock:
            self._last_status_message = message

    def add_pending(self, request_id: str) -> None:
        with self._lock:
            if request_id not in self._pending_ids:
                self._pending_ids.append(request_id)

    def remove_pending(self, request_id: str) -> None:
        with self._lock:
            try:
                self._pending_ids.remove(request_id)
            except ValueError:
                pass

    def reset_failure(self) -> str:
        """Clear a FAILED state back to IDLE.  Returns the previous state name."""
        with self._lock:
            previous = self._state.value
            if self._state == BackendState.FAILED:
                self._state = BackendState.IDLE
                self._target_backend = None
                self._switch_started_at = None
                self._last_status_message = "Failure cleared; reset to IDLE."
            return previous


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

lifecycle_manager = LifecycleManager()

# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/backend", tags=["lifecycle"])


@router.get("/status", response_model=BackendStatusResponse)
def get_backend_status() -> BackendStatusResponse:
    """Return the current lifecycle state snapshot."""
    data = lifecycle_manager.get_status()
    return BackendStatusResponse(**data)


@router.get("/pending", response_model=PendingResponse)
def get_pending_requests() -> PendingResponse:
    """Return the list of pending request IDs."""
    return PendingResponse(pending_request_ids=lifecycle_manager.list_pending())


@router.post("/reset-failure", response_model=ResetFailureResponse)
def reset_failure() -> ResetFailureResponse:
    """Clear the FAILED state and return to IDLE.

    This is a no-op if the current state is not FAILED; the ``ok`` flag
    in the response reflects whether the state was actually changed.
    """
    previous = lifecycle_manager.reset_failure()
    current = lifecycle_manager.get_status()["state"]
    return ResetFailureResponse(
        ok=(previous == BackendState.FAILED.value),
        previous_state=previous,
        current_state=current,
    )
