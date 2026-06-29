"""Unit tests for router/status_api.py.

Tests exercise the three FastAPI endpoints via TestClient with the
lifecycle_manager singleton replaced by a controlled mock/stub so no
real subprocess work or disk I/O is required.

Endpoint coverage:
    GET  /backend/status        — 200, required fields, turbo_enabled value
    GET  /backend/pending       — list of pending request IDs
    POST /backend/reset-failure — clears FAILED state, no-ops on other states

Response model coverage:
    BackendStatusResponse, PendingResponse, ResetFailureResponse
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import router.status_api as status_api_module
from router.status_api import (
    BackendState,
    BackendStatusResponse,
    LifecycleManager,
    PendingResponse,
    ResetFailureResponse,
    router as status_router,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app() -> FastAPI:
    """Return a minimal FastAPI app with the status router mounted."""
    app = FastAPI()
    app.include_router(status_router)
    return app


def _status_payload(
    *,
    active_backend: str = "local-long-128k",
    state: str = "IDLE",
    switch_started_at: str | None = None,
    target_backend: str | None = None,
    pending_requests: int = 0,
    memory_pressure: str = "low",
    last_status_message: str = "",
    turbo_enabled: bool = False,
) -> dict[str, Any]:
    """Return a dict matching what LifecycleManager.get_status() returns."""
    return {
        "active_backend": active_backend,
        "state": state,
        "switch_started_at": switch_started_at,
        "target_backend": target_backend,
        "pending_requests": pending_requests,
        "memory_pressure": memory_pressure,
        "last_status_message": last_status_message,
        "turbo_enabled": turbo_enabled,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_manager() -> MagicMock:
    """Return a MagicMock that impersonates a LifecycleManager instance.

    Each test can configure return_value / side_effect on specific methods
    to exercise different scenarios.  The mock is patched onto the module's
    ``lifecycle_manager`` singleton so that the router handler functions
    pick it up automatically.
    """
    manager = MagicMock(spec=LifecycleManager)
    # Sensible defaults — tests override as needed.
    manager.get_status.return_value = _status_payload()
    manager.list_pending.return_value = []
    manager.reset_failure.return_value = BackendState.IDLE.value
    return manager


@pytest.fixture
def client(mock_manager: MagicMock) -> TestClient:
    """TestClient with the lifecycle_manager singleton replaced by the mock."""
    app = _make_app()
    with patch.object(status_api_module, "lifecycle_manager", mock_manager):
        yield TestClient(app)


# ---------------------------------------------------------------------------
# 1. GET /backend/status returns 200 with required fields
# ---------------------------------------------------------------------------

class TestGetBackendStatus:
    """GET /backend/status — presence and shape of required response fields."""

    REQUIRED_FIELDS = {
        "active_backend",
        "state",
        "switch_started_at",
        "target_backend",
        "pending_requests",
        "memory_pressure",
        "last_status_message",
        "turbo_enabled",
    }

    def test_returns_200(self, client: TestClient, mock_manager: MagicMock) -> None:
        mock_manager.get_status.return_value = _status_payload()
        r = client.get("/backend/status")
        assert r.status_code == 200

    def test_response_is_json(self, client: TestClient) -> None:
        r = client.get("/backend/status")
        assert r.headers["content-type"].startswith("application/json")

    def test_all_required_fields_present(self, client: TestClient) -> None:
        r = client.get("/backend/status")
        body = r.json()
        missing = self.REQUIRED_FIELDS - set(body.keys())
        assert not missing, f"Missing fields in response: {missing}"

    def test_active_backend_field(self, client: TestClient, mock_manager: MagicMock) -> None:
        mock_manager.get_status.return_value = _status_payload(
            active_backend="local-long-128k"
        )
        body = client.get("/backend/status").json()
        assert body["active_backend"] == "local-long-128k"

    def test_state_field(self, client: TestClient, mock_manager: MagicMock) -> None:
        mock_manager.get_status.return_value = _status_payload(
            state="SWITCHING_TO_TURBO_256K"
        )
        body = client.get("/backend/status").json()
        assert body["state"] == "SWITCHING_TO_TURBO_256K"

    def test_switch_started_at_can_be_null(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.get_status.return_value = _status_payload(switch_started_at=None)
        body = client.get("/backend/status").json()
        assert body["switch_started_at"] is None

    def test_switch_started_at_propagated_when_set(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        ts = "2026-06-29T12:00:00Z"
        mock_manager.get_status.return_value = _status_payload(
            switch_started_at=ts,
            state="SWITCHING_TO_TURBO_256K",
            target_backend="local-turbo-256k",
        )
        body = client.get("/backend/status").json()
        assert body["switch_started_at"] == ts

    def test_target_backend_can_be_null(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.get_status.return_value = _status_payload(target_backend=None)
        body = client.get("/backend/status").json()
        assert body["target_backend"] is None

    def test_pending_requests_field(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.get_status.return_value = _status_payload(pending_requests=3)
        body = client.get("/backend/status").json()
        assert body["pending_requests"] == 3

    def test_memory_pressure_field(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.get_status.return_value = _status_payload(
            memory_pressure="high"
        )
        body = client.get("/backend/status").json()
        assert body["memory_pressure"] == "high"

    def test_last_status_message_field(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        msg = "Switching to 256k TurboQuant backend; context is preserved."
        mock_manager.get_status.return_value = _status_payload(
            last_status_message=msg
        )
        body = client.get("/backend/status").json()
        assert body["last_status_message"] == msg

    def test_get_status_is_called_once(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        client.get("/backend/status")
        mock_manager.get_status.assert_called_once()


# ---------------------------------------------------------------------------
# 2. GET /backend/status returns correct turbo_enabled value
# ---------------------------------------------------------------------------

class TestTurboEnabled:
    """turbo_enabled reflects the manager's reported value faithfully."""

    def test_turbo_enabled_false(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.get_status.return_value = _status_payload(turbo_enabled=False)
        body = client.get("/backend/status").json()
        assert body["turbo_enabled"] is False

    def test_turbo_enabled_true(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.get_status.return_value = _status_payload(turbo_enabled=True)
        body = client.get("/backend/status").json()
        assert body["turbo_enabled"] is True

    def test_turbo_enabled_is_boolean(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """The field must be a JSON boolean, not a stringified value."""
        mock_manager.get_status.return_value = _status_payload(turbo_enabled=True)
        body = client.get("/backend/status").json()
        assert isinstance(body["turbo_enabled"], bool)


# ---------------------------------------------------------------------------
# 3. GET /backend/pending returns list of pending request IDs
# ---------------------------------------------------------------------------

class TestGetPending:
    """GET /backend/pending — correct IDs, empty case, list type."""

    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/backend/pending")
        assert r.status_code == 200

    def test_response_has_pending_request_ids_key(self, client: TestClient) -> None:
        body = client.get("/backend/pending").json()
        assert "pending_request_ids" in body

    def test_empty_list_when_no_pending(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.list_pending.return_value = []
        body = client.get("/backend/pending").json()
        assert body["pending_request_ids"] == []

    def test_single_pending_id(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.list_pending.return_value = ["req-abc123"]
        body = client.get("/backend/pending").json()
        assert body["pending_request_ids"] == ["req-abc123"]

    def test_multiple_pending_ids_preserved_order(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        ids = ["req-1", "req-2", "req-3"]
        mock_manager.list_pending.return_value = ids
        body = client.get("/backend/pending").json()
        assert body["pending_request_ids"] == ids

    def test_pending_ids_are_strings(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.list_pending.return_value = ["req-xyz"]
        body = client.get("/backend/pending").json()
        for item in body["pending_request_ids"]:
            assert isinstance(item, str)

    def test_list_pending_called_once(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        client.get("/backend/pending")
        mock_manager.list_pending.assert_called_once()


# ---------------------------------------------------------------------------
# 4. POST /backend/reset-failure clears FAILED state
# ---------------------------------------------------------------------------

class TestResetFailure:
    """POST /backend/reset-failure — state transitions and no-op behavior."""

    def test_returns_200(self, client: TestClient, mock_manager: MagicMock) -> None:
        mock_manager.reset_failure.return_value = BackendState.FAILED.value
        mock_manager.get_status.return_value = _status_payload(state="IDLE")
        r = client.post("/backend/reset-failure")
        assert r.status_code == 200

    def test_ok_true_when_state_was_failed(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.reset_failure.return_value = BackendState.FAILED.value
        mock_manager.get_status.return_value = _status_payload(state="IDLE")
        body = client.post("/backend/reset-failure").json()
        assert body["ok"] is True

    def test_previous_state_is_failed(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.reset_failure.return_value = BackendState.FAILED.value
        mock_manager.get_status.return_value = _status_payload(state="IDLE")
        body = client.post("/backend/reset-failure").json()
        assert body["previous_state"] == "FAILED"

    def test_current_state_is_idle_after_reset(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.reset_failure.return_value = BackendState.FAILED.value
        mock_manager.get_status.return_value = _status_payload(state="IDLE")
        body = client.post("/backend/reset-failure").json()
        assert body["current_state"] == "IDLE"

    def test_ok_false_when_state_was_not_failed(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        """reset-failure is a no-op if the current state is not FAILED."""
        mock_manager.reset_failure.return_value = BackendState.IDLE.value
        mock_manager.get_status.return_value = _status_payload(state="IDLE")
        body = client.post("/backend/reset-failure").json()
        assert body["ok"] is False

    def test_no_op_previous_state_returned(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.reset_failure.return_value = "ACTIVE_LONG_128K"
        mock_manager.get_status.return_value = _status_payload(
            state="ACTIVE_LONG_128K"
        )
        body = client.post("/backend/reset-failure").json()
        assert body["previous_state"] == "ACTIVE_LONG_128K"
        assert body["current_state"] == "ACTIVE_LONG_128K"

    def test_reset_failure_called_once(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        client.post("/backend/reset-failure")
        mock_manager.reset_failure.assert_called_once()

    def test_response_has_all_required_fields(
        self, client: TestClient, mock_manager: MagicMock
    ) -> None:
        mock_manager.reset_failure.return_value = BackendState.FAILED.value
        mock_manager.get_status.return_value = _status_payload(state="IDLE")
        body = client.post("/backend/reset-failure").json()
        assert {"ok", "previous_state", "current_state"} <= set(body.keys())


# ---------------------------------------------------------------------------
# 5. Response model field validation (all required fields from the plan)
# ---------------------------------------------------------------------------

class TestResponseModels:
    """Verify that each response model has the required fields from the plan."""

    def test_backend_status_response_fields(self) -> None:
        """BackendStatusResponse must carry every field in the plan's schema."""
        expected = {
            "active_backend",
            "state",
            "switch_started_at",
            "target_backend",
            "pending_requests",
            "memory_pressure",
            "last_status_message",
            "turbo_enabled",
        }
        model_fields = set(BackendStatusResponse.model_fields.keys())
        missing = expected - model_fields
        assert not missing, (
            f"BackendStatusResponse is missing plan-required fields: {missing}"
        )

    def test_pending_response_fields(self) -> None:
        expected = {"pending_request_ids"}
        model_fields = set(PendingResponse.model_fields.keys())
        missing = expected - model_fields
        assert not missing, (
            f"PendingResponse is missing plan-required fields: {missing}"
        )

    def test_reset_failure_response_fields(self) -> None:
        expected = {"ok", "previous_state", "current_state"}
        model_fields = set(ResetFailureResponse.model_fields.keys())
        missing = expected - model_fields
        assert not missing, (
            f"ResetFailureResponse is missing plan-required fields: {missing}"
        )

    def test_backend_status_response_instantiation(self) -> None:
        """BackendStatusResponse must accept valid plan-shaped data."""
        obj = BackendStatusResponse(
            active_backend="local-turbo-256k",
            state="ACTIVE_TURBO_256K",
            switch_started_at=None,
            target_backend=None,
            pending_requests=0,
            memory_pressure="medium",
            last_status_message="Switched.",
            turbo_enabled=True,
        )
        assert obj.active_backend == "local-turbo-256k"
        assert obj.turbo_enabled is True

    def test_pending_response_instantiation(self) -> None:
        obj = PendingResponse(pending_request_ids=["r1", "r2"])
        assert obj.pending_request_ids == ["r1", "r2"]

    def test_reset_failure_response_instantiation(self) -> None:
        obj = ResetFailureResponse(ok=True, previous_state="FAILED", current_state="IDLE")
        assert obj.ok is True
        assert obj.previous_state == "FAILED"
        assert obj.current_state == "IDLE"

    def test_lifecycle_manager_get_status_returns_all_required_keys(self) -> None:
        """LifecycleManager.get_status() must return a dict with all plan keys."""
        manager = LifecycleManager()
        snapshot = manager.get_status()
        expected = {
            "active_backend",
            "state",
            "switch_started_at",
            "target_backend",
            "pending_requests",
            "memory_pressure",
            "last_status_message",
            "turbo_enabled",
        }
        missing = expected - set(snapshot.keys())
        assert not missing, f"get_status() missing keys: {missing}"

    def test_lifecycle_manager_turbo_enabled_default_false(self) -> None:
        manager = LifecycleManager()
        assert manager.get_status()["turbo_enabled"] is False

    def test_lifecycle_manager_set_turbo_enabled(self) -> None:
        manager = LifecycleManager()
        manager.set_turbo_enabled(True)
        assert manager.get_status()["turbo_enabled"] is True

    def test_lifecycle_manager_add_and_list_pending(self) -> None:
        manager = LifecycleManager()
        manager.add_pending("req-1")
        manager.add_pending("req-2")
        ids = manager.list_pending()
        assert "req-1" in ids
        assert "req-2" in ids

    def test_lifecycle_manager_reset_failure_clears_failed(self) -> None:
        manager = LifecycleManager()
        manager.set_state(BackendState.FAILED)
        previous = manager.reset_failure()
        assert previous == "FAILED"
        assert manager.get_status()["state"] == "IDLE"

    def test_lifecycle_manager_reset_failure_noop_on_non_failed(self) -> None:
        manager = LifecycleManager()
        manager.set_state(BackendState.ACTIVE_LONG_128K)
        previous = manager.reset_failure()
        assert previous == "ACTIVE_LONG_128K"
        assert manager.get_status()["state"] == "ACTIVE_LONG_128K"
