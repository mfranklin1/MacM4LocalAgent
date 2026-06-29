"""Unit tests for router/backend_lifecycle.py.

Tests focus on the parts that do NOT require live processes:

  1. BackendState enum membership
  2. memory_pressure() parsing of macOS ``memory_pressure`` command output
  3. get_status() dict keys
  4. health_check() with mocked HTTP responses
  5. _backend_start_command() correct argv per backend kind
  6. switch_to() switch-lock prevents concurrent switches

Async tests use asyncio.run() to match the pattern used elsewhere in this
test suite (see test_router_active.py) rather than requiring the
pytest-asyncio plugin's asyncio_mode setting.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure the repo root is importable.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from router.backend_lifecycle import (
    BackendLifecycleManager,
    BackendState,
    _backend_start_command,
    health_check,
    lifecycle_manager,
    memory_pressure,
)


# ---------------------------------------------------------------------------
# 1. BackendState enum
# ---------------------------------------------------------------------------

class TestBackendStateEnum:
    """BackendState must carry all states documented in the lifecycle plan."""

    REQUIRED_STATES = {
        "IDLE",
        "ACTIVE_FAST",
        "ACTIVE_LONG_128K",
        "SWITCHING_TO_TURBO_256K",
        "ACTIVE_TURBO_256K",
        "SWITCHING_TO_TURBO_512K",
        "ACTIVE_TURBO_512K",
        "FALLING_BACK_EXTERNAL",
        "FAILED",
    }

    def test_all_required_states_present(self) -> None:
        member_names = {m.name for m in BackendState}
        missing = self.REQUIRED_STATES - member_names
        assert not missing, f"BackendState is missing states: {missing}"

    def test_idle_value(self) -> None:
        assert BackendState.IDLE.value == "IDLE"

    def test_active_fast_value(self) -> None:
        assert BackendState.ACTIVE_FAST.value == "ACTIVE_FAST"

    def test_active_long_128k_value(self) -> None:
        assert BackendState.ACTIVE_LONG_128K.value == "ACTIVE_LONG_128K"

    def test_switching_to_turbo_256k_value(self) -> None:
        assert BackendState.SWITCHING_TO_TURBO_256K.value == "SWITCHING_TO_TURBO_256K"

    def test_active_turbo_256k_value(self) -> None:
        assert BackendState.ACTIVE_TURBO_256K.value == "ACTIVE_TURBO_256K"

    def test_switching_to_turbo_512k_value(self) -> None:
        assert BackendState.SWITCHING_TO_TURBO_512K.value == "SWITCHING_TO_TURBO_512K"

    def test_active_turbo_512k_value(self) -> None:
        assert BackendState.ACTIVE_TURBO_512K.value == "ACTIVE_TURBO_512K"

    def test_falling_back_external_value(self) -> None:
        assert BackendState.FALLING_BACK_EXTERNAL.value == "FALLING_BACK_EXTERNAL"

    def test_failed_value(self) -> None:
        assert BackendState.FAILED.value == "FAILED"

    def test_state_count(self) -> None:
        assert len(BackendState) == len(self.REQUIRED_STATES)

    def test_states_are_str_enum(self) -> None:
        assert isinstance(BackendState.IDLE, str)

    def test_lookup_by_value(self) -> None:
        assert BackendState("ACTIVE_TURBO_256K") is BackendState.ACTIVE_TURBO_256K


# ---------------------------------------------------------------------------
# 2. memory_pressure()
# ---------------------------------------------------------------------------

class TestMemoryPressure:
    """memory_pressure() must parse the macOS ``memory_pressure`` output."""

    def _run_with_stdout(self, text: str) -> str:
        """Patch subprocess.run to return ``text`` on stdout, then call memory_pressure()."""
        mock_result = MagicMock()
        mock_result.stdout = text
        mock_result.stderr = ""
        with patch(
            "router.backend_lifecycle.subprocess.run",
            return_value=mock_result,
        ):
            return memory_pressure()

    def test_normal_returns_low(self) -> None:
        result = self._run_with_stdout("System Memory Pressure: NORMAL\n")
        assert result == "low"

    def test_warn_returns_medium(self) -> None:
        result = self._run_with_stdout("System Memory Pressure: WARN\n")
        assert result == "medium"

    def test_critical_returns_high(self) -> None:
        result = self._run_with_stdout("System Memory Pressure: CRITICAL\n")
        assert result == "high"

    def test_critical_takes_priority_over_normal(self) -> None:
        # If both appear (shouldn't happen in practice but guard against it)
        result = self._run_with_stdout(
            "System Memory Pressure: NORMAL\nSystem Memory Pressure: CRITICAL\n"
        )
        assert result == "high"

    def test_warn_takes_priority_over_normal(self) -> None:
        result = self._run_with_stdout(
            "System Memory Pressure: NORMAL\nSystem Memory Pressure: WARN\n"
        )
        # CRITICAL check runs first, then WARN. Both absent → WARN wins.
        assert result == "medium"

    def test_unknown_output_returns_unknown(self) -> None:
        result = self._run_with_stdout("something unexpected\n")
        assert result == "unknown"

    def test_empty_output_returns_unknown(self) -> None:
        result = self._run_with_stdout("")
        assert result == "unknown"

    def test_subprocess_exception_returns_unknown(self) -> None:
        with patch(
            "router.backend_lifecycle.subprocess.run",
            side_effect=FileNotFoundError("memory_pressure not found"),
        ):
            result = memory_pressure()
        assert result == "unknown"

    def test_timeout_exception_returns_unknown(self) -> None:
        with patch(
            "router.backend_lifecycle.subprocess.run",
            side_effect=subprocess_timeout(),
        ):
            result = memory_pressure()
        assert result == "unknown"

    def test_case_insensitive_detection(self) -> None:
        # Output is uppercased internally; test lowercase original string
        result = self._run_with_stdout("System Memory Pressure: normal\n")
        assert result == "low"

    def test_stderr_also_checked(self) -> None:
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "System Memory Pressure: WARN\n"
        with patch(
            "router.backend_lifecycle.subprocess.run",
            return_value=mock_result,
        ):
            result = memory_pressure()
        assert result == "medium"


def subprocess_timeout():
    """Helper: return a subprocess.TimeoutExpired instance for test use."""
    import subprocess
    return subprocess.TimeoutExpired(cmd=["memory_pressure"], timeout=5)


# ---------------------------------------------------------------------------
# 3. get_status() keys
# ---------------------------------------------------------------------------

class TestGetStatus:
    """LifecycleManager.get_status() must return a dict with all required keys."""

    REQUIRED_KEYS = {
        "active_backend",
        "state",
        "switch_started_at",
        "target_backend",
        "pending_requests",
        "memory_pressure",
        "last_status_message",
        "turbo_enabled",
    }

    def test_all_required_keys_present(self) -> None:
        manager = BackendLifecycleManager()
        snapshot = manager.get_status()
        missing = self.REQUIRED_KEYS - set(snapshot.keys())
        assert not missing, f"get_status() missing keys: {missing}"

    def test_state_is_string(self) -> None:
        manager = BackendLifecycleManager()
        snapshot = manager.get_status()
        assert isinstance(snapshot["state"], str)

    def test_default_state_is_idle(self) -> None:
        manager = BackendLifecycleManager()
        assert manager.get_status()["state"] == "IDLE"

    def test_default_active_backend(self) -> None:
        manager = BackendLifecycleManager()
        assert manager.get_status()["active_backend"] == "local-long-128k"

    def test_default_pending_requests_is_zero(self) -> None:
        manager = BackendLifecycleManager()
        assert manager.get_status()["pending_requests"] == 0

    def test_default_turbo_enabled_is_false(self) -> None:
        manager = BackendLifecycleManager()
        assert manager.get_status()["turbo_enabled"] is False

    def test_default_switch_started_at_is_none(self) -> None:
        manager = BackendLifecycleManager()
        assert manager.get_status()["switch_started_at"] is None

    def test_default_target_backend_is_none(self) -> None:
        manager = BackendLifecycleManager()
        assert manager.get_status()["target_backend"] is None

    def test_pending_requests_reflects_added_ids(self) -> None:
        manager = BackendLifecycleManager()
        manager.add_pending("req-1")
        manager.add_pending("req-2")
        assert manager.get_status()["pending_requests"] == 2

    def test_memory_pressure_key_is_string(self) -> None:
        manager = BackendLifecycleManager()
        assert isinstance(manager.get_status()["memory_pressure"], str)

    def test_set_turbo_enabled_reflected_in_status(self) -> None:
        manager = BackendLifecycleManager()
        manager.set_turbo_enabled(True)
        assert manager.get_status()["turbo_enabled"] is True

    def test_set_state_reflected_in_status(self) -> None:
        manager = BackendLifecycleManager()
        manager.set_state(BackendState.ACTIVE_TURBO_256K)
        assert manager.get_status()["state"] == "ACTIVE_TURBO_256K"

    def test_begin_switch_sets_target_and_timestamp(self) -> None:
        manager = BackendLifecycleManager()
        manager.begin_switch("local-turbo-256k", message="switching now")
        snapshot = manager.get_status()
        assert snapshot["target_backend"] == "local-turbo-256k"
        assert snapshot["switch_started_at"] is not None
        assert snapshot["last_status_message"] == "switching now"

    def test_complete_switch_clears_target_and_timestamp(self) -> None:
        manager = BackendLifecycleManager()
        manager.begin_switch("local-turbo-256k")
        manager.complete_switch("local-turbo-256k")
        snapshot = manager.get_status()
        assert snapshot["target_backend"] is None
        assert snapshot["switch_started_at"] is None
        assert snapshot["active_backend"] == "local-turbo-256k"


# ---------------------------------------------------------------------------
# 4. health_check()
# ---------------------------------------------------------------------------

class TestHealthCheck:
    """health_check() with mocked HTTP responses.

    All async coroutines are driven via asyncio.run() to avoid requiring a
    specific pytest-asyncio asyncio_mode configuration.
    """

    def _make_mock_aiohttp_200(self) -> MagicMock:
        """Return a mock aiohttp module that simulates a 200 response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.TCPConnector = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        return mock_aiohttp

    def _make_mock_aiohttp_error(self, exc: Exception) -> MagicMock:
        """Return a mock aiohttp module that raises ``exc`` on session enter."""
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(side_effect=exc)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.TCPConnector = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        return mock_aiohttp

    def _make_mock_aiohttp_status(self, status: int) -> MagicMock:
        """Return a mock aiohttp module that responds with ``status``."""
        mock_response = AsyncMock()
        mock_response.status = status
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_aiohttp = MagicMock()
        mock_aiohttp.ClientSession = MagicMock(return_value=mock_session)
        mock_aiohttp.TCPConnector = MagicMock(return_value=MagicMock())
        mock_aiohttp.ClientTimeout = MagicMock(return_value=MagicMock())
        return mock_aiohttp

    def test_200_response_returns_true(self) -> None:
        mock_aiohttp = self._make_mock_aiohttp_200()

        async def _run() -> bool:
            with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
                return await health_check("http://127.0.0.1:11434/api/tags")

        result = asyncio.run(_run())
        assert result is True

    def test_connection_error_returns_false(self) -> None:
        mock_aiohttp = self._make_mock_aiohttp_error(OSError("connection refused"))

        async def _run() -> bool:
            with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
                return await health_check("http://127.0.0.1:11434/api/tags")

        result = asyncio.run(_run())
        assert result is False

    def test_404_response_returns_false(self) -> None:
        mock_aiohttp = self._make_mock_aiohttp_status(404)

        async def _run() -> bool:
            with patch.dict("sys.modules", {"aiohttp": mock_aiohttp}):
                return await health_check("http://127.0.0.1:11434/api/tags")

        result = asyncio.run(_run())
        assert result is False

    def test_stdlib_fallback_200_returns_true(self) -> None:
        """When aiohttp is not available, falls back to urllib."""
        mock_req = MagicMock()
        mock_req.status = 200

        async def _run() -> bool:
            with patch.dict("sys.modules", {"aiohttp": None}):
                with patch("urllib.request.urlopen", return_value=mock_req):
                    return await health_check("http://127.0.0.1:11434/api/tags")

        result = asyncio.run(_run())
        assert result is True

    def test_stdlib_fallback_connection_error_returns_false(self) -> None:
        """Stdlib path: OSError -> returns False."""
        async def _run() -> bool:
            with patch.dict("sys.modules", {"aiohttp": None}):
                with patch(
                    "urllib.request.urlopen",
                    side_effect=OSError("connection refused"),
                ):
                    return await health_check("http://127.0.0.1:11434/api/tags")

        result = asyncio.run(_run())
        assert result is False


# ---------------------------------------------------------------------------
# 5. _backend_start_command()
# ---------------------------------------------------------------------------

class TestBackendStartCommand:
    """_backend_start_command() must return correct argv for each backend kind."""

    def test_mlx_kind_returns_list(self) -> None:
        cmd = _backend_start_command(
            "mlx", port=8080, model_path="/models/fast", max_context=16384
        )
        assert isinstance(cmd, list)
        assert len(cmd) > 0

    def test_mlx_kind_includes_port(self) -> None:
        cmd = _backend_start_command(
            "mlx", port=8080, model_path="/models/fast", max_context=16384
        )
        assert "--port" in cmd
        assert "8080" in cmd

    def test_mlx_kind_includes_model_path(self) -> None:
        cmd = _backend_start_command(
            "mlx", port=8080, model_path="/models/fast", max_context=16384
        )
        assert "/models/fast" in cmd

    def test_mlx_kind_includes_max_context(self) -> None:
        cmd = _backend_start_command(
            "mlx", port=8080, model_path="/models/fast", max_context=16384
        )
        assert "16384" in cmd

    def test_mlx_kind_uses_mlx_lm_server(self) -> None:
        cmd = _backend_start_command("mlx", port=8080, model_path="/m", max_context=1024)
        cmd_str = " ".join(cmd)
        assert "mlx_lm.server" in cmd_str or "mlx-lm" in cmd_str

    def test_ollama_kind_returns_list(self) -> None:
        cmd = _backend_start_command("ollama")
        assert isinstance(cmd, list)
        assert len(cmd) > 0

    def test_ollama_kind_invokes_ollama(self) -> None:
        cmd = _backend_start_command("ollama")
        assert "ollama" in cmd

    def test_ollama_kind_includes_serve(self) -> None:
        cmd = _backend_start_command("ollama")
        assert "serve" in cmd

    def test_mlx_turbo_kind_returns_list(self) -> None:
        cmd = _backend_start_command(
            "mlx-turbo",
            port=8084,
            model_path="/models/turbo",
            max_context=262144,
            kv_bits=3,
            fp16_layers=2,
        )
        assert isinstance(cmd, list)
        assert len(cmd) > 0

    def test_mlx_turbo_kind_includes_port(self) -> None:
        cmd = _backend_start_command(
            "mlx-turbo", port=8084, model_path="/m", max_context=262144
        )
        assert "--port" in cmd
        assert "8084" in cmd

    def test_mlx_turbo_kind_includes_kv_bits(self) -> None:
        cmd = _backend_start_command(
            "mlx-turbo", port=8084, model_path="/m", max_context=262144, kv_bits=3
        )
        assert "--kv-bits" in cmd
        assert "3" in cmd

    def test_mlx_turbo_kind_includes_fp16_layers(self) -> None:
        cmd = _backend_start_command(
            "mlx-turbo", port=8084, model_path="/m", max_context=262144, fp16_layers=4
        )
        assert "--fp16-layers" in cmd
        assert "4" in cmd

    def test_mlx_turbo_kind_includes_max_context(self) -> None:
        cmd = _backend_start_command(
            "mlx-turbo", port=8084, model_path="/m", max_context=524288
        )
        assert "524288" in cmd

    def test_unknown_kind_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend kind"):
            _backend_start_command("unknown-kind")

    def test_mlx_model_fallback_to_model_arg(self) -> None:
        # When model_path is empty, model= should be used
        cmd = _backend_start_command("mlx", model="my-model-name")
        assert "my-model-name" in cmd

    def test_mlx_turbo_model_fallback_to_model_arg(self) -> None:
        cmd = _backend_start_command("mlx-turbo", model="turbo-model-name")
        assert "turbo-model-name" in cmd

    def test_mlx_commands_are_strings(self) -> None:
        cmd = _backend_start_command("mlx", port=8080, model_path="/m", max_context=100)
        for item in cmd:
            assert isinstance(item, str), f"Command item {item!r} is not a string"

    def test_ollama_commands_are_strings(self) -> None:
        cmd = _backend_start_command("ollama")
        for item in cmd:
            assert isinstance(item, str)

    def test_mlx_turbo_commands_are_strings(self) -> None:
        cmd = _backend_start_command("mlx-turbo", port=8085, model_path="/m", max_context=100)
        for item in cmd:
            assert isinstance(item, str)


# ---------------------------------------------------------------------------
# 6. switch_to() — lock prevents concurrent switches
# ---------------------------------------------------------------------------

class TestSwitchToLock:
    """switch_to() must hold the lock so two concurrent callers serialize.

    All async coroutines are driven via asyncio.run() to match the project
    convention established in test_router_active.py.
    """

    def test_switch_to_returns_true_when_backend_changes(self) -> None:
        manager = BackendLifecycleManager()
        manager._active_backend = "local-long-128k"
        result = asyncio.run(manager.switch_to("local-turbo-256k"))
        assert result is True

    def test_switch_to_returns_false_when_already_active(self) -> None:
        manager = BackendLifecycleManager()
        manager._active_backend = "local-turbo-256k"
        result = asyncio.run(manager.switch_to("local-turbo-256k"))
        assert result is False

    def test_switch_to_updates_active_backend(self) -> None:
        manager = BackendLifecycleManager()
        asyncio.run(manager.switch_to("local-turbo-256k"))
        assert manager.get_status()["active_backend"] == "local-turbo-256k"

    def test_switch_to_256k_sets_active_turbo_256k_state(self) -> None:
        manager = BackendLifecycleManager()
        asyncio.run(manager.switch_to("local-turbo-256k"))
        assert manager.get_status()["state"] == "ACTIVE_TURBO_256K"

    def test_switch_to_512k_sets_active_turbo_512k_state(self) -> None:
        manager = BackendLifecycleManager()
        asyncio.run(manager.switch_to("local-turbo-512k"))
        assert manager.get_status()["state"] == "ACTIVE_TURBO_512K"

    def test_switch_to_fast_sets_active_fast_state(self) -> None:
        manager = BackendLifecycleManager()
        manager._active_backend = "local-long-128k"
        asyncio.run(manager.switch_to("local-fast"))
        assert manager.get_status()["state"] == "ACTIVE_FAST"

    def test_switch_to_128k_sets_active_long_128k_state(self) -> None:
        manager = BackendLifecycleManager()
        manager._active_backend = "local-fast"
        asyncio.run(manager.switch_to("local-long-128k"))
        assert manager.get_status()["state"] == "ACTIVE_LONG_128K"

    def test_switch_lock_serializes_concurrent_calls(self) -> None:
        """Two concurrent switch_to calls must not overlap.

        asyncio.gather fires both coroutines in the same event loop.
        Because switch_to holds the lock for the duration, the second
        caller must block and then see the backend already active (False).
        """
        manager = BackendLifecycleManager()
        manager._active_backend = "local-long-128k"

        async def _run() -> list[bool]:
            return list(
                await asyncio.gather(
                    manager.switch_to("local-turbo-256k"),
                    manager.switch_to("local-turbo-256k"),
                )
            )

        results = asyncio.run(_run())
        # Exactly one should have actually switched (True), the other
        # found the target already active (False).
        assert sorted(results) == [False, True]

    def test_second_concurrent_switch_sees_updated_state(self) -> None:
        """After the first switch completes the second call returns False."""
        manager = BackendLifecycleManager()
        manager._active_backend = "local-long-128k"

        async def _run() -> list[bool]:
            return list(
                await asyncio.gather(
                    manager.switch_to("local-turbo-256k"),
                    manager.switch_to("local-turbo-256k"),
                )
            )

        results = asyncio.run(_run())
        assert True in results
        assert False in results

    def test_sequential_switches_work(self) -> None:
        manager = BackendLifecycleManager()
        manager._active_backend = "local-long-128k"

        async def _run() -> tuple[bool, bool]:
            r1 = await manager.switch_to("local-turbo-256k")
            r2 = await manager.switch_to("local-turbo-512k")
            return r1, r2

        r1, r2 = asyncio.run(_run())
        assert r1 is True
        assert r2 is True
        assert manager.get_status()["active_backend"] == "local-turbo-512k"

    def test_switch_to_clears_target_backend_on_completion(self) -> None:
        manager = BackendLifecycleManager()
        asyncio.run(manager.switch_to("local-turbo-256k"))
        assert manager.get_status()["target_backend"] is None

    def test_switch_to_custom_transition_state(self) -> None:
        manager = BackendLifecycleManager()
        asyncio.run(
            manager.switch_to(
                "local-turbo-256k",
                transition_state=BackendState.SWITCHING_TO_TURBO_256K,
            )
        )
        assert manager.get_status()["state"] == "ACTIVE_TURBO_256K"

    def test_switch_to_records_status_message(self) -> None:
        manager = BackendLifecycleManager()
        asyncio.run(
            manager.switch_to("local-turbo-256k", message="Switching to 256k tier.")
        )
        assert manager.get_status()["last_status_message"] == "Switching to 256k tier."


# ---------------------------------------------------------------------------
# 7. Module-level singleton
# ---------------------------------------------------------------------------

class TestModuleLevelSingleton:
    """lifecycle_manager should be a BackendLifecycleManager instance."""

    def test_lifecycle_manager_is_instance(self) -> None:
        assert isinstance(lifecycle_manager, BackendLifecycleManager)

    def test_lifecycle_manager_has_get_status(self) -> None:
        assert callable(lifecycle_manager.get_status)

    def test_lifecycle_manager_get_status_returns_dict(self) -> None:
        assert isinstance(lifecycle_manager.get_status(), dict)
