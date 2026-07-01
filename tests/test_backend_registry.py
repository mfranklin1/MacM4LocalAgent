"""Unit tests for router/backend_registry.py.

Tests load the real config/backend-registry.yaml (committed to the repo, not
mocked). TURBO_ENABLED is manipulated via monkeypatch so the env never
bleeds between tests. The `reg_turbo_on`/`reg_turbo_off` fixtures pin
env_path to a nonexistent file so max_context comes only from the static
YAML defaults, not the host's real hardware-detected config/detected.env
(which varies runner-to-runner and previously made routing-boundary tests
pass on a developer's Mac but fail on GitHub's CI runner).
"""

from __future__ import annotations

import os
import pathlib
import sys
import textwrap

import pytest

# Ensure the repo root is importable.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_registry():
    """Import (or re-import) the backend_registry module.

    We import inside tests rather than at module load time so that
    monkeypatching os.environ before the import takes effect — the module
    reads TURBO_ENABLED once at import time via the singleton initialiser.
    For tests that need a fresh singleton we reimport the module.
    """
    import importlib
    if "router.backend_registry" in sys.modules:
        return importlib.reload(sys.modules["router.backend_registry"])
    return __import__("router.backend_registry", fromlist=["*"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_module():
    """Ensure each test gets a clean import of backend_registry so that
    module-level env reads (TURBO_ENABLED) reflect any monkeypatches."""
    # Remove the cached module before each test; the test body will import it.
    sys.modules.pop("router.backend_registry", None)
    yield
    sys.modules.pop("router.backend_registry", None)


# Env vars that override a backend's max_context (see max_context_env in
# config/backend-registry.yaml). Cleared in reg_turbo_on/reg_turbo_off so
# routing-boundary tests are hermetic -- see those fixtures for why.
_MAX_CONTEXT_ENV_KEYS = ("LOCAL_LONG_CTX",)


@pytest.fixture()
def reg_turbo_on(monkeypatch, tmp_path):
    """Registry loaded with TURBO_ENABLED=1.

    Points env_path at a nonexistent file AND clears any already-set
    max_context_env vars (e.g. LOCAL_LONG_CTX) so max_context values come
    solely from the static YAML defaults (131072 for local-long-128k,
    etc.) -- not from the host's real hardware-detected config/detected.env,
    which varies runner-to-runner. Both guards are needed:
    BackendRegistry falls back from env_path's file to the real
    os.environ, and other modules imported during the same pytest session
    (e.g. claude_proxy/server.py, which runs os.environ.setdefault(key,
    val) for every key in config/detected.env at import time) can leak
    LOCAL_LONG_CTX into the real environment before this fixture runs --
    this previously made the test pass in isolation but fail as part of
    the full suite (and on CI, where hardware detection yields a smaller
    value).
    """
    monkeypatch.setenv("TURBO_ENABLED", "1")
    for key in _MAX_CONTEXT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    mod = _import_registry()
    return mod.BackendRegistry(env_path=tmp_path / "nonexistent.env")


@pytest.fixture()
def reg_turbo_off(monkeypatch, tmp_path):
    """Registry loaded with TURBO_ENABLED=0. See reg_turbo_on for why
    env_path is pinned to a nonexistent file and max_context_env keys
    are cleared."""
    monkeypatch.setenv("TURBO_ENABLED", "0")
    for key in _MAX_CONTEXT_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    mod = _import_registry()
    return mod.BackendRegistry(env_path=tmp_path / "nonexistent.env")


@pytest.fixture()
def mod_turbo_on(monkeypatch):
    """Full module with TURBO_ENABLED=1 (gives access to helpers too)."""
    monkeypatch.setenv("TURBO_ENABLED", "1")
    return _import_registry()


@pytest.fixture()
def mod_turbo_off(monkeypatch):
    """Full module with TURBO_ENABLED=0."""
    monkeypatch.setenv("TURBO_ENABLED", "0")
    return _import_registry()


# ---------------------------------------------------------------------------
# 1. Registry loads from the real YAML file
# ---------------------------------------------------------------------------

class TestRegistryLoads:
    def test_registry_singleton_exists(self, reg_turbo_on):
        mod = sys.modules["router.backend_registry"]
        assert hasattr(mod, "registry")

    def test_backend_names_present(self, reg_turbo_on):
        names = reg_turbo_on.tier_order()
        assert "local-long-128k" in names
        assert "local-turbo-256k" in names
        assert "local-turbo-512k" in names

    def test_fallback_target(self, reg_turbo_on):
        assert reg_turbo_on.fallback_target() == "claude-external"

    def test_tier_order_is_list(self, reg_turbo_on):
        order = reg_turbo_on.tier_order()
        assert isinstance(order, list)
        assert len(order) >= 2

    def test_tier_order_correct_sequence(self, reg_turbo_on):
        order = reg_turbo_on.tier_order()
        assert order == [
            "local-long-128k",
            "local-turbo-256k",
            "local-turbo-512k",
        ]


# ---------------------------------------------------------------------------
# 2. BackendConfig dataclass
# ---------------------------------------------------------------------------

class TestBackendConfig:
    def test_local_long_config(self, reg_turbo_on):
        cfg = reg_turbo_on.backend("local-long-128k")
        assert cfg is not None
        assert cfg.name == "local-long-128k"
        assert cfg.kind == "ollama"

    def test_local_turbo_256k_config(self, reg_turbo_on):
        cfg = reg_turbo_on.backend("local-turbo-256k")
        assert cfg is not None
        assert cfg.name == "local-turbo-256k"
        assert cfg.kind == "mlx-turbo"
        assert cfg.max_context == 262144

    def test_local_turbo_512k_config(self, reg_turbo_on):
        cfg = reg_turbo_on.backend("local-turbo-512k")
        assert cfg is not None
        assert cfg.name == "local-turbo-512k"
        assert cfg.kind == "mlx-turbo"
        assert cfg.max_context == 524288

    def test_unknown_backend_returns_none(self, reg_turbo_on):
        assert reg_turbo_on.backend("does-not-exist") is None

    def test_config_is_dataclass(self, reg_turbo_on):
        import dataclasses
        cfg = reg_turbo_on.backend("local-long-128k")
        assert dataclasses.is_dataclass(cfg)

    def test_resident_policy_local_long(self, reg_turbo_on):
        cfg = reg_turbo_on.backend("local-long-128k")
        assert cfg.resident_policy == "warm_optional"

    def test_resident_policy_turbo(self, reg_turbo_on):
        cfg = reg_turbo_on.backend("local-turbo-256k")
        assert cfg.resident_policy == "on_demand"


# ---------------------------------------------------------------------------
# 3. choose_backend() — with turbo enabled
# ---------------------------------------------------------------------------

class TestChooseBackendTurboOn:
    def test_8k_tokens_routes_to_local_long(self, reg_turbo_on):
        assert reg_turbo_on.choose_backend(8_000) == "local-long-128k"

    def test_50k_tokens_routes_to_local_long(self, reg_turbo_on):
        assert reg_turbo_on.choose_backend(50_000) == "local-long-128k"

    def test_150k_tokens_routes_to_turbo_256k(self, reg_turbo_on):
        assert reg_turbo_on.choose_backend(150_000) == "local-turbo-256k"

    def test_300k_tokens_routes_to_turbo_512k(self, reg_turbo_on):
        assert reg_turbo_on.choose_backend(300_000) == "local-turbo-512k"

    def test_600k_tokens_routes_to_fallback(self, reg_turbo_on):
        fallback = reg_turbo_on.fallback_target()
        assert reg_turbo_on.choose_backend(600_000) == fallback

    def test_zero_tokens_routes_to_local_long(self, reg_turbo_on):
        # Edge case: zero tokens should take the cheapest tier.
        assert reg_turbo_on.choose_backend(0) == "local-long-128k"

    def test_exact_turbo_256k_max(self, reg_turbo_on):
        cfg = reg_turbo_on.backend("local-turbo-256k")
        assert reg_turbo_on.choose_backend(cfg.max_context) == "local-turbo-256k"

    def test_just_over_turbo_256k_max_routes_to_512k(self, reg_turbo_on):
        cfg = reg_turbo_on.backend("local-turbo-256k")
        assert reg_turbo_on.choose_backend(cfg.max_context + 1) == "local-turbo-512k"


# ---------------------------------------------------------------------------
# 4. choose_backend() — with turbo disabled
# ---------------------------------------------------------------------------

class TestChooseBackendTurboOff:
    def test_8k_tokens_still_local_long(self, reg_turbo_off):
        assert reg_turbo_off.choose_backend(8_000) == "local-long-128k"

    def test_50k_tokens_still_local_long(self, reg_turbo_off):
        assert reg_turbo_off.choose_backend(50_000) == "local-long-128k"

    def test_150k_tokens_fallback_not_turbo(self, reg_turbo_off):
        # When TURBO_ENABLED=0, 150k exceeds local-long and should NOT go to turbo.
        result = reg_turbo_off.choose_backend(150_000)
        assert result != "local-turbo-256k"
        assert result == reg_turbo_off.fallback_target()

    def test_300k_tokens_fallback_not_turbo(self, reg_turbo_off):
        result = reg_turbo_off.choose_backend(300_000)
        assert result != "local-turbo-512k"
        assert result == reg_turbo_off.fallback_target()

    def test_600k_tokens_fallback(self, reg_turbo_off):
        assert reg_turbo_off.choose_backend(600_000) == reg_turbo_off.fallback_target()

    def test_turbo_not_in_effective_tier_order(self, reg_turbo_off):
        # When turbo is disabled, the turbo tiers should not appear in the
        # effective routing sequence used by choose_backend().
        # We verify indirectly: no token count should route to a turbo backend.
        for tokens in [150_000, 262_144, 300_000, 524_288]:
            result = reg_turbo_off.choose_backend(tokens)
            assert "turbo" not in result, (
                f"choose_backend({tokens}) returned {result!r} with TURBO_ENABLED=0"
            )


# ---------------------------------------------------------------------------
# 5. is_turbo_enabled()
# ---------------------------------------------------------------------------

class TestIsTurboEnabled:
    def test_turbo_on_when_env_is_1(self, reg_turbo_on):
        assert reg_turbo_on.is_turbo_enabled() is True

    def test_turbo_off_when_env_is_0(self, reg_turbo_off):
        assert reg_turbo_off.is_turbo_enabled() is False

    def test_turbo_off_by_default(self, monkeypatch):
        # When TURBO_ENABLED is not set at all, should default to disabled.
        monkeypatch.delenv("TURBO_ENABLED", raising=False)
        sys.modules.pop("router.backend_registry", None)
        mod = _import_registry()
        # Default should be False/off so the user doesn't accidentally start turbo.
        assert mod.registry.is_turbo_enabled() is False

    def test_truthy_values_enable_turbo(self, monkeypatch):
        for val in ("1", "true", "yes", "on"):
            monkeypatch.setenv("TURBO_ENABLED", val)
            sys.modules.pop("router.backend_registry", None)
            mod = _import_registry()
            assert mod.registry.is_turbo_enabled() is True, f"Expected turbo on for {val!r}"

    def test_falsy_values_disable_turbo(self, monkeypatch):
        for val in ("0", "false", "no", "off"):
            monkeypatch.setenv("TURBO_ENABLED", val)
            sys.modules.pop("router.backend_registry", None)
            mod = _import_registry()
            assert mod.registry.is_turbo_enabled() is False, f"Expected turbo off for {val!r}"


# ---------------------------------------------------------------------------
# 6. tier_order() and fallback_target()
# ---------------------------------------------------------------------------

class TestTierOrder:
    def test_tier_order_returns_three_tiers(self, reg_turbo_on):
        assert len(reg_turbo_on.tier_order()) == 3

    def test_tier_order_first_is_cheapest(self, reg_turbo_on):
        assert reg_turbo_on.tier_order()[0] == "local-long-128k"

    def test_tier_order_last_is_most_expensive(self, reg_turbo_on):
        assert reg_turbo_on.tier_order()[-1] == "local-turbo-512k"

    def test_tier_order_is_ascending_by_context(self, reg_turbo_on):
        order = reg_turbo_on.tier_order()
        ctx_sizes = []
        for name in order:
            cfg = reg_turbo_on.backend(name)
            if cfg is not None and cfg.max_context is not None:
                ctx_sizes.append(cfg.max_context)
        assert ctx_sizes == sorted(ctx_sizes), (
            "tier_order() should be in ascending max_context order"
        )


class TestFallbackTarget:
    def test_fallback_target_is_string(self, reg_turbo_on):
        assert isinstance(reg_turbo_on.fallback_target(), str)

    def test_fallback_target_value(self, reg_turbo_on):
        assert reg_turbo_on.fallback_target() == "claude-external"

    def test_fallback_same_regardless_of_turbo(self, reg_turbo_on, reg_turbo_off):
        assert reg_turbo_on.fallback_target() == reg_turbo_off.fallback_target()


# ---------------------------------------------------------------------------
# 7. _read_env_file()
# ---------------------------------------------------------------------------

class TestReadEnvFile:
    """Tests for the _read_env_file() helper inside backend_registry."""

    def _get_fn(self, mod_turbo_on):
        return mod_turbo_on._read_env_file

    def test_empty_file_returns_empty_dict(self, mod_turbo_on, tmp_path):
        f = tmp_path / "empty.env"
        f.write_text("")
        result = self._get_fn(mod_turbo_on)(f)
        assert result == {}

    def test_missing_file_returns_empty_dict(self, mod_turbo_on, tmp_path):
        result = self._get_fn(mod_turbo_on)(tmp_path / "nonexistent.env")
        assert result == {}

    def test_simple_key_value(self, mod_turbo_on, tmp_path):
        f = tmp_path / "simple.env"
        f.write_text("FOO=bar\n")
        assert self._get_fn(mod_turbo_on)(f) == {"FOO": "bar"}

    def test_double_quoted_value(self, mod_turbo_on, tmp_path):
        f = tmp_path / "quoted.env"
        f.write_text('KEY="hello world"\n')
        result = self._get_fn(mod_turbo_on)(f)
        assert result["KEY"] == "hello world"

    def test_single_quoted_value(self, mod_turbo_on, tmp_path):
        f = tmp_path / "squoted.env"
        f.write_text("KEY='hello world'\n")
        result = self._get_fn(mod_turbo_on)(f)
        assert result["KEY"] == "hello world"

    def test_comment_lines_are_skipped(self, mod_turbo_on, tmp_path):
        f = tmp_path / "comments.env"
        f.write_text(
            textwrap.dedent("""\
                # this is a comment
                FOO=bar
                # another comment
                BAZ=qux
            """)
        )
        result = self._get_fn(mod_turbo_on)(f)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_blank_lines_are_skipped(self, mod_turbo_on, tmp_path):
        f = tmp_path / "blanks.env"
        f.write_text("\nFOO=bar\n\n")
        result = self._get_fn(mod_turbo_on)(f)
        assert result == {"FOO": "bar"}

    def test_lines_without_equals_are_skipped(self, mod_turbo_on, tmp_path):
        f = tmp_path / "noeq.env"
        f.write_text("JUSTAKEYNOVALUE\nFOO=bar\n")
        result = self._get_fn(mod_turbo_on)(f)
        assert result == {"FOO": "bar"}

    def test_value_with_equals_sign(self, mod_turbo_on, tmp_path):
        # Values that themselves contain '=' must survive (split on first '=' only).
        f = tmp_path / "eqinval.env"
        f.write_text("URL=http://example.com/a=b\n")
        result = self._get_fn(mod_turbo_on)(f)
        assert result["URL"] == "http://example.com/a=b"

    def test_multiple_entries(self, mod_turbo_on, tmp_path):
        f = tmp_path / "multi.env"
        f.write_text(
            textwrap.dedent("""\
                MLX_PORT=8080
                OLLAMA_PORT=11434
                TURBO_ENABLED=0
                LOCAL_LONG_CTX=131072
            """)
        )
        result = self._get_fn(mod_turbo_on)(f)
        assert result["MLX_PORT"] == "8080"
        assert result["OLLAMA_PORT"] == "11434"
        assert result["TURBO_ENABLED"] == "0"
        assert result["LOCAL_LONG_CTX"] == "131072"

    def test_whitespace_around_key_and_value(self, mod_turbo_on, tmp_path):
        f = tmp_path / "ws.env"
        f.write_text("  KEY  =  value  \n")
        result = self._get_fn(mod_turbo_on)(f)
        assert result["KEY"] == "value"


# ---------------------------------------------------------------------------
# 8. BackendRegistry class surface / structure
# ---------------------------------------------------------------------------

class TestBackendRegistryClass:
    def test_registry_is_instance_of_backend_registry_class(self, mod_turbo_on):
        assert isinstance(mod_turbo_on.registry, mod_turbo_on.BackendRegistry)

    def test_backend_config_class_exists(self, mod_turbo_on):
        assert hasattr(mod_turbo_on, "BackendConfig")

    def test_choose_backend_returns_string(self, reg_turbo_on):
        result = reg_turbo_on.choose_backend(1000)
        assert isinstance(result, str)

    def test_fallback_target_never_a_turbo_backend(self, reg_turbo_on):
        target = reg_turbo_on.fallback_target()
        assert "turbo" not in target

    def test_all_backends_in_tier_order_are_reachable(self, reg_turbo_on):
        for name in reg_turbo_on.tier_order():
            cfg = reg_turbo_on.backend(name)
            assert cfg is not None, f"backend({name!r}) returned None"

    def test_choose_backend_result_is_valid_backend_or_fallback(self, reg_turbo_on):
        valid = set(reg_turbo_on.tier_order()) | {reg_turbo_on.fallback_target()}
        for tokens in [0, 100, 8_000, 50_000, 150_000, 300_000, 600_000]:
            result = reg_turbo_on.choose_backend(tokens)
            assert result in valid, (
                f"choose_backend({tokens}) returned {result!r} which is not in {valid}"
            )
