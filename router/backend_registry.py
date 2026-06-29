"""Backend registry: loads config/backend-registry.yaml and provides routing helpers.

Module-level singleton ``registry`` is the entry point for the router and lifecycle
manager.  TURBO_ENABLED is read from the environment at import time so that tests
can monkeypatch os.environ before importing (or use importlib.reload).
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REGISTRY_YAML = _REPO_ROOT / "config" / "backend-registry.yaml"
_DETECTED_ENV = _REPO_ROOT / "config" / "detected.env"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_env_file(path: pathlib.Path) -> dict[str, str]:
    """Parse a shell-style KEY=VALUE env file; return a dict of strings.

    Skips comments (#), blank lines, and lines without '='.
    Strips surrounding quotes and whitespace from keys and values.
    Splits on the first '=' only so values may contain '='.
    """
    result: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return result
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
            val = val[1:-1]
        result[key] = val
    return result


# ---------------------------------------------------------------------------
# BackendConfig dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class BackendConfig:
    name: str
    kind: str
    max_context: Optional[int]
    resident_policy: str
    port: Optional[int] = None
    port_env: Optional[str] = None
    model_env: Optional[str] = None
    model_repo: Optional[str] = None
    model_path_env: Optional[str] = None
    max_context_env: Optional[str] = None
    turbo_kv_bits: Optional[int] = None
    turbo_fp16_layers: Optional[int] = None
    startup_timeout_seconds: int = 60
    idle_timeout_seconds: int = 0


# ---------------------------------------------------------------------------
# BackendRegistry
# ---------------------------------------------------------------------------

class BackendRegistry:
    """Loads backend-registry.yaml and answers routing questions."""

    def __init__(self, yaml_path: pathlib.Path = _REGISTRY_YAML,
                 env_path: pathlib.Path = _DETECTED_ENV) -> None:
        with yaml_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        detected = _read_env_file(env_path)

        self._backends: dict[str, BackendConfig] = {}
        for name, spec in raw["backends"].items():
            max_ctx: Optional[int] = spec.get("max_context")
            # Runtime env overrides the static YAML default when available.
            if "max_context_env" in spec:
                env_key = spec["max_context_env"]
                raw_val = detected.get(env_key) or os.environ.get(env_key)
                if raw_val is not None:
                    max_ctx = int(raw_val)
            self._backends[name] = BackendConfig(
                name=name,
                kind=spec["kind"],
                max_context=max_ctx,
                resident_policy=spec.get("resident_policy", "on_demand"),
                port=spec.get("port"),
                port_env=spec.get("port_env"),
                model_env=spec.get("model_env"),
                model_repo=spec.get("model_repo"),
                model_path_env=spec.get("model_path_env"),
                max_context_env=spec.get("max_context_env"),
                turbo_kv_bits=spec.get("turbo_kv_bits"),
                turbo_fp16_layers=spec.get("turbo_fp16_layers"),
                startup_timeout_seconds=spec.get("startup_timeout_seconds", 60),
                idle_timeout_seconds=spec.get("idle_timeout_seconds", 0),
            )

        self._tier_order: list[str] = raw["tier_order"]
        self._fallback_target: str = raw["fallback"]["target"]
        self._turbo_enabled: bool = _resolve_turbo_enabled()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backend(self, name: str) -> Optional[BackendConfig]:
        return self._backends.get(name)

    def tier_order(self) -> list[str]:
        return list(self._tier_order)

    def fallback_target(self) -> str:
        return self._fallback_target

    def is_turbo_enabled(self) -> bool:
        return self._turbo_enabled

    def choose_backend(self, tokens: int) -> str:
        """Return the cheapest backend that can handle *tokens* context tokens.

        Turbo tiers are skipped when TURBO_ENABLED is false.  If no local tier
        fits, returns fallback_target().
        """
        for name in self._tier_order:
            cfg = self._backends.get(name)
            if cfg is None:
                continue
            if "turbo" in name and not self._turbo_enabled:
                continue
            if cfg.max_context is not None and tokens <= cfg.max_context:
                return name
        return self._fallback_target


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_turbo_enabled() -> bool:
    """Read TURBO_ENABLED from the environment; default False."""
    val = os.environ.get("TURBO_ENABLED", "").strip().lower()
    return val in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

registry = BackendRegistry()


# ---------------------------------------------------------------------------
# Convenience helpers (called by route_by_size._try_turbo_escalation)
# ---------------------------------------------------------------------------

def pick_turbo_backend(tokens: int) -> Optional[str]:
    """Return a turbo backend name that can handle *tokens*, or None.

    Only looks at turbo tiers.  Returns None when TURBO_ENABLED is False,
    no turbo tier can handle the token count, or the token count exceeds
    all turbo tiers (should fall back to claude-code instead).
    """
    if not registry.is_turbo_enabled():
        return None
    for name in registry.tier_order():
        if "turbo" not in name:
            continue
        cfg = registry.backend(name)
        if cfg is not None and cfg.max_context is not None and tokens <= cfg.max_context:
            return name
    return None
