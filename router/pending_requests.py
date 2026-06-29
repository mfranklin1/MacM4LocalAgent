"""
pending_requests.py — Persistence layer for in-flight requests during backend switches.

Writes one JSON file per request to a configurable pending directory, moves failed
requests to a separate failed directory, and supports atomic writes + thread safety.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class PendingRequest:
    """Represents a single in-flight request persisted to disk."""

    request_id: str
    created_at: str                          # ISO-8601 string, e.g. "2026-06-29T12:00:00+00:00"
    estimated_tokens: int
    target_backend: str
    client: str
    payload: dict
    compression_metadata: Optional[dict] = field(default=None)
    original_payload_ref: Optional[str] = field(default=None)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict matching the canonical schema."""
        d: dict = {
            "request_id": self.request_id,
            "created_at": self.created_at,
            "estimated_tokens": self.estimated_tokens,
            "target_backend": self.target_backend,
            "client": self.client,
            "payload": self.payload,
        }
        if self.compression_metadata is not None:
            d["compression_metadata"] = self.compression_metadata
        if self.original_payload_ref is not None:
            d["original_payload_ref"] = self.original_payload_ref
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "PendingRequest":
        """Reconstruct a PendingRequest from a deserialised JSON dict."""
        return cls(
            request_id=data["request_id"],
            created_at=data["created_at"],
            estimated_tokens=data["estimated_tokens"],
            target_backend=data["target_backend"],
            client=data["client"],
            payload=data["payload"],
            compression_metadata=data.get("compression_metadata"),
            original_payload_ref=data.get("original_payload_ref"),
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class PendingRequestStore:
    """
    File-backed store for pending requests.

    - Atomic writes (write to .tmp then os.replace).
    - Thread-safe via a single instance-level Lock.
    - pending_dir/<request_id>.json holds active requests.
    - failed_dir/<request_id>.json holds failed requests (with appended error metadata).
    """

    _SUFFIX = ".json"
    _TMP_SUFFIX = ".json.tmp"

    def __init__(self, pending_dir: str | os.PathLike, failed_dir: str | os.PathLike) -> None:
        self._pending_dir = Path(pending_dir)
        self._failed_dir = Path(failed_dir)
        self._pending_dir.mkdir(parents=True, exist_ok=True)
        self._failed_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def persist(self, request: PendingRequest) -> None:
        """Write *request* to pending_dir/<request_id>.json atomically."""
        target = self._pending_dir / f"{request.request_id}{self._SUFFIX}"
        tmp = self._pending_dir / f"{request.request_id}{self._TMP_SUFFIX}"
        data = json.dumps(request.to_dict(), indent=2, ensure_ascii=False)
        with self._lock:
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, target)

    def load(self, request_id: str) -> PendingRequest:
        """Read and return the PendingRequest identified by *request_id*.

        Raises FileNotFoundError if the request is not in the pending directory.
        """
        target = self._pending_dir / f"{request_id}{self._SUFFIX}"
        with self._lock:
            raw = target.read_text(encoding="utf-8")
        return PendingRequest.from_dict(json.loads(raw))

    def complete(self, request_id: str) -> None:
        """Delete the pending file for *request_id* (request fulfilled successfully)."""
        target = self._pending_dir / f"{request_id}{self._SUFFIX}"
        with self._lock:
            try:
                target.unlink()
            except FileNotFoundError:
                pass  # idempotent — already removed

    def fail(self, request_id: str, error: str) -> None:
        """Move the pending file to failed_dir with appended error metadata.

        If the pending file does not exist, a minimal failed record is written
        directly to failed_dir (so the error is never silently lost).
        """
        pending = self._pending_dir / f"{request_id}{self._SUFFIX}"
        failed = self._failed_dir / f"{request_id}{self._SUFFIX}"
        failed_tmp = self._failed_dir / f"{request_id}{self._TMP_SUFFIX}"

        with self._lock:
            # Load existing pending data if present; otherwise use a stub.
            if pending.exists():
                data = json.loads(pending.read_text(encoding="utf-8"))
            else:
                data = {"request_id": request_id}

            # Append error metadata.
            data["_error"] = {
                "message": error,
                "failed_at": datetime.now(tz=timezone.utc).isoformat(),
            }

            serialised = json.dumps(data, indent=2, ensure_ascii=False)
            failed_tmp.write_text(serialised, encoding="utf-8")
            os.replace(failed_tmp, failed)

            # Remove the pending file (best-effort).
            try:
                pending.unlink()
            except FileNotFoundError:
                pass

    def list_pending(self) -> list[str]:
        """Return the list of pending request IDs (no particular order)."""
        with self._lock:
            return [
                p.stem
                for p in self._pending_dir.iterdir()
                if p.suffix == self._SUFFIX and not p.name.endswith(self._TMP_SUFFIX)
            ]

    def sweep_stale(self, max_age_seconds: float = 3600) -> int:
        """Remove pending files older than *max_age_seconds*.

        Returns the number of files removed.
        """
        now = datetime.now(tz=timezone.utc).timestamp()
        removed = 0
        with self._lock:
            for path in list(self._pending_dir.iterdir()):
                if path.suffix != self._SUFFIX or path.name.endswith(self._TMP_SUFFIX):
                    continue
                try:
                    age = now - path.stat().st_mtime
                    if age > max_age_seconds:
                        path.unlink()
                        removed += 1
                except FileNotFoundError:
                    pass  # already removed by another thread between iterdir and stat
        return removed


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

def _resolve_dir(env_key: str, default: str) -> str:
    """Return the directory path from *env_key*, falling back to *default*."""
    return os.environ.get(env_key, default)


PENDING_REQUEST_DIR: str = _resolve_dir("PENDING_REQUEST_DIR", ".runtime/pending_requests")
FAILED_REQUEST_DIR: str = _resolve_dir("FAILED_REQUEST_DIR", ".runtime/failed_requests")

#: Singleton store — import and use directly, e.g.::
#:
#:   from router.pending_requests import store
#:   store.persist(req)
store = PendingRequestStore(
    pending_dir=PENDING_REQUEST_DIR,
    failed_dir=FAILED_REQUEST_DIR,
)
