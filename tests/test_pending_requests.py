"""Unit tests for router/pending_requests.py."""

from __future__ import annotations

import json
import pathlib
import sys
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from router.pending_requests import PendingRequest, PendingRequestStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(request_id: str = "req-001", **kwargs) -> PendingRequest:
    defaults = {
        "request_id": request_id,
        "created_at": "2026-06-29T12:00:00+00:00",
        "estimated_tokens": 50_000,
        "target_backend": "local-long-128k",
        "client": "cline",
        "payload": {"model": "hybrid-auto", "messages": []},
    }
    defaults.update(kwargs)
    return PendingRequest(**defaults)


def _make_store(tmp_path: pathlib.Path) -> PendingRequestStore:
    return PendingRequestStore(
        pending_dir=tmp_path / "pending",
        failed_dir=tmp_path / "failed",
    )


# ---------------------------------------------------------------------------
# PendingRequest serialisation
# ---------------------------------------------------------------------------

class TestPendingRequestSerialisation:
    def test_to_dict_required_fields(self):
        req = _make_request()
        d = req.to_dict()
        assert d["request_id"] == "req-001"
        assert d["estimated_tokens"] == 50_000
        assert d["target_backend"] == "local-long-128k"
        assert d["client"] == "cline"
        assert isinstance(d["payload"], dict)

    def test_to_dict_omits_none_optional_fields(self):
        req = _make_request()
        d = req.to_dict()
        assert "compression_metadata" not in d
        assert "original_payload_ref" not in d

    def test_to_dict_includes_optional_when_set(self):
        req = _make_request(
            compression_metadata={"ratio": 0.65},
            original_payload_ref="s3://bucket/key",
        )
        d = req.to_dict()
        assert d["compression_metadata"] == {"ratio": 0.65}
        assert d["original_payload_ref"] == "s3://bucket/key"

    def test_from_dict_round_trip(self):
        req = _make_request()
        d = req.to_dict()
        restored = PendingRequest.from_dict(d)
        assert restored.request_id == req.request_id
        assert restored.estimated_tokens == req.estimated_tokens
        assert restored.target_backend == req.target_backend
        assert restored.client == req.client

    def test_from_dict_optional_fields_absent(self):
        req = _make_request()
        d = req.to_dict()
        restored = PendingRequest.from_dict(d)
        assert restored.compression_metadata is None
        assert restored.original_payload_ref is None

    def test_from_dict_optional_fields_present(self):
        req = _make_request(
            compression_metadata={"ratio": 0.7},
            original_payload_ref="/tmp/ref.json",
        )
        restored = PendingRequest.from_dict(req.to_dict())
        assert restored.compression_metadata == {"ratio": 0.7}
        assert restored.original_payload_ref == "/tmp/ref.json"

    def test_json_serialisable(self):
        req = _make_request()
        d = req.to_dict()
        raw = json.dumps(d)
        assert '"request_id"' in raw


# ---------------------------------------------------------------------------
# PendingRequestStore.persist() + load()
# ---------------------------------------------------------------------------

class TestPersistAndLoad:
    def test_persist_creates_file(self, tmp_path):
        store = _make_store(tmp_path)
        req = _make_request("req-persist")
        store.persist(req)
        assert (tmp_path / "pending" / "req-persist.json").exists()

    def test_load_returns_correct_request(self, tmp_path):
        store = _make_store(tmp_path)
        req = _make_request("req-load", estimated_tokens=99_000)
        store.persist(req)
        loaded = store.load("req-load")
        assert loaded.request_id == "req-load"
        assert loaded.estimated_tokens == 99_000

    def test_load_raises_file_not_found(self, tmp_path):
        store = _make_store(tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load("does-not-exist")

    def test_persist_is_idempotent(self, tmp_path):
        store = _make_store(tmp_path)
        req = _make_request("req-idem", estimated_tokens=10_000)
        store.persist(req)
        req2 = _make_request("req-idem", estimated_tokens=20_000)
        store.persist(req2)
        loaded = store.load("req-idem")
        assert loaded.estimated_tokens == 20_000

    def test_persist_file_is_valid_json(self, tmp_path):
        store = _make_store(tmp_path)
        req = _make_request("req-json")
        store.persist(req)
        raw = (tmp_path / "pending" / "req-json.json").read_text()
        parsed = json.loads(raw)
        assert parsed["request_id"] == "req-json"


# ---------------------------------------------------------------------------
# PendingRequestStore.complete()
# ---------------------------------------------------------------------------

class TestComplete:
    def test_complete_removes_file(self, tmp_path):
        store = _make_store(tmp_path)
        req = _make_request("req-done")
        store.persist(req)
        store.complete("req-done")
        assert not (tmp_path / "pending" / "req-done.json").exists()

    def test_complete_is_idempotent(self, tmp_path):
        store = _make_store(tmp_path)
        req = _make_request("req-done2")
        store.persist(req)
        store.complete("req-done2")
        store.complete("req-done2")  # should not raise

    def test_complete_nonexistent_is_noop(self, tmp_path):
        store = _make_store(tmp_path)
        store.complete("no-such-id")  # should not raise


# ---------------------------------------------------------------------------
# PendingRequestStore.fail()
# ---------------------------------------------------------------------------

class TestFail:
    def test_fail_moves_to_failed_dir(self, tmp_path):
        store = _make_store(tmp_path)
        req = _make_request("req-fail")
        store.persist(req)
        store.fail("req-fail", "backend timed out")
        assert not (tmp_path / "pending" / "req-fail.json").exists()
        assert (tmp_path / "failed" / "req-fail.json").exists()

    def test_fail_adds_error_metadata(self, tmp_path):
        store = _make_store(tmp_path)
        req = _make_request("req-err")
        store.persist(req)
        store.fail("req-err", "connection refused")
        data = json.loads((tmp_path / "failed" / "req-err.json").read_text())
        assert "_error" in data
        assert data["_error"]["message"] == "connection refused"
        assert "failed_at" in data["_error"]

    def test_fail_without_pending_file_writes_stub(self, tmp_path):
        store = _make_store(tmp_path)
        store.fail("orphan-id", "mystery error")
        data = json.loads((tmp_path / "failed" / "orphan-id.json").read_text())
        assert data["request_id"] == "orphan-id"
        assert data["_error"]["message"] == "mystery error"

    def test_fail_preserves_original_fields(self, tmp_path):
        store = _make_store(tmp_path)
        req = _make_request("req-preserve", target_backend="local-turbo-256k")
        store.persist(req)
        store.fail("req-preserve", "crash")
        data = json.loads((tmp_path / "failed" / "req-preserve.json").read_text())
        assert data["target_backend"] == "local-turbo-256k"


# ---------------------------------------------------------------------------
# PendingRequestStore.list_pending()
# ---------------------------------------------------------------------------

class TestListPending:
    def test_empty_store_returns_empty_list(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.list_pending() == []

    def test_single_pending_request(self, tmp_path):
        store = _make_store(tmp_path)
        store.persist(_make_request("req-a"))
        assert store.list_pending() == ["req-a"]

    def test_multiple_pending_requests(self, tmp_path):
        store = _make_store(tmp_path)
        store.persist(_make_request("req-x"))
        store.persist(_make_request("req-y"))
        store.persist(_make_request("req-z"))
        pending = store.list_pending()
        assert sorted(pending) == ["req-x", "req-y", "req-z"]

    def test_completed_not_listed(self, tmp_path):
        store = _make_store(tmp_path)
        store.persist(_make_request("req-1"))
        store.persist(_make_request("req-2"))
        store.complete("req-1")
        assert store.list_pending() == ["req-2"]

    def test_failed_not_listed(self, tmp_path):
        store = _make_store(tmp_path)
        store.persist(_make_request("req-fail-list"))
        store.fail("req-fail-list", "err")
        assert store.list_pending() == []


# ---------------------------------------------------------------------------
# PendingRequestStore.sweep_stale()
# ---------------------------------------------------------------------------

class TestSweepStale:
    def test_sweep_removes_old_files(self, tmp_path, monkeypatch):
        store = _make_store(tmp_path)
        req = _make_request("req-old")
        store.persist(req)
        # Backdate the file's mtime by 2 hours.
        p = tmp_path / "pending" / "req-old.json"
        old_mtime = p.stat().st_mtime - 7200
        import os
        os.utime(p, (old_mtime, old_mtime))
        removed = store.sweep_stale(max_age_seconds=3600)
        assert removed == 1
        assert not p.exists()

    def test_sweep_leaves_fresh_files(self, tmp_path):
        store = _make_store(tmp_path)
        req = _make_request("req-fresh")
        store.persist(req)
        removed = store.sweep_stale(max_age_seconds=3600)
        assert removed == 0
        assert (tmp_path / "pending" / "req-fresh.json").exists()

    def test_sweep_returns_count(self, tmp_path):
        store = _make_store(tmp_path)
        for i in range(3):
            store.persist(_make_request(f"req-{i}"))
        # Backdate all of them.
        import os
        for p in (tmp_path / "pending").iterdir():
            old_mtime = p.stat().st_mtime - 7200
            os.utime(p, (old_mtime, old_mtime))
        removed = store.sweep_stale(max_age_seconds=3600)
        assert removed == 3

    def test_sweep_empty_dir_returns_zero(self, tmp_path):
        store = _make_store(tmp_path)
        assert store.sweep_stale() == 0


# ---------------------------------------------------------------------------
# Directories are created on init
# ---------------------------------------------------------------------------

class TestStoreInit:
    def test_creates_pending_dir(self, tmp_path):
        d = tmp_path / "deep" / "pending"
        assert not d.exists()
        PendingRequestStore(pending_dir=d, failed_dir=tmp_path / "failed")
        assert d.is_dir()

    def test_creates_failed_dir(self, tmp_path):
        d = tmp_path / "deep" / "failed"
        assert not d.exists()
        PendingRequestStore(pending_dir=tmp_path / "pending", failed_dir=d)
        assert d.is_dir()
