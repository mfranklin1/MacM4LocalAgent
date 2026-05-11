"""bench.runners.litellm_arm.run_arm + bench.runners.cursor_session.ingest:
end-to-end with mocked LiteLLM streaming + a synthetic task."""
from __future__ import annotations

import json
import pathlib
import textwrap

import httpx
import pytest

from bench import db, grader
from bench.runners import litellm_arm, cursor_session


@pytest.fixture
def synthetic_task(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    test_file = tasks_dir / "test_addone.py"
    test_file.write_text(textwrap.dedent("""
        import addone
        def test_one(): assert addone.add(1) == 2
        def test_two(): assert addone.add(2) == 3
    """).lstrip())
    task = {
        "id": "addone",
        "prompt": "Implement add(x).",
        "grading": {
            "save_as": "addone.py",
            "test_file": "test_addone.py",
            "weights": {"passes_tests": 0.8, "syntactic_validity": 0.1, "docstring": 0.1},
        },
    }
    (tasks_dir / "addone.json").write_text(json.dumps(task))
    monkeypatch.setattr(grader, "TASKS_DIR", tasks_dir, raising=True)
    return task


def _patch_streaming(monkeypatch: pytest.MonkeyPatch, body_chunks: list[str],
                     usage: dict[str, int], status: int = 200) -> dict:
    """Replace `litellm_arm.call_streaming` with a stub returning canned data."""
    captured: dict = {}

    def fake_stream(model, prompt, *, base_url="x", timeout=900.0):
        captured["model"] = model
        captured["prompt"] = prompt
        out = "".join(body_chunks)
        return {
            "ok": True, "model": model, "model_reported": model,
            "output": out,
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "wall_ms": 1234, "ttft_ms": 200,
        }

    monkeypatch.setattr(litellm_arm, "call_streaming", fake_stream)
    return captured


GOOD_RESPONSE = textwrap.dedent('''
    Here you go.

    ```python
    """A tiny module that adds one. We document the rationale here at length
    so that the docstring word-count check ticks over to true; production
    code would have a much richer rationale than this example sentence does
    in test fixtures, but for the purposes of exercising the harness this is
    enough text and then some more padding to get well past the threshold."""
    def add(x: int) -> int:
        return x + 1
    ```
''').lstrip()


def test_run_arm_local_persists_and_grades(tmp_db, synthetic_task,
                                            monkeypatch: pytest.MonkeyPatch,
                                            tmp_path: pathlib.Path) -> None:
    _patch_streaming(monkeypatch, [GOOD_RESPONSE],
                     {"prompt_tokens": 120, "completion_tokens": 80})
    work = tmp_path / "work"
    row = litellm_arm.run_arm(
        arm="local-only", model="local-long",
        task=synthetic_task, work_dir=work,
    )
    assert row["id"] >= 1
    assert row["arm"] == "local-only"
    assert row["pytest_total"] == 2
    assert row["pytest_passed"] == 2
    assert row["actual_cost"] == 0.0
    assert row["shadow_cost"] > 0.0
    assert row["composite_score"] >= 0.9


def test_run_arm_claude_charges_actual(tmp_db, synthetic_task,
                                       monkeypatch: pytest.MonkeyPatch,
                                       tmp_path: pathlib.Path) -> None:
    _patch_streaming(monkeypatch, [GOOD_RESPONSE],
                     {"prompt_tokens": 1000, "completion_tokens": 500})
    work = tmp_path / "work"
    row = litellm_arm.run_arm(
        arm="claude-only", model="claude-code",
        task=synthetic_task, work_dir=work,
    )
    # `claude-code` is the proxy's default Claude tier -- currently Opus 4.7.
    # 1000*5e-6 + 500*25e-6 = 0.005 + 0.0125 = 0.0175
    assert row["actual_cost"] == pytest.approx(0.0175, rel=1e-6)
    assert row["arm"] == "claude-only"


def test_run_arm_handles_no_code(tmp_db, synthetic_task,
                                 monkeypatch: pytest.MonkeyPatch,
                                 tmp_path: pathlib.Path) -> None:
    # Stub returns prose with no fence -> extract_python returns the prose,
    # which is a syntax error -> grader reports 0 pytest, score < 0.5.
    monkeypatch.setattr(litellm_arm, "call_streaming",
                        lambda *a, **k: {
                            "ok": True, "model": "x", "model_reported": "x",
                            "output": "I'm sorry, I refuse.",
                            "input_tokens": 10, "output_tokens": 5,
                            "wall_ms": 100, "ttft_ms": 50,
                        })
    work = tmp_path / "work"
    row = litellm_arm.run_arm(
        arm="local-only", model="local-long",
        task=synthetic_task, work_dir=work,
    )
    assert row["pytest_total"] == 0 or row["pytest_passed"] == 0
    assert row["composite_score"] < 0.5


# ---- cursor_session ingest --------------------------------------------------

def test_cursor_session_ingest_grades_and_persists(
    tmp_db, synthetic_task, tmp_path: pathlib.Path,
) -> None:
    work = tmp_path / "session"
    work.mkdir()
    (work / "output.txt").write_text(GOOD_RESPONSE)
    session = {
        "arm": "cursor-no-proxy",
        "task_id": "addone",
        "model": "claude-sonnet-4-6",
        "start_ts": 1714000000,
        "end_ts":   1714000087,
        "ttft_ms":  1300,
        "input_tokens":  4321,
        "output_tokens": 1840,
        "notes": "Ask mode, no rules.",
    }
    (work / "session.json").write_text(json.dumps(session))

    row = cursor_session.ingest(
        session_path=work / "session.json",
        output_path=work / "output.txt",
    )
    assert row["arm"] == "cursor-no-proxy"
    assert row["task_id"] == "addone"
    assert row["wall_ms"] == 87_000
    assert row["ttft_ms"] == 1_300
    assert row["input_tok"] == 4321
    assert row["output_tok"] == 1840
    assert row["actual_cost"] == 0.0          # not anchored yet
    assert row["shadow_cost"] > 0.0
    assert row["pytest_total"] == 2
    assert row["pytest_passed"] == 2
    persisted = db.list_runs(task_id="addone", arm="cursor-no-proxy")
    assert len(persisted) == 1


def test_cursor_session_rejects_unknown_arm(
    tmp_db, synthetic_task, tmp_path: pathlib.Path,
) -> None:
    work = tmp_path / "session"
    work.mkdir()
    (work / "output.txt").write_text(GOOD_RESPONSE)
    session = {"arm": "claude-only", "task_id": "addone",
               "start_ts": 1, "end_ts": 2}
    (work / "session.json").write_text(json.dumps(session))
    with pytest.raises(ValueError):
        cursor_session.ingest(
            session_path=work / "session.json",
            output_path=work / "output.txt",
        )
