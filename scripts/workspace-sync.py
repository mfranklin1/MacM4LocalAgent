"""Sync MacM4LocalAgent.code-workspace to match the current git worktrees.

Run directly or via `make worktree-sync`. Called automatically by
`make worktree` and `make worktree-rm` so the Cursor workspace file
stays in step without any manual editing.

The workspace file is always written to the MAIN worktree root (the first
entry in `git worktree list --porcelain`), regardless of which worktree
this script is called from. This means it works correctly whether invoked
from the main repo or from inside a linked worktree like .worktrees/feat-x.

Folder ordering in the workspace file:
  1. { "path": "." }  — always the main worktree
  2. One entry per linked worktree, relative path when inside the main
     repo tree, absolute path when the worktree lives elsewhere
     (e.g. a sibling directory).

Usage:
    python3 scripts/workspace-sync.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


def _parse_worktrees() -> tuple[pathlib.Path, list[pathlib.Path]]:
    """Return (main_root, [linked_worktree_paths]) by parsing git porcelain output.

    Works regardless of which worktree the script is called from because
    git always lists the main worktree first.
    """
    raw = subprocess.check_output(
        ["git", "worktree", "list", "--porcelain"],
        text=True,
        # No cwd needed — git walks up to the repo from wherever we are.
    )
    paths: list[pathlib.Path] = []
    for line in raw.splitlines():
        if line.startswith("worktree "):
            paths.append(pathlib.Path(line[len("worktree "):].strip()).resolve())

    if not paths:
        raise RuntimeError("git worktree list returned no entries")

    main_root = paths[0]
    linked = paths[1:]
    return main_root, linked


def _folder_entry(wt_abs: pathlib.Path, main_root: pathlib.Path) -> dict:
    """Return a .code-workspace folder dict, using a relative path where possible."""
    try:
        rel = wt_abs.relative_to(main_root)
        return {"path": str(rel)}
    except ValueError:
        # Worktree is outside the main repo tree (e.g. a sibling directory).
        return {"path": str(wt_abs)}


def sync(dry_run: bool = False) -> None:
    main_root, linked = _parse_worktrees()
    workspace_file = main_root / f"{main_root.name}.code-workspace"

    folders: list[dict] = [{"path": "."}]
    for wt in linked:
        folders.append(_folder_entry(wt, main_root))

    # Preserve any existing non-folder keys (settings, extensions, etc.)
    # so user customisations stored in the workspace file are not lost.
    existing: dict = {}
    if workspace_file.exists():
        try:
            existing = json.loads(workspace_file.read_text())
        except json.JSONDecodeError:
            pass

    workspace = {"folders": folders, **{k: v for k, v in existing.items() if k != "folders"}}
    content = json.dumps(workspace, indent=2) + "\n"

    if dry_run:
        print(content)
        return

    workspace_file.write_text(content)
    rel_ws = workspace_file.relative_to(main_root)
    print(f"[workspace-sync] {rel_ws}")
    for entry in folders:
        tag = "  ← main" if entry["path"] == "." else ""
        print(f"  {entry['path']}{tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print result without writing")
    args = parser.parse_args()
    try:
        sync(dry_run=args.dry_run)
    except Exception as e:
        print(f"[workspace-sync] error: {e}", file=sys.stderr)
        sys.exit(1)
