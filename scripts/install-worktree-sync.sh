#!/usr/bin/env bash
# install-worktree-sync.sh
#
# One-shot installer. Run once; works in every git repo from then on.
#
# What it does:
#   1. Writes ~/.local/bin/cursor-workspace-sync — a self-contained Python
#      script that reads `git worktree list` and syncs the repo's
#      .code-workspace file. No per-repo scripts or Makefiles needed.
#   2. Adds a `gwt` shell function to ~/.zshrc that wraps
#      `git worktree add` and `git worktree remove` so the sync runs
#      automatically after every worktree operation in any repo.
#
# After install:
#   gwt add .worktrees/feat-x feat/x   # create worktree + sync workspace
#   gwt rm  .worktrees/feat-x          # remove worktree + sync workspace
#   gwt list                           # git worktree list (passthrough)
#   cursor-workspace-sync              # manual repair in any git repo
set -euo pipefail

BIN="$HOME/.local/bin"
SCRIPT="$BIN/cursor-workspace-sync"
ZSHRC="$HOME/.zshrc"
MARKER="# >>> cursor-workspace-sync"

mkdir -p "$BIN"

# ── 1. Write the standalone sync script ─────────────────────────────────────
cat > "$SCRIPT" << 'PYEOF'
#!/usr/bin/env python3
"""Sync a <repo>.code-workspace file to match the current git worktrees.

Self-contained: no repo-level scripts or Makefiles required. Works in
any git repo. Invoke directly or via the `gwt` shell wrapper.

Workspace file naming convention:
  <main-worktree-dirname>.code-workspace in the main worktree root.
  e.g. ~/code/my-project/.../my-project.code-workspace

Folder ordering:
  1. { "path": "." }  — main worktree (always first)
  2. One entry per linked worktree: relative path when inside the main
     repo tree, absolute path when the worktree is an external sibling.
"""
from __future__ import annotations
import json, pathlib, subprocess, sys


def _parse_worktrees() -> tuple[pathlib.Path, list[pathlib.Path]]:
    raw = subprocess.check_output(
        ["git", "worktree", "list", "--porcelain"], text=True
    )
    paths: list[pathlib.Path] = []
    for line in raw.splitlines():
        if line.startswith("worktree "):
            paths.append(pathlib.Path(line[len("worktree "):].strip()).resolve())
    if not paths:
        raise SystemExit("not inside a git repo")
    return paths[0], paths[1:]


def _entry(wt: pathlib.Path, root: pathlib.Path) -> dict:
    try:
        return {"path": str(wt.relative_to(root))}
    except ValueError:
        return {"path": str(wt)}


def main() -> None:
    root, linked = _parse_worktrees()
    ws_file = root / f"{root.name}.code-workspace"

    folders = [{"path": "."}] + [_entry(wt, root) for wt in linked]

    existing: dict = {}
    if ws_file.exists():
        try:
            existing = json.loads(ws_file.read_text())
        except json.JSONDecodeError:
            pass

    workspace = {"folders": folders, **{k: v for k, v in existing.items() if k != "folders"}}
    ws_file.write_text(json.dumps(workspace, indent=2) + "\n")

    print(f"[workspace-sync] {ws_file.name}")
    for f in folders:
        tag = "  ← main" if f["path"] == "." else ""
        print(f"  {f['path']}{tag}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[workspace-sync] {e}", file=sys.stderr)
        sys.exit(1)
PYEOF
chmod +x "$SCRIPT"
echo "wrote $SCRIPT"

# ── 2. Add gwt shell wrapper to ~/.zshrc ────────────────────────────────────
# Idempotent: skip if the marker is already present.
if grep -q "$MARKER" "$ZSHRC" 2>/dev/null; then
  echo "~/.zshrc already has gwt wrapper — skipping"
else
  cat >> "$ZSHRC" << 'ZSHEOF'

# >>> cursor-workspace-sync
# gwt = git worktree wrapper that auto-syncs the Cursor .code-workspace file.
# Works in any git repo. No per-repo Makefile needed.
#
#   gwt add .worktrees/feat-x feat/x   create worktree + sync workspace
#   gwt rm  .worktrees/feat-x          remove worktree + sync workspace
#   gwt list                           alias for git worktree list
#   gwt <anything>                     passthrough to git worktree <anything>
#
gwt() {
  git worktree "$@"
  local exit_code=$?
  # Sync on operations that change the worktree set; passthrough on others.
  case "${1:-}" in
    add|remove|rm|prune)
      cursor-workspace-sync 2>/dev/null || true
      ;;
  esac
  return $exit_code
}
# <<< cursor-workspace-sync
ZSHEOF
  echo "added gwt wrapper to $ZSHRC"
fi

echo ""
echo "Done. Run \`source ~/.zshrc\` or open a new terminal, then:"
echo "  gwt add .worktrees/<slug> <branch>   # create worktree, sync workspace"
echo "  gwt rm  .worktrees/<slug>             # remove worktree, sync workspace"
echo "  cursor-workspace-sync                 # manual sync in any git repo"
PYEOF