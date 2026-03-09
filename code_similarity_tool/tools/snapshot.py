#!/usr/bin/env python3
"""CLI entry point: code-sim-snapshot

Manage ChromaDB snapshots for evaluation workflows.

The active database (CODE_SIM_DB_PATH or ~/.code-sim/chroma) is the source
for ``save`` and the restore target for ``load``.  Use ``--snapshot-dir``
to control where named snapshots are stored on disk.

Usage:
    code-sim-snapshot-save   <name> [--snapshot-dir DIR] [--force]
    code-sim-snapshot-list          [--snapshot-dir DIR]
    code-sim-snapshot-load   <name> [--snapshot-dir DIR]
    code-sim-snapshot-delete <name> [--snapshot-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..core.runtime import load_runtime_context

_META_FILE = "_snapshot_meta.json"
_SNAPSHOTS_SUBDIR = "snapshots"
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _active_db_path() -> Path:
    """Return the active ChromaDB path (from env or default)."""
    ctx = load_runtime_context()
    return ctx.db_path


def _default_snapshots_dir() -> Path:
    """Default directory for named snapshots: <db-path-parent>/snapshots/."""
    return _active_db_path().parent / _SNAPSHOTS_SUBDIR


def _resolve_snapshots_dir(override: Optional[str]) -> Path:
    """Return the snapshots directory, using *override* when provided."""
    if override:
        return Path(override).expanduser().resolve()
    return _default_snapshots_dir()


def _validate_name(name: str) -> None:
    if not _NAME_RE.match(name):
        print(f"Error: snapshot name must match [a-zA-Z0-9_-]+, got '{name}'", file=sys.stderr)
        sys.exit(1)


def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} TB"


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def cmd_save(name: str, snapshot_dir: Optional[str], force: bool) -> None:
    _validate_name(name)
    db_path = _active_db_path()

    if not db_path.is_dir():
        print(f"Error: database directory does not exist: {db_path}", file=sys.stderr)
        sys.exit(1)

    snap_root = _resolve_snapshots_dir(snapshot_dir)
    dest = snap_root / name

    if dest.exists():
        if not force:
            print(f"Error: snapshot '{name}' already exists at {dest}. Use --force to overwrite.", file=sys.stderr)
            sys.exit(1)
        shutil.rmtree(dest)

    snap_root.mkdir(parents=True, exist_ok=True)
    print(f"Copying {db_path} -> {dest} ...")
    shutil.copytree(db_path, dest)

    meta = {
        "name": name,
        "source_path": str(db_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (dest / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size = _dir_size(dest)
    print(f"Snapshot '{name}' saved ({_fmt_size(size)})")


def cmd_list(snapshot_dir: Optional[str]) -> None:
    snap_root = _resolve_snapshots_dir(snapshot_dir)

    if not snap_root.is_dir():
        print(f"No snapshots found in {snap_root}.")
        return

    dirs = sorted(d for d in snap_root.iterdir() if d.is_dir())
    if not dirs:
        print(f"No snapshots found in {snap_root}.")
        return

    print(f"Snapshots in {snap_root}:\n")
    print(f"{'Name':<30} {'Size':>10}   {'Created'}")
    print("-" * 70)
    for d in dirs:
        meta_path = d / _META_FILE
        created = "unknown"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                raw = meta.get("created_at", "")
                if raw:
                    dt = datetime.fromisoformat(raw)
                    created = dt.strftime("%Y-%m-%d %H:%M UTC")
            except (json.JSONDecodeError, ValueError):
                pass
        size = _fmt_size(_dir_size(d))
        print(f"{d.name:<30} {size:>10}   {created}")


def cmd_load(name: str, snapshot_dir: Optional[str] = None) -> None:
    _validate_name(name)
    snap_root = _resolve_snapshots_dir(snapshot_dir)
    src = snap_root / name

    if not src.is_dir():
        print(f"Error: snapshot '{name}' not found in {snap_root}.", file=sys.stderr)
        sys.exit(1)

    db_path = _active_db_path()
    print(f"Restoring snapshot '{name}' -> {db_path} ...")
    if db_path.exists():
        shutil.rmtree(db_path)
    shutil.copytree(src, db_path, ignore=shutil.ignore_patterns(_META_FILE))

    print(f"Snapshot '{name}' loaded.")


def cmd_delete(name: str, snapshot_dir: Optional[str] = None) -> None:
    _validate_name(name)
    snap_root = _resolve_snapshots_dir(snapshot_dir)
    target = snap_root / name

    if not target.is_dir():
        print(f"Error: snapshot '{name}' not found in {snap_root}.", file=sys.stderr)
        sys.exit(1)

    shutil.rmtree(target)
    print(f"Snapshot '{name}' deleted.")


def resolve_snapshot(snapshot: str, snapshot_dir: Optional[str] = None) -> Path:
    """Resolve a snapshot argument to a directory path.

    Resolution order:
    1. If *snapshot* is an existing directory (absolute or relative), use it directly.
    2. Look up *snapshot* as a name under *snapshot_dir* (or the default snapshots root).
    """
    candidate = Path(snapshot).expanduser().resolve()
    if candidate.is_dir():
        return candidate

    snap_root = _resolve_snapshots_dir(snapshot_dir)
    snap_dir = snap_root / snapshot
    if snap_dir.is_dir():
        return snap_dir

    print(f"Error: snapshot '{snapshot}' not found.", file=sys.stderr)
    print(f"  Checked as path: {candidate}", file=sys.stderr)
    print(f"  Checked as name: {snap_dir}", file=sys.stderr)
    sys.exit(1)


_SNAPSHOT_DIR_HELP = (
    "Directory for named snapshots "
    "(default: <CODE_SIM_DB_PATH>/../snapshots/)"
)


def main_save() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-snapshot-save",
        description="Save current DB as a named snapshot.",
    )
    parser.add_argument("name", help="Snapshot name (alphanumeric, hyphens, underscores)")
    parser.add_argument("--snapshot-dir", default=None, help=_SNAPSHOT_DIR_HELP)
    parser.add_argument("--force", action="store_true", help="Overwrite existing snapshot")
    args = parser.parse_args()
    cmd_save(args.name, args.snapshot_dir, args.force)


def main_list() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-snapshot-list",
        description="List all snapshots.",
    )
    parser.add_argument("--snapshot-dir", default=None, help=_SNAPSHOT_DIR_HELP)
    args = parser.parse_args()
    cmd_list(args.snapshot_dir)


def main_load() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-snapshot-load",
        description="Replace active DB with a snapshot.",
    )
    parser.add_argument("name", help="Snapshot name to load")
    parser.add_argument("--snapshot-dir", default=None, help=_SNAPSHOT_DIR_HELP)
    args = parser.parse_args()
    cmd_load(args.name, args.snapshot_dir)


def main_delete() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-snapshot-delete",
        description="Delete a snapshot.",
    )
    parser.add_argument("name", help="Snapshot name to delete")
    parser.add_argument("--snapshot-dir", default=None, help=_SNAPSHOT_DIR_HELP)
    args = parser.parse_args()
    cmd_delete(args.name, args.snapshot_dir)
