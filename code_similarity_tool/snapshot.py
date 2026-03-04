#!/usr/bin/env python3
"""CLI entry point: code-sim-snapshot

Manage ChromaDB snapshots for evaluation workflows.
Snapshots are stored at <db-path>/snapshots/<name>/.

Usage:
    code-sim-snapshot save <name> [--db-path PATH] [--force]
    code-sim-snapshot list [--db-path PATH]
    code-sim-snapshot load <name> [--db-path PATH]
    code-sim-snapshot delete <name> [--db-path PATH]
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

from .runtime import load_runtime_context

_META_FILE = "_snapshot_meta.json"
_SNAPSHOTS_SUBDIR = "snapshots"
_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _resolve_db_path(override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    ctx = load_runtime_context()
    return ctx.db_path


def _snapshots_root(db_path: Path) -> Path:
    return db_path.parent / _SNAPSHOTS_SUBDIR


def _snapshot_dir(db_path: Path, name: str) -> Path:
    return _snapshots_root(db_path) / name


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


def cmd_save(name: str, db_path_override: Optional[str], force: bool) -> None:
    _validate_name(name)
    db_path = _resolve_db_path(db_path_override)

    if not db_path.is_dir():
        print(f"Error: database directory does not exist: {db_path}", file=sys.stderr)
        sys.exit(1)

    dest = _snapshot_dir(db_path, name)
    if dest.exists():
        if not force:
            print(f"Error: snapshot '{name}' already exists. Use --force to overwrite.", file=sys.stderr)
            sys.exit(1)
        shutil.rmtree(dest)

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


def cmd_list(db_path_override: Optional[str]) -> None:
    db_path = _resolve_db_path(db_path_override)
    snap_root = _snapshots_root(db_path)

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


def cmd_load(name: str, db_path_override: Optional[str] = None) -> None:
    _validate_name(name)
    db_path = _resolve_db_path(db_path_override)
    src = _snapshot_dir(db_path, name)

    if not src.is_dir():
        print(f"Error: snapshot '{name}' not found in {db_path}.", file=sys.stderr)
        sys.exit(1)

    print(f"Restoring snapshot '{name}' -> {db_path} ...")
    if db_path.exists():
        shutil.rmtree(db_path)
    shutil.copytree(src, db_path, ignore=shutil.ignore_patterns(_META_FILE))

    print(f"Snapshot '{name}' loaded.")


def cmd_delete(name: str, db_path_override: Optional[str] = None) -> None:
    _validate_name(name)
    db_path = _resolve_db_path(db_path_override)
    target = _snapshot_dir(db_path, name)

    if not target.is_dir():
        print(f"Error: snapshot '{name}' not found in {db_path}.", file=sys.stderr)
        sys.exit(1)

    shutil.rmtree(target)
    print(f"Snapshot '{name}' deleted.")


_DB_PATH_HELP = "Override database path (default: CODE_SIM_DB_PATH or ~/.code-sim/chroma)"


def main_save() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-snapshot-save",
        description="Save current DB as a named snapshot.",
    )
    parser.add_argument("name", help="Snapshot name (alphanumeric, hyphens, underscores)")
    parser.add_argument("--db-path", default=None, help=_DB_PATH_HELP)
    parser.add_argument("--force", action="store_true", help="Overwrite existing snapshot")
    args = parser.parse_args()
    cmd_save(args.name, args.db_path, args.force)


def main_list() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-snapshot-list",
        description="List all snapshots.",
    )
    parser.add_argument("--db-path", default=None, help=_DB_PATH_HELP)
    args = parser.parse_args()
    cmd_list(args.db_path)


def main_load() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-snapshot-load",
        description="Replace active DB with a snapshot.",
    )
    parser.add_argument("name", help="Snapshot name to load")
    parser.add_argument("--db-path", default=None, help=_DB_PATH_HELP)
    args = parser.parse_args()
    cmd_load(args.name, args.db_path)


def main_delete() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-snapshot-delete",
        description="Delete a snapshot.",
    )
    parser.add_argument("name", help="Snapshot name to delete")
    parser.add_argument("--db-path", default=None, help=_DB_PATH_HELP)
    args = parser.parse_args()
    cmd_delete(args.name, args.db_path)
