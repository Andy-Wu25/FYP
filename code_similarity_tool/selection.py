from __future__ import annotations
from pathlib import Path
from fnmatch import fnmatch

def within_selection(repo_root: Path, abs_path: Path, cfg: dict) -> bool:
    try:
        rel = abs_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False  # outside repo

    rel_str = str(rel).replace("\\", "/")

    # exclude patterns first
    for pat in cfg.get("exclude_patterns", []):
        if fnmatch(rel_str, pat):
            return False

    inc_dirs = [Path(d) for d in cfg.get("include_dirs", [])]
    inc_files = set(cfg.get("include_files", []))

    if rel_str in inc_files:
        return True

    for d in inc_dirs:
        try:
            rel.relative_to(d)
            return True
        except ValueError:
            pass

    # If nothing selected, default to allow (legacy behaviour)
    if not inc_dirs and not inc_files:
        return True
    return False