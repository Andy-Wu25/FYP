from __future__ import annotations
from pathlib import Path
from fnmatch import fnmatch

def within_selection(repo_root: Path, abs_path: Path, cfg: dict) -> bool:
    rel = abs_path.resolve().relative_to(repo_root.resolve())
    rel_str = str(rel).replace("\\", "/")

    # exclude patterns first
    for pat in cfg.get("exclude_patterns", []):
        if fnmatch(rel_str, pat):
            return False

    inc_dirs = [Path(d) for d in cfg.get("include_dirs", [])]
    inc_files = set(cfg.get("include_files", []))

    # included by file
    if rel_str in inc_files:
        return True

    # included by directory
    for d in inc_dirs:
        try:
            rel.relative_to(d)
            return True
        except ValueError:
            pass

    # if nothing selected at all, fallback to old behavior:
    if not inc_dirs and not inc_files:
        return True  # “watch everything” under previous logic
    return False
