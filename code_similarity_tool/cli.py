#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

DEFAULT_IGNORE = """# Tool internals and build junk
/code_similarity_tool/
**/__pycache__/
venv/
node_modules/
.git/

# Example: ignore synthetic or test samples
src/test/
tests/
"""

def init_ignore() -> None:
    """
    Create a .code-simignore file in the current repo root if it does not exist.

    Usage (after installing the package):
        code-sim-init-ignore
    """
    repo_root = Path.cwd()
    path = repo_root / ".code-simignore"

    if path.exists():
        print(".code-simignore already exists; not overwriting.")
        print(f"Location: {path}")
        return

    path.write_text(DEFAULT_IGNORE, encoding="utf-8")
    print("Created .code-simignore with default rules.")
    print(f"Location: {path}")