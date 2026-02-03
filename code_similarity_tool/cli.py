# cli.py
from __future__ import annotations
from pathlib import Path

def init_ignore():
    repo_root = Path.cwd()
    path = repo_root / ".code-simignore"
    if path.exists():
        print(f".code-simignore already exists at {path}")
        return
    template = """# Code similarity ignore file
    /code_similarity_tool/
    **/__pycache__/
    venv/
    node_modules/
    .git/

    # Example: ignore tests
    tests/
    """
    path.write_text(template, encoding="utf-8")
    print(f"Created {path}")

def install_git_hook():
    repo_root = Path.cwd()
    git_dir = repo_root / ".git"
    hook_path = git_dir / "hooks" / "pre-commit"

    if not git_dir.is_dir():
        print("Error: .git directory not found; run this at the root of a Git repo.")
        return

    hook_path.parent.mkdir(parents=True, exist_ok=True)

    hook_body = """#!/bin/sh
# Code similarity pre-commit hook
code-sim-check
"""

    if hook_path.exists():
        # Very simple “append if not already there”
        text = hook_path.read_text(encoding="utf-8")
        if "code-sim-check" in text:
            print(f"pre-commit hook already contains code-sim-check at {hook_path}")
            return
        hook_path.write_text(text + "\n" + hook_body, encoding="utf-8")
        print(f"Appended code-sim-check to existing hook: {hook_path}")
    else:
        hook_path.write_text(hook_body, encoding="utf-8")
        print(f"Created pre-commit hook at {hook_path}")

    hook_path.chmod(0o755)
    print("Git hook installed. It will run code-sim-check before each commit.")