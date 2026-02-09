from __future__ import annotations

import argparse
from pathlib import Path

from .runtime import find_repo_root


IGNORE_TEMPLATE = """# Code similarity ignore file
# Patterns follow gitignore-style rules.
.git/
**/__pycache__/
venv/
.venv/
node_modules/
dist/
build/
"""

MARKER_START = "# >>> code-sim hook >>>"
MARKER_END = "# <<< code-sim hook <<<"


def init_ignore() -> None:
    repo_root = find_repo_root()
    ignore_path = repo_root / ".code-simignore"

    if ignore_path.exists():
        print(f".code-simignore already exists at {ignore_path}")
        return

    ignore_path.write_text(IGNORE_TEMPLATE, encoding="utf-8")
    print(f"Created {ignore_path}")


def _hook_block(stage: str) -> str:
    if stage == "pre-push":
        command = "code-sim-update"
    else:
        command = "code-sim-check"

    return (
        f"{MARKER_START}\n"
        f"# stage: {stage}\n"
        f"set -e\n"
        f"{command}\n"
        f"{MARKER_END}\n"
    )


def _inject_block(existing_text: str, block: str) -> str:
    if MARKER_START in existing_text and MARKER_END in existing_text:
        before, _, tail = existing_text.partition(MARKER_START)
        _, _, after = tail.partition(MARKER_END)
        return f"{before}{block}{after.lstrip()}"

    if existing_text and not existing_text.endswith("\n"):
        existing_text += "\n"
    return f"{existing_text}{block}"


def install_git_hook() -> None:
    parser = argparse.ArgumentParser(prog="code-sim-install-hook")
    parser.add_argument(
        "--stage",
        choices=["pre-push", "pre-commit"],
        default="pre-push",
        help="Hook stage to install. Default is pre-push (recommended).",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    git_dir = repo_root / ".git"
    if not git_dir.is_dir():
        print("Error: .git directory not found. Run this inside a Git repository.")
        return

    hook_path = git_dir / "hooks" / args.stage
    hook_path.parent.mkdir(parents=True, exist_ok=True)

    shebang = "#!/bin/sh\n"
    block = _hook_block(args.stage)

    if hook_path.exists():
        current = hook_path.read_text(encoding="utf-8")
        if not current.startswith("#!/"):
            current = shebang + current
        updated = _inject_block(current, block)
        hook_path.write_text(updated, encoding="utf-8")
        print(f"Updated {hook_path}")
    else:
        hook_path.write_text(shebang + block, encoding="utf-8")
        print(f"Created {hook_path}")

    hook_path.chmod(0o755)
    print(f"Installed code-sim hook at stage '{args.stage}'.")
