from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .language_detection import has_language_hint, is_probably_binary

_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


@dataclass(frozen=True)
class RuntimeContext:
    repo_root: Path
    repo_name: str
    repo_id: str
    org_id: str
    db_path: Path
    private_collection_name: str
    public_collection_name: str
    metric: str


def _git_cmd(repo_root: Path, args: Sequence[str], *, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo_root),
        capture_output=True,
        check=True,
        text=text,
    )


def _git_cmd_optional(repo_root: Path, args: Sequence[str], *, text: bool = True) -> Optional[subprocess.CompletedProcess]:
    try:
        return _git_cmd(repo_root, args, text=text)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def find_repo_root(start: Optional[Path] = None) -> Path:
    base = (start or Path.cwd()).resolve()
    proc = _git_cmd_optional(base, ["rev-parse", "--show-toplevel"], text=True)
    if proc and proc.stdout.strip():
        return Path(proc.stdout.strip()).resolve()
    return base


def _remote_origin_url(repo_root: Path) -> Optional[str]:
    proc = _git_cmd_optional(repo_root, ["config", "--get", "remote.origin.url"], text=True)
    if not proc:
        return None
    out = proc.stdout.strip()
    return out or None


def _default_repo_id(repo_root: Path) -> str:
    source = _remote_origin_url(repo_root)
    if source:
        basis = f"remote:{source}"
    else:
        basis = f"path:{repo_root}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()


def load_runtime_context(start: Optional[Path] = None) -> RuntimeContext:
    repo_root = find_repo_root(start)

    org_id = os.getenv("CODE_SIM_ORG_ID", "local").strip() or "local"
    repo_id = os.getenv("CODE_SIM_REPO_ID", "").strip() or _default_repo_id(repo_root)
    db_path = Path(
        os.getenv("CODE_SIM_DB_PATH", str(Path.home() / ".code-sim" / "chroma"))
    ).expanduser().resolve()

    base_collection = os.getenv("CODE_SIM_COLLECTION", "project_code").strip() or "project_code"
    private_collection_name = (
        os.getenv("CODE_SIM_PRIVATE_COLLECTION", f"{base_collection}_private").strip()
        or f"{base_collection}_private"
    )
    public_collection_name = (
        os.getenv("CODE_SIM_PUBLIC_COLLECTION", f"{base_collection}_public").strip()
        or f"{base_collection}_public"
    )
    metric = os.getenv("CODE_SIM_METRIC", "cosine").strip() or "cosine"

    return RuntimeContext(
        repo_root=repo_root,
        repo_name=repo_root.name,
        repo_id=repo_id,
        org_id=org_id,
        db_path=db_path,
        private_collection_name=private_collection_name,
        public_collection_name=public_collection_name,
        metric=metric,
    )


def read_index_blob(repo_root: Path, rel_path: str) -> Optional[bytes]:
    proc = _git_cmd_optional(repo_root, ["show", f":{rel_path}"], text=False)
    if not proc:
        return None
    return proc.stdout


def staged_added_modified_renamed(repo_root: Path) -> List[str]:
    proc = _git_cmd_optional(
        repo_root,
        ["diff", "--cached", "--name-only", "--diff-filter=AMR", "-z"],
        text=False,
    )
    if not proc:
        return []

    rel_paths = [p.decode("utf-8") for p in proc.stdout.split(b"\x00") if p]
    return sorted(set(rel_paths))


def parse_staged_new_ranges(diff_text: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []

    for line in diff_text.splitlines():
        match = _HUNK_RE.match(line)
        if not match:
            continue

        start = int(match.group("start"))
        count_raw = match.group("count")
        count = int(count_raw) if count_raw is not None else 1
        if count <= 0:
            end = start
        else:
            end = start + count - 1

        ranges.append((start, end))

    if not ranges:
        return []

    ranges.sort()
    merged: List[Tuple[int, int]] = [ranges[0]]
    for start, end in ranges[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def staged_hunk_line_ranges(repo_root: Path, rel_path: str) -> List[Tuple[int, int]]:
    proc = _git_cmd_optional(
        repo_root,
        ["diff", "--cached", "--unified=0", "--no-color", "--", rel_path],
        text=True,
    )
    if not proc:
        return []
    return parse_staged_new_ranges(proc.stdout)


def iter_repo_source_files(repo_root: Path, matcher) -> List[Path]:
    try:
        max_file_bytes = max(1, int(os.getenv("CODE_SIM_MAX_FILE_BYTES", "250000")))
    except ValueError:
        max_file_bytes = 250000
    out: List[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if not matcher.allows(path, is_dir=False):
            continue
        if not has_language_hint(path):
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                continue
            if is_probably_binary(path.read_bytes()[:4096]):
                continue
        except OSError:
            continue
        out.append(path)
    return sorted(out)
