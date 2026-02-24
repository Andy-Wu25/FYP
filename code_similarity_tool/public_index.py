from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .clients import CodeVectorStore
from .code_parser import CodeElement, extract_code_elements
from .embeddings import EmbeddingClient
from .ignore import load_ignore_file
from .language_detection import has_language_hint, is_probably_binary
from .public_links import build_github_blob_url, build_github_commit_url
from .runtime import load_runtime_context
from .source_filtering import is_noise_source_path


GITHUB_URL_RE = re.compile(
    r"^https://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+?)(?:\.git)?/?$"
)

GNU_PREFIX_ALLOWLIST = ("GPL-", "LGPL-", "AGPL-")
LICENSE_CANDIDATE_NAMES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "COPYING",
    "COPYING.txt",
    "COPYING.md",
)


def _configure_logging() -> logging.Logger:
    log_level = os.getenv("CODE_SIM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="[%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def parse_github_url(url: str) -> Tuple[str, str, str]:
    raw = url.strip()
    match = GITHUB_URL_RE.match(raw)
    if not match:
        raise ValueError(
            "URL must be a valid GitHub repository URL in the form "
            "https://github.com/<owner>/<repo>[.git]"
        )

    owner = match.group("owner")
    repo = match.group("repo")
    canonical = f"https://github.com/{owner}/{repo}"
    return owner, repo, canonical


def _run_git(args: List[str], cwd: Optional[Path] = None, *, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        check=True,
        text=text,
    )


def _get_file_last_commit(repo_dir: Path, rel_path: str, fallback: str) -> str:
    """Return the SHA of the last commit that modified *rel_path* inside *repo_dir*.

    Falls back to *fallback* (the repo HEAD SHA) if git produces no output or
    raises (e.g. shallow clones without full history, or test environments).
    """
    try:
        result = _run_git(["log", "-1", "--format=%H", "--", rel_path], cwd=repo_dir)
        sha = result.stdout.strip()
        return sha if sha else fallback
    except Exception:  # noqa: BLE001
        return fallback


def resolve_remote_commit(url: str, ref: Optional[str]) -> str:
    if ref:
        out = _run_git(["ls-remote", url, ref], text=True).stdout.strip().splitlines()
        if not out:
            raise RuntimeError(f"Unable to resolve ref '{ref}' for {url}")
        first = out[0].split()
        if not first:
            raise RuntimeError(f"Unexpected ls-remote output for ref '{ref}' and url {url}")
        return first[0]

    head_line = _run_git(["ls-remote", url, "HEAD"], text=True).stdout.strip().splitlines()
    if not head_line:
        raise RuntimeError(f"Unable to resolve HEAD for {url}")
    first = head_line[0].split()
    if not first:
        raise RuntimeError(f"Unexpected ls-remote output for HEAD and url {url}")
    return first[0]


def clone_repo_at_commit(url: str, commit_sha: str, repo_dir: Path) -> None:
    _run_git(["clone", "--quiet", "--no-checkout", url, str(repo_dir)], text=True)
    _run_git(["-C", str(repo_dir), "checkout", "--quiet", commit_sha], text=True)


def _iter_license_candidates(repo_root: Path) -> Iterable[Path]:
    for name in LICENSE_CANDIDATE_NAMES:
        candidate = repo_root / name
        if candidate.is_file():
            yield candidate

    for candidate in repo_root.iterdir():
        if not candidate.is_file():
            continue
        upper = candidate.name.upper()
        if upper.startswith("LICENSE") or upper.startswith("COPYING"):
            yield candidate


def _read_small_text(path: Path, max_bytes: int = 128_000) -> str:
    data = path.read_bytes()[:max_bytes]
    return data.decode("utf-8", errors="replace")


def detect_license_spdx(repo_root: Path) -> Optional[str]:
    # First pass: explicit SPDX identifier in license files
    spdx_re = re.compile(r"SPDX-License-Identifier:\s*([A-Za-z0-9+_.-]+)", re.IGNORECASE)
    for path in _iter_license_candidates(repo_root):
        text = _read_small_text(path)
        match = spdx_re.search(text)
        if not match:
            continue
        return match.group(1).upper()

    # Second pass: heuristic GNU family detection
    for path in _iter_license_candidates(repo_root):
        text = _read_small_text(path).upper()

        if "GNU AFFERO GENERAL PUBLIC LICENSE" in text:
            if "VERSION 3" in text:
                return "AGPL-3.0"
            return "AGPL-UNKNOWN"

        if "GNU LESSER GENERAL PUBLIC LICENSE" in text:
            if "VERSION 3" in text:
                return "LGPL-3.0"
            if "VERSION 2.1" in text:
                return "LGPL-2.1"
            return "LGPL-UNKNOWN"

        if "GNU GENERAL PUBLIC LICENSE" in text:
            if "VERSION 3" in text:
                return "GPL-3.0"
            if "VERSION 2" in text:
                return "GPL-2.0"
            return "GPL-UNKNOWN"

    return None


def ensure_gnu_license(repo_root: Path) -> str:
    spdx = detect_license_spdx(repo_root)
    if not spdx:
        raise RuntimeError("No detectable license found in repository root.")

    normalized = spdx.upper()
    if not normalized.startswith(GNU_PREFIX_ALLOWLIST):
        raise RuntimeError(
            f"License '{spdx}' is not in GNU allowlist ({', '.join(GNU_PREFIX_ALLOWLIST)})."
        )
    return normalized


def _public_source_id(canonical_url: str, commit_sha: str) -> str:
    raw = f"public:{canonical_url}@{commit_sha}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _public_element_id(public_source_id: str, rel_path: str, content_hash: str) -> str:
    raw = f"{public_source_id}:{rel_path}:{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _public_element_label(index_1_based: int, rel_path: str, element: CodeElement) -> str:
    return (
        f"#{index_1_based} file={rel_path} "
        f"name={element.get('name', '<unknown>')} "
        f"kind={element.get('kind', '<unknown>')} "
        f"lines={element.get('start_line', '?')}-{element.get('end_line', '?')} "
        f"chars={len(element.get('text', ''))}"
    )


def iter_public_repo_source_files(repo_root: Path) -> List[Path]:
    matcher = load_ignore_file(repo_root)
    try:
        max_file_bytes = max(1, int(os.getenv("CODE_SIM_MAX_FILE_BYTES", "250000")))
    except ValueError:
        max_file_bytes = 250000

    out: List[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if not matcher.allows(path, is_dir=False):
            continue
        if is_noise_source_path(path):
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


def index_public_github_repo(
    url: str,
    ref: Optional[str] = None,
    *,
    debug_element_index: Optional[int] = None,
) -> int:
    log = _configure_logging()
    ctx = load_runtime_context()

    owner, repo, canonical_url = parse_github_url(url)
    commit_sha = resolve_remote_commit(canonical_url, ref)
    public_source_id = _public_source_id(canonical_url, commit_sha)

    log.info("[public-index] source=%s commit=%s", canonical_url, commit_sha)
    log.info(
        "[public-index] db=%s private_collection=%s public_collection=%s",
        ctx.db_path,
        ctx.private_collection_name,
        ctx.public_collection_name,
    )

    all_elements: List[CodeElement] = []
    rel_paths_for_element: List[str] = []
    file_commits: Dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="code-sim-public-") as tmp:
        repo_dir = Path(tmp) / "repo"
        clone_repo_at_commit(canonical_url, commit_sha, repo_dir)

        license_id = ensure_gnu_license(repo_dir)
        files = iter_public_repo_source_files(repo_dir)

        for file_path in files:
            rel_path = str(file_path.relative_to(repo_dir))
            elements = extract_code_elements(file_path, file_path.read_bytes())
            for element in elements:
                element["id"] = _public_element_id(public_source_id, rel_path, element["hash"])

            if elements:
                all_elements.extend(elements)
                rel_paths_for_element.extend([rel_path] * len(elements))
                file_commits[rel_path] = _get_file_last_commit(repo_dir, rel_path, commit_sha)

    if debug_element_index is not None:
        if debug_element_index < 1 or debug_element_index > len(all_elements):
            raise RuntimeError(
                f"--debug-element-index must be between 1 and {len(all_elements)} for this source."
            )
        idx0 = debug_element_index - 1
        element = all_elements[idx0]
        rel_path = rel_paths_for_element[idx0]
        print(_public_element_label(debug_element_index, rel_path, element))
        preview = element.get("text", "")
        preview = preview[:800]
        print("----- element preview (first 800 chars) -----")
        print(preview)
        print("---------------------------------------------")
        return 0

    store = CodeVectorStore(
        path=str(ctx.db_path),
        private_collection_name=ctx.private_collection_name,
        public_collection_name=ctx.public_collection_name,
        metric=ctx.metric,
    )

    if not all_elements:
        removed = store.delete_public_source_entries(public_source_id)
        if removed:
            log.info("[public-index] removed %d stale element(s) for source id %s", removed, public_source_id)
        log.info("[public-index] no indexable functions/methods found.")
        return 0

    embedder = EmbeddingClient()
    labels = [
        _public_element_label(i + 1, rel_path, element)
        for i, (rel_path, element) in enumerate(zip(rel_paths_for_element, all_elements))
    ]
    vectors = embedder.embed_documents(
        [element["text"] for element in all_elements],
        labels=labels,
    )

    by_file_elements: Dict[str, List[CodeElement]] = defaultdict(list)
    by_file_vectors: Dict[str, List[List[float]]] = defaultdict(list)

    for element, vector, rel_path in zip(all_elements, vectors, rel_paths_for_element):
        by_file_elements[rel_path].append(element)
        by_file_vectors[rel_path].append(vector)

    total = 0
    for rel_path in by_file_elements:
        file_commit = file_commits.get(rel_path, commit_sha)
        store.upsert_public_code_elements(
            by_file_elements[rel_path],
            by_file_vectors[rel_path],
            base_metadata={
                "org_id": "public",
                "repo_id": public_source_id,
                "repo_name": f"{owner}/{repo}",
                "file_path": rel_path,
                "public_source_id": public_source_id,
                "source_url": canonical_url,
                "source_owner": owner,
                "source_repo": repo,
                "source_commit": commit_sha,
                "source_commit_url": build_github_commit_url(canonical_url, commit_sha),
                "source_file_url": build_github_blob_url(canonical_url, commit_sha, rel_path),
                "source_file_commit": file_commit,
                "license": license_id,
            },
        )
        total += len(by_file_elements[rel_path])

    current_ids = [element["id"] for element in all_elements]
    removed = store.delete_public_source_stale_entries(public_source_id, keep_ids=current_ids)
    if removed:
        log.info("[public-index] removed %d stale element(s) for source id %s", removed, public_source_id)

    log.info("[public-index] indexed %d code element(s) from %s@%s", total, canonical_url, commit_sha)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-index-public",
        description="Operator command: index GNU-licensed public GitHub repository into central DB.",
    )
    parser.add_argument("url", help="GitHub repository URL (https://github.com/<owner>/<repo>)")
    parser.add_argument("--ref", default=None, help="Optional git ref (branch/tag/sha). Defaults to HEAD.")
    parser.add_argument(
        "--debug-element-index",
        type=int,
        default=None,
        help=(
            "Print one collected element (1-based index in embedding order) and exit "
            "without indexing. Useful for diagnosing embedding failures."
        ),
    )
    args = parser.parse_args()

    try:
        index_public_github_repo(args.url, ref=args.ref, debug_element_index=args.debug_element_index)
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
