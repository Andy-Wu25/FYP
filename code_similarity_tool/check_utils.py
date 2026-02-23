from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .code_parser import CodeElement, extract_code_elements
from .ignore import load_ignore_file
from .language_detection import has_language_hint, is_probably_binary
from .runtime import (
    RuntimeContext,
    iter_repo_source_files,
    load_runtime_context,
    read_index_blob,
    staged_added_modified_renamed,
    staged_hunk_line_ranges,
)

HitFilter = Callable[[Dict, str, str], bool]
QUERY_SCOPES = ("staged", "files", "repo")


def configure_logging() -> logging.Logger:
    log_level = os.getenv("CODE_SIM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="[%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def resolve_optional_paths(raw_paths: List[str], repo_root: Path) -> List[str]:
    rels: List[str] = []
    for raw in raw_paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()

        try:
            rels.append(str(candidate.relative_to(repo_root)))
        except ValueError:
            continue
    return rels


def add_scope_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        choices=QUERY_SCOPES,
        default="staged",
        help=(
            "Query scope: staged (default, staged hunks only), files (whole provided files), "
            "repo (whole repository)."
        ),
    )


def validate_scope_args(parser: argparse.ArgumentParser, args) -> None:
    if args.scope == "files" and not args.paths:
        parser.error("--scope files requires at least one path.")
    if args.scope == "repo" and args.paths:
        parser.error("--scope repo does not accept explicit paths.")


def _max_file_bytes() -> int:
    try:
        return max(1, int(os.getenv("CODE_SIM_MAX_FILE_BYTES", "250000")))
    except ValueError:
        return 250000


def collect_staged_query_elements(repo_root: Path, rel_paths: List[str]) -> List[Tuple[str, CodeElement]]:
    matcher = load_ignore_file(repo_root)
    max_file_bytes = _max_file_bytes()
    out: List[Tuple[str, CodeElement]] = []

    for rel_path in rel_paths:
        abs_path = (repo_root / rel_path).resolve()
        if ".git" in abs_path.parts:
            continue
        if not matcher.allows(abs_path, is_dir=False):
            continue
        if not has_language_hint(abs_path):
            continue

        staged_content = read_index_blob(repo_root, rel_path)
        if staged_content is None:
            continue
        if len(staged_content) > max_file_bytes:
            continue
        if is_probably_binary(staged_content[:4096]):
            continue

        changed_ranges = staged_hunk_line_ranges(repo_root, rel_path)
        if not changed_ranges:
            continue

        for element in extract_code_elements(Path(rel_path), staged_content):
            if not _element_overlaps_any_changed_range(
                element["start_line"], element["end_line"], changed_ranges
            ):
                continue
            out.append((rel_path, element))

    return out


def collect_full_file_query_elements(repo_root: Path, rel_paths: List[str]) -> List[Tuple[str, CodeElement]]:
    matcher = load_ignore_file(repo_root)
    max_file_bytes = _max_file_bytes()
    out: List[Tuple[str, CodeElement]] = []

    for rel_path in rel_paths:
        abs_path = (repo_root / rel_path).resolve()
        if not abs_path.is_file():
            continue
        if ".git" in abs_path.parts:
            continue
        if not matcher.allows(abs_path, is_dir=False):
            continue
        if not has_language_hint(abs_path):
            continue

        try:
            if abs_path.stat().st_size > max_file_bytes:
                continue
            content = abs_path.read_bytes()
        except OSError:
            continue

        if is_probably_binary(content[:4096]):
            continue

        for element in extract_code_elements(Path(rel_path), content):
            out.append((rel_path, element))

    return out


def _element_overlaps_any_changed_range(
    start_line: int, end_line: int, changed_ranges: List[Tuple[int, int]]
) -> bool:
    for changed_start, changed_end in changed_ranges:
        if changed_end < start_line:
            continue
        if changed_start > end_line:
            continue
        return True
    return False


def extract_hits(
    results: Dict,
    *,
    top_k: int,
    max_distance: float | None,
    query_rel: str,
    query_hash: str,
    include_hit: HitFilter,
) -> List[Tuple[float, Dict]]:
    ids = (results.get("ids") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    hits: List[Tuple[float, Dict]] = []
    for _, dist, meta in zip(ids, dists, metas):
        if not isinstance(meta, dict):
            continue

        if not include_hit(meta, query_rel, query_hash):
            continue

        if max_distance is not None and isinstance(dist, (int, float)) and dist > max_distance:
            continue

        if not isinstance(dist, (int, float)):
            continue

        hits.append((float(dist), meta))
        if len(hits) >= top_k:
            break

    return hits


def query_elements_from_args(paths: List[str], scope: str) -> Tuple[RuntimeContext, List[str], List[Tuple[str, CodeElement]]]:
    """Return (runtime_context, rel_paths, query_elements) for requested scope."""
    ctx = load_runtime_context()

    if scope == "staged":
        staged = set(staged_added_modified_renamed(ctx.repo_root))
        requested = set(resolve_optional_paths(paths, ctx.repo_root))
        if requested:
            rel_paths = sorted(staged.intersection(requested))
        else:
            rel_paths = sorted(staged)
        query_elements = collect_staged_query_elements(ctx.repo_root, rel_paths)
        return ctx, rel_paths, query_elements

    if scope == "files":
        rel_paths = sorted(set(resolve_optional_paths(paths, ctx.repo_root)))
        query_elements = collect_full_file_query_elements(ctx.repo_root, rel_paths)
        return ctx, rel_paths, query_elements

    if scope == "repo":
        matcher = load_ignore_file(ctx.repo_root)
        rel_paths = [str(path.relative_to(ctx.repo_root)) for path in iter_repo_source_files(ctx.repo_root, matcher)]
        query_elements = collect_full_file_query_elements(ctx.repo_root, rel_paths)
        return ctx, rel_paths, query_elements

    raise ValueError(f"Unsupported scope '{scope}'.")
