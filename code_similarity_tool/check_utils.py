from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from .code_parser import CodeElement, extract_code_elements
from .ignore import load_ignore_file
from .runtime import (
    RuntimeContext,
    SUPPORTED_SUFFIXES,
    load_runtime_context,
    read_index_blob,
    staged_added_modified_renamed,
    staged_hunk_line_ranges,
)

HitFilter = Callable[[Dict, str, str], bool]


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


def collect_staged_query_elements(repo_root: Path, rel_paths: List[str]) -> List[Tuple[str, CodeElement]]:
    matcher = load_ignore_file(repo_root)
    out: List[Tuple[str, CodeElement]] = []

    for rel_path in rel_paths:
        abs_path = (repo_root / rel_path).resolve()
        if abs_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if ".git" in abs_path.parts:
            continue
        if not matcher.allows(abs_path, is_dir=False):
            continue

        staged_content = read_index_blob(repo_root, rel_path)
        if staged_content is None:
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


def staged_elements_from_args(paths: List[str]) -> Tuple[RuntimeContext, List[str], List[Tuple[str, CodeElement]]]:
    """Return (runtime_context, rel_paths, staged_query_elements)."""
    ctx = load_runtime_context()

    staged = set(staged_added_modified_renamed(ctx.repo_root))
    staged.update(resolve_optional_paths(paths, ctx.repo_root))

    rel_paths = sorted(staged)
    query_elements = collect_staged_query_elements(ctx.repo_root, rel_paths)
    return ctx, rel_paths, query_elements
