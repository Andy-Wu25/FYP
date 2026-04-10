from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ── ANSI codes ─────────────────────────────────────────────────────────────────
_BOLD   = "\033[1m"
_RESET  = "\033[0m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_DIM    = "\033[2m"

_W = 76  # dashboard line width


def disable_color() -> None:
    global _BOLD, _RESET, _CYAN, _GREEN, _YELLOW, _RED, _DIM
    _BOLD = _RESET = _CYAN = _GREEN = _YELLOW = _RED = _DIM = ""


def _vis_len(s: str) -> int:
    """String length ignoring ANSI CSI and OSC escape sequences."""
    s = re.sub(r"\033\[[^a-zA-Z]*[a-zA-Z]", "", s)
    s = re.sub(r"\033\][^\033\007]*(?:\033\\|\007)", "", s)
    return len(s)


def _dist_badge(d: float) -> str:
    if d < 0.15:
        return f"{_RED}{_BOLD}{d:.4f}{_RESET}"
    if d < 0.35:
        return f"{_YELLOW}{_BOLD}{d:.4f}{_RESET}"
    return f"{_BOLD}{d:.4f}{_RESET}"


def fmt_header_box(prog: str, scope: str, org_id: str, n_elements: int) -> str:
    inner_w = _W - 2
    title = "CODE SIMILARITY CHECK"
    pad_l = (inner_w - len(title)) // 2
    pad_r = inner_w - len(title) - pad_l
    return "\n".join([
        "╔" + "═" * inner_w + "╗",
        "║" + " " * pad_l + _BOLD + title + _RESET + " " * pad_r + "║",
        "╠" + "═" * inner_w + "╣",
        "║  " + f"Command    {prog:<28} Scope      {scope}".ljust(inner_w - 2) + "  ║",
        "║  " + f"Org        {org_id:<28} Elements   {n_elements}".ljust(inner_w - 2) + "  ║",
        "╚" + "═" * inner_w + "╝",
    ])


def fmt_query_header(name: str, file: str, start: int, end: int) -> str:
    inner = f"  {_BOLD}{_CYAN}{name}{_RESET}  ·"
    badge = f"  {_DIM}{file}  {start}–{end}{_RESET}  "
    sep_len = _W - _vis_len(inner) - _vis_len(badge)
    return "\n" + "━" * 4 + inner + "━" * max(0, sep_len) + badge


def fmt_check_sub_header(label: str, count: Optional[int] = None) -> str:
    if count is None:
        count_str = ""
    elif count == 0:
        count_str = f"  {_DIM}no matches{_RESET}"
    else:
        count_str = f"  {_DIM}{count} match{'es' if count != 1 else ''}{_RESET}"
    line = f"  ── {_CYAN}{label}{_RESET}{count_str} "
    pad = _W - _vis_len(line) - 1
    return line + "─" * max(0, pad)


def fmt_private_hit(idx: int, distance: float, meta: Dict) -> str:
    repo = meta.get("repo_name", "<unknown>")
    file = meta.get("file_path", "<unknown>")
    func = meta.get("function_name", "<unknown>")
    sl   = meta.get("start_line", "?")
    el   = meta.get("end_line", "?")
    return (
        f"  #{idx}  {_dist_badge(distance)}   "
        f"{_BOLD}{repo}{_RESET}  ›  {file}:{sl}–{el}  ›  {func}"
    )


def fmt_public_hit(
    idx: int,
    distance: float,
    meta: Dict,
    permalink: Optional[str],
    file_commit_url: Optional[str],
    commit_url: Optional[str],
    license_display: str,
) -> List[str]:
    repo = meta.get("repo_name", "<unknown>")
    file = meta.get("file_path", "<unknown>")
    func = meta.get("function_name", "<unknown>")
    sl   = meta.get("start_line", "?")
    el   = meta.get("end_line", "?")
    if license_display and license_display != "<unknown>":
        lic_str = f"  {_GREEN}[{license_display}]{_RESET}"
    else:
        lic_str = f"  {_DIM}[?]{_RESET}"
    rows = [
        f"  #{idx}  {_dist_badge(distance)}{lic_str}   "
        f"{_BOLD}{repo}{_RESET}  ›  {file}:{sl}–{el}  ›  {func}"
    ]
    pad = "             "
    if permalink:
        rows.append(f"{pad}{_DIM}↳{_RESET}  {permalink}")
    if file_commit_url:
        rows.append(f"{pad}{_DIM}↳  file commit:{_RESET}  {file_commit_url}")
    if commit_url:
        rows.append(f"{pad}{_DIM}↳  commit:{_RESET}  {commit_url}")
    return rows


def fmt_footer(parts: List[str]) -> str:
    summary = "  " + f"  ·  ".join(parts)
    return "\n" + "━" * _W + "\n" + summary + "\n" + "━" * _W

from ..core.code_parser import CodeElement, extract_code_elements
from ..core.ignore import load_ignore_file
from ..core.language_detection import has_language_hint, is_probably_binary
from ..core.runtime import (
    RuntimeContext,
    iter_repo_source_files,
    load_runtime_context,
    read_index_blob,
    staged_added_modified_renamed,
    staged_hunk_line_ranges,
)
from ..core.source_filtering import is_noise_source_path

HitFilter = Callable[[Dict, str, str], bool]
QUERY_SCOPES = ("staged", "files", "directory", "repo")


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
            raise SystemExit(
                f"Error: '{raw}' is outside the current repository ({repo_root})."
            )
    return rels


def add_scope_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--scope",
        choices=QUERY_SCOPES,
        default="staged",
        help=(
            "Query scope: staged (default, staged hunks only), files (whole provided files), "
            "directory (current working directory or provided directory path), "
            "repo (whole repository)."
        ),
    )


def validate_scope_args(parser: argparse.ArgumentParser, args) -> None:
    if args.scope == "files" and not args.paths:
        parser.error("--scope files requires at least one path.")
    if args.scope == "directory" and args.paths and len(args.paths) > 1:
        parser.error("--scope directory accepts at most one directory path.")
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
        if not matcher.allows(abs_path, is_dir=False):
            continue
        if is_noise_source_path(abs_path):
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
        if not matcher.allows(abs_path, is_dir=False):
            continue
        if is_noise_source_path(abs_path):
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


def _hit_line_count(meta: Dict) -> int | None:
    """Return the line count of a hit from its metadata, or None if unknown."""
    start = meta.get("start_line")
    end = meta.get("end_line")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        return int(end) - int(start) + 1
    return None


def extract_hits(
    results: Dict,
    *,
    top_k: int,
    max_distance: float | None,
    query_rel: str,
    query_hash: str,
    include_hit: HitFilter,
    min_lines: int = 0,
) -> List[Tuple[float, Dict, str]]:
    """Return list of (distance, metadata, document_text) tuples.

    *document_text* is the stored source code when available, empty string otherwise.
    """
    ids = (results.get("ids") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    docs = (results.get("documents") or [[]])[0]

    hits: List[Tuple[float, Dict, str]] = []
    for idx, (_, dist, meta) in enumerate(zip(ids, dists, metas)):
        if not isinstance(meta, dict):
            continue

        if not include_hit(meta, query_rel, query_hash):
            continue

        if max_distance is not None and isinstance(dist, (int, float)) and dist > max_distance:
            continue

        if not isinstance(dist, (int, float)):
            continue

        if min_lines > 0:
            lc = _hit_line_count(meta)
            if lc is not None and lc < min_lines:
                continue

        doc = docs[idx] if idx < len(docs) and docs[idx] else ""
        hits.append((float(dist), meta, doc))
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

    if scope == "directory":
        log = logging.getLogger(__name__)
        if paths:
            target = Path(paths[0])
            if not target.is_absolute():
                target = (Path.cwd() / target).resolve()
            else:
                target = target.resolve()
        else:
            target = Path.cwd().resolve()
        if not target.is_dir():
            raise ValueError(f"Not a directory: {target}")
        if not target.is_relative_to(ctx.repo_root):
            raise SystemExit(
                f"Error: '{target}' is outside the current repository ({ctx.repo_root})."
            )
        matcher = load_ignore_file(ctx.repo_root)
        rel_paths = [
            str(path.relative_to(ctx.repo_root))
            for path in iter_repo_source_files(target, matcher)
        ]
        query_elements = collect_full_file_query_elements(ctx.repo_root, rel_paths)
        return ctx, rel_paths, query_elements

    if scope == "repo":
        matcher = load_ignore_file(ctx.repo_root)
        rel_paths = [str(path.relative_to(ctx.repo_root)) for path in iter_repo_source_files(ctx.repo_root, matcher)]
        query_elements = collect_full_file_query_elements(ctx.repo_root, rel_paths)
        return ctx, rel_paths, query_elements

    raise ValueError(f"Unsupported scope '{scope}'.")
