#!/usr/bin/env python3
"""CLI entry point: code-sim-stats

Display a statistics dashboard for the private and public code similarity
indexes stored in ChromaDB, with optional JSON export.

Usage:
    code-sim-stats [--json FILE] [--no-color]
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..infra.clients import CodeVectorStore
from ..core.runtime import load_runtime_context

# ── ANSI codes ─────────────────────────────────────────────────────────────────
_BOLD  = "\033[1m"
_RESET = "\033[0m"
_CYAN  = "\033[96m"
_GREEN = "\033[92m"
_DIM   = "\033[2m"

_W = 76  # visual line width

# ── Language lookup ─────────────────────────────────────────────────────────────
_LANG_EXT: Dict[str, str] = {
    ".py": "Python",       ".pyw": "Python",
    ".js": "JavaScript",   ".mjs": "JavaScript",  ".jsx": "JavaScript",
    ".ts": "TypeScript",   ".tsx": "TypeScript",
    ".java": "Java",       ".kt": "Kotlin",
    ".go": "Go",           ".rs": "Rust",
    ".cpp": "C++",         ".cc": "C++",          ".cxx": "C++",
    ".c": "C",             ".h": "C/C++",
    ".cs": "C#",           ".rb": "Ruby",         ".php": "PHP",
    ".swift": "Swift",     ".scala": "Scala",
    ".r": "R",             ".sh": "Shell",        ".bash": "Shell",
    ".sql": "SQL",         ".lua": "Lua",         ".dart": "Dart",
    ".vue": "Vue",         ".html": "HTML",
}


def _lang(path: str) -> str:
    return _LANG_EXT.get(Path(path).suffix.lower(), "Other")


def _vis_len(s: str) -> int:
    """Length of string ignoring ANSI CSI and OSC (hyperlink) escape sequences."""
    # Strip CSI sequences: ESC [ ... m  (colours, bold, etc.)
    s = re.sub(r"\033\[[^a-zA-Z]*[a-zA-Z]", "", s)
    # Strip OSC sequences: ESC ] ... ST  (hyperlinks; ST = ESC\ or BEL)
    s = re.sub(r"\033\][^\033\007]*(?:\033\\|\007)", "", s)
    return len(s)


def _rpad(s: str, width: int) -> str:
    """Right-pad string to visual width, accounting for ANSI codes."""
    return s + " " * max(0, width - _vis_len(s))


def _hyperlink(url: str, text: str) -> str:
    """Wrap text in an OSC 8 terminal hyperlink (supported by iTerm2, etc.)."""
    return f"\033]8;;{url}\033\\{text}\033]8;;\033\\"


def _github_url(repo_name: str) -> Optional[str]:
    """Return a GitHub URL if repo_name looks like owner/repo, else None."""
    parts = repo_name.split("/")
    if len(parts) == 2 and all(parts):
        return f"https://github.com/{repo_name}"
    return None


def _bar(fraction: float, width: int = 28) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024  # type: ignore[assignment]
    return f"{n_bytes:.1f} TB"


# ── Data fetching ──────────────────────────────────────────────────────────────

def _get_all_metadatas(collection, where: Optional[Dict] = None) -> List[Dict]:
    """Fetch every metadata record from a ChromaDB collection (paginated)."""
    page_size = 5_000
    offset = 0
    all_metas: List[Dict] = []
    while True:
        kw: Dict[str, Any] = {
            "include": ["metadatas"],
            "limit": page_size,
            "offset": offset,
        }
        if where:
            kw["where"] = where
        result = collection.get(**kw)
        metas: List[Dict] = result.get("metadatas") or []
        all_metas.extend(metas)
        if len(metas) < page_size:
            break
        offset += page_size
    return all_metas


def _compute_stats(metas: List[Dict]) -> Dict[str, Any]:
    repos = Counter(m.get("repo_name", "<unknown>") for m in metas)
    files = len({(m.get("repo_name", ""), m.get("file_path", "")) for m in metas})
    kinds = Counter(m.get("kind", "function") for m in metas)
    langs = Counter(_lang(m.get("file_path", "")) for m in metas)
    lics = Counter(
        (m.get("license") or m.get("license_spdx") or "Unknown").upper()
        for m in metas
    )
    repo_commits: Dict[str, str] = {}
    for m in metas:
        rn = m.get("repo_name", "")
        if rn and rn not in repo_commits:
            commit = m.get("source_commit", "")
            if commit:
                repo_commits[rn] = str(commit)
    return {
        "total": len(metas),
        "repos": repos,
        "files": files,
        "kinds": kinds,
        "langs": langs,
        "licenses": lics,
        "repo_commits": repo_commits,
    }


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _dist_rows(
    counter: Counter,
    top_n: int = 8,
    name_w: int = 22,
    bar_w: int = 26,
    make_url: Optional[Any] = None,
) -> List[str]:
    """Format a Counter as aligned bar-chart rows."""
    total = sum(counter.values()) or 1
    rows = []
    for name, count in counter.most_common(top_n):
        short = name if len(name) <= name_w else name[: name_w - 1] + "\u2026"
        if make_url:
            url = make_url(name)
            if url:
                short = _hyperlink(url, short)
        frac = count / total
        rows.append(
            f"  {_rpad(short, name_w)}  {_bar(frac, bar_w)}  {frac * 100:5.1f}%  {count:,}"
        )
    return rows


def _section_header(label: str, count: int) -> str:
    badge = f"  {_BOLD}{_GREEN}{count:,} elements{_RESET}  "
    inner = f"  {_BOLD}{_CYAN}{label}{_RESET}  \u00b7"
    sep_len = _W - _vis_len(inner) - _vis_len(badge)
    return "\n" + "\u2501" * 4 + inner + "\u2501" * max(0, sep_len) + badge


def _sub_header(label: str) -> str:
    line = f"  \u2500\u2500 {_CYAN}{label}{_RESET} "
    pad = _W - _vis_len(line) - 1
    return line + "\u2500" * max(0, pad)


def _summary_row(stats: Dict) -> str:
    n_repos = len(stats["repos"])
    avg_repo = stats["total"] / max(n_repos, 1)
    avg_file = stats["total"] / max(stats["files"], 1)
    return (
        f"  Repos   {_BOLD}{n_repos:>4}{_RESET}   "
        f"Files  {_BOLD}{stats['files']:>5}{_RESET}   "
        f"Avg/repo  {_BOLD}{avg_repo:>7.1f}{_RESET}   "
        f"Avg/file  {_BOLD}{avg_file:>6.1f}{_RESET}"
    )


def _render_index_block(stats: Dict, show_licenses: bool = False) -> List[str]:
    lines: List[str] = []

    if stats["total"] == 0:
        lines.append(
            f"\n  {_DIM}Index is empty. "
            + ("Run code-sim-index-public to populate." if show_licenses
               else "Run code-sim-index to populate.")
            + _RESET
        )
        return lines

    lines.append("")
    lines.append(_summary_row(stats))
    lines.append("")

    repo_commits = stats.get("repo_commits", {})

    def _make_repo_url(name: str) -> Optional[str]:
        parts = name.split("/")
        if len(parts) != 2 or not all(parts):
            return None
        commit = repo_commits.get(name)
        if commit:
            return f"https://github.com/{name}/tree/{commit}"
        return _github_url(name)

    # Repositories
    lines.append(_sub_header("Repositories"))
    lines.extend(_dist_rows(stats["repos"], top_n=len(stats["repos"]), make_url=_make_repo_url))
    lines.append("")

    # Licenses (public only) or Element Kinds (private)
    if show_licenses and stats["licenses"]:
        lines.append(_sub_header("Licenses"))
        lines.extend(_dist_rows(stats["licenses"], top_n=6))
        lines.append("")

    # Element Kinds
    lines.append(_sub_header("Element Kinds"))
    lines.extend(_dist_rows(stats["kinds"], top_n=5, name_w=14, bar_w=22))
    lines.append("")

    # Languages
    lines.append(_sub_header("Languages"))
    lines.extend(_dist_rows(stats["langs"], top_n=6, name_w=14, bar_w=22))

    return lines


# ── Top-level render ───────────────────────────────────────────────────────────

def render_dashboard(
    private_stats: Dict,
    public_stats: Dict,
    db_path: str,
    org_id: str,
) -> str:
    lines: List[str] = []
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    sqlite_path = Path(db_path) / "chroma.sqlite3"
    db_size = _fmt_size(sqlite_path.stat().st_size) if sqlite_path.exists() else "N/A"
    db_name = Path(db_path).name

    # ── Header box ──────────────────────────────────────────────────────────────
    inner_w = _W - 2
    lines.append("\u2554" + "\u2550" * inner_w + "\u2557")

    title = "CODE SIMILARITY INDEX  \u00b7  STATISTICS DASHBOARD"
    pad_l = (inner_w - len(title)) // 2
    pad_r = inner_w - len(title) - pad_l
    lines.append("\u2551" + " " * pad_l + _BOLD + title + _RESET + " " * pad_r + "\u2551")

    lines.append("\u2560" + "\u2550" * inner_w + "\u2563")
    lines.append("\u2551  " + f"Database   {db_name:<36} Size       {db_size:<10}" .ljust(inner_w - 2) + "  \u2551")
    lines.append("\u2551  " + f"Org        {org_id:<36} Generated  {now}" .ljust(inner_w - 2) + "  \u2551")
    lines.append("\u255a" + "\u2550" * inner_w + "\u255d")

    # ── Private index ───────────────────────────────────────────────────────────
    lines.append(_section_header("PRIVATE INDEX", private_stats["total"]))
    lines.extend(_render_index_block(private_stats, show_licenses=False))

    # ── Public index ────────────────────────────────────────────────────────────
    lines.append(_section_header("PUBLIC INDEX", public_stats["total"]))
    lines.extend(_render_index_block(public_stats, show_licenses=True))

    # ── Footer ──────────────────────────────────────────────────────────────────
    total = private_stats["total"] + public_stats["total"]
    lines.append("")
    lines.append("\u2501" * _W)
    if total > 0:
        priv_pct = private_stats["total"] / total * 100
        pub_pct  = public_stats["total"]  / total * 100
        summary = (
            f"  Total  {_BOLD}{_GREEN}{total:,} elements{_RESET}"
            f"  \u00b7  Private {priv_pct:.1f}%"
            f"  \u00b7  Public {pub_pct:.1f}%"
            f"  \u00b7  DB {db_size}"
        )
    else:
        summary = f"  {_DIM}No elements indexed yet.{_RESET}"
    lines.append(summary)
    lines.append("\u2501" * _W)

    return "\n".join(lines)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-stats",
        description="Display a statistics dashboard for the code similarity indexes.",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        nargs="?",
        const="-",
        default=None,
        help="Write statistics as JSON. Omit FILE to print to stdout, or provide a path to write to a file.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output.",
    )
    args = parser.parse_args()

    if args.no_color:
        global _BOLD, _RESET, _CYAN, _GREEN, _DIM
        _BOLD = _RESET = _CYAN = _GREEN = _DIM = ""

    ctx = load_runtime_context()
    store = CodeVectorStore(
        path=str(ctx.db_path),
        private_collection_name=ctx.private_collection_name,
        public_collection_name=ctx.public_collection_name,
        metric=ctx.metric,
    )

    private_metas = _get_all_metadatas(
        store.private_collection,
        where={"org_id": ctx.org_id} if ctx.org_id else None,
    )
    public_metas = _get_all_metadatas(store.public_collection)

    private_stats = _compute_stats(private_metas)
    public_stats  = _compute_stats(public_metas)

    print(render_dashboard(private_stats, public_stats, str(ctx.db_path), ctx.org_id))

    if args.json:
        sqlite_path = Path(ctx.db_path) / "chroma.sqlite3"
        data: Dict[str, Any] = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "org_id": ctx.org_id,
            "database": {
                "path": str(ctx.db_path),
                "size_bytes": sqlite_path.stat().st_size if sqlite_path.exists() else None,
            },
            "private_index": {
                "total_elements": private_stats["total"],
                "total_repos": len(private_stats["repos"]),
                "total_files": private_stats["files"],
                "avg_elements_per_repo": round(
                    private_stats["total"] / max(len(private_stats["repos"]), 1), 2
                ),
                "avg_elements_per_file": round(
                    private_stats["total"] / max(private_stats["files"], 1), 2
                ),
                "by_repo":     dict(private_stats["repos"].most_common()),
                "by_kind":     dict(private_stats["kinds"].most_common()),
                "by_language": dict(private_stats["langs"].most_common()),
            },
            "public_index": {
                "total_elements": public_stats["total"],
                "total_repos": len(public_stats["repos"]),
                "total_files": public_stats["files"],
                "avg_elements_per_repo": round(
                    public_stats["total"] / max(len(public_stats["repos"]), 1), 2
                ),
                "avg_elements_per_file": round(
                    public_stats["total"] / max(public_stats["files"], 1), 2
                ),
                "by_repo":     dict(public_stats["repos"].most_common()),
                "by_license":  dict(public_stats["licenses"].most_common()),
                "by_kind":     dict(public_stats["kinds"].most_common()),
                "by_language": dict(public_stats["langs"].most_common()),
            },
        }
        out = json.dumps(data, indent=2)
        if args.json == "-":
            print(out)
        else:
            Path(args.json).write_text(out, encoding="utf-8")
            print(f"\nJSON written \u2192 {args.json}")


if __name__ == "__main__":
    main()
