#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import difflib
import json
import re
from pathlib import Path
from typing import Callable, Dict, List

from .check_utils import (
    _DIM,
    _RESET,
    add_scope_argument,
    configure_logging,
    disable_color,
    extract_hits,
    fmt_check_sub_header,
    fmt_footer,
    fmt_header_box,
    fmt_public_hit,
    fmt_query_header,
    query_elements_from_args,
    validate_scope_args,
)
from .interactive import ResultEntry, launch_interactive_viewer
from ..infra.clients import CodeVectorStore
from ..infra.embeddings import EmbeddingClient, load_embedding_config
from ..infra.public_links import build_github_commit_url, build_public_commit_permalink, build_public_match_permalink


KNOWN_LICENSE_KEYWORDS = (
    "0BSD",
    "AFL-3.0",
    "AGPL-3.0",
    "APACHE-2.0",
    "ARTISTIC-2.0",
    "BSD-2-CLAUSE",
    "BSD-3-CLAUSE",
    "BSD-3-CLAUSE-CLEAR",
    "BSD-4-CLAUSE",
    "BSL-1.0",
    "CC",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC0-1.0",
    "ECL-2.0",
    "EPL-1.0",
    "EPL-2.0",
    "EUPL-1.1",
    "GPL",
    "GPL-2.0",
    "GPL-3.0",
    "ISC",
    "LGPL",
    "LGPL-2.1",
    "LGPL-3.0",
    "LPPL-1.3C",
    "MIT",
    "MPL-2.0",
    "MS-PL",
    "NCSA",
    "OFL-1.1",
    "OSL-3.0",
    "POSTGRESQL",
    "UNLICENSE",
    "WTFPL",
    "ZLIB",
)
_NORMALIZED_LICENSE_KEYWORDS = {
    re.sub(r"[^A-Z0-9]+", "", keyword): keyword for keyword in KNOWN_LICENSE_KEYWORDS
}


def _normalize_license_filters(raw_values: List[str] | None) -> List[str]:
    out: List[str] = []
    for raw in raw_values or []:
        for token in raw.split(","):
            value = token.strip().upper()
            if not value:
                continue
            if value not in out:
                out.append(value)
    return out


def _suggest_license_keywords(keyword: str, *, max_suggestions: int = 3) -> List[str]:
    value = keyword.strip().upper()
    if not value:
        return []
    suggestions: List[str] = []
    by_token = difflib.get_close_matches(value, KNOWN_LICENSE_KEYWORDS, n=max_suggestions, cutoff=0.55)
    suggestions.extend(by_token)
    normalized = re.sub(r"[^A-Z0-9]+", "", value)
    if normalized:
        normalized_candidates = difflib.get_close_matches(
            normalized, list(_NORMALIZED_LICENSE_KEYWORDS.keys()), n=max_suggestions, cutoff=0.65,
        )
        for candidate in normalized_candidates:
            mapped = _NORMALIZED_LICENSE_KEYWORDS.get(candidate)
            if mapped and mapped not in suggestions:
                suggestions.append(mapped)
    return suggestions[:max_suggestions]


def _public_hit_filter_for_licenses(licenses: List[str]) -> Callable[[Dict, str, str], bool]:
    allowed = set(licenses)

    def _include_public_hit(meta: Dict, query_rel: str, query_hash: str) -> bool:
        _ = query_rel
        _ = query_hash
        if not allowed:
            return True
        raw_license = meta.get("license", "") or meta.get("license_spdx", "")
        license_value = str(raw_license).strip().upper()
        return license_value in allowed

    return _include_public_hit


def _resolve_urls(meta: Dict):
    permalink = build_public_match_permalink(meta)
    commit_url = None
    if not permalink:
        commit_url = meta.get("source_commit_url")
        if not isinstance(commit_url, str) or not commit_url.strip():
            commit_url = build_public_commit_permalink(meta)
    file_commit = meta.get("source_file_commit", "")
    file_commit_url = None
    if isinstance(file_commit, str) and file_commit.strip():
        file_commit_url = build_github_commit_url(str(meta.get("source_url", "")), file_commit.strip())
    license_display = meta.get("license")
    if not isinstance(license_display, str) or not license_display.strip():
        license_display = meta.get("license_spdx", "<unknown>")
    return permalink, file_commit_url, commit_url, license_display


def main() -> None:
    log = configure_logging()

    parser = argparse.ArgumentParser(
        prog="code-sim-check-public",
        description="Similarity check against the central public index (scope: staged/files/repo).",
    )
    parser.add_argument("paths", nargs="*", help="Optional paths (used by --scope staged/files).")
    add_scope_argument(parser)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Maximum allowed distance (inclusive). Lower values are more similar.",
    )
    parser.add_argument(
        "--min-lines",
        type=int,
        default=0,
        help="Hide matched segments shorter than this many lines (default: 0, no filter).",
    )
    parser.add_argument(
        "--license",
        dest="licenses",
        action="append",
        default=[],
        help="Optional SPDX license filter (repeatable or comma-separated). If omitted, all licenses are included.",
    )
    parser.add_argument("--json", metavar="FILE", default=None, help="Write results as JSON to FILE.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colour output.")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive result browser.")
    args = parser.parse_args()
    validate_scope_args(parser, args)

    if args.no_color:
        disable_color()

    top_k = max(1, args.top_k)
    min_lines = max(0, args.min_lines)
    license_filters = _normalize_license_filters(args.licenses)
    include_public_hit = _public_hit_filter_for_licenses(license_filters)

    ctx, rel_paths, query_elements = query_elements_from_args(args.paths, args.scope)

    if not rel_paths:
        log.info("No %s files matched the active scope and filters.",
                 "repository" if args.scope == "repo" else "provided" if args.scope == "files" else "staged")
        return
    if not query_elements:
        log.info("No queryable code elements found after filters and ignore rules.")
        return

    log.info("Checking %d code element(s) from %d file(s) in scope '%s' against public index.",
             len(query_elements), len(rel_paths), args.scope)
    if license_filters:
        log.info("Applying public license filter: %s", ", ".join(license_filters))

    # Load index-time embedding config to match query preparation
    index_cfg = load_embedding_config(ctx.db_path, "public")
    resolved_truncate = index_cfg.get("truncate_tokens") if index_cfg else None

    embedder = EmbeddingClient(truncate_tokens=resolved_truncate)
    vectors = embedder.embed_documents([element["text"] for _, element in query_elements])

    store = CodeVectorStore(
        path=str(ctx.db_path),
        private_collection_name=ctx.private_collection_name,
        public_collection_name=ctx.public_collection_name,
        metric=ctx.metric,
    )
    pool_size = max(top_k * 4, 20)
    total_hits = 0
    query_results = []
    all_entries: List[ResultEntry] = []

    print(fmt_header_box("code-sim-check-public", args.scope, ctx.org_id, len(query_elements)))

    for (rel_path, element), vector in zip(query_elements, vectors):
        print(fmt_query_header(element["name"], rel_path, element["start_line"], element["end_line"]))

        results = store.query_public_by_embedding(vector, n_results=pool_size, licenses=license_filters)
        hits = extract_hits(
            results,
            top_k=top_k,
            max_distance=args.max_distance,
            query_rel=rel_path,
            query_hash=element["hash"],
            include_hit=include_public_hit,
            min_lines=min_lines,
        )

        print(fmt_check_sub_header("Public", len(hits)))
        if not hits:
            print(f"  {_DIM}No matches in public index.{_RESET}")
        else:
            total_hits += len(hits)
            for idx, (distance, meta, doc) in enumerate(hits, start=1):
                permalink, file_commit_url, commit_url, license_display = _resolve_urls(meta)
                for row in fmt_public_hit(idx, distance, meta, permalink, file_commit_url, commit_url, license_display):
                    print(row)
                all_entries.append(ResultEntry(
                    query_name=element["name"], query_file=rel_path,
                    rank=idx, distance=distance, meta=meta, code=doc, hit_type="public",
                ))

        hit_records: List[Dict] = []
        for i, (d, m, _doc) in enumerate(hits, start=1):
            permalink, file_commit_url, commit_url, license_display = _resolve_urls(m)
            hit: Dict = {
                "rank": i,
                "distance": round(d, 6),
                "repo": m.get("repo_name", "<unknown>"),
                "file": m.get("file_path", "<unknown>"),
                "function": m.get("function_name", "<unknown>"),
                "start_line": m.get("start_line"),
                "end_line": m.get("end_line"),
                "license": license_display,
                "commit": m.get("source_commit", "<unknown>"),
            }
            if permalink:
                hit["permalink"] = permalink
            if file_commit_url:
                hit["file_commit_url"] = file_commit_url
            if commit_url:
                hit["commit_url"] = commit_url
            hit_records.append(hit)

        query_results.append({
            "name": element["name"],
            "file": rel_path,
            "start_line": element["start_line"],
            "end_line": element["end_line"],
            "hits": hit_records,
        })

    print(fmt_footer([
        f"Checked  {len(query_elements)} element{'s' if len(query_elements) != 1 else ''}",
        f"Public  {total_hits} match{'es' if total_hits != 1 else ''}",
    ]))

    if not args.no_interactive:
        launch_interactive_viewer(all_entries)

    if license_filters and total_hits == 0:
        unknown_filters = [lic for lic in license_filters if lic not in KNOWN_LICENSE_KEYWORDS]
        for bad_filter in unknown_filters:
            suggestions = _suggest_license_keywords(bad_filter)
            if suggestions:
                print(f"License filter '{bad_filter}' not recognized. Did you mean: {', '.join(suggestions)}?")

    if args.json:
        data = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command": "code-sim-check-public",
            "scope": args.scope,
            "top_k": top_k,
            "max_distance": args.max_distance,
            "min_lines": min_lines,
            "license_filters": license_filters,
            "queries": query_results,
            "summary": {
                "total_elements_checked": len(query_elements),
                "total_hits": total_hits,
            },
        }
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nJSON written \u2192 {args.json}")


if __name__ == "__main__":
    main()
