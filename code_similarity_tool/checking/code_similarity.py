#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict, List

from .check_utils import (
    add_scope_argument,
    configure_logging,
    disable_color,
    extract_hits,
    fmt_check_sub_header,
    fmt_footer,
    fmt_header_box,
    fmt_private_hit,
    fmt_public_hit,
    fmt_query_header,
    query_elements_from_args,
    validate_scope_args,
)
from .interactive import ResultEntry, launch_interactive_viewer
from ..infra.clients import CodeVectorStore
from ..infra.embeddings import EmbeddingClient, load_embedding_config
from ..infra.public_links import build_github_commit_url, build_public_commit_permalink, build_public_match_permalink


def _include_private_hit_factory(query_repo_id: str):
    def _include_hit(meta: Dict, query_rel: str, query_hash: str) -> bool:
        if (
            meta.get("repo_id") == query_repo_id
            and meta.get("file_path") == query_rel
            and meta.get("content_hash") == query_hash
        ):
            return False
        return True
    return _include_hit


def _include_all(meta: Dict, query_rel: str, query_hash: str) -> bool:
    return True


def _resolve_public_hit_urls(meta: Dict):
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


def _run_check(prog: str, description: str, check_private: bool, check_public: bool) -> None:
    log = configure_logging()

    parser = argparse.ArgumentParser(prog=prog, description=description)
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
    parser.add_argument("--json", metavar="FILE", default=None, help="Write results as JSON to FILE.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colour output.")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive result browser.")
    args = parser.parse_args()
    validate_scope_args(parser, args)

    if args.no_color:
        disable_color()

    top_k = max(1, args.top_k)
    min_lines = max(0, args.min_lines)
    ctx, rel_paths, query_elements = query_elements_from_args(args.paths, args.scope)

    if not rel_paths:
        log.info("No %s files matched the active scope and filters.",
                 "repository" if args.scope == "repo" else "provided" if args.scope == "files" else "staged")
        return
    if not query_elements:
        log.info("No queryable code elements found after filters and ignore rules.")
        return

    log.info("Checking %d code element(s) from %d file(s) in scope '%s'.",
             len(query_elements), len(rel_paths), args.scope)

    # Load index-time embedding config to match query preparation
    index_cfg_priv = load_embedding_config(ctx.db_path, "private") if check_private else None
    index_cfg_pub = load_embedding_config(ctx.db_path, "public") if check_public else None
    configs = [c for c in [index_cfg_priv, index_cfg_pub] if c is not None]

    resolved_truncate = None
    if len(configs) == 1:
        resolved_truncate = configs[0].get("truncate_tokens")
    elif len(configs) == 2 and configs[0] == configs[1]:
        resolved_truncate = configs[0].get("truncate_tokens")
    elif len(configs) == 2:
        log.warning(
            "Private and public indexes have different embedding configs: "
            "private=%s public=%s. Using env/default config for queries.",
            index_cfg_priv, index_cfg_pub,
        )

    embedder = EmbeddingClient(truncate_tokens=resolved_truncate)
    vectors = embedder.embed_documents([element["text"] for _, element in query_elements])

    store = CodeVectorStore(
        path=str(ctx.db_path),
        private_collection_name=ctx.private_collection_name,
        public_collection_name=ctx.public_collection_name,
        metric=ctx.metric,
    )
    pool_size = max(top_k * 4, 20)
    include_private = _include_private_hit_factory(ctx.repo_id)

    total_private_hits = 0
    total_public_hits = 0
    query_results = []
    all_entries: List[ResultEntry] = []

    print(fmt_header_box(prog, args.scope, ctx.org_id, len(query_elements)))

    for (rel_path, element), vector in zip(query_elements, vectors):
        print(fmt_query_header(element["name"], rel_path, element["start_line"], element["end_line"]))

        query_record: Dict = {
            "name": element["name"],
            "file": rel_path,
            "start_line": element["start_line"],
            "end_line": element["end_line"],
        }

        # ── Private ──────────────────────────────────────────────────────────
        if check_private:
            private_results = store.query_private_by_embedding(
                vector, org_id=ctx.org_id, n_results=pool_size
            )
            private_hits = extract_hits(
                private_results,
                top_k=top_k,
                max_distance=args.max_distance,
                query_rel=rel_path,
                query_hash=element["hash"],
                include_hit=include_private,
                min_lines=min_lines,
            )
            print(fmt_check_sub_header("Private", len(private_hits)))
            if not private_hits:
                from .check_utils import _DIM, _RESET
                print(f"  {_DIM}No matches in private org index.{_RESET}")
            else:
                total_private_hits += len(private_hits)
                for idx, (distance, meta, doc) in enumerate(private_hits, start=1):
                    print(fmt_private_hit(idx, distance, meta))
                    all_entries.append(ResultEntry(
                        query_name=element["name"], query_file=rel_path,
                        rank=idx, distance=distance, meta=meta, code=doc, hit_type="private",
                    ))

            query_record["private_hits"] = [
                {
                    "rank": i,
                    "distance": round(d, 6),
                    "repo": m.get("repo_name", "<unknown>"),
                    "file": m.get("file_path", "<unknown>"),
                    "function": m.get("function_name", "<unknown>"),
                    "start_line": m.get("start_line"),
                    "end_line": m.get("end_line"),
                }
                for i, (d, m, _doc) in enumerate(private_hits, start=1)
            ]

        # ── Public ───────────────────────────────────────────────────────────
        if check_public:
            public_results = store.query_public_by_embedding(vector, n_results=pool_size)
            public_hits = extract_hits(
                public_results,
                top_k=top_k,
                max_distance=args.max_distance,
                query_rel=rel_path,
                query_hash=element["hash"],
                include_hit=_include_all,
                min_lines=min_lines,
            )
            print(fmt_check_sub_header("Public", len(public_hits)))
            if not public_hits:
                from .check_utils import _DIM, _RESET
                print(f"  {_DIM}No matches in public index.{_RESET}")
            else:
                total_public_hits += len(public_hits)
                for idx, (distance, meta, doc) in enumerate(public_hits, start=1):
                    permalink, file_commit_url, commit_url, license_display = _resolve_public_hit_urls(meta)
                    for row in fmt_public_hit(idx, distance, meta, permalink, file_commit_url, commit_url, license_display):
                        print(row)
                    all_entries.append(ResultEntry(
                        query_name=element["name"], query_file=rel_path,
                        rank=idx, distance=distance, meta=meta, code=doc, hit_type="public",
                    ))

            public_hit_records: List[Dict] = []
            for i, (d, m, _doc) in enumerate(public_hits if check_public else [], start=1):
                permalink, file_commit_url, commit_url, license_display = _resolve_public_hit_urls(m)
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
                public_hit_records.append(hit)
            query_record["public_hits"] = public_hit_records

        query_results.append(query_record)

    # ── Footer ───────────────────────────────────────────────────────────────
    footer_parts = [f"Checked  {len(query_elements)} element{'s' if len(query_elements) != 1 else ''}"]
    if check_private:
        footer_parts.append(f"Private  {total_private_hits} match{'es' if total_private_hits != 1 else ''}")
    if check_public:
        footer_parts.append(f"Public  {total_public_hits} match{'es' if total_public_hits != 1 else ''}")
    print(fmt_footer(footer_parts))

    if not args.no_interactive:
        launch_interactive_viewer(all_entries)

    if args.json:
        data: Dict = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command": prog,
            "scope": args.scope,
            "top_k": top_k,
            "max_distance": args.max_distance,
            "min_lines": min_lines,
            "org_id": ctx.org_id,
            "queries": query_results,
            "summary": {"total_elements_checked": len(query_elements)},
        }
        if check_private:
            data["summary"]["total_private_hits"] = total_private_hits
        if check_public:
            data["summary"]["total_public_hits"] = total_public_hits
        Path(args.json).write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"\nJSON written \u2192 {args.json}")


def main() -> None:
    _run_check(
        prog="code-sim-check",
        description="Similarity check against both private org-scoped and public indexes (scope: staged/files/repo).",
        check_private=True,
        check_public=True,
    )


def main_private() -> None:
    _run_check(
        prog="code-sim-check-private",
        description="Similarity check against private org-scoped index only (scope: staged/files/repo).",
        check_private=True,
        check_public=False,
    )


if __name__ == "__main__":
    main()
