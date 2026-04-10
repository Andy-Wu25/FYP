#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict

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
    fmt_private_hit,
    fmt_query_header,
    query_elements_from_args,
    validate_scope_args,
)
from .interactive import ResultEntry, launch_interactive_viewer
from ..infra.clients import CodeVectorStore
from ..infra.embeddings import EmbeddingClient, load_embedding_config


def _include_self_hit(meta: Dict, query_rel: str, query_hash: str) -> bool:
    if meta.get("file_path") == query_rel and meta.get("content_hash") == query_hash:
        return False
    return True


def main() -> None:
    log = configure_logging()

    parser = argparse.ArgumentParser(
        prog="code-sim-check-self",
        description="Similarity check against the current repository only (scope: staged/files/repo).",
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
    parser.add_argument("--json", metavar="FILE", default=None, help="Write results as JSON to FILE.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colour output.")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive result browser.")
    args = parser.parse_args()
    validate_scope_args(parser, args)

    if args.no_color:
        disable_color()

    top_k = max(1, args.top_k)
    ctx, rel_paths, query_elements = query_elements_from_args(args.paths, args.scope)

    if not rel_paths:
        log.info("No %s files matched the active scope and filters.",
                 "repository" if args.scope == "repo" else "provided" if args.scope == "files" else "staged")
        return
    if not query_elements:
        log.info("No queryable code elements found after filters and ignore rules.")
        return

    log.info("Checking %d code element(s) from %d file(s) in scope '%s' for self repo '%s'.",
             len(query_elements), len(rel_paths), args.scope, ctx.repo_name)

    # Load index-time embedding config to match query preparation
    index_cfg = load_embedding_config(ctx.db_path, "private")
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
    all_entries: list[ResultEntry] = []

    print(fmt_header_box("code-sim-check-self", args.scope, ctx.org_id, len(query_elements)))

    for (rel_path, element), vector in zip(query_elements, vectors):
        print(fmt_query_header(element["name"], rel_path, element["start_line"], element["end_line"]))

        results = store.query_private_repo_by_embedding(
            vector, org_id=ctx.org_id, repo_id=ctx.repo_id, n_results=pool_size,
        )
        hits = extract_hits(
            results,
            top_k=top_k,
            max_distance=args.max_distance,
            query_rel=rel_path,
            query_hash=element["hash"],
            include_hit=_include_self_hit,
        )

        print(fmt_check_sub_header("Self", len(hits)))
        if not hits:
            print(f"  {_DIM}No matches in this repository.{_RESET}")
        else:
            total_hits += len(hits)
            for idx, (distance, meta, doc) in enumerate(hits, start=1):
                print(fmt_private_hit(idx, distance, meta))
                all_entries.append(ResultEntry(
                    query_name=element["name"], query_file=rel_path,
                    rank=idx, distance=distance, meta=meta, code=doc, hit_type="self",
                ))

        query_record = {
            "name": element["name"],
            "file": rel_path,
            "start_line": element["start_line"],
            "end_line": element["end_line"],
            "hits": [
                {
                    "rank": i,
                    "distance": round(d, 6),
                    "repo": m.get("repo_name", "<unknown>"),
                    "file": m.get("file_path", "<unknown>"),
                    "function": m.get("function_name", "<unknown>"),
                    "start_line": m.get("start_line"),
                    "end_line": m.get("end_line"),
                }
                for i, (d, m, _doc) in enumerate(hits, start=1)
            ],
        }
        query_results.append(query_record)

    print(fmt_footer([
        f"Checked  {len(query_elements)} element{'s' if len(query_elements) != 1 else ''}",
        f"Self  {total_hits} match{'es' if total_hits != 1 else ''}  ·  repo: {ctx.repo_name}",
    ]))

    if total_hits == 0:
        print(f"\n{_DIM}Hint: no indexed data found. Run code-sim-index first.\033[0m")

    if not args.no_interactive:
        launch_interactive_viewer(all_entries)

    if args.json:
        data = {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "command": "code-sim-check-self",
            "scope": args.scope,
            "top_k": top_k,
            "max_distance": args.max_distance,
            "repo": ctx.repo_name,
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
