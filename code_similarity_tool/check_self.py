#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Dict

from .check_utils import (
    add_scope_argument,
    configure_logging,
    extract_hits,
    query_elements_from_args,
    validate_scope_args,
)
from .clients import CodeVectorStore
from .embeddings import EmbeddingClient


def _include_self_hit(meta: Dict, query_rel: str, query_hash: str) -> bool:
    # Skip exact self match from same file/content
    if meta.get("file_path") == query_rel and meta.get("content_hash") == query_hash:
        return False
    return True


def main() -> None:
    log = configure_logging()

    parser = argparse.ArgumentParser(
        prog="code-sim-check-self",
        description=(
            "Read-only similarity check against the current repository only "
            "(scope: staged/files/repo)."
        ),
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
        "--json",
        metavar="FILE",
        default=None,
        help="Write results as JSON to FILE.",
    )
    args = parser.parse_args()
    validate_scope_args(parser, args)

    top_k = max(1, args.top_k)

    ctx, rel_paths, query_elements = query_elements_from_args(args.paths, args.scope)

    if not rel_paths:
        if args.scope == "repo":
            log.info("No repository files matched the active scope and filters.")
        elif args.scope == "files":
            log.info("No provided files matched the active scope and filters.")
        else:
            log.info("No staged files matched the active scope and filters.")
        return

    if not query_elements:
        log.info("No queryable code elements found after filters and ignore rules.")
        return

    log.info(
        "Checking %d code element(s) from %d file(s) in scope '%s' for self repo '%s'.",
        len(query_elements),
        len(rel_paths),
        args.scope,
        ctx.repo_name,
    )

    embedder = EmbeddingClient()
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

    for (rel_path, element), vector in zip(query_elements, vectors):
        results = store.query_private_repo_by_embedding(
            vector,
            org_id=ctx.org_id,
            repo_id=ctx.repo_id,
            n_results=pool_size,
        )
        hits = extract_hits(
            results,
            top_k=top_k,
            max_distance=args.max_distance,
            query_rel=rel_path,
            query_hash=element["hash"],
            include_hit=_include_self_hit,
        )

        query_record = {
            "name": element["name"],
            "file": rel_path,
            "start_line": element["start_line"],
            "end_line": element["end_line"],
            "hits": [],
        }

        print("-" * 72)
        print(f"Query: {element['name']} ({rel_path}:{element['start_line']}-{element['end_line']})")

        if not hits:
            print("  No self-repo similar items found.")
            query_results.append(query_record)
            continue

        total_hits += len(hits)
        for idx, (distance, meta) in enumerate(hits, start=1):
            print(
                "  "
                f"{idx}. distance={distance:.4f} "
                f"repo={meta.get('repo_name', '<unknown>')} "
                f"file={meta.get('file_path', '<unknown>')} "
                f"function={meta.get('function_name', '<unknown>')} "
                f"lines={meta.get('start_line', '?')}-{meta.get('end_line', '?')}"
            )
            query_record["hits"].append({
                "rank": idx,
                "distance": round(distance, 6),
                "repo": meta.get("repo_name", "<unknown>"),
                "file": meta.get("file_path", "<unknown>"),
                "function": meta.get("function_name", "<unknown>"),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
            })

        query_results.append(query_record)

    print("=" * 72)
    print(
        f"Checked {len(query_elements)} code element(s); found {total_hits} self-repo similar match(es) in repo '{ctx.repo_name}'."
    )

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
