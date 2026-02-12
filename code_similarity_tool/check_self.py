#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Dict

from .check_utils import configure_logging, extract_hits, staged_elements_from_args
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
        description="Read-only similarity check for staged code against the current repository only.",
    )
    parser.add_argument("paths", nargs="*", help="Optional paths to include (must be staged).")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-distance", type=float, default=None)
    args = parser.parse_args()

    top_k = max(1, args.top_k)

    ctx, rel_paths, query_elements = staged_elements_from_args(args.paths)

    if not rel_paths:
        log.info("No staged Python/Java files to check.")
        return

    if not query_elements:
        log.info("No staged functions/methods found after filters and ignore rules.")
        return

    log.info(
        "Checking %d code element(s) from %d staged file(s) in self repo '%s'.",
        len(query_elements),
        len(rel_paths),
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

        print("-" * 72)
        print(f"Query: {element['name']} ({rel_path}:{element['start_line']}-{element['end_line']})")

        if not hits:
            print("  No self-repo similar items found.")
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

    print("=" * 72)
    print(
        f"Checked {len(query_elements)} code element(s); found {total_hits} self-repo similar match(es) in repo '{ctx.repo_name}'."
    )


if __name__ == "__main__":
    main()
