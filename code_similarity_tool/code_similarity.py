#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from .clients import CodeVectorStore
from .code_parser import CodeElement, extract_code_elements
from .embeddings import EmbeddingClient
from .ignore import load_ignore_file
from .runtime import (
    SUPPORTED_SUFFIXES,
    load_runtime_context,
    read_index_blob,
    staged_added_modified_renamed,
)


def _configure_logging() -> logging.Logger:
    log_level = os.getenv("CODE_SIM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="[%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def _resolve_optional_paths(raw_paths: List[str], repo_root: Path) -> List[str]:
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


def _collect_query_elements(
    repo_root: Path,
    rel_paths: List[str],
    matcher,
) -> List[Tuple[str, CodeElement]]:
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

        for element in extract_code_elements(Path(rel_path), staged_content):
            out.append((rel_path, element))

    return out


def _extract_hits(results: Dict, *, top_k: int, max_distance: float | None, query_rel: str, query_hash: str, query_repo_id: str) -> List[Tuple[float, Dict]]:
    ids = (results.get("ids") or [[]])[0]
    dists = (results.get("distances") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]

    hits: List[Tuple[float, Dict]] = []
    for _, dist, meta in zip(ids, dists, metas):
        if not isinstance(meta, dict):
            continue

        if (
            meta.get("repo_id") == query_repo_id
            and meta.get("file_path") == query_rel
            and meta.get("content_hash") == query_hash
        ):
            continue

        if max_distance is not None and isinstance(dist, (int, float)) and dist > max_distance:
            continue

        if not isinstance(dist, (int, float)):
            continue

        hits.append((float(dist), meta))
        if len(hits) >= top_k:
            break

    return hits


def main() -> None:
    log = _configure_logging()

    parser = argparse.ArgumentParser(
        prog="code-sim-check",
        description="Read-only similarity check for staged code against org-scoped vector DB.",
    )
    parser.add_argument("paths", nargs="*", help="Optional paths to include (must be staged).")
    parser.add_argument("--top-k", type=int, default=int(os.getenv("CODE_SIM_TOP_K", "5")))
    parser.add_argument("--max-distance", type=float, default=None)
    args = parser.parse_args()

    top_k = max(1, args.top_k)

    ctx = load_runtime_context()
    matcher = load_ignore_file(ctx.repo_root)

    staged = set(staged_added_modified_renamed(ctx.repo_root))
    staged.update(_resolve_optional_paths(args.paths, ctx.repo_root))

    if not staged:
        log.info("No staged Python/Java files to check.")
        return

    rel_paths = sorted(staged)
    query_elements = _collect_query_elements(ctx.repo_root, rel_paths, matcher)
    if not query_elements:
        log.info("No staged functions/methods found after filters and ignore rules.")
        return

    log.info(
        "Checking %d code element(s) from %d staged file(s) in org '%s'.",
        len(query_elements),
        len(rel_paths),
        ctx.org_id,
    )

    embedder = EmbeddingClient()
    vectors = embedder.embed_documents([element["text"] for _, element in query_elements])

    store = CodeVectorStore(path=str(ctx.db_path), collection_name=ctx.collection_name, metric=ctx.metric)

    pool_size = max(top_k * 4, 20)
    total_hits = 0

    for (rel_path, element), vector in zip(query_elements, vectors):
        results = store.query_by_embedding(vector, org_id=ctx.org_id, n_results=pool_size)
        hits = _extract_hits(
            results,
            top_k=top_k,
            max_distance=args.max_distance,
            query_rel=rel_path,
            query_hash=element["hash"],
            query_repo_id=ctx.repo_id,
        )

        print("-" * 72)
        print(
            f"Query: {element['name']} ({rel_path}:{element['start_line']}-{element['end_line']})"
        )

        if not hits:
            print("  No similar items found in current organization scope.")
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
        f"Checked {len(query_elements)} code element(s); found {total_hits} similar match(es) in org '{ctx.org_id}'."
    )


if __name__ == "__main__":
    main()
