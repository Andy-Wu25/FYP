#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import subprocess
from collections import defaultdict
from typing import Dict, List

from .clients import CodeVectorStore
from .code_parser import CodeElement, extract_code_elements, make_element_id
from .embeddings import EmbeddingClient
from .ignore import load_ignore_file
from .public_index import index_public_github_repo, parse_github_url
from .runtime import iter_repo_source_files, load_runtime_context


def _configure_logging() -> logging.Logger:
    log_level = os.getenv("CODE_SIM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="[%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def sync_current_repo(action_name: str = "index") -> int:
    log = _configure_logging()
    ctx = load_runtime_context()

    log.info("[%s] org=%s repo=%s root=%s", action_name, ctx.org_id, ctx.repo_name, ctx.repo_root)
    log.info(
        "[%s] db=%s private_collection=%s public_collection=%s",
        action_name,
        ctx.db_path,
        ctx.private_collection_name,
        ctx.public_collection_name,
    )

    matcher = load_ignore_file(ctx.repo_root)
    files = iter_repo_source_files(ctx.repo_root, matcher)

    all_elements: List[CodeElement] = []
    rel_paths_for_element: List[str] = []

    for file_path in files:
        rel_path = str(file_path.relative_to(ctx.repo_root))
        elements = extract_code_elements(file_path, file_path.read_bytes())
        for element in elements:
            element["id"] = make_element_id(ctx.org_id, ctx.repo_id, rel_path, element["hash"])

        if elements:
            all_elements.extend(elements)
            rel_paths_for_element.extend([rel_path] * len(elements))

    store = CodeVectorStore(
        path=str(ctx.db_path),
        private_collection_name=ctx.private_collection_name,
        public_collection_name=ctx.public_collection_name,
        metric=ctx.metric,
    )
    removed = store.delete_private_repo_entries(ctx.org_id, ctx.repo_id)
    if removed:
        log.info("[%s] removed %d stale element(s) for repo_id=%s", action_name, removed, ctx.repo_id)

    if not all_elements:
        log.info("[%s] no functions/methods found after ignore rules.", action_name)
        return 0

    embedder = EmbeddingClient()
    vectors = embedder.embed_documents([element["text"] for element in all_elements])

    by_file_elements: Dict[str, List[CodeElement]] = defaultdict(list)
    by_file_vectors: Dict[str, List[List[float]]] = defaultdict(list)

    for element, vector, rel_path in zip(all_elements, vectors, rel_paths_for_element):
        by_file_elements[rel_path].append(element)
        by_file_vectors[rel_path].append(vector)

    total = 0
    for rel_path in by_file_elements:
        store.upsert_private_code_elements(
            by_file_elements[rel_path],
            by_file_vectors[rel_path],
            base_metadata={
                "org_id": ctx.org_id,
                "repo_id": ctx.repo_id,
                "repo_name": ctx.repo_name,
                "file_path": rel_path,
            },
        )
        total += len(by_file_elements[rel_path])

    log.info("[%s] indexed %d code element(s) across %d file(s).", action_name, total, len(by_file_elements))
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-index",
        description=(
            "Index current repository into private org scope, or index a GNU-licensed "
            "public GitHub repository when a URL is provided."
        ),
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Optional GitHub URL. If omitted, index current repository private scope.",
    )
    parser.add_argument("--url", default=None, help="GitHub URL to index into public scope.")
    parser.add_argument("--ref", default=None, help="Optional git ref for public indexing.")
    args = parser.parse_args()

    public_url = args.url or args.target
    if public_url:
        try:
            parse_github_url(public_url)
            index_public_github_repo(public_url, ref=args.ref)
        except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
            parser.error(str(exc))
        return

    sync_current_repo(action_name="index")


if __name__ == "__main__":
    main()
