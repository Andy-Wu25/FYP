#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ..infra.clients import CodeVectorStore
from ..core.code_parser import CodeElement, extract_code_elements, make_element_id
from ..infra.embeddings import EmbeddingClient, add_embedding_args
from ..core.ignore import load_ignore_file
from .public_index import (
    clone_repo_at_commit,
    iter_public_repo_source_files,
    parse_github_url,
    resolve_remote_commit,
)
from ..core.runtime import iter_repo_source_files, load_runtime_context


def _configure_logging() -> logging.Logger:
    log_level = os.getenv("CODE_SIM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="[%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def sync_current_repo(
    action_name: str = "index",
    *,
    max_chars: int | None = None,
    long_text_mode: str | None = None,
) -> int:
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

    if not all_elements:
        removed = store.delete_private_repo_entries(ctx.org_id, ctx.repo_id)
        if removed:
            log.info("[%s] removed %d stale element(s) for repo_id=%s", action_name, removed, ctx.repo_id)
        log.info("[%s] no functions/methods found after ignore rules.", action_name)
        return 0

    embedder = EmbeddingClient(max_chars=max_chars, long_text_mode=long_text_mode)
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

    current_ids = [element["id"] for element in all_elements]
    removed = store.delete_private_repo_stale_entries(ctx.org_id, ctx.repo_id, keep_ids=current_ids)
    if removed:
        log.info("[%s] removed %d stale element(s) for repo_id=%s", action_name, removed, ctx.repo_id)

    log.info("[%s] indexed %d code element(s) across %d file(s).", action_name, total, len(by_file_elements))
    return total


def index_private_github_repo(
    url: str,
    ref: Optional[str] = None,
    *,
    max_chars: int | None = None,
    long_text_mode: str | None = None,
) -> int:
    log = _configure_logging()
    ctx = load_runtime_context()

    owner, repo, canonical_url = parse_github_url(url)
    repo_name = f"{owner}/{repo}"
    repo_id = hashlib.sha256(f"remote:{canonical_url}".encode("utf-8")).hexdigest()
    commit_sha = resolve_remote_commit(canonical_url, ref)

    log.info("[index-private] org=%s repo=%s commit=%s", ctx.org_id, repo_name, commit_sha)
    log.info(
        "[index-private] db=%s private_collection=%s",
        ctx.db_path,
        ctx.private_collection_name,
    )

    all_elements: List[CodeElement] = []
    rel_paths_for_element: List[str] = []

    with tempfile.TemporaryDirectory(prefix="code-sim-private-") as tmp:
        repo_dir = Path(tmp) / "repo"
        clone_repo_at_commit(canonical_url, commit_sha, repo_dir)
        files = iter_public_repo_source_files(repo_dir)

        for file_path in files:
            rel_path = str(file_path.relative_to(repo_dir))
            elements = extract_code_elements(file_path, file_path.read_bytes())
            for element in elements:
                element["id"] = make_element_id(ctx.org_id, repo_id, rel_path, element["hash"])
            if elements:
                all_elements.extend(elements)
                rel_paths_for_element.extend([rel_path] * len(elements))

    store = CodeVectorStore(
        path=str(ctx.db_path),
        private_collection_name=ctx.private_collection_name,
        public_collection_name=ctx.public_collection_name,
        metric=ctx.metric,
    )

    if not all_elements:
        removed = store.delete_private_repo_entries(ctx.org_id, repo_id)
        if removed:
            log.info("[index-private] removed %d stale element(s) for repo_id=%s", removed, repo_id)
        log.info("[index-private] no functions/methods found after ignore rules.")
        return 0

    embedder = EmbeddingClient(max_chars=max_chars, long_text_mode=long_text_mode)
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
                "repo_id": repo_id,
                "repo_name": repo_name,
                "file_path": rel_path,
            },
        )
        total += len(by_file_elements[rel_path])

    current_ids = [element["id"] for element in all_elements]
    removed = store.delete_private_repo_stale_entries(ctx.org_id, repo_id, keep_ids=current_ids)
    if removed:
        log.info("[index-private] removed %d stale element(s) for repo_id=%s", removed, repo_id)

    log.info("[index-private] indexed %d code element(s) from %s@%s", total, repo_name, commit_sha)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-index",
        description="Index the current repository into the private org-scoped index.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help=argparse.SUPPRESS,
    )
    add_embedding_args(parser)
    args = parser.parse_args()

    if args.target:
        try:
            parse_github_url(args.target)
            is_github = True
        except ValueError:
            is_github = False

        if is_github:
            parser.error(
                f"'{args.target}' looks like a GitHub URL.\n"
                "  To index a remote repo into your private org index:\n"
                "    code-sim-index-private " + args.target + "\n"
                "  To index a third-party public repo into the shared public index:\n"
                "    code-sim-index-public " + args.target
            )
        else:
            parser.error("code-sim-index does not accept arguments. Run it from inside the repository to index.")

    sync_current_repo(
        action_name="index",
        max_chars=args.max_chars,
        long_text_mode=args.long_text_mode,
    )


def main_private_url() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-index-private",
        description="Index a remote GitHub repository into the private org-scoped index.",
    )
    parser.add_argument("url", help="GitHub repository URL (https://github.com/<owner>/<repo>)")
    parser.add_argument("--ref", default=None, help="Optional git ref (branch/tag/sha). Defaults to HEAD.")
    add_embedding_args(parser)
    args = parser.parse_args()

    try:
        index_private_github_repo(
            args.url,
            ref=args.ref,
            max_chars=args.max_chars,
            long_text_mode=args.long_text_mode,
        )
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
