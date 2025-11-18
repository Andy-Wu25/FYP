#!/usr/bin/env python3
from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import List

from .code_similarity import extract_code_elements, EmbeddingClient
from .clients import CodeVectorStore
from .ignore import load_ignore_file

# -------- Logging --------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# -------- Paths / constants --------
THIS_FILE = Path(__file__).resolve()
TOOL_DIR  = THIS_FILE.parent                 # .../code_similarity_tool
REPO_ROOT = TOOL_DIR.parent                  # repo root

DB_PATH         = REPO_ROOT / ".git" / ".code-sim-db"
COLLECTION_NAME = "project_code"
METRIC          = "cosine"

SUPPORTED_SUFFIXES = {".py", ".java"}


def make_instance_id(rel_path: str, content_hash: str) -> str:
    """
    Stable per-file instance id: same body in different files => different id.
    """
    return hashlib.sha256(f"{rel_path}:{content_hash}".encode("utf-8")).hexdigest()


def _iter_candidate_files(repo_root: Path, matcher) -> List[Path]:
    """
    Walk the repo and return candidate source files, respecting .code-simignore.
    """
    out: List[Path] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        # be defensive: never index inside .git even if user forgets ignore
        if ".git" in p.parts:
            continue
        if not matcher.allows(p, is_dir=False):
            continue
        out.append(p)
    return sorted(out)


def main():
    # Ensure DB path exists
    DB_PATH.mkdir(parents=True, exist_ok=True)

    store    = CodeVectorStore(path=str(DB_PATH), collection_name=COLLECTION_NAME, metric=METRIC)
    embedder = EmbeddingClient()
    matcher  = load_ignore_file(REPO_ROOT)

    log.info("ChromaDB client initialized. Collection: %s", COLLECTION_NAME)
    log.info("--- Starting Initial Project Indexing ---")
    log.info("Resetting collection...")
    store.reset_collection()

    files = _iter_candidate_files(REPO_ROOT, matcher)
    for f in files:
        log.info("  -> Scanning: %s", f.relative_to(REPO_ROOT))

    # 1) Extract all elements first
    all_elements = []
    file_for_element: List[str] = []
    for f in files:
        buf = f.read_bytes()
        rel = str(f.relative_to(REPO_ROOT))

        els = extract_code_elements(f, buf)

        # Attach per-file instance ids (same body in different files => different id)
        for el in els:
            el["id"] = make_instance_id(rel, el["hash"])

        if els:
            all_elements.extend(els)
            file_for_element.extend([rel] * len(els))

    if not all_elements:
        log.info("\nFound a total of 0 functions/methods to index.")
        log.info("\n✅ Initial indexing complete!")
        log.info("DB path: %s", DB_PATH)
        return

    # 2) Single embed call for all functions/methods
    vectors = embedder.embed_documents([e["text"] for e in all_elements])
    if vectors is None:
        log.error("Failed to embed elements. Aborting.")
        return

    # 3) Upsert grouped per file
    from collections import defaultdict
    bucket = defaultdict(lambda: {"elements": [], "vectors": []})

    for e, v, rel in zip(all_elements, vectors, file_for_element):
        bucket[rel]["elements"].append(e)
        bucket[rel]["vectors"].append(v)

    total = 0
    for rel_path, pack in bucket.items():
        store.upsert_code_elements(pack["elements"], pack["vectors"], rel_path)
        total += len(pack["elements"])

    log.info("\nFound a total of %d functions/methods to index.", total)
    log.info("\n✅ Initial indexing complete!")
    log.info("DB path: %s", DB_PATH)


if __name__ == "__main__":
    main()
