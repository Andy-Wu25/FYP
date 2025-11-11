#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .code_similarity import extract_code_elements, EmbeddingClient
from .clients import CodeVectorStore
from .config import load_config
from .selection import within_selection

# -------- Logging --------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# -------- Paths / constants --------
THIS_FILE = Path(__file__).resolve()
TOOL_DIR  = THIS_FILE.parent                 # .../code_similarity_tool
REPO_ROOT = TOOL_DIR.parent                  # repo root (tool lives at repo root)

# keep DB “in the repo” but under .git so it’s never tracked
DB_PATH          = REPO_ROOT / ".git" / ".code-sim-db"
COLLECTION_NAME  = "project_code"
METRIC           = "cosine"

SUPPORTED_SUFFIXES = {".py", ".java"}

# -------- Helpers --------
def _is_junk(p: Path) -> bool:
    parts = set(p.parts)
    # quick filters
    if any(seg in parts for seg in (".git", "venv", "__pycache__", "node_modules")):
        return True
    if TOOL_DIR in p.parents:  # don’t index the tool itself
        return True
    return False

def _iter_selected_files(repo_root: Path) -> List[Path]:
    """
    Walk the repo and yield files that:
      - are .py or .java
      - pass selection rules (if any)
      - skip tool/.git/venv/etc
    """
    cfg = load_config(repo_root)
    out: List[Path] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if _is_junk(p):
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if not within_selection(repo_root, p, cfg):
            continue
        out.append(p)
    return out

# -------- Main --------
def main():
    DB_PATH.mkdir(parents=True, exist_ok=True)

    store    = CodeVectorStore(path=str(DB_PATH), collection_name=COLLECTION_NAME, metric=METRIC)
    embedder = EmbeddingClient()

    log.info("ChromaDB client initialized. Collection: %s", COLLECTION_NAME)
    log.info("--- Starting Initial Project Indexing ---")
    log.info("Resetting collection...")
    store.reset_collection()

    files = _iter_selected_files(REPO_ROOT)
    for f in files:
        log.info("  -> Scanning: %s", f.relative_to(REPO_ROOT))

    # 1) extract all elements first
    all_elements = []
    file_for_element = []  # parallel list storing each element's file path
    for f in files:
        buf = f.read_bytes()
        els = extract_code_elements(f, buf)
        if els:
            all_elements.extend(els)
            file_for_element.extend([str(f.relative_to(REPO_ROOT))] * len(els))

    if not all_elements:
        log.info("\nFound a total of 0 functions/methods to index.")
        log.info("\n✅ Initial indexing complete!")
        log.info("DB path: %s", DB_PATH)
        return

    # 2) single embed call for all elements
    vectors = embedder.embed_documents([e["text"] for e in all_elements])
    if vectors is None:
        log.error("Failed to embed elements. Aborting.")
        return

    # 3) upsert grouped per file (but embeddings already computed)
    #    walk through elements and bucket by file path
    from collections import defaultdict
    bucket = defaultdict(lambda: {"elements": [], "vectors": []})

    for e, v, fpath in zip(all_elements, vectors, file_for_element):
        bucket[fpath]["elements"].append(e)
        bucket[fpath]["vectors"].append(v)

    total = 0
    for rel_path, pack in bucket.items():
        store.upsert_code_elements(pack["elements"], pack["vectors"], rel_path)
        total += len(pack["elements"])

    log.info("\nFound a total of %d functions/methods to index.", total)
    log.info("\n✅ Initial indexing complete!")
    log.info("DB path: %s", DB_PATH)

if __name__ == "__main__":
    main()
