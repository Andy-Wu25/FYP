#!/usr/bin/env python3
from pathlib import Path
from typing import List
import logging

from .code_similarity import extract_code_elements, EmbeddingClient
from .clients import CodeVectorStore

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# Paths
THIS_FILE = Path(__file__).resolve()
TOOL_DIR = THIS_FILE.parent            # .../src/code_similarity_tool
SRC_ROOT = TOOL_DIR.parent             # .../src
REPO_ROOT = SRC_ROOT.parent            # repo root

# ✅ Keep DB "in the repo" but under .git so it is never tracked/pushed and doesn't affect pre-commit
DB_PATH = REPO_ROOT / ".git" / ".code-sim-db"

COLLECTION_NAME = "project_code"
METRIC = "cosine"

def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def _iter_source_files(src_root: Path, tool_dir: Path):
    """Yield all source files under src/, excluding the tool dir and common junk."""
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        if _is_under(p, tool_dir):
            continue
        if any(x in p.parts for x in ("__pycache__", "node_modules", "venv", ".git")):
            continue
        yield p

def main():
    # Ensure DB directory exists
    DB_PATH.mkdir(parents=True, exist_ok=True)

    embedder = EmbeddingClient()
    store = CodeVectorStore(path=str(DB_PATH), collection_name=COLLECTION_NAME, metric=METRIC)
    log.info(f"ChromaDB client initialized. Collection: {COLLECTION_NAME}")

    log.info("--- Starting Initial Project Indexing ---")
    log.info("Resetting collection...")
    store.reset_collection()

    files = list(_iter_source_files(SRC_ROOT, TOOL_DIR))
    for f in files:
        log.info(f"  -> Scanning: {f.relative_to(REPO_ROOT)}")

    total = 0
    for f in files:
        buf = f.read_bytes()
        elements = extract_code_elements(f, buf)
        if not elements:
            continue

        vectors = embedder.embed_documents([e["text"] for e in elements])
        if vectors is None:
            log.error(f"Failed to embed {f}. Skipping.")
            continue

        # Upsert this file's elements with correct relative path metadata
        rel_path = str(f.relative_to(REPO_ROOT))
        store.upsert_code_elements(elements, vectors, rel_path)
        total += len(elements)

    log.info(f"\nFound a total of {total} functions/methods to index.")
    if total == 0:
        log.info("Nothing to index. Exiting.")
        return

    log.info("\n✅ Initial indexing complete!")
    log.info(f"DB path: {DB_PATH}")
    log.info(f"Collection: {COLLECTION_NAME}")

if __name__ == "__main__":
    main()
