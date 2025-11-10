#!/usr/bin/env python3
from pathlib import Path
from typing import List
import hashlib, subprocess
import logging

from .code_similarity import extract_code_elements, EmbeddingClient  # reuse implementations
from .clients import CodeVectorStore

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

THIS_FILE = Path(__file__).resolve()
TOOL_DIR  = THIS_FILE.parent
SRC_ROOT  = TOOL_DIR.parent
REPO_ROOT = SRC_ROOT.parent

def _repo_slug() -> str:
    try:
        url = subprocess.check_output(
            ["git", "config", "--get", "remote.origin.url"],
            cwd=str(REPO_ROOT)
        ).decode().strip()
    except Exception:
        url = str(REPO_ROOT.resolve())
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{REPO_ROOT.name}-{h}"

DB_BASE = Path.home() / ".code-sim-db"
DB_PATH = DB_BASE / _repo_slug()
COLLECTION_NAME = "project_code"
METRIC = "cosine"

def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def _iter_source_files(src_root: Path, tool_dir: Path):
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        if _is_under(p, tool_dir):
            continue
        if any(x in p.parts for x in ("__pycache__", "node_modules", "venv", ".git")):
            continue
        yield p

def main():
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

    elements = []
    for f in files:
        # Read from disk (indexer always uses working tree)
        buf = f.read_bytes()
        elements.extend(extract_code_elements(f, buf))

    log.info(f"\nFound a total of {len(elements)} functions/methods to index.")
    if not elements:
        log.info("Nothing to index. Exiting.")
        return

    log.info("Getting embeddings from Voyage AI (this may take a moment)...")
    vectors = embedder.embed_documents([e["text"] for e in elements])
    if vectors is None:
        log.error("Failed to get embeddings.")
        return

    log.info("Embeddings received. Adding to vector database...")
    # store expects file_path as string relative to repo root (we’ll provide per element)
    store.add_many(elements, base_repo=str(REPO_ROOT))

    log.info("\n✅ Initial indexing complete!")
    log.info(f"DB path: {DB_PATH}")
    log.info(f"Collection: {COLLECTION_NAME}")

if __name__ == "__main__":
    main()
