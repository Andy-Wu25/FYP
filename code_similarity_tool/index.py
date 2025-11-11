#!/usr/bin/env python3
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .code_similarity import (
    extract_code_elements,
    EmbeddingClient,
    git_repo_root,   # reuse the robust root detection
    detect_lang,     # reuse helpers
    is_junk,
    is_tool_dir,
    lang_allowed,
)
from .clients import CodeVectorStore
from .config import load_config
from .selection import within_selection

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

PKG_DIR   = Path(__file__).resolve().parent
REPO_ROOT = git_repo_root()
DB_PATH   = REPO_ROOT / ".git" / ".code-sim-db"

COLLECTION_NAME = "project_code"
METRIC = "cosine"

def iter_candidates(repo_root: Path, cfg: dict) -> List[Path]:
    """Yield files in selection (respects include dirs/files, excludes junk/tool), language-filtered and capped."""
    languages = cfg.get("languages", ["python", "java"])
    max_files = int(cfg.get("max_files", 200))

    out: List[Path] = []
    for p in repo_root.rglob("*"):
        if not p.is_file():
            continue
        if is_junk(p):
            continue
        if is_tool_dir(p):
            continue
        if not within_selection(repo_root, p, cfg):
            continue
        if detect_lang(p) is None or not lang_allowed(p, languages):
            continue
        out.append(p)
        if len(out) >= max_files:
            break
    return out

def main():
    DB_PATH.mkdir(parents=True, exist_ok=True)

    cfg = load_config(REPO_ROOT)
    files = iter_candidates(REPO_ROOT, cfg)

    store = CodeVectorStore(path=str(DB_PATH), collection_name=COLLECTION_NAME, metric=METRIC)
    log.info("ChromaDB client initialized. Collection: %s", COLLECTION_NAME)

    log.info("--- Starting Initial Project Indexing ---")
    log.info("Resetting collection...")
    store.reset_collection()

    for f in files:
        log.info("  -> Scanning: %s", f.resolve().relative_to(REPO_ROOT))

    total_funcs = 0
    embedder = EmbeddingClient()

    # Index file-by-file (keeps memory modest; free Voyage tier friendly)
    for f in files:
        buf = f.read_bytes()
        elements = extract_code_elements(f, buf)
        if not elements:
            continue

        vectors = embedder.embed_documents([e["text"] for e in elements])
        if vectors is None:
            log.error("Failed to embed %s. Skipping.", f)
            continue

        rel_path = str(f.resolve().relative_to(REPO_ROOT))
        store.upsert_code_elements(elements, vectors, rel_path)
        total_funcs += len(elements)

    log.info("\nFound a total of %d functions/methods to index.", total_funcs)
    log.info("\n✅ Initial indexing complete!")
    log.info("DB path: %s", DB_PATH)
    log.info("Collection: %s", COLLECTION_NAME)

if __name__ == "__main__":
    main()
