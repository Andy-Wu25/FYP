#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, TypedDict

import voyageai
from tree_sitter_language_pack import get_language, get_parser

from .clients import CodeVectorStore
from .config import load_config
from .selection import within_selection

# -------- Logging --------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# -------- Repo / paths --------
PKG_DIR = Path(__file__).resolve().parent  # <repo>/code_similarity_tool

def git_repo_root() -> Path:
    """Return absolute repo root using Git. Fallback by walking up to .git."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--show-toplevel'],
            check=True, capture_output=True, text=True
        ).stdout.strip()
        return Path(out).resolve()
    except Exception:
        cur = PKG_DIR
        for p in [cur, *cur.parents]:
            if (p / '.git').exists():
                return p.resolve()
        # last resort: parent of tool dir
        return PKG_DIR.parent.resolve()

REPO_ROOT = git_repo_root()
DB_PATH   = REPO_ROOT / ".git" / ".code-sim-db"

COLLECTION_NAME = "project_code"
METRIC = "cosine"

# -------- Types --------
class CodeElement(TypedDict):
    id: str
    name: str
    kind: str
    start_line: int
    end_line: int
    text: str
    hash: str

class QueryElement(TypedDict):
    name: str
    file_path: str

# -------- Helpers --------
def detect_lang(path: Path) -> Optional[str]:
    s = path.suffix.lower()
    if s == ".py":
        return "python"
    if s == ".java":
        return "java"
    return None

def _slice(buf: bytes, node) -> str:
    return buf[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

def get_file_content_from_git(commit_hash: str, file_path_rel_repo: str) -> Optional[bytes]:
    """
    file_path_rel_repo must be repo-root relative.
    commit_hash '' means staged (index), otherwise e.g. 'HEAD'
    """
    try:
        git_spec = f"{commit_hash}:{file_path_rel_repo}" if commit_hash else f":{file_path_rel_repo}"
        res = subprocess.run(['git', 'show', git_spec], capture_output=True, check=True)
        return res.stdout  # bytes
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def extract_code_elements(file_path: Path, buf: Optional[bytes]) -> List[CodeElement]:
    if not buf:
        return []
    lang = detect_lang(file_path)
    if not lang:
        return []
    language = get_language(lang)
    parser   = get_parser(lang)
    tree     = parser.parse(buf)
    root     = tree.root_node

    if lang == "python":
        query_str = r"(function_definition) @decl"
        kind_map  = {"function_definition": "function"}
    else:  # java
        query_str = r"""
          (method_declaration) @decl
          (constructor_declaration) @decl
        """
        kind_map  = {"method_declaration": "method", "constructor_declaration": "constructor"}

    query = language.query(query_str)
    items: List[CodeElement] = []

    for _, caps in query.matches(root):
        nodes = caps.get("decl")
        if not nodes:
            continue
        d = nodes[0]
        name_node = d.child_by_field_name("name")
        name = _slice(buf, name_node) if name_node else "<no-name>"
        text = _slice(buf, d)

        content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        items.append({
            "id": content_sha,
            "name": name,
            "kind": kind_map.get(d.type, d.type),
            "start_line": d.start_point[0] + 1,
            "end_line": d.end_point[0] + 1,
            "text": text,
            "hash": content_sha,
        })
    return items

def is_junk(p: Path) -> bool:
    parts = set(p.parts)
    return any(x in parts for x in ('.git', 'venv', '__pycache__', 'node_modules'))

def is_tool_dir(p: Path) -> bool:
    try:
        p.relative_to(PKG_DIR)
        return True
    except ValueError:
        return False

def lang_allowed(p: Path, languages: List[str]) -> bool:
    ext = p.suffix.lower()
    return (ext == '.py' and 'python' in languages) or (ext == '.java' and 'java' in languages)

# -------- Embedding client --------
class EmbeddingClient:
    def __init__(self, model: str = "voyage-code-2"):
        try:
            self.client = voyageai.Client()
            self.model  = model
            log.info("Voyage AI client initialized.")
        except Exception as e:
            log.error(f"Voyage AI client init failed (set VOYAGE_API_KEY): {e}")
            sys.exit(1)

    def embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return []
        try:
            res = self.client.embed(texts, model=self.model, input_type="document")
            return res.embeddings
        except Exception as e:
            log.error(f"Voyage embedding failed: {e}")
            return None

# -------- Orchestrator --------
class CodeProcessor:
    def __init__(self, store: CodeVectorStore, embedder: EmbeddingClient):
        self.store = store
        self.embedder = embedder

    def process_modified_file(self, abs_path: Path):
        rel_repo = str(abs_path.resolve().relative_to(REPO_ROOT))
        log.info(f"Processing modified file: {rel_repo}")

        before = get_file_content_from_git('HEAD', rel_repo)
        after  = get_file_content_from_git('',     rel_repo)
        if not after:
            log.warning(f"Could not read staged content for {rel_repo}. Skipping.")
            return

        before_el = extract_code_elements(abs_path, before) if before else []
        after_el  = extract_code_elements(abs_path, after)

        b_map = {e['hash']: e for e in before_el}
        a_map = {e['hash']: e for e in after_el}

        to_delete = list(set(b_map) - set(a_map))
        to_add    = list(set(a_map) - set(b_map))
        new_el    = [a_map[h] for h in to_add]

        if to_delete:
            log.info("Detected changes, updating database...")
            for h in to_delete:
                log.info(f"- Deleting old version of: {b_map[h]['name']}")
            self.store.delete_by_ids(to_delete)

        if not new_el:
            log.info("No new or modified functions to add.")
            return

        for el in new_el:
            log.info(f"+ Adding new version of: {el['name']}")

        embeddings = self.embedder.embed_documents([e["text"] for e in new_el])
        if embeddings is None:
            log.error("Failed to get embeddings. Aborting update.")
            return

        self.store.upsert_code_elements(new_el, embeddings, rel_repo)
        log.info("Database updated successfully.")

        # Similarity display
        log.info("\nRunning similarity queries for new/modified functions...")
        for i, el in enumerate(new_el):
            res = self.store.query_by_embedding(embeddings[i], n_results=6)
            self._show_query_results(res, {"name": el['name'], "file_path": rel_repo})

    def _show_query_results(self, results: Dict, query_element: QueryElement):
        print("-" * 25)
        print(f"Query for code similar to '{query_element['name']}' in '{query_element['file_path']}':")
        if not results.get('ids') or not results['ids'][0]:
            print("  -> Query returned no results.")
            return
        ids = results['ids'][0]
        dists = results['distances'][0]
        metas = results['metadatas'][0]
        if len(ids) < 2:
            print("  -> No other similar items found in the database.")
            return
        for i in range(1, len(ids)):  # skip exact self
            m = metas[i]
            print(f"\n  -> Found similar item (distance: {dists[i]:.4f})")
            print(f"     File: {m['file_path']}")
            print(f"     Function: {m['function_name']} (lines {m['start_line']}-{m['end_line']})")
        print("-" * 25)

# -------- main (pre-commit) --------
def main():
    # Ensure DB exists
    DB_PATH.mkdir(parents=True, exist_ok=True)

    # Parse args (support --deleted)
    args = sys.argv[1:]
    deleted_mode = False
    if args and args[0] == "--deleted":
        deleted_mode = True
        args = args[1:]

    # Staged files come as argv (pre-commit passes them)
    staged_abs: List[Path] = [Path(p).resolve() for p in args]
    if not staged_abs:
        log.info("No staged files to process.")
        sys.exit(0)

    cfg = load_config(REPO_ROOT)
    languages = cfg.get("languages", ["python", "java"])

    # Filter staged
    filtered: List[Path] = []
    for p in staged_abs:
        if is_junk(p):
            continue
        if is_tool_dir(p):
            continue
        if not within_selection(REPO_ROOT, p, cfg):
            continue
        if detect_lang(p) is None or not lang_allowed(p, languages):
            continue
        filtered.append(p)

    if not filtered:
        log.info("No relevant files to process after filtering.")
        sys.exit(0)

    store = CodeVectorStore(path=str(DB_PATH), collection_name=COLLECTION_NAME, metric=METRIC)

    if deleted_mode:
        for fp in filtered:
            rel = str(fp.resolve().relative_to(REPO_ROOT))
            log.info(f"File deleted. Removing DB entries for: {rel}")
            n = store.delete_by_file_path(rel)
            log.info(f"Deleted {n} function(s).")
        sys.exit(0)

    embedder  = EmbeddingClient()
    processor = CodeProcessor(store, embedder)

    log.info(f"Processing {len(filtered)} staged file(s)...")
    for fp in filtered:
        log.info("-" * 40)
        log.info(f"=> Processing: {fp.resolve().relative_to(REPO_ROOT)}")
        processor.process_modified_file(fp)

    sys.exit(0)

if __name__ == "__main__":
    main()
