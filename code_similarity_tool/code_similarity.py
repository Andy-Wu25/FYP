#!/usr/bin/env python3
from __future__ import annotations

import sys
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, TypedDict

import voyageai
from tree_sitter_language_pack import get_language, get_parser

# --- local store wrapper ---
from .clients import CodeVectorStore

# -------- Logging --------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# -------- Paths / constants --------
THIS_FILE = Path(__file__).resolve()
TOOL_DIR = THIS_FILE.parent                 # .../code_similarity_tool
REPO_ROOT = TOOL_DIR.parent                 # repo root (tool lives at repo root)
DB_PATH = REPO_ROOT / ".git" / ".code-sim-db"
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

def get_file_content_from_git(commit_hash: str, file_path: str) -> Optional[bytes]:
    """
    Read content from git. If commit_hash == '', read the INDEX (staged).
    Returns None if the blob doesn't exist.
    """
    try:
        spec = f"{commit_hash}:{file_path}" if commit_hash else f":{file_path}"
        res = subprocess.run(['git', 'show', spec], capture_output=True, check=True, text=False)
        return res.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def extract_code_elements(file_path: Path, buf: Optional[bytes]) -> List[CodeElement]:
    """Extract functions/methods using tree-sitter. Returns [] if buf is None or lang unsupported."""
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
        decl_nodes = caps.get("decl")
        if not decl_nodes:
            continue
        d = decl_nodes[0]
        name_node = d.child_by_field_name("name")
        name = _slice(buf, name_node) if name_node else "<no-name>"
        text = _slice(buf, d)

        # content-based id so renames/moves don't re-embed
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

# -------- Embedding --------
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

# -------- Main Processor --------
class CodeProcessor:
    def __init__(self, store: CodeVectorStore, embedder: EmbeddingClient):
        self.store = store
        self.embedder = embedder

    def process_modified_file(self, file_path: Path):
        """Compare HEAD vs INDEX for a file, update DB, run similarity queries."""
        rel_for_git = str(file_path.relative_to(REPO_ROOT))
        log.info(f"Processing modified file: {rel_for_git}")

        before = get_file_content_from_git('HEAD', rel_for_git)
        after  = get_file_content_from_git('',     rel_for_git)
        if not after:
            log.warning(f"Could not read staged content for {rel_for_git}. Skipping.")
            return

        before_el = extract_code_elements(file_path, before) if before else []
        after_el  = extract_code_elements(file_path, after)

        before_map = {e['hash']: e for e in before_el}
        after_map  = {e['hash']: e for e in after_el}

        to_delete = list(set(before_map.keys()) - set(after_map.keys()))
        to_add    = list(set(after_map.keys())  - set(before_map.keys()))
        new_elems = [after_map[h] for h in to_add]

        if to_delete:
            log.info("Detected changes, updating database...")
            for h in to_delete:
                log.info(f"- Deleting old version of: {before_map[h]['name']}")
            self.store.delete_by_ids(to_delete)

        if not new_elems:
            log.info("No new or modified functions to add.")
            return

        for el in new_elems:
            log.info(f"+ Adding new version of: {el['name']}")

        # Embed & upsert
        embeddings = self.embedder.embed_documents([e["text"] for e in new_elems])
        if embeddings is None:
            log.error("Failed to get embeddings. Aborting update.")
            return

        self.store.upsert_code_elements(new_elems, embeddings, rel_for_git)
        log.info("Database updated successfully.")

        # Similarity queries
        log.info("\nRunning similarity queries for new/modified functions...")
        for i, element in enumerate(new_elems):
            similar = self.store.query_by_embedding(embeddings[i], n_results=6)
            self._show_query_results(similar, {"name": element['name'], "file_path": rel_for_git})

    def process_deleted_file(self, rel_path: str):
        """Remove all DB entries for a deleted file path (repo-relative)."""
        log.info(f"File deleted. Removing DB entries for: {rel_path}")
        num = self.store.delete_by_file_path(rel_path)
        log.info(f"Deleted {num} function(s).")

    def _show_query_results(self, results: Dict, query_element: QueryElement):
        # Leading separator only; no trailing line to keep output tidy.
        print("-" * 25)
        print(f"Query for code similar to '{query_element['name']}' in '{query_element['file_path']}':")
        if not results.get('ids') or not results['ids'][0]:
            print("  -> Query returned no results.")
            print()
            return

        ids = results['ids'][0]
        dists = results['distances'][0]
        metas = results['metadatas'][0]
        if len(ids) < 2:
            print("  -> No other similar items found in the database.")
            print()
            return

        for i in range(1, len(ids)):  # skip exact self
            m = metas[i]
            print(f"\n  -> Found similar item (distance: {dists[i]:.4f})")
            print(f"     File: {m['file_path']}")
            print(f"     Function: {m['function_name']} (lines {m['start_line']}-{m['end_line']})")
        print()  # blank line between blocks

# -------- Deletions helper (portable) --------
def _get_staged_deletions() -> List[Path]:
    """
    Return repo-absolute Paths for files staged as deleted (diff-filter=D).
    Uses -z to be robust to spaces. Returns [] on error.
    """
    try:
        out = subprocess.check_output(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=D', '-z'],
            cwd=str(REPO_ROOT)
        )
        parts = out.split(b'\x00')
        files = [p.decode('utf-8') for p in parts if p]
        return [(REPO_ROOT / f).resolve() for f in files]
    except Exception:
        return []

# -------- Entry point for pre-commit --------
def main():
    # Ensure DB exists
    DB_PATH.mkdir(parents=True, exist_ok=True)

    # Parse args (support --deleted and --with-deletions)
    args = sys.argv[1:]
    deleted_mode = False
    with_deletions = False

    if args and args[0] == "--deleted":
        deleted_mode = True
        args = args[1:]
    elif args and args[0] == "--with-deletions":
        with_deletions = True
        args = args[1:]

    staged_am = [Path(p).resolve() for p in args]

    # Filter: only files we actually support + exclude junk/tool
    filtered_am: List[Path] = []
    for p in staged_am:
        # must be inside repo
        try:
            p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        # skip tool & common junk
        if TOOL_DIR in p.parents:
            continue
        if any(x in p.parts for x in ("__pycache__", "node_modules", "venv", ".git")):
            continue
        # only supported languages
        if detect_lang(p) is None:
            continue
        filtered_am.append(p)

    store = CodeVectorStore(path=str(DB_PATH), collection_name=COLLECTION_NAME, metric=METRIC)

    # Only deletions
    if deleted_mode:
        if not filtered_am:
            log.info("No relevant files to process after filtering.")
            sys.exit(0)
        for fp in filtered_am:
            rel = str(fp.relative_to(REPO_ROOT))
            log.info(f"File deleted. Removing DB entries for: {rel}")
            n = store.delete_by_file_path(rel)
            log.info(f"Deleted {n} function(s).")
        sys.exit(0)

    # AM + optional staged deletions
    embedder = EmbeddingClient()
    processor = CodeProcessor(store, embedder)

    if filtered_am:
        log.info(f"Processing {len(filtered_am)} staged file(s)...")
        for fp in filtered_am:
            rel = fp.relative_to(REPO_ROOT)
            log.info("-" * 40)
            log.info(f"=> Processing: {rel}")
            processor.process_modified_file(fp)

    staged_deleted: List[Path] = []
    if with_deletions:
        staged_deleted = _get_staged_deletions()
        if staged_deleted:
            log.info("Processing deletions...")
            for fp in staged_deleted:
                try:
                    rel = str(fp.relative_to(REPO_ROOT))
                except ValueError:
                    continue
                processor.process_deleted_file(rel)

    if not filtered_am and not staged_deleted:
        log.info("No relevant files to process after filtering.")

    sys.exit(0)

if __name__ == "__main__":
    main()
