#!/usr/bin/env python3
import sys
import os
import hashlib
import subprocess
import logging
import difflib
from pathlib import Path
from typing import List, Dict, Optional, Any, TypedDict

import voyageai
from tree_sitter_language_pack import get_language, get_parser

# --- third-party store wrapper (local) ---
from .clients import CodeVectorStore
# If you keep extractors here, that's fine; otherwise import from code_parser

# -------- Setup Logging --------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# -------- Paths / constants (shared between indexer & hook) --------
THIS_FILE = Path(__file__).resolve()
TOOL_DIR = THIS_FILE.parent                 # .../src/code_similarity_tool
SRC_ROOT = TOOL_DIR.parent                  # .../src
REPO_ROOT = SRC_ROOT.parent                 # repo root
DB_PATH = REPO_ROOT / ".git" / ".code-sim-db"      # single persistent DB location
COLLECTION_NAME = "project_code"
METRIC = "cosine"

# -------- Type Definitions --------
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

# -------- Git & Parsing Helpers (Stateless) --------
def get_file_content_from_git(commit_hash: str, file_path: str) -> Optional[bytes]:
    """Gets the content of a file from a specific git state (index if commit_hash is '')."""
    try:
        git_spec = f"{commit_hash}:{file_path}" if commit_hash else f":{file_path}"
        result = subprocess.run(['git', 'show', git_spec], capture_output=True, check=True, text=False)
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def detect_lang(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext == ".java":
        return "java"
    return None

def _slice_text(buf: bytes, node) -> str:
    return buf[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

def extract_code_elements(file_path: Path, buf: Optional[bytes]) -> List[CodeElement]:
    """Extracts functions/methods using tree-sitter. Returns [] if buf is None."""
    if not buf:
        return []
    lang = detect_lang(file_path)
    if not lang:
        return []

    language = get_language(lang)
    parser = get_parser(lang)
    tree = parser.parse(buf)
    root = tree.root_node

    if lang == "python":
        query_str = r"(function_definition) @decl"
        kind_map = {"function_definition": "function"}
    else:  # java
        query_str = r"""
          (method_declaration) @decl
          (constructor_declaration) @decl
        """
        kind_map = {"method_declaration": "method", "constructor_declaration": "constructor"}

    query = language.query(query_str)
    items: List[CodeElement] = []

    for _, caps in query.matches(root):
        captured_nodes = caps.get("decl")
        if not captured_nodes:
            continue
        d = captured_nodes[0]
        name_node = d.child_by_field_name("name")
        name = _slice_text(buf, name_node) if name_node else "<no-name>"
        text = _slice_text(buf, d)

        # content-based ids (avoid file-path in hash so moves don't force re-embedding)
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

# -------- Embedding Client --------
class EmbeddingClient:
    """Wraps Voyage AI embeddings (batched)."""
    def __init__(self, model: str = "voyage-code-2"):
        try:
            self.client = voyageai.Client()
            self.model = model
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

# -------- Main Application Logic --------
class CodeProcessor:
    def __init__(self, vector_store: CodeVectorStore, embed_client: EmbeddingClient):
        self.vector_store = vector_store
        self.embed_client = embed_client

    def process_modified_file(self, file_path: Path):
        """Compare HEAD vs INDEX, update DB, run similarity queries (with paired logging)."""
        rel_for_git = str(file_path.relative_to(REPO_ROOT))
        log.info(f"Processing modified file: {rel_for_git}")

        before = get_file_content_from_git('HEAD', rel_for_git)
        after  = get_file_content_from_git('',    rel_for_git)

        if not after:
            log.warning(f"Could not read staged content for {rel_for_git}. Skipping.")
            return

        elems_before = extract_code_elements(file_path, before) if before else []
        elems_after  = extract_code_elements(file_path, after)

        # Maps / sets by hash
        b_map = {e['hash']: e for e in elems_before}
        a_map = {e['hash']: e for e in elems_after}
        b_hashes = set(b_map.keys())
        a_hashes = set(a_map.keys())

        # Plain set math for DB ops
        hashes_to_delete = list(b_hashes - a_hashes)
        hashes_to_add    = list(a_hashes - b_hashes)
        to_delete_elems  = [b_map[h] for h in hashes_to_delete]
        to_add_elems     = [a_map[h] for h in hashes_to_add]

        # ---------- Pairing for nicer logs ----------
        # Index remaining (unmatched-by-hash) by name for pairing heuristics
        b_by_name = {}
        for e in to_delete_elems:
            b_by_name.setdefault(e['name'], []).append(e)
        a_by_name = {}
        for e in to_add_elems:
            a_by_name.setdefault(e['name'], []).append(e)

        paired = []          # list of tuples (old_el, new_el)
        consumed_del = set() # ids of old elements we've paired
        consumed_add = set() # ids of new elements we've paired

        # Pass 1: same-name, different-content pairs (likely edits to same function)
        for name in set(b_by_name.keys()).intersection(a_by_name.keys()):
            old_list = b_by_name[name]
            new_list = a_by_name[name]
            # Greedy pair by order of appearance
            for old_el, new_el in zip(old_list, new_list):
                paired.append((old_el, new_el))
                consumed_del.add(old_el['id'])
                consumed_add.add(new_el['id'])

        # Pass 2 (optional): fuzzy name pairing for renames (e.g., heyalice -> heyace)
        # Only try to pair remaining unmatched items
        remaining_dels = [e for e in to_delete_elems if e['id'] not in consumed_del]
        remaining_adds = [e for e in to_add_elems   if e['id'] not in consumed_add]
        if remaining_dels and remaining_adds:
            # Build best-match table by name similarity
            pairs = []
            for old_el in remaining_dels:
                best = None
                best_score = 0.0
                for new_el in remaining_adds:
                    score = difflib.SequenceMatcher(None, old_el['name'], new_el['name']).ratio()
                    if score > best_score:
                        best_score = score
                        best = new_el
                # Require a decent similarity to call it a rename; tweak threshold as you like
                if best is not None and best_score >= 0.6 and best['id'] not in consumed_add:
                    pairs.append((old_el, best, best_score))

            # Greedy one-to-one matching by highest score
            pairs.sort(key=lambda t: t[2], reverse=True)
            used_new_ids = set()
            for old_el, new_el, _ in pairs:
                if new_el['id'] in used_new_ids or old_el['id'] in consumed_del:
                    continue
                paired.append((old_el, new_el))
                consumed_del.add(old_el['id'])
                consumed_add.add(new_el['id'])
                used_new_ids.add(new_el['id'])

        # Anything not paired remains solo delete/add
        solo_deletes = [e for e in to_delete_elems if e['id'] not in consumed_del]
        solo_adds    = [e for e in to_add_elems   if e['id'] not in consumed_add]

        # ---------- DB updates ----------
        if hashes_to_delete:
            log.info("Detected changes, updating database...")
            # We still delete all old hashes (includes pairs + solo deletes)
            self.vector_store.delete_by_ids(hashes_to_delete)

        # ---------- Pretty, ordered logs ----------
        # Sort for determinism (by old start_line if available, else name)
        def _sort_key_old(e): return (e['start_line'], e['name'])
        def _sort_key_new(e): return (e['start_line'], e['name'])

        paired.sort(key=lambda pr: _sort_key_old(pr[0]))
        solo_deletes.sort(key=_sort_key_old)
        solo_adds.sort(key=_sort_key_new)

        for old_el, new_el in paired:
            log.info(f"- Deleting old version of: {old_el['name']}")
            log.info(f"+ Adding new version of: {new_el['name']}")

        for old_el in solo_deletes:
            log.info(f"- Deleting old version of: {old_el['name']}")

        # Embed only the *new* elements (paired new + solo adds)
        new_elems = [new for (_, new) in paired] + solo_adds
        if not new_elems:
            log.info("No new or modified functions to add.")
            return

        for el in new_elems:
            log.info(f"+ Adding new version of: {el['name']}")

        embeddings = self.embed_client.embed_documents([e["text"] for e in new_elems])
        if embeddings is None:
            log.error("Failed to get embeddings. Aborting update.")
            return

        rel_fp = str(file_path.relative_to(REPO_ROOT))
        self.vector_store.upsert_code_elements(new_elems, embeddings, rel_fp)
        log.info("Database updated successfully.")

        # ---------- Similarity queries ----------
        log.info("\nRunning similarity queries for new/modified functions...")
        for i, element in enumerate(new_elems):
            similar = self.vector_store.query_by_embedding(embeddings[i], n_results=6)
            self._show_query_results(similar, {"name": element['name'], "file_path": rel_fp})


    def process_deleted_file(self, rel_path: str):
        log.info(f"File deleted. Removing DB entries for: {rel_path}")
        num = self.vector_store.delete_by_file_path(rel_path)
        log.info(f"Deleted {num} function(s).")

    def _show_query_results(self, results: Dict, query_element: QueryElement):
        print("-" * 25)
        print(f"Query for code similar to '{query_element['name']}' in '{query_element['file_path']}':")

        if not results.get('ids') or not results['ids'][0]:
            print("  -> Query returned no results.")
            return

        ids = results['ids'][0]
        distances = results['distances'][0]
        metas = results['metadatas'][0]

        if len(ids) < 2:
            print("  -> No other similar items found in the database.")
            return

        for i in range(1, len(ids)):  # skip exact self
            m = metas[i]
            print(f"\n  -> Found similar item (distance: {distances[i]:.4f})")
            print(f"     File: {m['file_path']}")
            print(f"     Function: {m['function_name']} (lines {m['start_line']}-{m['end_line']})")
        print("-" * 25)

# -------- Entry point for pre-commit --------
def main():
    # Ensure DB path exists
    DB_PATH.mkdir(parents=True, exist_ok=True)

    # pre-commit passes staged files as argv
    staged = [Path(p).resolve() for p in sys.argv[1:]]

    if not staged:
        log.info("No staged files to process.")
        sys.exit(0)

    # Filter: only under src/, never under the tool dir, and skip noise
    filtered: List[Path] = []
    for p in staged:
        # must be under src/
        try:
            p.relative_to(SRC_ROOT)
        except ValueError:
            continue
        # exclude tool code
        try:
            p.relative_to(TOOL_DIR)
            continue
        except ValueError:
            pass
        if any(x in p.parts for x in ("__pycache__", "node_modules", "venv", ".git")):
            continue
        filtered.append(p)

    if not filtered:
        log.info("No relevant files to process after filtering.")
        sys.exit(0)

    log.info(f"Processing {len(filtered)} staged file(s)...")
    try:
        embed_client = EmbeddingClient()
        store = CodeVectorStore(path=str(DB_PATH), collection_name=COLLECTION_NAME, metric=METRIC)
        processor = CodeProcessor(store, embed_client)

        for fp in filtered:
            rel = fp.relative_to(REPO_ROOT)
            log.info("-" * 40)
            log.info(f"=> Processing: {rel}")
            processor.process_modified_file(fp)

    except Exception as e:
        log.exception(f"Unexpected error: {e}")
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
