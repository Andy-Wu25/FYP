#!/usr/bin/env python3
from __future__ import annotations

import sys
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, TypedDict, Set

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

SUPPORTED_SUFFIXES = {".py", ".java"}

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

def _get_staged_added_modified() -> List[Path]:
    """Return repo-absolute Paths for files staged as added/modified (diff-filter=AM)."""
    try:
        out = subprocess.check_output(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=AM', '-z'],
            cwd=str(REPO_ROOT)
        )
        parts = out.split(b'\x00')
        files = [p.decode('utf-8') for p in parts if p]
        return [(REPO_ROOT / f).resolve() for f in files]
    except Exception:
        return []

def _get_staged_deletions() -> List[Path]:
    """Return repo-absolute Paths for files staged as deleted (diff-filter=D)."""
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

def _is_junk(p: Path) -> bool:
    parts = set(p.parts)
    if any(seg in parts for seg in (".git", "venv", "__pycache__", "node_modules")):
        return True
    if TOOL_DIR in p.parents:
        return True
    return False

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

    def analyze_modified_file(self, file_path: Path):
        """
        Compare HEAD vs INDEX and return:
          rel_for_git, new_elems, deleted_ids, deleted_names
        """
        rel_for_git = str(file_path.relative_to(REPO_ROOT))
        log.info(f"Processing modified file: {rel_for_git}")

        before = get_file_content_from_git('HEAD', rel_for_git)
        after  = get_file_content_from_git('',     rel_for_git)
        if not after:
            log.info(f"[skip] Cannot read staged content for: {rel_for_git}")
            return rel_for_git, [], [], []

        before_el = extract_code_elements(file_path, before) if before else []
        after_el  = extract_code_elements(file_path, after)

        before_map = {e['hash']: e for e in before_el}
        after_map  = {e['hash']: e for e in after_el}

        deleted_ids   = list(set(before_map.keys()) - set(after_map.keys()))
        to_add_ids    = list(set(after_map.keys())  - set(before_map.keys()))
        new_elems     = [after_map[h] for h in to_add_ids]
        deleted_names = [before_map[h]['name'] for h in deleted_ids if h in before_map]

        if not new_elems and not deleted_ids:
            log.info(f"[skip] No function-level changes in: {rel_for_git}")

        return rel_for_git, new_elems, deleted_ids, deleted_names

    def process_deleted_file(self, rel_path: str):
        """Remove all DB entries for a deleted file path (repo-relative)."""
        log.info(f"File deleted. Removing DB entries for: {rel_path}")
        num = self.store.delete_by_file_path(rel_path)
        log.info(f"Deleted {num} function(s).")

    def _show_query_results(self, results: Dict, query_element: QueryElement):
        # Leading separator only; no trailing dashed line
        print("-" * 25)
        print(f"Query for code similar to '{query_element['name']}' in '{query_element['file_path']}':")
        if not results.get('ids') or not results['ids'][0]:
            print("  -> Query returned no results.\n")
            return

        ids = results['ids'][0]
        dists = results['distances'][0]
        metas = results['metadatas'][0]
        if len(ids) < 2:
            print("  -> No other similar items found in the database.\n")
            return

        for i in range(1, len(ids)):  # skip exact self
            m = metas[i]
            print(f"\n  -> Found similar item (distance: {dists[i]:.4f})")
            print(f"     File: {m['file_path']}")
            print(f"     Function: {m['function_name']} (lines {m['start_line']}-{m['end_line']})")
        print()  # blank line between blocks

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

    # 1) Start with files provided by pre-commit (argv)
    from_precommit = [Path(p).resolve() for p in args]

    # 2) Union with the *actual* staged-added/modified list from Git
    staged_am_git = _get_staged_added_modified()
    union: List[Path] = []
    seen: Set[Path] = set()
    for p in from_precommit + staged_am_git:
        try:
            p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if _is_junk(p):
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        if p not in seen:
            seen.add(p)
            union.append(p)

    store    = CodeVectorStore(path=str(DB_PATH), collection_name=COLLECTION_NAME, metric=METRIC)
    embedder = EmbeddingClient()
    proc     = CodeProcessor(store, embedder)

    # Legacy deletion mode (treat argv as deletions only)
    if deleted_mode:
        if not union:
            log.info("No relevant files to process after filtering.")
            sys.exit(0)
        for fp in union:
            rel = str(fp.relative_to(REPO_ROOT))
            proc.process_deleted_file(rel)
        sys.exit(0)

    # Analyze AM changes; batch embeddings across ALL files
    all_new: List[CodeElement] = []
    all_new_rel_paths: List[str] = []

    if union:
        log.info(f"Processing {len(union)} staged file(s)...")
        for fp in union:
            rel_for_git, new_elems, deleted_ids, deleted_names = proc.analyze_modified_file(fp)

            if deleted_ids:
                log.info("Detected changes, updating database...")
                for nm in deleted_names:
                    log.info(f"- Deleting old version of: {nm}")
                store.delete_by_ids(deleted_ids)

            if new_elems:
                for el in new_elems:
                    log.info(f"+ Adding new version of: {el['name']}")
                all_new.extend(new_elems)
                all_new_rel_paths.extend([rel_for_git] * len(new_elems))
    else:
        log.info("No relevant files to process after filtering.")

    # Single embed call for all new/changed functions
    if all_new:
        vectors = embedder.embed_documents([e["text"] for e in all_new])
        if vectors is None:
            log.error("Failed to get embeddings. Aborting update.")
            sys.exit(1)

        # Upsert grouped by file path
        from collections import defaultdict
        by_file_elems: Dict[str, List[CodeElement]] = defaultdict(list)
        by_file_vecs:  Dict[str, List[List[float]]] = defaultdict(list)
        for el, vec, rel in zip(all_new, vectors, all_new_rel_paths):
            by_file_elems[rel].append(el)
            by_file_vecs[rel].append(vec)

        for rel in by_file_elems:
            store.upsert_code_elements(by_file_elems[rel], by_file_vecs[rel], rel)
        log.info("Database updated successfully.")

        # Similarity queries
        idx = 0
        for rel in by_file_elems:
            for el in by_file_elems[rel]:
                similar = store.query_by_embedding(vectors[idx], n_results=6)
                proc._show_query_results(similar, {"name": el['name'], "file_path": rel})
                idx += 1

    # Optional: also purge staged deletions if requested
    if with_deletions:
        staged_deleted = _get_staged_deletions()
        if staged_deleted:
            log.info("Processing deletions...")
            for fp in staged_deleted:
                try:
                    rel = str(fp.relative_to(REPO_ROOT))
                except ValueError:
                    continue
                proc.process_deleted_file(rel)

    sys.exit(0)

if __name__ == "__main__":
    main()
