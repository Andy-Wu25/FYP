#!/usr/bin/env python3
from __future__ import annotations

import sys
import hashlib
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, TypedDict, Tuple

import voyageai
from tree_sitter_language_pack import get_language, get_parser

from .clients import CodeVectorStore

# -------- Logging --------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

# -------- Paths / constants --------
THIS_FILE = Path(__file__).resolve()
TOOL_DIR  = THIS_FILE.parent                 # .../code_similarity_tool
REPO_ROOT = TOOL_DIR.parent                  # repo root (tool lives at repo root)
DB_PATH   = REPO_ROOT / ".git" / ".code-sim-db"
COLLECTION_NAME = "project_code"
METRIC          = "cosine"

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

# -------- Git staged helpers (portable) --------
def _get_staged_deletions() -> List[str]:
    """Repo-relative paths staged as deleted."""
    try:
        out = subprocess.check_output(
            ['git', 'diff', '--cached', '--name-only', '--diff-filter=D', '-z'],
            cwd=str(REPO_ROOT)
        )
        parts = [p for p in out.split(b'\x00') if p]
        return [p.decode('utf-8') for p in parts]
    except Exception:
        return []

def _get_staged_renames() -> List[Tuple[str, str]]:
    """
    Returns list of (old_rel, new_rel) for staged renames.
    Uses -M to detect renames; parses -z for safety.
    """
    try:
        out = subprocess.check_output(
            ['git', 'diff', '--cached', '-M', '--name-status', '-z'],
            cwd=str(REPO_ROOT)
        )
        b = out.split(b'\x00')
        i, renames = 0, []
        while i < len(b):
            if not b[i]:
                break
            entry = b[i].decode('utf-8', errors='replace')
            # Examples: "R100", "R087", etc. Next two fields are old and new names.
            if entry.startswith('R'):
                oldp = b[i+1].decode('utf-8', errors='replace')
                newp = b[i+2].decode('utf-8', errors='replace')
                renames.append((oldp, newp))
                i += 3
            else:
                # Statuses like "A", "M", "D" are followed by one path
                i += 2
        return renames
    except Exception:
        return []

# -------- Main Processor --------
class CodeProcessor:
    def __init__(self, store: CodeVectorStore, embedder: EmbeddingClient):
        self.store = store
        self.embedder = embedder

    def analyze_modified_file(self, file_path: Path) -> Tuple[str, List[CodeElement], List[str], List[str]]:
        """
        Compare HEAD vs INDEX for a file and compute:
          - rel_for_git (str)
          - new_elems: list[CodeElement] to (re)embed/upsert
          - deleted_ids: list[str] element ids to delete
          - deleted_names: list[str] names for logging
        """
        rel_for_git = str(file_path.relative_to(REPO_ROOT))
        log.info(f"Processing modified file: {rel_for_git}")

        before = get_file_content_from_git('HEAD', rel_for_git)
        after  = get_file_content_from_git('',     rel_for_git)
        if not after:
            log.warning(f"Could not read staged content for {rel_for_git}. Skipping.")
            return rel_for_git, [], [], []

        before_el = extract_code_elements(file_path, before) if before else []
        after_el  = extract_code_elements(file_path, after)

        b_map = {e['hash']: e for e in before_el}
        a_map = {e['hash']: e for e in after_el}

        deleted_ids   = list(set(b_map.keys()) - set(a_map.keys()))
        to_add_ids    = list(set(a_map.keys()) - set(b_map.keys()))
        new_elems     = [a_map[h] for h in to_add_ids]
        deleted_names = [b_map[h]['name'] for h in deleted_ids if h in b_map]

        return rel_for_git, new_elems, deleted_ids, deleted_names

    def process_deleted_file(self, rel_path: str):
        """Remove all DB entries for a deleted file path (repo-relative)."""
        log.info(f"File deleted. Removing DB entries for: {rel_path}")
        num = self.store.delete_by_file_path(rel_path)
        log.info(f"Deleted {num} function(s).")

    def show_query_results(self, results: Dict, query_element: QueryElement):
        # Leading separator only; no trailing dashed line—keeps output tidy.
        print("-" * 25)
        print(f"Query for code similar to '{query_element['name']}' in '{query_element['file_path']}':")
        if not results.get('ids') or not results['ids'][0]:
            print("  -> Query returned no results.\n")
            return

        ids   = results['ids'][0]
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

    # Parse args (support --deleted / --with-deletions / --with-renames)
    args = sys.argv[1:]
    deleted_mode   = False
    with_deletions = False
    with_renames   = False

    while args and args[0].startswith("--"):
        if args[0] == "--deleted":
            deleted_mode = True
        elif args[0] == "--with-deletions":
            with_deletions = True
        elif args[0] == "--with-renames":
            with_renames = True
        else:
            log.warning(f"Unknown flag {args[0]} ignored.")
        args = args[1:]

    # Files pre-commit passes are the staged AM candidates
    staged_am_abs = [Path(p).resolve() for p in args]

    # Filter: inside repo, right types, skip junk/tool
    filtered_am: List[Path] = []
    for p in staged_am_abs:
        try:
            p.relative_to(REPO_ROOT)
        except ValueError:
            continue
        if _is_junk(p):
            continue
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        filtered_am.append(p)

    store    = CodeVectorStore(path=str(DB_PATH), collection_name=COLLECTION_NAME, metric=METRIC)
    embedder = EmbeddingClient()
    proc     = CodeProcessor(store, embedder)

    # Legacy "only deletions" mode: treat argv as files-to-delete
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

    # Optional: process staged renames (update metadata without re-embedding)
    if with_renames:
        renames = _get_staged_renames()
        for old_rel, new_rel in renames:
            # Only bother if target is a supported language and not junk
            np = (REPO_ROOT / new_rel).resolve()
            if _is_junk(np) or np.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            updated = store.move_file_path(old_rel, new_rel)
            if updated:
                log.info(f"Updated {updated} vector(s) for rename: {old_rel} -> {new_rel}")

    # Analyze AM changes first; batch embeddings across ALL files
    all_new: List[CodeElement] = []
    all_new_rel_paths: List[str] = []  # same length as all_new
    if filtered_am:
        log.info(f"Processing {len(filtered_am)} staged file(s)...")
        for fp in filtered_am:
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
                proc.show_query_results(similar, {"name": el['name'], "file_path": rel})
                idx += 1
    else:
        log.info("No new or modified functions to add.")

    # Optionally also purge staged deletions of whole files
    if with_deletions:
        staged_deleted = _get_staged_deletions()
        if staged_deleted:
            log.info("Processing deletions...")
            for rel in staged_deleted:
                # Only purge supported types; keeps DB tidy.
                p = (REPO_ROOT / rel).resolve()
                if p.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                proc.process_deleted_file(rel)

    if not filtered_am and not with_deletions:
        log.info("No relevant files to process after filtering.")

    sys.exit(0)

if __name__ == "__main__":
    main()
