#!/usr/bin/env python3
import sys
import hashlib
import subprocess
import voyageai
import chromadb
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, TypedDict
from tree_sitter_language_pack import get_language, get_parser

# -------- Setup Logging --------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger(__name__)

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
    """Gets the content of a file from a specific git state."""
    try:
        git_spec = f"{commit_hash}:{file_path}" if commit_hash else f":{file_path}"
        command = ['git', 'show', git_spec]
        result = subprocess.run(command, capture_output=True, check=True, text=False)
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

def slice_text(buf: bytes, node) -> str:
    return buf[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

def extract_code_elements(file_path: Path, buf: bytes) -> List[CodeElement]:
    """Extracts functions/methods from code content using tree-sitter."""
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
    else: # java
        query_str = r"""
          (method_declaration) @decl
          (constructor_declaration) @decl
        """
        kind_map = {
            "method_declaration": "method",
            "constructor_declaration": "constructor",
        }
    
    query = language.query(query_str)
    items: List[CodeElement] = []
    
    for _, capdict in query.matches(root):
        captured_nodes = capdict.get("decl")
        if not captured_nodes:
            continue
        
        d = captured_nodes[0]
        name_node = d.child_by_field_name("name")
        name = slice_text(buf, name_node) if name_node else "<no-name>"
        text = slice_text(buf, d)
        
        unique_string = f"{str(file_path)}::{text}"
        content_hash = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()

        items.append({
            "id": content_hash, "name": name, "kind": kind_map.get(d.type, d.type),
            "start_line": d.start_point[0] + 1, "end_line": d.end_point[0] + 1,
            "text": text, "hash": content_hash,
        })
    return items

# -------- Embedding Client --------
class EmbeddingClient:
    """Wraps all interactions with the Voyage AI embedding API."""
    def __init__(self):
        try:
            self.client = voyageai.Client()
            log.info("Voyage AI client initialized.")
        except Exception as e:
            log.error(f"Voyage AI client init failed (set VOYAGE_API_KEY): {e}")
            sys.exit(1)

    def embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Embeds a list of code snippets."""
        if not texts:
            return []
        try:
            result = self.client.embed(texts, model="voyage-code-2", input_type="document")
            return result.embeddings
        except Exception as e:
            log.error(f"Voyage embedding failed: {e}")
            return None

# -------- Vector Database Client --------
class CodeVectorStore:
    """Wraps all interactions with the ChromaDB vector store."""
    def __init__(self, path: str = "vector_db", collection_name: str = "project_code"):
        try:
            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            log.info(f"ChromaDB client initialized. Collection: {collection_name}")
        except Exception as e:
            log.error(f"ChromaDB client init failed: {e}")
            sys.exit(1)

    def upsert_code_elements(self, elements: List[CodeElement], embeddings: List[List[float]], file_path: Path):
        """Adds or updates code elements in the database."""
        ids = [el['id'] for el in elements]
        metadatas = [{
            "file_path": str(file_path),
            "function_name": el['name'],
            "kind": el['kind'],
            "start_line": el['start_line'],
            "end_line": el['end_line'],
            "content_hash": el['hash']
        } for el in elements]
        documents = [el["text"] for el in elements]
        
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def delete_by_ids(self, ids: List[str]):
        """Deletes a list of elements by their unique IDs."""
        if ids:
            self.collection.delete(ids=ids)

    def delete_by_file_path(self, file_path_str: str) -> int:
        """Deletes all vector DB entries associated with a file."""
        existing_items = self.collection.get(where={"file_path": file_path_str})
        if existing_items and existing_items['ids']:
            num_deleted = len(existing_items['ids'])
            self.collection.delete(ids=existing_items['ids'])
            return num_deleted
        return 0

    def query_by_embedding(self, embedding: List[float], n_results: int = 6) -> Dict:
        """Finds similar items to a given embedding."""
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )

# -------- Main Application Logic --------
class CodeProcessor:
    """
    Orchestrates the process of diffing, embedding, and storing
    code elements.
    """
    def __init__(self, vector_store: CodeVectorStore, embed_client: EmbeddingClient):
        self.vector_store = vector_store
        self.embed_client = embed_client

    def process_modified_file(self, file_path: Path):
        """
        Compares staged vs. HEAD versions of a file, updates the DB,
        and runs similarity queries.
        """
        log.info(f"Processing modified file: {file_path}")
        content_before = get_file_content_from_git('HEAD', str(file_path))
        content_after = get_file_content_from_git('', str(file_path))

        if not content_after:
            log.warning(f"Could not read staged content for {file_path}. Skipping.")
            return

        elements_before = extract_code_elements(file_path, content_before) if content_before else []
        elements_after = extract_code_elements(file_path, content_after)

        before_map = {el['hash']: el for el in elements_before}
        after_map = {el['hash']: el for el in elements_after}

        hashes_before = set(before_map.keys())
        hashes_after = set(after_map.keys())

        hashes_to_delete = list(hashes_before - hashes_after)
        hashes_to_add = list(hashes_after - hashes_before)
        elements_to_embed = [after_map[h] for h in hashes_to_add]

        if hashes_to_delete:
            log.info("Detected changes, updating database...")
            for h in hashes_to_delete:
                log.info(f"- Deleting old version of: {before_map[h]['name']}")
            self.vector_store.delete_by_ids(hashes_to_delete)

        if not elements_to_embed:
            log.info("No new or modified functions to add.")
            return

        for el in elements_to_embed:
            log.info(f"+ Adding new version of: {el['name']}")

        # 1. Embed
        payloads = [it["text"] for it in elements_to_embed]
        embeddings = self.embed_client.embed_documents(payloads)
        
        if embeddings is None:
            log.error("Failed to get embeddings. Aborting update.")
            return

        # 2. Upsert
        self.vector_store.upsert_code_elements(elements_to_embed, embeddings, file_path)
        log.info("Database updated successfully.")

        # 3. Query
        log.info("\nRunning similarity queries for new/modified functions...")
        for i, element in enumerate(elements_to_embed):
            query_details: QueryElement = {"name": element['name'], "file_path": str(file_path)}
            similar_items = self.vector_store.query_by_embedding(embeddings[i], n_results=6)
            self._show_query_results(similar_items, query_details)

    def process_deleted_file(self, file_path_str: str):
        """Handles a file deletion event by cleaning the database."""
        log.info(f"File was deleted. Removing all associated functions from DB: {file_path_str}")
        num_deleted = self.vector_store.delete_by_file_path(file_path_str)
        if num_deleted > 0:
            log.info(f"Deleted {num_deleted} function(s) for deleted file: {file_path_str}")
        else:
            log.info("No functions found in the database for this file.")

    def _show_query_results(self, results: Dict, query_element: QueryElement):
        """Parses and prints ChromaDB query results in a readable format."""
        print("-" * 25)
        print(f"Query for code similar to '{query_element['name']}' in '{query_element['file_path']}':")
        
        if not results['ids'] or not results['ids'][0]:
            print("  -> Query returned no results.")
            return

        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        if len(ids) < 2:
            print("  -> No other similar items found in the database.")
            return

        # Start from the second item (index 1) to skip the identical match
        for i in range(1, len(ids)):
            metadata = metadatas[i]
            print(f"\n  -> Found similar item (distance: {distances[i]:.4f})")
            print(f"     File: {metadata['file_path']}")
            print(f"     Function: {metadata['function_name']} (lines {metadata['start_line']}-{metadata['end_line']})")
        print("-" * 25)

# ... (all your other code: imports, helpers, classes, etc. - UNCHANGED) ...


# -------- NEW: Main execution for pre-commit --------
def main():
    # pre-commit passes all staged files as arguments
    # sys.argv[0] is the script name
    # sys.argv[1:] is the list of files
    staged_files = sys.argv[1:]
    
    if not staged_files:
        log.info("No staged files to process.")
        sys.exit(0) # Exit successfully

    # Filter out files in venv or other ignored paths
    # This is a safety check in case .gitignore is missing
    files_to_process = []
    for f in staged_files:
        target_path = Path(f)
        if target_path.parts and target_path.parts[0] in ('venv', 'vector_db', '.git'):
            log.info(f"Skipping ignored-path file: {f}")
            continue
        files_to_process.append(target_path)
    
    if not files_to_process:
        log.info("No relevant files to process after filtering.")
        sys.exit(0)
        
    log.info(f"Processing {len(files_to_process)} staged file(s)...")

    # We will assume a single script failure should block the commit
    # We use exit_code to track this.
    exit_code = 0 
    
    try:
        # Initialize services ONCE
        embed_client = EmbeddingClient()
        vector_store = CodeVectorStore()
        processor = CodeProcessor(vector_store, embed_client)
        
        # Loop over each staged file
        for file_path in files_to_process:
            log.info("-" * 40)
            log.info(f"=> Processing: {file_path}")
            
            # The pre-commit framework doesn't run on deleted files
            # by default, so we only need to handle modifications/additions.
            processor.process_modified_file(file_path)

    except Exception as e:
        log.exception(f"An unexpected error occurred: {e}")
        exit_code = 1 # Mark failure

    # Exit with 1 to block the commit, 0 to allow it
    sys.exit(exit_code) 

if __name__ == "__main__":
    main()