#!/usr/bin/env python3
import sys
import hashlib
import subprocess
import voyageai
import chromadb
from pathlib import Path
from typing import List, Dict, Optional
from tree_sitter_language_pack import get_language, get_parser

# -------- Voyage AI client --------
try:
    voyage_ai = voyageai.Client()
except Exception as e:
    print("Voyage AI client init failed (set VOYAGE_API_KEY):", e)
    sys.exit(1)

# -------- ChromaDB client --------
chroma_client = chromadb.PersistentClient(path="vector_db")
code_collection = chroma_client.get_or_create_collection(name="project_code")

# -------- Git interaction functions --------
def get_file_content_from_git(commit_hash: str, file_path: str) -> Optional[bytes]:
    """
    Gets the content of a file from a specific git state.
    - Use 'HEAD' for the last commit.
    - Use '' (empty string) for the staged version.
    Returns None if the file doesn't exist in that state.
    """
    try:
        # Using an empty string for the commit hash tells git to read from the index (staging area)
        git_spec = f"{commit_hash}:{file_path}" if commit_hash else f":{file_path}"
        command = ['git', 'show', git_spec]
        result = subprocess.run(command, capture_output=True, check=True)
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

# -------- Helper functions --------
def detect_lang(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext == ".java":
        return "java"
    return None

def slice_text(buf: bytes, node) -> str:
    return buf[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

# -------- Code extraction now uses content and adds hashing --------
def extract_code_elements(file_path: Path, buf: bytes) -> List[Dict]:
    lang = detect_lang(file_path)
    if not lang:
        return []

    language = get_language(lang)
    parser = get_parser(lang)
    tree = parser.parse(buf)
    root = tree.root_node

    if lang == "python":
        query = language.query(r"(function_definition) @decl")
        kind_map = {"function_definition": "function"}
    else: # java
        query = language.query(r"""
          (method_declaration) @decl
          (constructor_declaration) @decl
        """)
        kind_map = {
            "method_declaration": "method",
            "constructor_declaration": "constructor",
        }

    items: List[Dict] = []
    for _, capdict in query.matches(root):
        captured_nodes = capdict.get("decl")
        if not captured_nodes:
            continue
        d = captured_nodes[0]
        name_node = d.child_by_field_name("name")
        name = slice_text(buf, name_node) if name_node else "<no-name>"
        text = slice_text(buf, d)
        # Create a unique string by combining the file path and the function text
        unique_string = f"{str(file_path)}::{text}"
        content_hash = hashlib.sha256(unique_string.encode("utf-8")).hexdigest()

        items.append({
            "id": content_hash, "name": name, "kind": kind_map.get(d.type, d.type),
            "start_line": d.start_point[0] + 1, "end_line": d.end_point[0] + 1,
            "text": text, "hash": content_hash,
        })
    return items

# -------- Function to show query results --------
def show_query_results(results: Dict, query_element: Dict):
    """Parses and prints ChromaDB query results in a readable format."""
    print("-" * 25)
    print(f"Query for code similar to '{query_element['name']}' in '{query_element['file_path']}':")
    
    ids = results['ids'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    # The first result is always the item itself, so we need at least 2 to find a match.
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

# -------- Core processing logic --------
def process_modified_file(file_path: Path):
    """
    Compares the staged version of a file with its HEAD version,
    updates the DB, and runs similarity queries. This also handles newly added files.
    """
    content_before = get_file_content_from_git('HEAD', str(file_path))
    content_after = get_file_content_from_git('', str(file_path))

    if not content_after:
        print(f"[warn] Could not read staged content for {file_path}. Skipping.")
        return

    elements_before = extract_code_elements(file_path, content_before) if content_before else []
    elements_after = extract_code_elements(file_path, content_after)

    before_map = {el['hash']: el for el in elements_before}
    after_map = {el['hash']: el for el in elements_after}

    hashes_before = set(before_map.keys())
    hashes_after = set(after_map.keys())

    hashes_to_delete = hashes_before - hashes_after
    hashes_to_add = hashes_after - hashes_before

    if hashes_to_delete:
        print("Detected changes, updating database...")
        for h in hashes_to_delete:
            # Look up the function name from the 'before' state
            print(f"  - Deleting old version of: {before_map[h]['name']}")
        code_collection.delete(ids=list(hashes_to_delete))

    if not hashes_to_add:
        print("No new or modified functions to add.")
        return

    for h in hashes_to_add:
        # Look up the function name from the 'after' state
        print(f"  + Adding new version of: {after_map[h]['name']}")
    elements_to_embed = [after_map[h] for h in hashes_to_add]

    try:
        payloads = [it["text"] for it in elements_to_embed]
        result = voyage_ai.embed(payloads, model="voyage-code-2", input_type="document")
        embeddings = result.embeddings
    except Exception as e:
        print(f"[error] Voyage embedding failed: {e}")
        return

    ids = [el['id'] for el in elements_to_embed]
    metadatas = [{
        "file_path": str(file_path),
        "function_name": el['name'],
        "kind": el['kind'],
        "start_line": el['start_line'],
        "end_line": el['end_line'],
        "content_hash": el['hash']
    } for el in elements_to_embed]
    documents = [el["text"] for el in elements_to_embed]
    
    code_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print("Database updated successfully.")

    print("\nRunning similarity queries for new/modified functions...")
    for i, element in enumerate(elements_to_embed):
        query_details = {"name": element['name'], "file_path": str(file_path)}
        similar_items = code_collection.query(
            query_embeddings=[embeddings[i]],
            n_results=3
        )
        show_query_results(similar_items, query_details)

def delete_all_elements_for_file(file_path_str: str):
    """Deletes all vector DB entries associated with a deleted file."""
    print(f"File was deleted. Removing all associated functions from the database...")
    existing_items = code_collection.get(where={"file_path": file_path_str})
    if existing_items and existing_items['ids']:
        print(f"Deleting {len(existing_items['ids'])} function(s) for deleted file: {file_path_str}")
        code_collection.delete(ids=existing_items['ids'])
    else:
        print("No functions found in the database for this file.")

# -------- Main function is now a command-line router --------
def main():
    if len(sys.argv) < 3:
        print("Usage: python3 code_similarity.py <command> <path-to-file>")
        print("Commands: --modified, --deleted")
        sys.exit(3)

    command = sys.argv[1]
    target_path_str = sys.argv[2]
    
    if command == "--modified":
        process_modified_file(Path(target_path_str))
    elif command == "--deleted":
        delete_all_elements_for_file(target_path_str)
    else:
        print(f"Unknown command: {command}")
        sys.exit(4)

if __name__ == "__main__":
    main()