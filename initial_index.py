#!/usr/bin/env python3
import os
import sys
import hashlib
import voyageai
import chromadb
from pathlib import Path
from typing import List, Dict, Optional
from tree_sitter_language_pack import get_language, get_parser

# -------- Voyage AI & ChromaDB Clients --------
try:
    voyage_ai = voyageai.Client()
except Exception as e:
    print(f"[error] Voyage AI client init failed (set VOYAGE_API_KEY): {e}", file=sys.stderr)
    sys.exit(1)

DB_PATH = "vector_db"
COLLECTION_NAME = "project_code"
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# -------- Helper & Extraction Functions --------
def detect_lang(path: Path) -> Optional[str]:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext == ".java":
        return "java"
    return None

def slice_text(buf: bytes, node) -> str:
    return buf[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

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

# -------- Main Indexing Logic --------
def index_entire_project(root_dir: str = "."):
    """
    Scans an entire project directory, deletes the old database,
    and builds a new one from scratch.
    """
    print("--- Starting Initial Project Indexing ---")

    # FIX: Use a try-except block to safely delete the collection if it exists.
    try:
        print(f"Attempting to delete existing collection: '{COLLECTION_NAME}'...")
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print("Previous collection deleted.")
    except chromadb.errors.NotFoundError:
        print("No existing collection found. Starting fresh.")
    except Exception as e:
        # Catch other potential errors during deletion
        print(f"[error] Failed to delete collection: {e}", file=sys.stderr)
        sys.exit(1)

    # Create a fresh collection
    code_collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    print(f"Created new collection: '{COLLECTION_NAME}'")

    all_elements_to_add = []
    
    for subdir, _, files in os.walk(root_dir):
        if any(part.startswith('.') for part in Path(subdir).parts) or 'venv' in subdir:
            continue

        for file in files:
            file_path = Path(os.path.join(subdir, file))
            if file_path.suffix not in ['.py', '.java']:
                continue

            print(f"  -> Scanning: {file_path}")
            try:
                content = file_path.read_bytes()
                elements = extract_code_elements(file_path, content)
                if elements:
                    for el in elements:
                        el['file_path'] = str(file_path)
                    all_elements_to_add.extend(elements)
            except Exception as e:
                print(f"    [warn] Could not process file: {e}", file=sys.stderr)

    if not all_elements_to_add:
        print("No code elements found to index.")
        return

    print(f"\nFound a total of {len(all_elements_to_add)} functions/methods to index.")
    print("Getting embeddings from Voyage AI (this may take a moment)...")

    try:
        payloads = [it["text"] for it in all_elements_to_add]
        result = voyage_ai.embed(payloads, model="voyage-code-2", input_type="document")
        embeddings = result.embeddings
    except Exception as e:
        print(f"[error] Voyage embedding failed: {e}", file=sys.stderr)
        return

    print("Embeddings received. Adding to vector database...")

    ids = [el['id'] for el in all_elements_to_add]
    documents = [el['text'] for el in all_elements_to_add]
    metadatas = [{
        "file_path": el['file_path'], "function_name": el['name'], "kind": el['kind'],
        "start_line": el['start_line'], "end_line": el['end_line'], "content_hash": el['hash']
    } for el in all_elements_to_add]

    code_collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    print("\n✅ Initial indexing complete!")
    print("The pre-commit hook will now handle incremental updates.")

if __name__ == "__main__":
    # FIX: Assume the script is run from the project root. No need to ask.
    index_entire_project()