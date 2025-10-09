#!/usr/bin/env python3
import sys
import voyageai
import chromadb
from pathlib import Path
from typing import List, Dict
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

# -------- Helper functions --------
def load_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        print(f"[error] File not found: {path}")
        sys.exit(1)

def detect_lang(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext == ".java":
        return "java"
    return None

def slice_text(buf: bytes, node) -> str:
    return buf[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

# -------- Code extraction function --------
def extract_code_elements(path: Path) -> List[Dict]:
    lang = detect_lang(path)
    if not lang:
        print(f"[error] Unsupported file type: {path} (expected .py or .java)")
        sys.exit(2)

    language = get_language(lang)
    parser = get_parser(lang)
    buf = load_bytes(path)
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
        decls = capdict.get("decl") or capdict.get(b"decl") or []
        if not decls: continue
        d = decls[0]
        name_node = d.child_by_field_name("name")
        name = slice_text(buf, name_node) if name_node else "<no-name>"
        items.append({
            "name": name,
            "kind": kind_map.get(d.type, d.type),
            "start_line": d.start_point[0] + 1,
            "end_line": d.end_point[0] + 1,
            "text": slice_text(buf, d),
        })
    return items

# -------- Function to format query results --------
def format_query_results(results: Dict, query_element: Dict):
    """Parses and prints ChromaDB query results in a readable format."""
    print("=" * 50)
    print(f"Query for code similar to '{query_element['name']}' in '{query_element['file_path']}':")
    print("=" * 50)

    ids = results['ids'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    documents = results['documents'][0]

    if len(ids) < 2:
        print("No other similar items found in the database.")
        return

    # Start from the second item (index 1) to skip the identical match to itself
    for i in range(1, len(ids)):
        metadata = metadatas[i]
        print(f"\nFound a similar item with score (distance): {distances[i]:.4f}")
        print(f"  File: {metadata['file_path']}")
        print(f"  Function: {metadata['function_name']} (lines {metadata['start_line']}-{metadata['end_line']})")
        print("  Code:")
        
        # Perform the replacement before the f-string
        indented_code = documents[i].strip().replace('\n', '\n    ')
        print(f"    {indented_code}")


# -------- Main execution logic --------
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 your_script.py <path-to-file(.py|.java)>")
        sys.exit(3)

    target_path = Path(sys.argv[1])
    elements = extract_code_elements(target_path)

    if not elements:
        print(f"No functions/methods found in {target_path}.")
        return

    print(f"--- Found {len(elements)} code elements in {target_path} ---")

    # --- Step 1: Get Embeddings from Voyage AI ---
    print("\n--- Getting embeddings from Voyage AI (voyage-code-2) ---")
    try:
        payloads = [it["text"] for it in elements]
        result = voyage_ai.embed(payloads, model="voyage-code-2", input_type="document")
        embeddings = result.embeddings
        print(f"Successfully received {len(embeddings)} embeddings.")
    except Exception as e:
        print("Voyage embedding error:", e)
        return

    # --- Step 2: Prepare and Add Data to ChromaDB ---
    print(f"\n--- Adding/updating {len(elements)} elements in ChromaDB ---")
    ids = [f"{target_path}:{el['name']}:{el['start_line']}" for el in elements]
    metadatas = [
        {
            "file_path": str(target_path),
            "function_name": el['name'],
            "kind": el['kind'],
            "start_line": el['start_line'],
            "end_line": el['end_line']
        } for el in elements
    ]
    documents = [el["text"] for el in elements]
    # Use upsert to add new items or update existing ones
    code_collection.upsert(
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print("Data upserted to ChromaDB successfully.")

    # --- MODIFIED: Step 3: Query for each element in the file ---
    print("\n--- Running Similarity Queries for Each New Element ---")
    if embeddings:
        # Loop through each element we just processed
        for i, element in enumerate(elements):
            query_element_details = {
                "name": element['name'],
                "file_path": str(target_path)
            }
            
            # Use the corresponding embedding for the query
            query_embedding = embeddings[i]
            
            # Find the top 3 most similar items (will include the item itself)
            similar_items = code_collection.query(
                query_embeddings=[query_embedding],
                n_results=3 
            )
            
            # Format and print the results for this specific element
            format_query_results(similar_items, query_element_details)

if __name__ == "__main__":
    main()