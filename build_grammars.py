#!/usr/bin/env python3
import sys
import voyageai
from pathlib import Path
from typing import List, Dict
from tree_sitter_language_pack import get_language, get_parser

# -------- Voyage client --------
try:
    voyage_ai = voyageai.Client()  # reads VOYAGE_API_KEY from env
except Exception as e:
    print("Voyage AI client init failed (set VOYAGE_API_KEY):", e)
    sys.exit(1)

# -------- helpers --------
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

# -------- extraction --------
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
        # capture whole function declarations; get name from field
        query = language.query(r"(function_definition) @decl")
        kind_map = {"function_definition": "function"}
    else:  # java
        query = language.query(r"""
          (method_declaration) @decl
          (constructor_declaration) @decl
        """)
        kind_map = {
            "method_declaration": "method",
            "constructor_declaration": "constructor",
        }

    items: List[Dict] = []
    for _, capdict in query.matches(root):  # 0.23.x API
        decls = capdict.get("decl") or capdict.get(b"decl") or []
        if not decls:
            continue
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

# -------- main --------
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 build_grammars.py <path-to-file(.py|.java)>")
        sys.exit(3)

    target = Path(sys.argv[1])
    elements = extract_code_elements(target)

    if not elements:
        print(f"No functions/methods found in {target}.")
        return

    print(f"--- Found {len(elements)} code elements in {target} ---")
    for it in elements:
        print("-" * 20)
        print(f"{it['kind']}: {it['name']} (lines {it['start_line']}-{it['end_line']})")
        print(it["text"].strip())

    print("\n--- Getting embeddings from Voyage AI (voyage-code-2) ---")
    try:
        payloads = [it["text"] for it in elements]
        result = voyage_ai.embed(payloads, model="voyage-code-2", input_type="document")
        print(f"Successfully received {len(result.embeddings)} embeddings.")
        for i, emb in enumerate(result.embeddings, 1):
            print(f"Embedding {i}: dim={len(emb)}, preview={emb[:4]}")
    except Exception as e:
        print("Voyage embedding error:", e)

if __name__ == "__main__":
    main()
