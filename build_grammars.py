#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List, Dict
from tree_sitter_language_pack import get_language, get_parser

# ---------- helpers ----------
def load_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        print(f"[error] File not found: {path}")
        sys.exit(1)

def decode_slice(buf: bytes, node) -> str:
    return buf[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

def print_items(header: str, items: List[Dict]):
    print(header)
    for it in items:
        print(f"{it['kind']}: {it['name']} (lines {it['start_line']}-{it['end_line']})")
        print(it["text"].strip(), "\n")

def detect_lang_from_ext(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext == ".py": return "python"
    if ext == ".java": return "java"
    return None

# ---------- extraction ----------
def extract_python_functions(code: bytes) -> List[Dict]:
    L = get_language("python")
    P = get_parser("python")
    tree = P.parse(code)
    root = tree.root_node

    q = L.query(r"(function_definition) @decl")
    items = []
    for _, caps in q.matches(root):  # 0.23.x returns (pattern_idx, {cap: [nodes...]})
        decls = caps.get("decl") or caps.get(b"decl") or []
        if not decls: continue
        d = decls[0]
        name_node = d.child_by_field_name("name")
        name = decode_slice(code, name_node) if name_node else "<no-name>"
        items.append({
            "name": name,
            "kind": "function",
            "start_line": d.start_point[0] + 1,
            "end_line": d.end_point[0] + 1,
            "text": decode_slice(code, d),
        })
    return items

def extract_java_methods(code: bytes) -> List[Dict]:
    L = get_language("java")
    P = get_parser("java")
    tree = P.parse(code)
    root = tree.root_node

    q = L.query(r"""
      (method_declaration) @decl
      (constructor_declaration) @decl
    """)
    items = []
    for _, caps in q.matches(root):
        decls = caps.get("decl") or caps.get(b"decl") or []
        if not decls: continue
        d = decls[0]
        name_node = d.child_by_field_name("name")
        name = decode_slice(code, name_node) if name_node else "<no-name>"
        kind = "method" if d.type == "method_declaration" else "constructor"
        items.append({
            "name": name,
            "kind": kind,
            "start_line": d.start_point[0] + 1,
            "end_line": d.end_point[0] + 1,
            "text": decode_slice(code, d),
        })
    return items

# ---------- main ----------
def main():
    if len(sys.argv) != 2:
        print("Usage: python3 build_grammars.py <path-to-file(.py|.java)>")
        sys.exit(2)

    target = Path(sys.argv[1])
    code = load_bytes(target)
    lang = detect_lang_from_ext(target)
    if not lang:
        print(f"[error] Unsupported file type: {target} (expected .py or .java)")
        sys.exit(3)

    if lang == "python":
        items = extract_python_functions(code)
        print_items(f"--- Found in Python Code ({target}) ---", items)
    else:
        items = extract_java_methods(code)
        print_items(f"--- Found in Java Code ({target}) ---", items)

if __name__ == "__main__":
    main()
