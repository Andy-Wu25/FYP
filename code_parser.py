#!/usr/bin/env python3
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, TypedDict
from tree_sitter_language_pack import get_language, get_parser

# -------- Type Definition --------
class CodeElement(TypedDict):
    id: str
    name: str
    kind: str
    start_line: int
    end_line: int
    text: str
    hash: str
    file_path: str # Add file_path here

# -------- Parsing & Extraction Functions --------

def detect_lang(path: Path) -> Optional[str]:
    """Detects the language from a file extension."""
    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext == ".java":
        return "java"
    return None

def slice_text(buf: bytes, node) -> str:
    """Slices text from the buffer given a tree-sitter node."""
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
            "file_path": str(file_path)
        })
    return items