#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, TypedDict

from tree_sitter_language_pack import get_language, get_parser


class CodeElement(TypedDict):
    id: str
    name: str
    kind: str
    start_line: int
    end_line: int
    text: str
    hash: str


def detect_lang(path: Path) -> Optional[str]:
    suffix = path.suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".java":
        return "java"
    return None


def make_element_id(org_id: str, repo_id: str, rel_path: str, content_hash: str) -> str:
    raw = f"{org_id}:{repo_id}:{rel_path}:{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _slice(buf: bytes, node) -> str:
    return buf[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def extract_code_elements(file_path: Path, buf: Optional[bytes]) -> List[CodeElement]:
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
        query = language.query(r"(function_definition) @decl")
        kind_map = {"function_definition": "function"}
    else:
        query = language.query(
            r"""
            (method_declaration) @decl
            (constructor_declaration) @decl
            """
        )
        kind_map = {
            "method_declaration": "method",
            "constructor_declaration": "constructor",
        }

    items: List[CodeElement] = []
    for _, caps in query.matches(root):
        decl_nodes = caps.get("decl")
        if not decl_nodes:
            continue

        decl = decl_nodes[0]
        name_node = decl.child_by_field_name("name")
        name = _slice(buf, name_node) if name_node else "<no-name>"
        text = _slice(buf, decl)
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        items.append(
            {
                "id": content_hash,
                "name": name,
                "kind": kind_map.get(decl.type, decl.type),
                "start_line": decl.start_point[0] + 1,
                "end_line": decl.end_point[0] + 1,
                "text": text,
                "hash": content_hash,
            }
        )

    return items
