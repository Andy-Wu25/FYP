#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Optional, TypedDict

from tree_sitter_language_pack import get_language, get_parser

from .language_detection import detect_tree_sitter_language


class CodeElement(TypedDict):
    id: str
    name: str
    kind: str
    start_line: int
    end_line: int
    text: str
    hash: str


def detect_lang(path: Path) -> Optional[str]:
    return detect_tree_sitter_language(path)


def make_element_id(org_id: str, repo_id: str, rel_path: str, content_hash: str) -> str:
    raw = f"{org_id}:{repo_id}:{rel_path}:{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _slice(buf: bytes, node) -> str:
    return buf[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


DECL_PRIMARY_TOKENS = (
    "function",
    "method",
    "constructor",
    "func",
    "procedure",
    "subprogram",
    "subroutine",
)
DECL_CONTEXT_TOKENS = (
    "declaration",
    "definition",
    "declarator",
    "item",
    "body",
    "statement",
    "spec",
)
DECL_EXCLUDE_TOKENS = (
    "call",
    "type",
    "parameter",
    "argument",
    "literal",
    "expression",
    "comment",
    "signature",
)
IDENTIFIER_TYPE_RE = re.compile(r"(identifier|name)$")


def _line_count(text: str) -> int:
    if not text:
        return 1
    return text.count("\n") + 1


def _file_level_elements(file_path: Path, buf: bytes) -> List[CodeElement]:
    text = buf.decode("utf-8", errors="replace")
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return [
        {
            "id": content_hash,
            "name": f"<file:{file_path.name}>",
            "kind": "file",
            "start_line": 1,
            "end_line": _line_count(text),
            "text": text,
            "hash": content_hash,
        }
    ]


def _is_declaration_like_node_type(node_type: str) -> bool:
    lowered = node_type.lower()
    if not any(token in lowered for token in DECL_PRIMARY_TOKENS):
        return False
    if any(token in lowered for token in DECL_EXCLUDE_TOKENS):
        return False
    if any(token in lowered for token in DECL_CONTEXT_TOKENS):
        return True
    return lowered in {
        "func_item",
        "function_item",
        "method_item",
        "constructor_item",
    }


def _kind_from_node_type(node_type: str) -> str:
    lowered = node_type.lower()
    if "constructor" in lowered:
        return "constructor"
    if "method" in lowered:
        return "method"
    if any(token in lowered for token in ("function", "func", "procedure", "subprogram", "subroutine")):
        return "function"
    return lowered


def _iter_named_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_named:
            yield node
        children = getattr(node, "children", None) or []
        for child in reversed(children):
            stack.append(child)


def _first_identifier_descendant(node, *, max_depth: int = 4):
    queue = [(node, 0)]
    while queue:
        current, depth = queue.pop(0)
        current_type = getattr(current, "type", "")
        if IDENTIFIER_TYPE_RE.search(current_type):
            return current
        if depth >= max_depth:
            continue
        children = getattr(current, "children", None) or []
        for child in children:
            queue.append((child, depth + 1))
    return None


def _name_for_declaration(buf: bytes, decl) -> str:
    for field_name in ("name", "declarator", "identifier"):
        node = decl.child_by_field_name(field_name)
        if node is None:
            continue
        identifier = _first_identifier_descendant(node)
        if identifier is not None:
            text = _slice(buf, identifier).strip()
            if text:
                return text
        text = _slice(buf, node).strip()
        if text:
            return text.splitlines()[0][:160]

    identifier = _first_identifier_descendant(decl)
    if identifier is not None:
        text = _slice(buf, identifier).strip()
        if text:
            return text
    return "<no-name>"


def extract_code_elements(file_path: Path, buf: Optional[bytes]) -> List[CodeElement]:
    if not buf:
        return []

    lang = detect_lang(file_path)
    if not lang:
        return _file_level_elements(file_path, buf)

    try:
        _ = get_language(lang)
        parser = get_parser(lang)
        tree = parser.parse(buf)
    except Exception:  # noqa: BLE001
        return _file_level_elements(file_path, buf)

    root = tree.root_node

    items: List[CodeElement] = []
    seen_hashes = set()
    for decl in _iter_named_nodes(root):
        node_type = getattr(decl, "type", "")
        if not _is_declaration_like_node_type(node_type):
            continue

        text = _slice(buf, decl).strip()
        if not text:
            continue
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        name = _name_for_declaration(buf, decl)

        items.append(
            {
                "id": content_hash,
                "name": name,
                "kind": _kind_from_node_type(node_type),
                "start_line": decl.start_point[0] + 1,
                "end_line": decl.end_point[0] + 1,
                "text": text,
                "hash": content_hash,
            }
        )

    if not items:
        return _file_level_elements(file_path, buf)

    return items
