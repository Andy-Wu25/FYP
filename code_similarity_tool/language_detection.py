from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, get_args

from identify.identify import tags_from_filename, tags_from_path
from tree_sitter_language_pack import SupportedLanguage, get_language


GENERIC_IDENTIFY_TAGS = {
    "text",
    "binary",
    "file",
    "directory",
    "symlink",
    "non-executable",
    "executable",
}

HINT_TO_TREE_SITTER = {
    "bash": "bash",
    "c": "c",
    "c#": "csharp",
    "c++": "cpp",
    "clojure": "clojure",
    "cmake": "cmake",
    "cpp": "cpp",
    "cs": "csharp",
    "css": "css",
    "dart": "dart",
    "dockerfile": "dockerfile",
    "elixir": "elixir",
    "erlang": "erlang",
    "fortran": "fortran",
    "fsharp": "fsharp",
    "go": "go",
    "graphql": "graphql",
    "haskell": "haskell",
    "html": "html",
    "java": "java",
    "javascript": "javascript",
    "json": "json",
    "kotlin": "kotlin",
    "lua": "lua",
    "makefile": "make",
    "markdown": "markdown",
    "matlab": "matlab",
    "nix": "nix",
    "objective-c": "objc",
    "objective-c++": "objc",
    "ocaml": "ocaml",
    "perl": "perl",
    "php": "php",
    "powershell": "powershell",
    "proto": "proto",
    "python": "python",
    "r": "r",
    "ruby": "ruby",
    "rust": "rust",
    "scala": "scala",
    "shell": "bash",
    "solidity": "solidity",
    "sql": "sql",
    "svelte": "svelte",
    "swift": "swift",
    "terraform": "terraform",
    "toml": "toml",
    "ts": "typescript",
    "tsx": "tsx",
    "typescript": "typescript",
    "vue": "vue",
    "xml": "xml",
    "yaml": "yaml",
    "zig": "zig",
}

SUFFIX_TO_TREE_SITTER = {
    ".cs": "csharp",
    ".cxx": "cpp",
    ".hxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".cc": "cpp",
    ".c++": "cpp",
    ".cpp": "cpp",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".kts": "kotlin",
    ".zsh": "bash",
    ".sh": "bash",
}

BASENAME_TO_TREE_SITTER = {
    "dockerfile": "dockerfile",
    "makefile": "make",
    "gnumakefile": "make",
    "cmakelists.txt": "cmake",
}

SUPPORTED_TREE_SITTER_LANGUAGES = set(get_args(SupportedLanguage))


def is_probably_binary(data: bytes) -> bool:
    if not data:
        return False
    if b"\x00" in data:
        return True
    sample = data[:4096]
    non_text = sum(1 for b in sample if b < 9 or (13 < b < 32))
    return (non_text / len(sample)) > 0.30


@lru_cache(maxsize=None)
def _is_tree_sitter_language_available(language: str) -> bool:
    try:
        get_language(language)
    except Exception:  # noqa: BLE001
        return False
    return True


def _identify_language_hints(path: Path) -> List[str]:
    try:
        if path.exists():
            tags = tags_from_path(str(path))
        else:
            tags = tags_from_filename(path.name)
    except Exception:  # noqa: BLE001
        tags = tags_from_filename(path.name)

    hints: List[str] = []
    for tag in sorted(tags):
        lowered = tag.lower()
        if lowered in GENERIC_IDENTIFY_TAGS:
            continue
        hints.append(lowered)
    return hints


def language_hints(path: Path) -> List[str]:
    hints: List[str] = []

    def _push(raw: str) -> None:
        value = raw.strip().lower()
        if not value:
            return
        if value not in hints:
            hints.append(value)

    for hint in _identify_language_hints(path):
        _push(hint)

    basename = path.name.lower()
    mapped_basename = BASENAME_TO_TREE_SITTER.get(basename)
    if mapped_basename:
        _push(mapped_basename)

    suffix = path.suffix.lower()
    mapped_suffix = SUFFIX_TO_TREE_SITTER.get(suffix)
    if mapped_suffix:
        _push(mapped_suffix)
    if suffix:
        _push(suffix)
        _push(suffix.lstrip("."))

    return hints


def has_language_hint(path: Path) -> bool:
    if language_hints(path):
        return True
    return bool(path.suffix)


def _hint_to_tree_sitter_language(hint: str) -> Optional[str]:
    lowered = hint.lower().strip()
    if lowered in HINT_TO_TREE_SITTER:
        return HINT_TO_TREE_SITTER[lowered]
    if lowered.startswith("."):
        mapped = SUFFIX_TO_TREE_SITTER.get(lowered)
        if mapped:
            return mapped
        lowered = lowered[1:]
    if lowered in SUPPORTED_TREE_SITTER_LANGUAGES:
        return lowered
    compact = lowered.replace("-", "").replace("_", "")
    for lang in SUPPORTED_TREE_SITTER_LANGUAGES:
        if compact == lang.replace("_", ""):
            return lang
    return None


def detect_tree_sitter_language(path: Path) -> Optional[str]:
    for hint in language_hints(path):
        lang = _hint_to_tree_sitter_language(hint)
        if not lang:
            continue
        if _is_tree_sitter_language_available(lang):
            return lang
    return None
