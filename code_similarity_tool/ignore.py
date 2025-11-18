#!/usr/bin/env python3
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass
class _Rule:
    pattern: str
    negated: bool
    dir_only: bool
    anchored: bool
    regex: re.Pattern

def _glob_to_regex(pat: str, anchored: bool) -> re.Pattern:
    """
    Convert a gitignore-like glob to a regex over forward-slash paths.
    Supports **, *, ?, and anchoring.
    """
    out = []
    i = 0
    while i < len(pat):
        c = pat[i]
        if c == '*':
            if i + 1 < len(pat) and pat[i + 1] == '*':
                out.append(".*")
                i += 2
                continue
            else:
                out.append("[^/]*")
        elif c == '?':
            out.append("[^/]")
        else:
            out.append(re.escape(c))
        i += 1

    body = "".join(out)
    if anchored:
        # match the path itself, or any descendant (for dir-like patterns)
        return re.compile(r"^" + body + r"(/.*)?$")
    # unanchored: allow match anywhere in the path segments
    return re.compile(r"(^|.*/)" + body + r"($|/.*)")

def _parse_line(raw: str) -> _Rule | None:
    line = raw.strip()
    if not line or line.startswith("#"):
        return None

    negated = False
    if line.startswith("!"):
        negated = True
        line = line[1:]

    dir_only = line.endswith("/")
    if dir_only:
        line = line[:-1]

    anchored = line.startswith("/")
    if anchored:
        line = line[1:]

    if not line:
        return None

    return _Rule(
        pattern=line,
        negated=negated,
        dir_only=dir_only,
        anchored=anchored,
        regex=_glob_to_regex(line, anchored=anchored),
    )

class IgnoreMatcher:
    def __init__(self, repo_root: Path, rules: List[_Rule]):
        self.repo_root = repo_root.resolve()
        self.rules = rules

    def allows(self, abs_path: Path, is_dir: bool | None = None) -> bool:
        """
        True => NOT ignored (allowed).
        False => ignored.
        Last matching rule wins, following gitignore-like semantics.
        """
        p = abs_path.resolve()
        try:
            rel = p.relative_to(self.repo_root)
        except Exception:
            # outside repo: treat as ignored
            return False

        rel_s = str(rel).replace("\\", "/")
        if is_dir is None:
            is_dir = p.is_dir()

        decision = True  # default allow

        for r in self.rules:
            if not r.regex.search(rel_s):
                continue

            if r.dir_only:
                # Special-case: pattern like "build/" should apply to:
                #   - the directory "build"
                #   - anything under "build/" (files or dirs)
                # but NOT to a regular file literally named "build".
                if rel_s == r.pattern and not is_dir:
                    # It's a file named exactly like the dir pattern -> skip this rule
                    continue
                # else: it's the directory itself or a descendant -> rule applies

            # Not negated => ignore; Negated => include
            decision = True if r.negated else False

        return decision

def load_ignore_file(repo_root: Path, filename: str = ".code-simignore") -> IgnoreMatcher:
    rules: List[_Rule] = []
    path = repo_root / filename
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            rule = _parse_line(raw)
            if rule:
                rules.append(rule)
    return IgnoreMatcher(repo_root, rules)
