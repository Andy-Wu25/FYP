from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import quote


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def build_github_blob_url(source_url: str, source_commit: str, file_path: str) -> Optional[str]:
    base = (source_url or "").strip().rstrip("/")
    commit = (source_commit or "").strip()
    rel = (file_path or "").replace("\\", "/").strip("/")
    if not base or not commit or not rel:
        return None
    encoded_path = quote(rel, safe="/._-~")
    return f"{base}/blob/{commit}/{encoded_path}"


def build_github_commit_url(source_url: str, source_commit: str) -> Optional[str]:
    base = (source_url or "").strip().rstrip("/")
    commit = (source_commit or "").strip()
    if not base or not commit:
        return None
    return f"{base}/commit/{commit}"


def _line_anchor(start_line: Any, end_line: Any) -> Optional[str]:
    start = _as_int(start_line)
    end = _as_int(end_line)
    if start is None or start <= 0:
        return None
    if end is None or end < start:
        return f"#L{start}"
    if end == start:
        return f"#L{start}"
    return f"#L{start}-L{end}"


def build_public_match_permalink(meta: Dict[str, Any]) -> Optional[str]:
    file_url = meta.get("source_file_url")
    if isinstance(file_url, str) and file_url.strip():
        base = file_url.strip()
    else:
        base = build_github_blob_url(
            str(meta.get("source_url", "")),
            str(meta.get("source_commit", "")),
            str(meta.get("file_path", "")),
        )
        if base is None:
            return None

    anchor = _line_anchor(meta.get("start_line"), meta.get("end_line"))
    if anchor:
        return f"{base}{anchor}"
    return base


def build_public_commit_permalink(meta: Dict[str, Any]) -> Optional[str]:
    return build_github_commit_url(
        str(meta.get("source_url", "")),
        str(meta.get("source_commit", "")),
    )
