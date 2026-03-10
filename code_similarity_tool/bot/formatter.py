"""GitHub PR comment formatter.

Converts a list of ElementFinding objects into a GitHub-flavoured
Markdown string suitable for posting as a PR comment.

Design constraints:
- GitHub PR comments are capped at 65,536 characters.
- Uses <details>/<summary> for collapsible sections.
- Exactly one COMMENT_SIGNATURE marker for idempotent upsert.
- Pure function — no I/O, no side effects.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..infra.public_links import build_public_match_permalink
from .github_api import COMMENT_SIGNATURE
from .pr_analyzer import ElementFinding

# GitHub hard limit for comment body size
_GITHUB_COMMENT_MAX_CHARS = 65_000  # leave a small buffer below 65536


def format_report(
    findings: List[ElementFinding],
    owner: str,
    repo: str,
    pr_number: int,
    head_sha: str,
    max_distance: float,
    truncated: bool = False,
    top_k: int = 3,
    check_private: bool = True,
    check_public: bool = True,
    embedding_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Build the full PR comment body.

    Returns None if there are no findings and a silent mode is desired
    (caller should check comment_on_zero_hits before posting).
    """
    if not findings:
        return None

    total_elements = len(findings)
    findings_with_hits = [f for f in findings if f.has_hits]
    public_findings = [f for f in findings if f.public_hits]
    private_findings = [f for f in findings if f.private_hits]

    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    repo_full = f"{owner}/{repo}"

    lines: List[str] = []

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    lines.append(COMMENT_SIGNATURE)
    lines.append("## Code Similarity Report")
    lines.append("")

    scope_summary = (
        f"> **Scope**: {total_elements} element(s) checked "
        f"| threshold ≤ {max_distance} | *{now_utc}*"
    )
    if truncated:
        scope_summary += " ⚠️ *(element cap reached — run locally for full results)*"
    lines.append(scope_summary)
    lines.append("")

    run_cmd = f"`code-sim-check-public --scope files <paths>`"
    lines.append(f"*Run locally: {run_cmd}*")
    lines.append("")
    lines.append(_format_config_block(max_distance, top_k, check_private, check_public, embedding_config))
    lines.append("")
    lines.append("---")
    lines.append("")

    # ------------------------------------------------------------------ #
    # Findings section
    # ------------------------------------------------------------------ #
    if not findings_with_hits:
        lines.append("### ✅ No Similarities Found")
        lines.append("")
        lines.append(
            f"Checked **{total_elements}** code element(s) — "
            "no matches above the distance threshold."
        )
        lines.append("")
        lines.append(_checked_elements_details(findings))
    else:
        hit_count = sum(len(f.public_hits) + len(f.private_hits) for f in findings_with_hits)
        lines.append(
            f"### ⚠️ {len(findings_with_hits)} Element(s) with Similarities "
            f"({hit_count} total match(es))"
        )
        lines.append("")

        # Public findings
        if public_findings:
            lines.append("#### Public Index Matches")
            lines.append("")
            for finding in public_findings:
                lines.extend(_format_finding_block(finding, "public"))
                lines.append("")

        # Private findings
        if private_findings:
            lines.append("#### Private Org Matches")
            lines.append("")
            for finding in private_findings:
                lines.extend(_format_finding_block(finding, "private"))
                lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(_checked_elements_details(findings))

    body = "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Enforce GitHub comment size limit
    # ------------------------------------------------------------------ #
    if len(body) > _GITHUB_COMMENT_MAX_CHARS:
        truncation_note = (
            "\n\n---\n"
            "> ⚠️ *Comment truncated — too many findings to display. "
            "Run `code-sim-check-public --scope files <paths>` locally for full output.*\n"
            f"\n{COMMENT_SIGNATURE}"
        )
        body = body[: _GITHUB_COMMENT_MAX_CHARS - len(truncation_note)] + truncation_note

    return body


def format_zero_findings_report(
    total_elements: int,
    findings: List[ElementFinding],
    max_distance: float,
    top_k: int = 3,
    check_private: bool = True,
    check_public: bool = True,
    embedding_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Compact comment for when nothing similar is found."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        COMMENT_SIGNATURE,
        "## Code Similarity Report",
        "",
        f"> ✅ No similarities found — {total_elements} element(s) checked "
        f"| threshold ≤ {max_distance} | *{now_utc}*",
        "",
        _format_config_block(max_distance, top_k, check_private, check_public, embedding_config),
        "",
    ]
    if findings:
        lines.append(_checked_elements_details(findings))
    return "\n".join(lines)


def format_error_report(error_msg: str) -> str:
    """Fallback comment posted when analysis fails unexpectedly."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"{COMMENT_SIGNATURE}\n"
        "## Code Similarity Report\n\n"
        f"> ❌ Analysis failed at {now_utc}: {error_msg}\n\n"
        "*Please check the bot server logs or re-run locally.*"
    )


# ------------------------------------------------------------------ #
# Internal helpers
# ------------------------------------------------------------------ #


def _format_config_block(
    max_distance: float,
    top_k: int,
    check_private: bool,
    check_public: bool,
    embedding_config: Optional[Dict[str, Any]],
) -> str:
    """Collapsible block showing the active analysis configuration."""
    scope_parts = []
    if check_private:
        scope_parts.append("private")
    if check_public:
        scope_parts.append("public")
    scope = " + ".join(scope_parts) if scope_parts else "none"

    rows = [
        f"| Max distance | `{max_distance}` |",
        f"| Top-K | `{top_k}` |",
        f"| Indexes | `{scope}` |",
    ]
    if embedding_config:
        max_chars = embedding_config.get("max_chars", 0)
        max_chars_display = f"`{max_chars}`" if max_chars else "`0` (full context)"
        rows.append(f"| Embedding max chars | {max_chars_display} |")
        if max_chars:
            rows.append(f"| Long text mode | `{embedding_config.get('long_text_mode', 'chunk')}` |")
            rows.append(f"| Chunk overlap | `{embedding_config.get('chunk_overlap', 512)}` |")
        rows.append(f"| Model | `{embedding_config.get('model', '?')}` |")

    table = "\n".join(rows)
    return (
        "<details>\n"
        "<summary>Configuration</summary>\n\n"
        "| Setting | Value |\n"
        "|---------|-------|\n"
        f"{table}\n\n"
        "</details>"
    )


def _format_finding_block(finding: ElementFinding, index_type: str) -> List[str]:
    """Render one element's hits as a markdown block."""
    el = finding.element
    hits = finding.public_hits if index_type == "public" else finding.private_hits
    if not hits:
        return []

    header = (
        f"**`{finding.rel_path}` → `{el['name']}` "
        f"(lines {el['start_line']}–{el['end_line']})**"
    )
    lines = [header, ""]

    # Table of hits
    if index_type == "public":
        lines.append("| # | Repo | File | Function | Distance | License | Link |")
        lines.append("|---|------|------|----------|----------|---------|------|")
        for idx, (dist, meta) in enumerate(hits, start=1):
            func = meta.get("function_name", "<unknown>")
            repo_name = meta.get("repo_name", "<unknown>")
            file_path = meta.get("file_path", "<unknown>")
            license_val = meta.get("license", "?")
            permalink = build_public_match_permalink(meta)
            link = f"[view]({permalink})" if permalink else "—"
            lines.append(
                f"| {idx} | `{repo_name}` | `{file_path}` | `{func}` "
                f"| {dist:.4f} | {license_val} | {link} |"
            )
    else:
        lines.append("| # | Match | Distance | Repo | File |")
        lines.append("|---|-------|----------|------|------|")
        for idx, (dist, meta) in enumerate(hits, start=1):
            func = meta.get("function_name", "<unknown>")
            repo_name = meta.get("repo_name", "<unknown>")
            file_path = meta.get("file_path", "<unknown>")
            start = meta.get("start_line", "?")
            end = meta.get("end_line", "?")
            lines.append(
                f"| {idx} | `{func}` | {dist:.4f} "
                f"| `{repo_name}` | `{file_path}:{start}-{end}` |"
            )

    return lines


def _checked_elements_details(findings: List[ElementFinding]) -> str:
    """Collapsible table listing every element that was checked."""
    total = len(findings)
    hits_count = sum(1 for f in findings if f.has_hits)
    summary_text = (
        f"✅ {total - hits_count} / {total} elements had no matches"
        if hits_count
        else f"✅ All {total} elements had no matches"
    )

    rows = ["| File | Element | Kind | Lines |", "|------|---------|------|-------|"]
    for f in findings:
        el = f.element
        rows.append(
            f"| `{f.rel_path}` | `{el['name']}` | {el['kind']} "
            f"| {el['start_line']}–{el['end_line']} |"
        )

    table = "\n".join(rows)
    return (
        f"<details>\n"
        f"<summary>{summary_text} — click to expand all checked elements</summary>\n\n"
        f"{table}\n\n"
        f"</details>"
    )
