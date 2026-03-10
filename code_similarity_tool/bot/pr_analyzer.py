"""Core PR analysis engine.

Takes PR file metadata + file contents, runs the existing
extract → embed → query → filter pipeline (unchanged from CLI),
and returns structured findings per code element.

This module is synchronous — the async webhook handler offloads
it to a thread pool so the asyncio event loop never blocks.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..checking.check_utils import extract_hits
from ..infra.clients import CodeVectorStore
from ..core.code_parser import CodeElement, extract_code_elements
from ..infra.embeddings import EmbeddingClient, load_embedding_config
from ..core.language_detection import has_language_hint, is_probably_binary
from ..core.runtime import load_runtime_context, parse_staged_new_ranges
from ..core.source_filtering import is_noise_source_path
from .config import BotConfig
from .github_api import PRFile

log = logging.getLogger(__name__)


@dataclass
class ElementFinding:
    """All similarity hits for a single code element in the PR."""

    rel_path: str
    element: CodeElement
    public_hits: List[Tuple[float, Dict]] = field(default_factory=list)
    private_hits: List[Tuple[float, Dict]] = field(default_factory=list)

    @property
    def has_hits(self) -> bool:
        return bool(self.public_hits or self.private_hits)


@dataclass
class AnalysisResult:
    """Bundle of findings + the resolved embedding config for the PR comment."""

    findings: List[ElementFinding]
    embedding_config: Optional[Dict[str, Any]] = None


def _max_file_bytes() -> int:
    try:
        return max(1, int(os.getenv("CODE_SIM_MAX_FILE_BYTES", "250000")))
    except ValueError:
        return 250000


def _include_public_hit(meta: Dict, query_rel: str, query_hash: str) -> bool:
    return True


def _make_private_hit_filter(org_id: str, repo_id: str):
    """Exclude exact self-matches (same org + repo + file + content)."""

    def _include_hit(meta: Dict, query_rel: str, query_hash: str) -> bool:
        return not (
            meta.get("org_id") == org_id
            and meta.get("repo_id") == repo_id
            and meta.get("file_path") == query_rel
            and meta.get("content_hash") == query_hash
        )

    return _include_hit


def _element_overlaps_any_changed_range(
    start_line: int,
    end_line: int,
    changed_ranges: List[Tuple[int, int]],
) -> bool:
    for changed_start, changed_end in changed_ranges:
        if changed_end < start_line:
            continue
        if changed_start > end_line:
            continue
        return True
    return False


def analyze_pr(
    owner: str,
    repo: str,
    pr_number: int,
    head_sha: str,
    pr_files: List[PRFile],
    file_contents: Dict[str, Optional[bytes]],
    cfg: BotConfig,
) -> AnalysisResult:
    """Run the full similarity analysis pipeline on a PR.

    Parameters
    ----------
    owner, repo, pr_number, head_sha:
        Identify the PR.
    pr_files:
        Changed file list from GitHub API (includes per-file unified diff patch).
    file_contents:
        Map of filename → raw bytes fetched from GitHub Contents API.
        Values may be None if the file could not be fetched.
    cfg:
        Bot configuration.

    Returns
    -------
    AnalysisResult containing findings and the resolved embedding config.
    """
    ctx = load_runtime_context()
    max_bytes = _max_file_bytes()

    # ------------------------------------------------------------------ #
    # Step 1 – collect (rel_path, element) pairs for changed code only
    # ------------------------------------------------------------------ #
    candidates: List[Tuple[str, CodeElement]] = []

    for pr_file in pr_files:
        filename = pr_file.filename

        if pr_file.status == "removed":
            continue

        path_obj = Path(filename)
        if is_noise_source_path(path_obj):
            continue
        if not has_language_hint(path_obj):
            continue

        content = file_contents.get(filename)
        if content is None:
            log.debug("Skipping %s: content unavailable", filename)
            continue
        if len(content) > max_bytes:
            log.warning("Skipping %s: file too large (%d bytes)", filename, len(content))
            continue
        if is_probably_binary(content[:4096]):
            log.debug("Skipping %s: detected as binary", filename)
            continue

        # Parse changed line ranges from the per-file unified diff patch.
        # parse_staged_new_ranges() expects the same @@ ... @@ hunk header
        # format that GitHub returns — no adaptation needed.
        patch = pr_file.patch or ""
        changed_ranges = parse_staged_new_ranges(patch)

        elements = extract_code_elements(path_obj, content)

        for element in elements:
            # Newly added files: include all elements (no existing content to diff)
            if pr_file.status == "added" or not changed_ranges:
                candidates.append((filename, element))
            elif _element_overlaps_any_changed_range(
                element["start_line"], element["end_line"], changed_ranges
            ):
                candidates.append((filename, element))

    if not candidates:
        log.info(
            "[bot] No queryable code elements found in PR %s/%s#%d", owner, repo, pr_number
        )
        return AnalysisResult(findings=[])

    # Cap total elements to protect against huge PRs
    truncated = len(candidates) > cfg.max_elements_per_pr
    if truncated:
        log.warning(
            "[bot] PR %s/%s#%d has %d elements; capping at %d",
            owner,
            repo,
            pr_number,
            len(candidates),
            cfg.max_elements_per_pr,
        )
        candidates = candidates[: cfg.max_elements_per_pr]

    # ------------------------------------------------------------------ #
    # Step 2 – embed all elements in one batch
    # ------------------------------------------------------------------ #
    log.info(
        "[bot] Embedding %d element(s) from PR %s/%s#%d", len(candidates), owner, repo, pr_number
    )
    # Load index-time embedding config to match query preparation
    index_cfg_priv = load_embedding_config(ctx.db_path, "private")
    index_cfg_pub = load_embedding_config(ctx.db_path, "public")
    configs = [c for c in [index_cfg_priv, index_cfg_pub] if c is not None]

    resolved_max_chars = None
    resolved_long_text_mode = None
    if len(configs) == 1:
        resolved_max_chars = configs[0].get("max_chars")
        resolved_long_text_mode = configs[0].get("long_text_mode")
    elif len(configs) == 2 and configs[0] == configs[1]:
        resolved_max_chars = configs[0].get("max_chars")
        resolved_long_text_mode = configs[0].get("long_text_mode")
    elif len(configs) == 2:
        log.warning(
            "[bot] Private and public indexes have different embedding configs: "
            "private=%s public=%s. Using env/default config for queries.",
            index_cfg_priv, index_cfg_pub,
        )

    embedder = EmbeddingClient(max_chars=resolved_max_chars, long_text_mode=resolved_long_text_mode)
    labels = [f"{rel}:{el['name']}" for rel, el in candidates]

    vectors = embedder.embed_documents(
        [el["text"] for _, el in candidates],
        labels=labels,
    )

    # ------------------------------------------------------------------ #
    # Step 3 – query ChromaDB and collect hits
    # ------------------------------------------------------------------ #
    store = CodeVectorStore(
        path=str(ctx.db_path),
        private_collection_name=ctx.private_collection_name,
        public_collection_name=ctx.public_collection_name,
        metric=ctx.metric,
    )

    private_hit_filter = _make_private_hit_filter(ctx.org_id, ctx.repo_id)
    findings: List[ElementFinding] = []

    for (rel_path, element), vector in zip(candidates, vectors):
        public_hits: List[Tuple[float, Dict]] = []
        private_hits: List[Tuple[float, Dict]] = []

        if cfg.check_public:
            results = store.query_public_by_embedding(vector, n_results=cfg.pool_size)
            public_hits = extract_hits(
                results,
                top_k=cfg.top_k,
                max_distance=cfg.max_distance,
                query_rel=rel_path,
                query_hash=element["hash"],
                include_hit=_include_public_hit,
            )

        if cfg.check_private:
            results = store.query_private_by_embedding(
                vector, org_id=ctx.org_id, n_results=cfg.pool_size
            )
            private_hits = extract_hits(
                results,
                top_k=cfg.top_k,
                max_distance=cfg.max_distance,
                query_rel=rel_path,
                query_hash=element["hash"],
                include_hit=private_hit_filter,
            )

        findings.append(
            ElementFinding(
                rel_path=rel_path,
                element=element,
                public_hits=public_hits,
                private_hits=private_hits,
            )
        )

    return AnalysisResult(
        findings=findings,
        embedding_config={
            "max_chars": embedder.max_chars,
            "long_text_mode": embedder.long_text_mode,
            "chunk_overlap": embedder.chunk_overlap,
            "model": embedder.model,
        },
    )
