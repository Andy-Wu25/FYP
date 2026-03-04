from __future__ import annotations

import argparse
import logging
import os

from .clients import CodeVectorStore
from .public_index import parse_github_url
from .runtime import load_runtime_context


def _configure_logging() -> logging.Logger:
    log_level = os.getenv("CODE_SIM_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="[%(levelname)s] %(message)s")
    return logging.getLogger(__name__)


def remove_public_github_repo(url: str) -> int:
    log = _configure_logging()
    ctx = load_runtime_context()

    _owner, _repo, canonical_url = parse_github_url(url)
    store = CodeVectorStore(
        path=str(ctx.db_path),
        private_collection_name=ctx.private_collection_name,
        public_collection_name=ctx.public_collection_name,
        metric=ctx.metric,
    )

    removed = store.delete_public_repo_entries(canonical_url)
    if removed:
        log.info("[public-remove] removed %d public element(s) for %s", removed, canonical_url)
    else:
        log.info("[public-remove] no public entries found for %s", canonical_url)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-remove-public",
        description="Remove all indexed public entries for a GitHub repository from the central DB.",
    )
    parser.add_argument("url", help="GitHub repository URL (https://github.com/<owner>/<repo>)")
    args = parser.parse_args()

    try:
        remove_public_github_repo(args.url)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
