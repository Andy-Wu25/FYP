#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ..infra.embeddings import add_embedding_args
from .index import sync_current_repo


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-update",
        description="Re-index the current repository (alias for code-sim-index).",
    )
    add_embedding_args(parser)
    args = parser.parse_args()

    sync_current_repo(
        action_name="update",
        max_chars=args.max_chars,
        long_text_mode=args.long_text_mode,
    )


if __name__ == "__main__":
    main()
