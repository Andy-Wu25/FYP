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
    parser.add_argument("--store-code", action="store_true", help="Store source code text in the vector database.")
    add_embedding_args(parser)
    args = parser.parse_args()

    sync_current_repo(
        action_name="update",
        truncate_tokens=args.truncate,
        store_code=args.store_code,
    )


if __name__ == "__main__":
    main()
