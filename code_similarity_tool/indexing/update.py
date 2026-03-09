#!/usr/bin/env python3
from __future__ import annotations

from .index import sync_current_repo


def main() -> None:
    sync_current_repo(action_name="update")


if __name__ == "__main__":
    main()
