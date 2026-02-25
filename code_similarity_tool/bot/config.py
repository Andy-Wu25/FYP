"""Bot configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BotConfig:
    # GitHub App identity
    app_id: str
    private_key: str          # PEM text of the GitHub App private key
    webhook_secret: bytes     # raw bytes of GITHUB_WEBHOOK_SECRET

    # Analysis thresholds
    max_distance: float       # cosine distance ceiling (lower = more similar)
    top_k: int                # max hits to show per element
    pool_size: int            # n_results fetched from ChromaDB before filtering

    # Feature flags
    check_public: bool        # query public index
    check_private: bool       # query private org index
    comment_on_zero_hits: bool  # post comment even when nothing found

    # Scope guards
    allowed_orgs: frozenset   # GitHub owner names; empty = allow all
    max_files_per_pr: int     # skip PRs with more changed files than this
    max_elements_per_pr: int  # cap total code elements analysed per PR

    @classmethod
    def from_env(cls) -> "BotConfig":
        app_id = os.environ.get("GITHUB_APP_ID", "").strip()
        if not app_id:
            raise ValueError(
                "GITHUB_APP_ID is required. "
                "Set it to the numeric App ID shown on your GitHub App settings page."
            )

        # Accept key as inline PEM text or as a path to the .pem file
        private_key = os.environ.get("GITHUB_APP_PRIVATE_KEY", "").strip()
        if not private_key:
            key_path = os.environ.get("GITHUB_APP_PRIVATE_KEY_PATH", "").strip()
            if key_path:
                private_key = Path(key_path).read_text(encoding="utf-8").strip()
        if not private_key:
            raise ValueError(
                "Either GITHUB_APP_PRIVATE_KEY (PEM text) or "
                "GITHUB_APP_PRIVATE_KEY_PATH (path to .pem file) is required."
            )

        secret_str = os.environ.get("GITHUB_WEBHOOK_SECRET", "").strip()
        if not secret_str:
            raise ValueError(
                "GITHUB_WEBHOOK_SECRET is required. "
                "Set it to the webhook secret you configured on your GitHub App."
            )

        allowed_orgs_raw = os.environ.get("GITHUB_BOT_ALLOWED_ORGS", "").strip()
        allowed_orgs = frozenset(
            o.strip() for o in allowed_orgs_raw.split(",") if o.strip()
        )

        top_k = max(1, int(os.environ.get("GITHUB_BOT_TOP_K", "3")))

        return cls(
            app_id=app_id,
            private_key=private_key,
            webhook_secret=secret_str.encode(),
            max_distance=float(os.environ.get("GITHUB_BOT_MAX_DISTANCE", "0.15")),
            top_k=top_k,
            pool_size=max(top_k * 4, 20),
            check_public=os.environ.get("GITHUB_BOT_CHECK_PUBLIC", "true").strip().lower()
            not in {"0", "false"},
            check_private=os.environ.get("GITHUB_BOT_CHECK_PRIVATE", "true").strip().lower()
            not in {"0", "false"},
            comment_on_zero_hits=os.environ.get("GITHUB_BOT_COMMENT_ZERO", "false").strip().lower()
            in {"1", "true"},
            allowed_orgs=allowed_orgs,
            max_files_per_pr=max(1, int(os.environ.get("GITHUB_BOT_MAX_FILES", "50"))),
            max_elements_per_pr=max(1, int(os.environ.get("GITHUB_BOT_MAX_ELEMENTS", "200"))),
        )
