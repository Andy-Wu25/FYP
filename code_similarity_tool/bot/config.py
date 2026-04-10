"""Bot configuration loaded from CLI flags, environment variables, or defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class BotConfig:
    # GitHub App identity (env-only — secrets should not be CLI flags)
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

    # Match filtering
    min_lines: int            # hide matched segments shorter than this (0 = no filter)

    # Scope guards
    allowed_orgs: frozenset   # GitHub owner names; empty = allow all
    max_files_per_pr: int     # skip PRs with more changed files than this
    max_elements_per_pr: int  # cap total code elements analysed per PR

    @classmethod
    def from_env(
        cls,
        *,
        max_distance: Optional[float] = None,
        top_k: Optional[int] = None,
        check_public: Optional[bool] = None,
        check_private: Optional[bool] = None,
        comment_on_zero_hits: Optional[bool] = None,
        min_lines: Optional[int] = None,
        allowed_orgs: Optional[str] = None,
        max_files_per_pr: Optional[int] = None,
        max_elements_per_pr: Optional[int] = None,
    ) -> "BotConfig":
        """Build config from CLI kwargs (if provided), then env vars, then defaults."""
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

        def _bool_env(var: str, default: bool) -> bool:
            raw = os.environ.get(var, "").strip().lower()
            if not raw:
                return default
            return raw not in {"0", "false"}

        resolved_max_distance = (
            max_distance if max_distance is not None
            else float(os.environ.get("GITHUB_BOT_MAX_DISTANCE", "0.15"))
        )
        resolved_top_k = max(1, (
            top_k if top_k is not None
            else int(os.environ.get("GITHUB_BOT_TOP_K", "3"))
        ))
        resolved_check_public = (
            check_public if check_public is not None
            else _bool_env("GITHUB_BOT_CHECK_PUBLIC", True)
        )
        resolved_check_private = (
            check_private if check_private is not None
            else _bool_env("GITHUB_BOT_CHECK_PRIVATE", True)
        )
        resolved_comment_zero = (
            comment_on_zero_hits if comment_on_zero_hits is not None
            else _bool_env("GITHUB_BOT_COMMENT_ZERO", True)
        )
        resolved_min_lines = max(0, (
            min_lines if min_lines is not None
            else int(os.environ.get("GITHUB_BOT_MIN_LINES", "0"))
        ))

        allowed_orgs_raw = (
            allowed_orgs if allowed_orgs is not None
            else os.environ.get("GITHUB_BOT_ALLOWED_ORGS", "").strip()
        )
        resolved_allowed_orgs = frozenset(
            o.strip() for o in allowed_orgs_raw.split(",") if o.strip()
        )

        resolved_max_files = max(1, (
            max_files_per_pr if max_files_per_pr is not None
            else int(os.environ.get("GITHUB_BOT_MAX_FILES", "50"))
        ))
        resolved_max_elements = max(1, (
            max_elements_per_pr if max_elements_per_pr is not None
            else int(os.environ.get("GITHUB_BOT_MAX_ELEMENTS", "200"))
        ))

        return cls(
            app_id=app_id,
            private_key=private_key,
            webhook_secret=secret_str.encode(),
            max_distance=resolved_max_distance,
            top_k=resolved_top_k,
            pool_size=max(resolved_top_k * 4, 20),
            check_public=resolved_check_public,
            check_private=resolved_check_private,
            comment_on_zero_hits=resolved_comment_zero,
            min_lines=resolved_min_lines,
            allowed_orgs=resolved_allowed_orgs,
            max_files_per_pr=resolved_max_files,
            max_elements_per_pr=resolved_max_elements,
        )
