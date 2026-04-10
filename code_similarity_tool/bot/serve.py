"""CLI entry point: code-sim-bot-serve

Starts the GitHub PR bot webhook server.

Usage:
    code-sim-bot-serve [--host HOST] [--port PORT] [--max-distance 0.15] ...

Required environment variables (secrets — not available as flags):
    GITHUB_APP_ID               Numeric GitHub App ID
    GITHUB_APP_PRIVATE_KEY      PEM text  (or GITHUB_APP_PRIVATE_KEY_PATH)
    GITHUB_WEBHOOK_SECRET       Webhook secret configured on the GitHub App

All flags fall back to GITHUB_BOT_* env vars, then to defaults.
All existing CODE_SIM_* and VLLM_* variables apply unchanged.
"""
from __future__ import annotations

import argparse
import logging
import sys

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="code-sim-bot-serve",
        description=(
            "Start the GitHub PR bot webhook server. "
            "Requires GITHUB_APP_ID, GITHUB_APP_PRIVATE_KEY (or PATH), "
            "and GITHUB_WEBHOOK_SECRET as environment variables."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Server options
    server = parser.add_argument_group("server options")
    server.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    server.add_argument("--port", type=int, default=3000, help="Port to listen on (default: 3000)")
    server.add_argument("--log-level", default=None, help="Override log level (default: CODE_SIM_LOG_LEVEL or INFO)")

    # Analysis options
    analysis = parser.add_argument_group("analysis options")
    analysis.add_argument(
        "--max-distance", type=float, default=None,
        help="Cosine distance ceiling — lower is more similar (default: GITHUB_BOT_MAX_DISTANCE or 0.15)",
    )
    analysis.add_argument(
        "--top-k", type=int, default=None,
        help="Max hits to show per element (default: GITHUB_BOT_TOP_K or 3)",
    )
    analysis.add_argument(
        "--min-lines", type=int, default=None,
        help="Hide matched segments shorter than this many lines (default: GITHUB_BOT_MIN_LINES or 0)",
    )
    analysis.add_argument(
        "--check-public", action="store_true", default=None, dest="check_public",
        help="Query public index (default: true)",
    )
    analysis.add_argument(
        "--no-check-public", action="store_false", dest="check_public",
        help="Disable public index queries",
    )
    analysis.add_argument(
        "--check-private", action="store_true", default=None, dest="check_private",
        help="Query private org index (default: true)",
    )
    analysis.add_argument(
        "--no-check-private", action="store_false", dest="check_private",
        help="Disable private index queries",
    )
    analysis.add_argument(
        "--comment-on-zero", action="store_true", default=None, dest="comment_on_zero",
        help="Post comment even when no hits found (default: true)",
    )
    analysis.add_argument(
        "--no-comment-on-zero", action="store_false", dest="comment_on_zero",
        help="Delete comment when no hits found",
    )

    # Scope guards
    guards = parser.add_argument_group("scope guards")
    guards.add_argument(
        "--allowed-orgs", default=None,
        help="Comma-separated GitHub owner names to allow (default: GITHUB_BOT_ALLOWED_ORGS or all)",
    )
    guards.add_argument(
        "--max-files", type=int, default=None,
        help="Skip PRs with more changed files (default: GITHUB_BOT_MAX_FILES or 50)",
    )
    guards.add_argument(
        "--max-elements", type=int, default=None,
        help="Cap total code elements analysed per PR (default: GITHUB_BOT_MAX_ELEMENTS or 200)",
    )

    args = parser.parse_args()

    # Configure logging early
    import os

    log_level_str = (args.log_level or os.getenv("CODE_SIM_LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, log_level_str, logging.INFO),
        format="[%(levelname)s] %(name)s: %(message)s",
    )

    # ------------------------------------------------------------------ #
    # Startup validation — fail fast before binding the port
    # ------------------------------------------------------------------ #
    try:
        from .config import BotConfig
        cfg = BotConfig.from_env(
            max_distance=args.max_distance,
            top_k=args.top_k,
            check_public=args.check_public,
            check_private=args.check_private,
            comment_on_zero_hits=args.comment_on_zero,
            min_lines=args.min_lines,
            allowed_orgs=args.allowed_orgs,
            max_files_per_pr=args.max_files,
            max_elements_per_pr=args.max_elements,
        )
        log.info(
            "Bot config: max_distance=%.3f top_k=%d min_lines=%d check_public=%s check_private=%s "
            "comment_zero=%s max_files=%d max_elements=%d",
            cfg.max_distance, cfg.top_k, cfg.min_lines, cfg.check_public, cfg.check_private,
            cfg.comment_on_zero_hits, cfg.max_files_per_pr, cfg.max_elements_per_pr,
        )
        if cfg.allowed_orgs:
            log.info("Bot config: allowed_orgs=%s", ", ".join(sorted(cfg.allowed_orgs)))
    except (ValueError, OSError) as exc:
        log.error("Configuration error: %s", exc)
        sys.exit(1)

    try:
        from .github_api import GitHubClient
        gh = GitHubClient(cfg.app_id, cfg.private_key)
        log.info("GitHub App client initialised (app_id=%s)", cfg.app_id)
    except Exception as exc:
        log.error("Failed to initialise GitHub client: %s", exc)
        sys.exit(1)

    # Build the FastAPI app
    try:
        from .server import _build_app
        app = _build_app(cfg, gh)
    except Exception as exc:
        log.error("Failed to build bot server: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Start uvicorn
    # ------------------------------------------------------------------ #
    try:
        import uvicorn
    except ImportError:
        log.error(
            "uvicorn is not installed. "
            "Install the bot extras: pip install -e '.[bot]'"
        )
        sys.exit(1)

    log.info("Starting code-sim-bot webhook server on %s:%d", args.host, args.port)
    log.info("GitHub webhook URL: http://<your-public-host>:%d/webhook", args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=log_level_str.lower(),
    )


if __name__ == "__main__":
    main()
