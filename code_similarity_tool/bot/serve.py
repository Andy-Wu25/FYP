"""CLI entry point: code-sim-bot-serve

Starts the GitHub PR bot webhook server. Validates all required environment
variables and connectivity (vLLM + ChromaDB) at startup so failures are
surfaced immediately rather than mid-webhook.

Usage:
    code-sim-bot-serve [--host HOST] [--port PORT]

Required environment variables:
    GITHUB_APP_ID               Numeric GitHub App ID
    GITHUB_APP_PRIVATE_KEY      PEM text  (or GITHUB_APP_PRIVATE_KEY_PATH)
    GITHUB_WEBHOOK_SECRET       Webhook secret configured on the GitHub App

Optional environment variables (same as CLI):
    GITHUB_BOT_MAX_DISTANCE     Cosine distance ceiling (default 0.15)
    GITHUB_BOT_TOP_K            Max hits per element (default 3)
    GITHUB_BOT_CHECK_PUBLIC     Query public index (default true)
    GITHUB_BOT_CHECK_PRIVATE    Query private index (default true)
    GITHUB_BOT_COMMENT_ZERO     Comment when zero hits found (default false)
    GITHUB_BOT_ALLOWED_ORGS     Comma-separated GitHub owner names (default all)
    GITHUB_BOT_MAX_FILES        Skip PRs with more changed files (default 50)
    GITHUB_BOT_MAX_ELEMENTS     Cap total elements per PR (default 200)

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
            "and GITHUB_WEBHOOK_SECRET."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to listen on (default: 3000)",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override log level (default: CODE_SIM_LOG_LEVEL env or INFO)",
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
        cfg = BotConfig.from_env()
        log.info("Bot config loaded: check_public=%s check_private=%s top_k=%d max_distance=%.3f",
                 cfg.check_public, cfg.check_private, cfg.top_k, cfg.max_distance)
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
