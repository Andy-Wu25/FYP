"""FastAPI webhook server for the GitHub PR bot."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Optional, Set

log = logging.getLogger(__name__)

_PR_EVENTS = {"opened", "synchronize", "reopened", "ready_for_review"}

# FastAPI imports at module level so type annotations resolve correctly
# when `from __future__ import annotations` is active (annotations are strings
# evaluated against the module's global namespace, not the local scope of
# _build_app where they used to live).
try:
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    _fastapi_available = True
except ImportError:
    _fastapi_available = False


def _build_app(cfg, gh):
    if not _fastapi_available:
        raise RuntimeError(
            "FastAPI is required for the bot. Install with: pip install -e '.[bot]'"
        )

    app = FastAPI(title="code-sim-bot", docs_url=None, redoc_url=None)
    in_flight: Set[str] = set()

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        log.error("FastAPI request validation error: %s", exc.errors())
        return JSONResponse({"status": "error", "detail": str(exc.errors())}, status_code=422)

    @app.get("/health")
    async def health():
        return {"status": "ok", "service": "code-sim-bot"}

    @app.post("/webhook")
    async def handle_webhook(request: Request):
        try:
            body = await request.body()

            sig_header = request.headers.get("X-Hub-Signature-256", "")
            if not _verify_signature(body, cfg.webhook_secret, sig_header):
                log.warning("Webhook signature mismatch — sig_header=%r", sig_header[:20] if sig_header else "")
                return JSONResponse({"status": "unauthorized"}, status_code=401)

            event = request.headers.get("X-GitHub-Event", "")
            log.info("Received GitHub event: %s", event)

            if event == "ping":
                return JSONResponse({"status": "pong"})
            if event != "pull_request":
                return JSONResponse({"status": "ignored", "event": event})

            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                log.error("Failed to parse webhook JSON: %s", exc)
                return JSONResponse({"status": "error", "detail": "invalid json"}, status_code=400)

            action = payload.get("action", "")
            if action not in _PR_EVENTS:
                return JSONResponse({"status": "ignored", "action": action})

            pr = payload.get("pull_request", {})
            repo_data = payload.get("repository", {})
            owner = (repo_data.get("owner") or {}).get("login", "")
            repo = repo_data.get("name", "")
            pr_number = pr.get("number")
            head_sha = (pr.get("head") or {}).get("sha", "")

            if not all([owner, repo, pr_number, head_sha]):
                log.warning("Incomplete PR payload: owner=%s repo=%s pr=%s", owner, repo, pr_number)
                return JSONResponse({"status": "ignored", "reason": "incomplete_payload"})

            if cfg.allowed_orgs and owner.lower() not in {o.lower() for o in cfg.allowed_orgs}:
                log.info("Ignoring PR from non-allowed owner '%s'", owner)
                return JSONResponse({"status": "ignored", "reason": "owner_not_allowed"})

            pr_key = f"{owner}/{repo}#{pr_number}"
            if pr_key in in_flight:
                log.info("PR %s already in flight — skipping duplicate event", pr_key)
                return JSONResponse({"status": "queued", "note": "already_in_flight"})

            log.info("Queuing analysis for PR %s (action=%s sha=%s)", pr_key, action, head_sha[:8])
            asyncio.create_task(
                _process_pr_event(
                    owner=owner,
                    repo=repo,
                    pr_number=pr_number,
                    head_sha=head_sha,
                    cfg=cfg,
                    gh=gh,
                    in_flight=in_flight,
                )
            )

            return JSONResponse({"status": "queued", "pr": pr_key})

        except Exception as exc:
            log.exception("Unhandled error in webhook handler: %s", exc)
            return JSONResponse({"status": "error", "detail": str(exc)}, status_code=500)

    return app


async def _process_pr_event(
    *,
    owner: str,
    repo: str,
    pr_number: int,
    head_sha: str,
    cfg,
    gh,
    in_flight: Set[str],
) -> None:
    from .formatter import format_error_report, format_report, format_zero_findings_report
    from .github_api import CHECKING_BODY
    from .pr_analyzer import analyze_pr

    pr_key = f"{owner}/{repo}#{pr_number}"
    in_flight.add(pr_key)
    comment_id: Optional[int] = None

    try:
        comment_id = await gh.find_bot_comment(owner, repo, pr_number)
        comment_id = await gh.upsert_pr_comment(
            owner, repo, pr_number, CHECKING_BODY, comment_id
        )

        pr_files = await gh.get_pr_files(owner, repo, pr_number)

        source_files = [f for f in pr_files if f.status != "removed"]
        if len(source_files) > cfg.max_files_per_pr:
            log.warning("PR %s has %d files; capping at %d", pr_key, len(source_files), cfg.max_files_per_pr)
            source_files = source_files[: cfg.max_files_per_pr]

        fetch_tasks = {
            f.filename: gh.get_file_content(owner, repo, f.filename, head_sha)
            for f in source_files
        }
        results = await asyncio.gather(*fetch_tasks.values(), return_exceptions=True)
        file_contents = {}
        for filename, result in zip(fetch_tasks.keys(), results):
            if isinstance(result, Exception):
                log.warning("Failed to fetch %s: %s", filename, result)
                file_contents[filename] = None
            else:
                file_contents[filename] = result

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: analyze_pr(
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                head_sha=head_sha,
                pr_files=source_files,
                file_contents=file_contents,
                cfg=cfg,
            ),
        )

        findings = result.findings
        embed_cfg = result.embedding_config
        truncated = len(findings) >= cfg.max_elements_per_pr
        has_hits = any(f.has_hits for f in findings)

        if not findings:
            if cfg.comment_on_zero_hits:
                body = format_zero_findings_report(
                    0, [], cfg.max_distance,
                    top_k=cfg.top_k, check_private=cfg.check_private,
                    check_public=cfg.check_public, embedding_config=embed_cfg,
                )
                await gh.upsert_pr_comment(owner, repo, pr_number, body, comment_id)
            else:
                await gh.delete_comment(owner, repo, comment_id)
                comment_id = None
        elif has_hits:
            body = format_report(
                findings=findings,
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                head_sha=head_sha,
                max_distance=cfg.max_distance,
                truncated=truncated,
                top_k=cfg.top_k,
                check_private=cfg.check_private,
                check_public=cfg.check_public,
                embedding_config=embed_cfg,
            )
            await gh.upsert_pr_comment(owner, repo, pr_number, body, comment_id)
        else:
            if cfg.comment_on_zero_hits:
                body = format_zero_findings_report(
                    len(findings), findings, cfg.max_distance,
                    top_k=cfg.top_k, check_private=cfg.check_private,
                    check_public=cfg.check_public, embedding_config=embed_cfg,
                )
                await gh.upsert_pr_comment(owner, repo, pr_number, body, comment_id)
            else:
                await gh.delete_comment(owner, repo, comment_id)
                comment_id = None

        log.info(
            "Completed analysis for PR %s: %d element(s), %d with hits",
            pr_key,
            len(findings),
            sum(1 for f in findings if f.has_hits),
        )

    except Exception as exc:
        log.exception("Unhandled error processing PR %s: %s", pr_key, exc)
        try:
            error_body = format_error_report(str(exc))
            await gh.upsert_pr_comment(owner, repo, pr_number, error_body, comment_id)
        except Exception:
            log.exception("Failed to post error comment for PR %s", pr_key)
    finally:
        in_flight.discard(pr_key)


def _verify_signature(body: bytes, secret: bytes, sig_header: str) -> bool:
    if not sig_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret, body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig_header)
