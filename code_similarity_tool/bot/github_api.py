"""Async GitHub REST API client for the PR bot.

Handles GitHub App authentication (JWT → installation access token),
PR file listing, file content fetching, and PR comment management.
"""
from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
COMMENT_SIGNATURE = "<!-- code-sim-bot-v1 -->"

# Shown while analysis is in progress
CHECKING_BODY = (
    f"{COMMENT_SIGNATURE}\n"
    "## Code Similarity Report\n\n"
    "> ⏳ Analysing PR changes… this usually takes under a minute.\n"
)


@dataclass
class PRFile:
    filename: str
    status: str          # added | modified | renamed | removed | unchanged
    patch: Optional[str]  # unified diff for this file (None for binary/large files)
    additions: int
    deletions: int


class GitHubClient:
    """Thin async GitHub API client using httpx.

    Manages GitHub App JWT generation and per-installation access token caching.
    Tokens are refreshed automatically 5 minutes before expiry.
    """

    def __init__(self, app_id: str, private_key: str) -> None:
        self._app_id = app_id
        self._private_key = private_key
        # owner → (token, expires_at_unix)
        self._token_cache: dict[str, tuple[str, float]] = {}

    # ------------------------------------------------------------------
    # Authentication helpers
    # ------------------------------------------------------------------

    def _generate_jwt(self) -> str:
        """Generate a signed GitHub App JWT valid for ~9 minutes."""
        try:
            import jwt as pyjwt
        except ImportError as exc:
            raise RuntimeError(
                "PyJWT is required for the bot. Install with: pip install -e '.[bot]'"
            ) from exc

        now = int(time.time())
        payload = {
            "iat": now - 60,   # allow 60 s clock skew
            "exp": now + 540,  # 9 minutes (GitHub max is 10)
            "iss": self._app_id,
        }
        return pyjwt.encode(payload, self._private_key, algorithm="RS256")

    async def _get_token(self, owner: str) -> str:
        """Return a valid installation access token for *owner*, refreshing if needed."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for the bot. Install with: pip install -e '.[bot]'"
            ) from exc

        cached = self._token_cache.get(owner)
        if cached:
            token, expires_at = cached
            if time.time() < expires_at - 300:  # 5-minute buffer
                return token

        jwt_token = self._generate_jwt()
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            # Find the installation for this owner
            resp = await client.get(f"{GITHUB_API}/app/installations", headers=headers)
            resp.raise_for_status()
            installations = resp.json()

            installation_id: Optional[int] = None
            for inst in installations:
                acct = inst.get("account", {})
                if acct.get("login", "").lower() == owner.lower():
                    installation_id = inst["id"]
                    break

            if installation_id is None:
                raise RuntimeError(
                    f"No GitHub App installation found for owner '{owner}'. "
                    "Make sure the App is installed on that account/org."
                )

            # Exchange for an installation access token
            resp = await client.post(
                f"{GITHUB_API}/app/installations/{installation_id}/access_tokens",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            token: str = data["token"]
            # Parse ISO 8601 expiry; fall back to 1 hour
            try:
                import datetime
                exp_str: str = data.get("expires_at", "")
                exp_dt = datetime.datetime.fromisoformat(exp_str.replace("Z", "+00:00"))
                expires_at = exp_dt.timestamp()
            except Exception:
                expires_at = time.time() + 3600

            self._token_cache[owner] = (token, expires_at)
            log.debug("Obtained installation token for '%s' (expires %.0f)", owner, expires_at)
            return token

    def _auth_headers(self, token: str) -> dict:
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    # ------------------------------------------------------------------
    # PR data fetching
    # ------------------------------------------------------------------

    async def get_pr_files(self, owner: str, repo: str, pr_number: int) -> list[PRFile]:
        """Return all files changed in a PR (paginated)."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for the bot.") from exc

        token = await self._get_token(owner)
        results: list[PRFile] = []
        page = 1

        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                resp = await client.get(
                    f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/files",
                    headers=self._auth_headers(token),
                    params={"per_page": 100, "page": page},
                )
                resp.raise_for_status()
                files = resp.json()
                if not files:
                    break
                for f in files:
                    results.append(
                        PRFile(
                            filename=f["filename"],
                            status=f["status"],
                            patch=f.get("patch"),
                            additions=f.get("additions", 0),
                            deletions=f.get("deletions", 0),
                        )
                    )
                if len(files) < 100:
                    break
                page += 1

        log.debug("Fetched %d file(s) from PR %s/%s#%d", len(results), owner, repo, pr_number)
        return results

    async def get_file_content(
        self, owner: str, repo: str, path: str, ref: str
    ) -> Optional[bytes]:
        """Fetch raw file content at *ref* via the GitHub Contents API.

        Returns None if the file does not exist, is a directory, exceeds the
        GitHub 1 MB API limit, or is binary-encoded in an unsupported way.
        Files > 1 MB are fetched via the raw download_url fallback.
        """
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for the bot.") from exc

        token = await self._get_token(owner)

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}",
                headers=self._auth_headers(token),
                params={"ref": ref},
            )
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, list):
                # Path resolved to a directory
                return None

            encoding = data.get("encoding", "")
            if encoding == "base64":
                raw_b64: str = data.get("content", "").replace("\n", "")
                return base64.b64decode(raw_b64)

            # Large files: GitHub returns encoding="none" and a download_url
            download_url = data.get("download_url")
            if download_url:
                raw_resp = await client.get(download_url, follow_redirects=True)
                raw_resp.raise_for_status()
                return raw_resp.content

        return None

    # ------------------------------------------------------------------
    # PR comment management
    # ------------------------------------------------------------------

    async def find_bot_comment(
        self, owner: str, repo: str, pr_number: int
    ) -> Optional[int]:
        """Search PR comments for the bot's signature and return its ID, or None."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for the bot.") from exc

        token = await self._get_token(owner)
        page = 1

        async with httpx.AsyncClient(timeout=15.0) as client:
            while True:
                resp = await client.get(
                    f"{GITHUB_API}/repos/{owner}/{repo}/issues/{pr_number}/comments",
                    headers=self._auth_headers(token),
                    params={"per_page": 100, "page": page},
                )
                resp.raise_for_status()
                comments = resp.json()
                if not comments:
                    break
                for comment in comments:
                    body: str = comment.get("body") or ""
                    if COMMENT_SIGNATURE in body:
                        return int(comment["id"])
                if len(comments) < 100:
                    break
                page += 1

        return None

    async def upsert_pr_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        comment_id: Optional[int] = None,
    ) -> int:
        """Create a new PR comment or update an existing one. Returns the comment ID."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for the bot.") from exc

        token = await self._get_token(owner)

        async with httpx.AsyncClient(timeout=15.0) as client:
            if comment_id:
                resp = await client.patch(
                    f"{GITHUB_API}/repos/{owner}/{repo}/issues/comments/{comment_id}",
                    headers=self._auth_headers(token),
                    json={"body": body},
                )
            else:
                resp = await client.post(
                    f"{GITHUB_API}/repos/{owner}/{repo}/issues/{pr_number}/comments",
                    headers=self._auth_headers(token),
                    json={"body": body},
                )
            resp.raise_for_status()
            return int(resp.json()["id"])

    async def delete_comment(self, owner: str, repo: str, comment_id: int) -> None:
        """Delete a PR comment. Silently ignores 404 (already deleted)."""
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for the bot.") from exc

        token = await self._get_token(owner)

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.delete(
                f"{GITHUB_API}/repos/{owner}/{repo}/issues/comments/{comment_id}",
                headers=self._auth_headers(token),
            )
            if resp.status_code not in (204, 404):
                resp.raise_for_status()
