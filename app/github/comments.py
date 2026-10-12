"""Posting review comments to GitHub as ONE batched pull request review.

Handles:
  - inline comment payloads (path, line, side="RIGHT", commit_id=head SHA)
  - duplicate prevention on re-push: fetches existing review comments from
    this app (identified by a marker in the body) and skips any (file, line,
    message) triple that already exists
  - GitHub rate limits (403 + Retry-After / X-RateLimit-Remaining) and 5xx
    with exponential backoff retries, all logged
"""
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from app.config import settings

logger = logging.getLogger(__name__)

# Marker so we can recognize our own comments among human ones.
BOT_MARKER = "<!-- smart-review-assistant -->"

MAX_ATTEMPTS = 4
BACKOFF_BASE_SECONDS = 2.0


class GitHubPostingError(RuntimeError):
    """Raised when posting the review fails after all retries."""


class ReviewPoster:
    """Posts one consolidated PR review with inline comments."""

    def __init__(self, client=None):
        # `client` is a GitHubApiClient (real or mock). Lazily constructed
        # so tests can inject a mock without network access.
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from app.github.client import GitHubApiClient
            self._client = GitHubApiClient()
        return self._client

    # --------------------------- public API -------------------------------

    def post_review(
        self,
        repo_full_name: str,
        pr_number: int,
        head_sha: str,
        findings: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Posts findings as inline comments in one review.

        Each finding dict needs: path, line, message, and optionally
        suggestion, severity, category. Findings are deduplicated against
        existing comments first (sync pushes produce the same lines again).
        Returns the GitHub review JSON, or None when everything was
        duplicate/nothing to post.
        """
        if not findings:
            logger.info("No findings for %s#%s; skipping review submission.",
                        repo_full_name, pr_number)
            return None

        existing = self._existing_bot_comments(repo_full_name, pr_number)
        # Key existing comments on (path, line, message) too — the message is
        # embedded in the bot body after the '**[SEV] Cat:**' prefix, so the
        # key builder re-derives it from the stored body.
        existing_keys = {
            self._dedupe_key(
                {"path": c.get("path"), "line": c.get("line"),
                 "message": self._message_from_body(c.get("body", ""))},
                c.get("body", ""),
            )
            for c in existing
        }

        comments: List[Dict[str, Any]] = []
        skipped_dupes = 0
        for f in findings:
            body = self._comment_body(f)
            if self._dedupe_key(f, body) in existing_keys:
                skipped_dupes += 1
                continue
            comments.append({
                "path": f["path"],
                "line": f["line"],       # new-side line number
                "side": "RIGHT",
                "body": body,
            })

        if not comments:
            logger.info("All %d finding(s) already posted for %s#%s.",
                        skipped_dupes, repo_full_name, pr_number)
            return None

        payload = {
            "commit_id": head_sha,
            "event": "COMMENT",
            "body": self._summary_body(findings, comments),
            "comments": comments,
        }
        return self._post_with_retry(repo_full_name, pr_number, payload)

    # --------------------------- internals --------------------------------

    def _comment_body(self, finding: Dict[str, Any]) -> str:
        parts = [BOT_MARKER, f"**[{finding.get('severity', 'INFO')}] "
                             f"{finding.get('category', 'Code Quality')}:** "
                             f"{finding['message']}"]
        suggestion = finding.get("suggestion", "")
        if suggestion:
            parts.append(f"\n```suggestion\n{suggestion}\n```")
        if finding.get("explanation"):
            parts.append(f"\n{finding['explanation']}")
        return "\n".join(parts)

    def _message_from_body(self, body: str) -> str:
        """Extracts the message part from a bot comment body.

        Bot bodies look like:
            <!-- smart-review-assistant -->\n**[SEV] Cat:** message...
        Returns the text after the '** Cat:** ' segment (i.e. the '** '
        closing bold, then ':** '), or the whole stripped body as a
        defensive fallback when the shape doesn't match.
        """
        marker = ":** "
        idx = body.find(marker)
        if idx == -1:
            return body.strip()
        tail = body[idx + len(marker):]
        return tail.split("\n", 1)[0].strip()

    def _dedupe_key(self, finding: Dict[str, Any], body: str) -> tuple:
        """Identity of a finding for duplicate detection.

        Spec says file + line + message; the other body parts (suggestion,
        explanation) may legitimately vary between re-runs of the model, so
        they must NOT be part of the key or legit re-pushes would double-post.
        """
        return (finding["path"], finding["line"], finding["message"])

    def _summary_body(self, findings: List[Dict[str, Any]], comments: List[Dict[str, Any]]) -> str:
        counts: Dict[str, int] = {}
        for f in findings:
            sev = str(f.get("severity", "INFO")).upper()
            counts[sev] = counts.get(sev, 0) + 1
        summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        return (
            f"{BOT_MARKER}\n"
            f"🤖 **Smart Review Assistant** reviewed {len(findings)} issue(s) "
            f"({summary}). {len(comments)} inline comment(s) below, sorted by "
            f"the author's added lines in this PR."
        )

    def _existing_bot_comments(self, repo_full_name: str, pr_number: int) -> List[Dict[str, Any]]:
        """Fetches existing review comments (all pages) and keeps our own.

        Uses GET /repos/{o}/{r}/pulls/{n}/comments; filters to bodies
        containing BOT_MARKER so human comments are never treated as dupes.
        """
        comments: List[Dict[str, Any]] = []
        page = 1
        while True:
            resp = self._retry_request(
                "GET",
                f"{self.client.base_url}/repos/{repo_full_name}/pulls/{pr_number}/comments",
                params={"per_page": 100, "page": page},
            )
            batch = resp.json() if resp.status_code == 200 else []
            if not isinstance(batch, list) or not batch:
                break
            comments.extend(c for c in batch if BOT_MARKER in c.get("body", ""))
            if len(batch) < 100:
                break
            page += 1
        return comments

    def _post_with_retry(self, repo_full_name: str, pr_number: int, payload: Dict[str, Any]):
        url = f"{self.client.base_url}/repos/{repo_full_name}/pulls/{pr_number}/reviews"
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = self._retry_request("POST", url, json=payload)
                if resp.status_code in (200, 201):
                    logger.info("Review submitted for %s#%s (%d comments).",
                                repo_full_name, pr_number, len(payload["comments"]))
                    return resp.json()
                # 4xx (other than rate limit) are permanent: don't retry.
                if 400 <= resp.status_code < 500 and resp.status_code != 429:
                    logger.error("Review POST failed permanently (%d) for %s#%s: %s",
                                 resp.status_code, repo_full_name, pr_number,
                                 resp.text[:200])
                    raise GitHubPostingError(
                        f"GitHub rejected review payload: {resp.status_code} {resp.text[:200]}"
                    )
                last_exc = GitHubPostingError(
                    f"HTTP {resp.status_code} posting review: {resp.text[:200]}"
                )
                logger.warning("Review POST attempt %d/%d failed for %s#%s: %s",
                               attempt, MAX_ATTEMPTS, repo_full_name, pr_number, last_exc)
            except requests.RequestException as exc:
                last_exc = exc
                logger.warning("Network error posting review (attempt %d/%d) %s#%s: %s",
                               attempt, MAX_ATTEMPTS, repo_full_name, pr_number, exc)

            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))

        raise GitHubPostingError(f"Giving up after {MAX_ATTEMPTS} attempts: {last_exc}")

    def _retry_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Wraps client.session with rate-limit awareness and logging."""
        sess: requests.Session = self.client.session
        headers = self.client._headers()
        timeout = 30
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = sess.request(method, url, headers=headers, timeout=timeout, **kwargs)
                if resp.status_code == 403 or resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                    wait = float(retry_after) if retry_after else BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                    logger.warning("GitHub rate limit hit (attempt %d/%d, remaining=%s); "
                                   "sleeping %.1fs: %s", attempt, MAX_ATTEMPTS,
                                   remaining, wait, url)
                    if attempt < MAX_ATTEMPTS:
                        time.sleep(wait)
                        last_exc = GitHubPostingError(f"Rate limited (HTTP {resp.status_code})")
                        continue
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning("Network error on %s %s (attempt %d/%d): %.1fs wait, %s",
                               method, url, attempt, MAX_ATTEMPTS, wait, exc)
                if attempt < MAX_ATTEMPTS:
                    time.sleep(wait)
        raise GitHubPostingError(f"Request failed after {MAX_ATTEMPTS} attempts: {last_exc}")
