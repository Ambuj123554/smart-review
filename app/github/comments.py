"""Posting review comments to GitHub as ONE batched pull request review.

Handles:
  - inline comment payloads (path, line, side="RIGHT", commit_id=head SHA)
  - duplicate prevention on re-push / webhook redelivery: fetches existing
    review comments from this app (identified by a marker in the body) via
    GET /repos/{o}/{r}/pulls/{n}/comments and skips any new finding whose
    (file, line, category-or-severity) already has a comment, regardless of
    the exact message text — LLM wording changes between runs, so keying on
    the message used to double-post the same issue.
  - re-review when the line changed: each posted comment embeds a hash of
    the commented line's content; a finding on a line whose content has
    changed since is NOT treated as a duplicate, even under a new commit SHA.
  - within-run merging: findings on the same file+line+category collapse
    into ONE comment keeping the most severe one.
  - GitHub rate limits (429, or 403 with rate-limit evidence such as
    X-RateLimit-Remaining: 0, Retry-After, or a secondary-rate-limit
    message) and 5xx with exponential backoff retries, all logged.
    A plain permission 403 is permanent and fails immediately with a
    message naming the needed GITHUB_TOKEN permission.
"""
import hashlib
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from app.config import settings

logger = logging.getLogger(__name__)

# Marker so we can recognize our own comments among human ones.
BOT_MARKER = "<!-- smart-review-assistant -->"

# Invisible marker embedded in posted bodies recording the hash of the
# commented line's content, so a later run can tell whether the line was
# edited since (edited line -> re-review, unchanged -> duplicate).
LINE_HASH_RE = re.compile(r"<!--\s*line-hash:([0-9a-f]+)\s*-->")

# Parses the '**[SEV] Cat:**' prefix of a bot comment body.
TAGS_RE = re.compile(r"\*\*\[([A-Za-z]+)\]\s*([^:*]+?):\*\*")

# Short hash keeps the invisible marker small; 64 bits of sha256 suffices.
LINE_HASH_LEN = 16

# Severity order used for the overall-risk rollup (highest wins).
RISK_ORDER = ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")


def severity_rank(value: Any) -> int:
    """Sort key for severities (higher = more severe); unknown -> lowest."""
    sev = str(value or "INFO").upper()
    try:
        return -RISK_ORDER.index(sev)
    except ValueError:
        return len(RISK_ORDER)


def line_hash(text: Optional[str]) -> Optional[str]:
    """Content hash embedded in comment bodies; None when text unknown."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:LINE_HASH_LEN]

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
        files_analysed: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Posts findings as inline comments in one review.

        Each finding dict needs: path, line, message, and optionally
        suggestion, severity, category, line_text (the content of the
        commented line — used to detect that a line was edited since a
        previous run). Findings are deduplicated against existing comments
        first (sync pushes and redelivered webhooks produce the same lines
        again, possibly with reworded LLM messages).
        When `files_analysed` is given, one PR-level summary comment
        (files, severity/category counts, overall risk) is posted after a
        successful inline review; it is never posted when there are no
        findings (early return) or when the inline post fails.
        Returns the GitHub review JSON, or None when everything was
        duplicate/nothing to post.
        """
        if not findings:
            logger.info("No findings for %s#%s; skipping review submission.",
                        repo_full_name, pr_number)
            return None

        # Within a single run: the model may report the same issue twice on
        # the same line with different wording — collapse same file+line+
        # category into one comment, keeping the most severe.
        findings = self._merge_same_line(findings)

        existing = self._existing_bot_comments(repo_full_name, pr_number)

        comments: List[Dict[str, Any]] = []
        skipped_dupes = 0
        for f in findings:
            if self._is_duplicate(f, existing):
                skipped_dupes += 1
                continue
            body = self._comment_body(f)
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
        review = self._post_with_retry(repo_full_name, pr_number, payload)

        # Summary comment: only after a successful inline review, and only
        # when the caller told us how many files were analysed (the real
        # pipeline always does).
        if files_analysed is not None:
            try:
                self._post_summary_comment(
                    repo_full_name,
                    pr_number,
                    self._summary_comment_body(findings, files_analysed),
                )
            except GitHubPostingError as err:
                logger.warning(
                    "Inline review posted but summary comment failed for %s#%s: %s",
                    repo_full_name, pr_number, err,
                )
        return review

    # --------------------------- internals --------------------------------

    @staticmethod
    def _merge_same_line(findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merges findings on the same file+line+category into ONE finding.

        Keeps the most severe one (CRITICAL > HIGH > MEDIUM > LOW > INFO)
        so a run that reports the same issue twice with different wording
        still yields a single inline comment. First occurrence's position
        wins; the caller's list is not mutated.
        """
        merged: List[Dict[str, Any]] = []
        index: Dict[tuple, int] = {}
        for f in findings:
            key = (f.get("path"), f.get("line"),
                   str(f.get("category") or "Code Quality"))
            pos = index.get(key)
            if pos is None:
                index[key] = len(merged)
                merged.append(f)
            elif severity_rank(f.get("severity")) > severity_rank(merged[pos].get("severity")):
                merged[pos] = f
        return merged

    def _is_duplicate(
        self,
        finding: Dict[str, Any],
        existing: List[Dict[str, Any]],
    ) -> bool:
        """True when a bot comment already covers this finding.

        Same file + same line + same category-or-severity tag counts as the
        same issue even when the LLM reworded the message. The commit SHA is
        deliberately NOT part of the identity: a redelivered webhook or a new
        push re-reports unchanged lines. Exception: when the line's content
        changed since the old comment was posted (embedded line-hash
        mismatch), the finding is new and must be posted again.
        """
        new_sev = str(finding.get("severity", "INFO")).upper()
        new_cat = str(finding.get("category") or "Code Quality")
        new_hash = line_hash(finding.get("line_text"))
        for c in existing:
            if (c.get("path") != finding.get("path")
                    or c.get("line") != finding.get("line")):
                continue
            body = c.get("body", "")
            old_hash = self._line_hash_from_body(body)
            # Line edited since the old comment: re-review, not a duplicate.
            if old_hash and new_hash and old_hash != new_hash:
                continue
            old_sev, old_cat = self._tags_from_body(body)
            if (old_cat and old_cat == new_cat) \
                    or (old_sev and old_sev == new_sev):
                return True
        return False

    @staticmethod
    def _tags_from_body(body: str) -> Tuple[Optional[str], Optional[str]]:
        """Extracts (severity, category) from a bot body's '**[SEV] Cat:**'
        prefix; (None, None) when the shape doesn't match."""
        match = TAGS_RE.search(body or "")
        if not match:
            return None, None
        return match.group(1).upper(), match.group(2).strip()

    @staticmethod
    def _line_hash_from_body(body: str) -> Optional[str]:
        """Extracts the embedded line-content hash from a bot body."""
        match = LINE_HASH_RE.search(body or "")
        return match.group(1) if match else None

    def _comment_body(self, finding: Dict[str, Any]) -> str:
        parts = [BOT_MARKER]
        body_hash = line_hash(finding.get("line_text"))
        if body_hash:
            # Invisible marker: lets a later run detect that the commented
            # line's text changed (then the finding is not a duplicate).
            parts.append(f"<!-- line-hash:{body_hash} -->")
        parts.append(f"**[{finding.get('severity', 'INFO')}] "
                     f"{finding.get('category', 'Code Quality')}:** "
                     f"{finding['message']}")
        suggestion = finding.get("suggestion", "")
        if suggestion:
            parts.append(f"\n```suggestion\n{suggestion}\n```")
        if finding.get("explanation"):
            parts.append(f"\n{finding['explanation']}")
        return "\n".join(parts)

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

    @staticmethod
    def _overall_risk(findings: List[Dict[str, Any]]) -> str:
        """Highest severity present (CRITICAL > HIGH > MEDIUM > LOW > INFO)."""
        sevs = {str(f.get("severity", "INFO")).upper() for f in findings}
        for level in RISK_ORDER:
            if level in sevs:
                return level
        return "NONE"

    def _summary_comment_body(
        self, findings: List[Dict[str, Any]], files_analysed: int
    ) -> str:
        """One PR-level comment: files analysed, severity/category counts,
        and the overall risk level."""
        sev_counts: Dict[str, int] = {}
        cat_counts: Dict[str, int] = {}
        for f in findings:
            sev = str(f.get("severity", "INFO")).upper()
            cat = str(f.get("category", "Code Quality"))
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        risk = self._overall_risk(findings)
        sev_str = ", ".join(
            f"{k}: {sev_counts[k]}" for k in RISK_ORDER if k in sev_counts
        )
        cat_str = ", ".join(
            f"{k}: {v}" for k, v in sorted(cat_counts.items())
        )
        return (
            f"{BOT_MARKER}\n"
            "📊 **Smart Review Assistant — PR review summary**\n"
            f"- Files analysed: {files_analysed}\n"
            f"- Findings: {len(findings)} total\n"
            f"- By severity: {sev_str}\n"
            f"- By category: {cat_str}\n"
            f"- Overall risk: **{risk}**"
        )

    def _post_summary_comment(
        self, repo_full_name: str, pr_number: int, body: str
    ) -> Optional[Dict[str, Any]]:
        """Posts ONE PR-level comment (issue-comment API) with retries."""
        url = (
            f"{self.client.base_url}/repos/{repo_full_name}"
            f"/issues/{pr_number}/comments"
        )
        resp = self._retry_request("POST", url, json={"body": body})
        if resp.status_code in (200, 201):
            logger.info("Summary comment posted for %s#%s.", repo_full_name, pr_number)
            return resp.json()
        logger.error("Summary comment rejected for %s#%s (%d): %s",
                     repo_full_name, pr_number, resp.status_code, resp.text[:200])
        raise GitHubPostingError(
            f"GitHub rejected summary comment: {resp.status_code} {resp.text[:200]}"
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

    @staticmethod
    def _is_rate_limit_response(resp: requests.Response) -> bool:
        """True only when the response is a genuine rate limit worth retrying.

        429 always counts. A 403 counts only when it carries rate-limit
        evidence: no remaining quota, a Retry-After header, or GitHub's
        secondary-rate-limit message. A plain 403 (e.g. "Resource not
        accessible by personal access token") is a permissions problem that
        retries cannot fix, so it must fail immediately.
        """
        if resp.status_code == 429:
            return True
        if resp.status_code != 403:
            return False
        remaining = str(resp.headers.get("X-RateLimit-Remaining", "")).strip()
        if remaining == "0":
            return True
        if resp.headers.get("Retry-After"):
            return True
        if "secondary rate limit" in (resp.text or "").lower():
            return True
        return False

    @staticmethod
    def _permission_error_message(resp: requests.Response) -> str:
        """Clear, actionable message for a non-rate-limit 403."""
        body = (resp.text or "").strip()[:200]
        return (
            f"GitHub denied access (HTTP 403): {body}. "
            "This is a permission error, not a rate limit, so it was not "
            "retried. GITHUB_TOKEN needs \"Pull requests: Read and write\" "
            "(classic PAT with `repo` scope, or a fine-grained token / "
            "GitHub App with Pull requests: Read and write permission)."
        )

    def _retry_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Wraps client.session with rate-limit awareness and logging.

        Retries only on 429, on 403 with rate-limit evidence, and on
        network errors / (handled by callers) 5xx. A plain permission 403
        raises GitHubPostingError immediately.
        """
        sess: requests.Session = self.client.session
        headers = self.client._headers()
        timeout = 30
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = sess.request(method, url, headers=headers, timeout=timeout, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                wait = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning("Network error on %s %s (attempt %d/%d): %.1fs wait, %s",
                               method, url, attempt, MAX_ATTEMPTS, wait, exc)
                if attempt < MAX_ATTEMPTS:
                    time.sleep(wait)
                continue

            if self._is_rate_limit_response(resp):
                retry_after = resp.headers.get("Retry-After")
                remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                wait = float(retry_after) if retry_after else BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
                logger.warning("GitHub rate limit hit (attempt %d/%d, remaining=%s); "
                               "sleeping %.1fs: %s", attempt, MAX_ATTEMPTS,
                               remaining, wait, url)
                last_exc = GitHubPostingError(f"Rate limited (HTTP {resp.status_code})")
                if attempt < MAX_ATTEMPTS:
                    time.sleep(wait)
                    continue
                raise GitHubPostingError(
                    f"Rate limited by GitHub (HTTP {resp.status_code}) after "
                    f"{MAX_ATTEMPTS} attempts (remaining={remaining}, "
                    f"Retry-After={retry_after!r}): {url}"
                )

            if resp.status_code == 403:
                # Plain permission error: permanent, retrying cannot help.
                msg = self._permission_error_message(resp)
                logger.error("%s (%s %s)", msg, method, url)
                raise GitHubPostingError(msg)

            return resp
        raise GitHubPostingError(f"Request failed after {MAX_ATTEMPTS} attempts: {last_exc}")
