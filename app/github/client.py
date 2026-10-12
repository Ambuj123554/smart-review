"""GitHub REST API client for PR enriched with GITHUB_TOKEN auth."""
import logging
from typing import Any, List, Optional
from urllib.parse import quote

import requests

from app.config import settings

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"
PAGE_SIZE = 100  # GitHub hard cap for per_page


class GitHubApiError(RuntimeError):
    """Raised when the GitHub API call fails after retries."""


class GitHubApiClient:
    """Small requests-based client for the endpoints this tool needs."""

    def __init__(self, token: Optional[str] = None, session: Optional[requests.Session] = None):
        self.token = token if token is not None else settings.GITHUB_TOKEN
        self.session = session or requests.Session()
        self.base_url = GITHUB_API_BASE

    def _headers(self, accept: str = "application/vnd.github+json") -> dict:
        headers = {"Accept": accept}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def get_pr_files(
        self,
        repo_full_name: str,
        pr_number: int,
        max_files: int = 0,
    ) -> List[dict]:
        """Fetches the changed-files array for one PR, following pagination.

        Returns the raw GitHub file dicts (filename, status, patch, ...).
        max_files (0 = no cap) limits how many entries are collected.
        """
        files: List[dict] = []
        page = 1
        while True:
            url = f"{self.base_url}/repos/{repo_full_name}/pulls/{pr_number}/files"
            resp = self.session.get(
                url,
                headers=self._headers(),
                params={"per_page": PAGE_SIZE, "page": page},
                timeout=30,
            )
            if resp.status_code == 404:
                raise GitHubApiError(f"PR or repo not found: {repo_full_name}#{pr_number}")
            if resp.status_code == 401:
                raise GitHubApiError("GitHub rejected the token (401). Check GITHUB_TOKEN.")
            if resp.status_code >= 400:
                raise GitHubApiError(f"GitHub API error {resp.status_code}: {resp.text[:200]}")

            batch = resp.json()
            if not isinstance(batch, list):
                raise GitHubApiError("Unexpected non-list response for PR files.")

            files.extend(batch)
            if max_files > 0 and len(files) >= max_files:
                files = files[:max_files]
                break

            if len(batch) < PAGE_SIZE:
                break  # last page
            page += 1

        return files

    def get_file_content(
        self, repo_full_name: str, path: str, ref: str
    ) -> Optional[str]:
        """Fetches a file's full text at a given ref (commit SHA).

        Used by the AST analysis step, which needs whole-file source rather
        than the patch. Returns None for missing/undecodable files so a
        single unfetchable file never breaks the review pipeline.
        """
        url = f"{self.base_url}/repos/{repo_full_name}/contents/{quote(path)}"
        headers = self._headers(accept="application/vnd.github.raw+json")
        try:
            resp = self.session.get(url, headers=headers, params={"ref": ref}, timeout=30)
        except requests.RequestException as exc:
            logger.warning("Could not fetch %s@%s: %s", path, ref[:12], exc)
            return None
        if resp.status_code == 401:
            raise GitHubApiError("GitHub rejected the token (401) fetching file content.")
        if resp.status_code != 200:
            logger.warning("File content unavailable for %s (%d)", path, resp.status_code)
            return None
        try:
            return resp.content.decode("utf-8")
        except UnicodeDecodeError:
            logger.info("Skipping non-UTF-8 file for AST analysis: %s", path)
            return None
