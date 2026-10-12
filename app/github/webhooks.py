import hashlib
import hmac
import json
import logging
import os
from typing import Any, List, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status

from app.ai.diff_reviewer import DiffReviewer, DiffReviewResult
from app.analyzer.engine import ASTAnalyzer
from app.config import settings
from app.database.db import DatabaseManager
from app.github.client import GitHubApiClient, GitHubApiError
from app.github.comments import ReviewPoster, GitHubPostingError
from app.github.diff import FilePatch, parse_files_payload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["GitHub Webhook"])

# PR lifecycle actions that trigger (re-)review. Everything else is ignored.
RELEVANT_ACTIONS = frozenset({"opened", "synchronize", "reopened"})


def verify_signature(secret: str, payload_body: bytes, signature_header: str) -> bool:
    """Validates the X-Hub-Signature-256 header against HMAC-SHA256(secret, body)."""
    if not secret or not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), payload_body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def extract_pr_info(
    payload: dict[str, Any],
) -> tuple[Optional[str], Optional[int], Optional[str], Optional[int], Optional[str]]:
    """Extracts (repo_full_name, pr_number, head_sha, installation_id, author)
    from a payload.

    Returns all-None when required fields are missing.
    """
    try:
        pr = payload["pull_request"]
        repo_full_name = payload["repository"]["full_name"]
        pr_number = pr["number"]
        head_sha = pr["head"]["sha"]
    except (KeyError, TypeError):
        logger.warning("Malformed pull_request payload; missing required fields.")
        return None, None, None, None, None

    # Only present when this is a GitHub App installation
    installation_id: Optional[int] = None
    raw_installation = payload.get("installation")
    if isinstance(raw_installation, dict):
        installation_id = raw_installation.get("id")

    # PR author: pull_request.user.login (absent for deleted users, etc.)
    author: Optional[str] = None
    raw_user = pr.get("user") if isinstance(pr, dict) else None
    if isinstance(raw_user, dict) and isinstance(raw_user.get("login"), str):
        author = raw_user["login"]

    return repo_full_name, pr_number, head_sha, installation_id, author


# Full-file source is only fetched for extensions the AST engine can parse
# (Python today). Everything else still gets LLM review of its added lines,
# but we don't waste an API call fetching content we cannot analyse.
SUPPORTED_AST_EXTENSIONS = frozenset({".py"})


def _ast_findings_for_file(
    client: GitHubApiClient,
    repo_full_name: str,
    head_sha: str,
    file_patch: FilePatch,
    analyzer: ASTAnalyzer,
) -> List[dict]:
    """Runs the deterministic AST rules on a file's full source at head_sha.

    AST analysis needs the whole file (not just the hunk), so the content is
    fetched at the head commit. Only findings whose line is one of the
    parsed diff's added lines are kept — same anti-hallucination contract
    as the LLM path (AST line numbers must map onto changed lines, otherwise
    the comment could not be placed inline).
    """
    ext = os.path.splitext(file_patch.path)[1].lower()
    if ext not in SUPPORTED_AST_EXTENSIONS:
        logger.debug("Skipping content fetch for unsupported file type: %s",
                     file_patch.path)
        return []
    content = client.get_file_content(repo_full_name, file_patch.path, head_sha)
    if not content:
        return []
    valid_lines = {al.line for al in file_patch.added_lines}
    out: List[dict] = []
    for f in analyzer.analyze(content):
        if f.line not in valid_lines:
            continue
        out.append({
            "path": file_patch.path,
            "line": f.line,
            "message": f.message,
            "suggestion": f.suggestion,
            "explanation": f.explanation,
            "severity": f.severity,
            "category": f.category,
        })
    return out


def _merge_findings(
    path: str, ast_findings: List[dict], llm_result: DiffReviewResult
) -> List[dict]:
    """Merges AST + LLM findings for one file, deduplicated per line.

    AST results are deterministic, so they win on (path, line) collisions;
    the LLM's finding for an already-covered line is dropped instead.
    """
    merged: List[dict] = list(ast_findings)
    covered = {(d["path"], d["line"]) for d in merged}
    for f in llm_result.findings:
        key = (path, f.line)
        if key in covered:
            continue
        merged.append({
            "path": path,
            "line": f.line,
            "message": f.message,
            "suggestion": f.suggestion,
            "explanation": f.explanation,
            "severity": f.severity,
            "category": f.category,
        })
        covered.add(key)
    return merged


def _process_review(
    repo_full_name: str,
    pr_number: int,
    head_sha: str,
    installation_id: Optional[int],
    author: Optional[str] = None,
) -> List[DiffReviewResult]:
    """Full PR review pipeline: fetch files, parse diffs, run AI review.

    Runs in a FastAPI background task after the 200 response is sent.
    Returns the per-file review results (used by tests via mocks).
    """
    # TODO(auth): if installation_id, mint a short-lived installation access
    #             token (POST /app/installations/{id}/access_tokens) instead of
    #             reusing the classic PAT from GITHUB_TOKEN. Requires the
    #             GitHub App's private key; not implemented yet.

    client = GitHubApiClient()
    try:
        # Fetch one more than the cap so "truncated" can be distinguished
        # from "exactly at the cap" (len == MAX_FILES_PER_PR is not a truncation).
        files = client.get_pr_files(
            repo_full_name,
            pr_number,
            max_files=settings.MAX_FILES_PER_PR + 1,
        )
    except GitHubApiError as err:
        logger.error("Failed to fetch changed files for %s#%s: %s",
                     repo_full_name, pr_number, err)
        return []

    if settings.MAX_FILES_PER_PR > 0 and len(files) > settings.MAX_FILES_PER_PR:
        logger.warning(
            "PR %s#%s truncated: %d+ files changed (MAX_FILES_PER_PR=%d); "
            "remaining changed files were not analysed.",
            repo_full_name, pr_number, len(files), settings.MAX_FILES_PER_PR,
        )
        files = files[:settings.MAX_FILES_PER_PR]

    parsed = parse_files_payload(
        files,
        max_files=settings.MAX_FILES_PER_PR,
        max_lines=settings.GITHUB_MAX_LINES,
    )
    if not parsed:
        logger.info("No analysable diff content for %s#%s.", repo_full_name, pr_number)
        return []

    if any(fp.truncated for fp in parsed):
        logger.warning(
            "PR %s#%s truncated: hit GITHUB_MAX_LINES=%d; later hunks "
            "were not analysed.",
            repo_full_name, pr_number, settings.GITHUB_MAX_LINES,
        )

    reviewer = DiffReviewer()
    ast_analyzer = ASTAnalyzer()
    results: List[DiffReviewResult] = []
    findings_payload: List[dict] = []
    for fp in parsed:
        try:
            llm_result = reviewer.review_file(fp)
        except Exception as err:  # noqa: BLE001 - one file must not kill the run
            logger.warning("Failed LLM review for %s in %s#%s: %s",
                           fp.path, repo_full_name, pr_number, err)
            llm_result = DiffReviewResult(fp.path)
        try:
            ast_findings = _ast_findings_for_file(
                client, repo_full_name, head_sha, fp, ast_analyzer
            )
        except Exception as err:  # noqa: BLE001
            logger.warning("Failed AST analysis for %s in %s#%s: %s",
                           fp.path, repo_full_name, pr_number, err)
            ast_findings = []
        results.append(llm_result)
        findings_payload.extend(_merge_findings(fp.path, ast_findings, llm_result))

    logger.info(
        "Review complete for %s#%s: %d file(s), %d merged finding(s).",
        repo_full_name,
        pr_number,
        len(results),
        len(findings_payload),
    )

    # Post ONE consolidated review with inline comments; duplicates from
    # previous pushes (same path/line/body) are skipped internally.
    try:
        ReviewPoster().post_review(
            repo_full_name, pr_number, head_sha, findings_payload,
            files_analysed=len(parsed),
        )
    except GitHubPostingError as err:
        logger.error("Could not post review for %s#%s: %s", repo_full_name, pr_number, err)

    # Persist PR + findings to the history/analytics store. DB failure must
    # not break an already-posted review; log and carry on.
    try:
        DatabaseManager().save_pr_review(
            repo_full_name, pr_number, head_sha, author, findings_payload
        )
    except Exception as err:  # noqa: BLE001
        logger.error("Could not save PR review to DB for %s#%s: %s",
                     repo_full_name, pr_number, err)

    return results


@router.post("/github", status_code=status.HTTP_200_OK)
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_hub_signature_256: str = Header(default=""),
    x_github_event: str = Header(default=""),
) -> dict[str, Optional[str]]:
    """Receives GitHub pull_request events and dispatches a background review."""
    body = await request.body()

    if not verify_signature(settings.GITHUB_WEBHOOK_SECRET, body, x_hub_signature_256):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing signature.",
        )

    event_type = x_github_event.strip().lower()
    if event_type != "pull_request":
        return {"status": "ignored", "reason": f"unsupported event: {event_type or 'unknown'}"}

    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Body is not valid JSON.",
        )

    action = payload.get("action")
    if action not in RELEVANT_ACTIONS:
        return {"status": "ignored", "reason": f"action not actionable: {action}"}

    repo_full_name, pr_number, head_sha, installation_id, author = extract_pr_info(payload)
    if repo_full_name is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed pull_request payload.",
        )

    # Fire-and-forget via FastAPI BackgroundTasks: runs after the response is
    # sent, so /webhook/github returns immediately and GitHub gets a fast 200.
    # When a real blocking pipeline lands (diff fetch + LLM call), wrap it with
    # run_in_threadpool so the event loop is never blocked.
    background_tasks.add_task(
        _process_review, repo_full_name, pr_number, head_sha, installation_id, author
    )

    return {
        "status": "accepted",
        "repo": repo_full_name,
        "pr": str(pr_number),
        "head_sha": head_sha,
        "installation_id": str(installation_id) if installation_id is not None else None,
        "author": author,
    }
