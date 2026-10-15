import hashlib
import hmac
import json
import logging
import os
from typing import Any, List, Optional

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request, status

from app.ai.diff_reviewer import DiffReviewer, DiffReviewResult
from app.analyzer.engine import ASTAnalyzer
from app.analyzer.rules.missing_tests import MissingTestsRule, is_test_file
from app.config import settings
from app.database.db import DatabaseManager
from app.github.client import GitHubApiClient, GitHubApiError
from app.github.comments import (
    ReviewPoster,
    GitHubPostingError,
    severity_rank,
)
from app.github.diff import BINARY_MARKERS, FilePatch, parse_files_payload
from app.review_config import ReviewConfig, fetch_review_config

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

# Changed files the review never analyses: lockfiles, minified bundles and
# generated code only add noise and huge diffs. Binary and deleted files
# are also skipped (no readable patch / nothing on the new side).
LOCK_FILE_NAMES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "pnpm-lock.yml",
    "poetry.lock", "pipfile.lock", "cargo.lock", "composer.lock",
    "gemfile.lock", "packages.lock.json", "go.sum", "flake.lock",
})
MINIFIED_SUFFIXES = (".min.js", ".min.css", ".min.map", ".js.map", ".css.map")
GENERATED_SUFFIXES = ("_pb2.py", "_pb2_grpc.py", ".pb.go", ".g.dart",
                      ".generated.py", ".generated.ts")


def _is_skippable_file(f: dict) -> bool:
    """True for changed files the review pipeline never analyses."""
    name = os.path.basename(f.get("filename", "") or "").lower()
    if f.get("status") == "removed":
        return True  # deleted: nothing to review on the new side
    patch = f.get("patch")
    if not patch or any(marker in patch[:200] for marker in BINARY_MARKERS):
        return True  # binary or no patch (e.g. mode-only changes)
    if name in LOCK_FILE_NAMES or name.endswith(".lock"):
        return True
    if name.endswith(MINIFIED_SUFFIXES):
        return True
    return name.endswith(GENERATED_SUFFIXES)


def _ast_findings_for_file(
    client: GitHubApiClient,
    repo_full_name: str,
    head_sha: str,
    file_patch: FilePatch,
    analyzer: ASTAnalyzer,
    content: Optional[str] = None,
) -> List[dict]:
    """Runs the deterministic AST rules on a file's full source at head_sha.

    AST analysis needs the whole file (not just the hunk), so the content is
    fetched at the head commit (or passed in pre-fetched to reuse one fetch
    between the AST rules and the missing-tests rule). Only findings whose
    line is one of the parsed diff's added lines are kept — same
    anti-hallucination contract as the LLM path (AST line numbers must map
    onto changed lines, otherwise the comment could not be placed inline).
    """
    ext = os.path.splitext(file_patch.path)[1].lower()
    if ext not in SUPPORTED_AST_EXTENSIONS:
        logger.debug("Skipping content fetch for unsupported file type: %s",
                     file_patch.path)
        return []
    if content is None:
        content = client.get_file_content(repo_full_name, file_patch.path, head_sha)
    if not content:
        return []
    # Line -> text of the added line, so findings carry the commented line's
    # content (used by the poster to detect that a line changed on re-push).
    added_by_line = {al.line: al.content for al in file_patch.added_lines}
    valid_lines = set(added_by_line)
    out: List[dict] = []
    for f in analyzer.analyze(content):
        if f.line not in valid_lines:
            continue
        out.append({
            "path": file_patch.path,
            "line": f.line,
            "rule_id": f.rule_id,
            "message": f.message,
            "suggestion": f.suggestion,
            "explanation": f.explanation,
            "severity": f.severity,
            "category": f.category,
            "line_text": added_by_line.get(f.line),
        })
    return out


def _merge_findings(
    path: str,
    ast_findings: List[dict],
    llm_result: DiffReviewResult,
    line_texts: Optional[dict] = None,
) -> List[dict]:
    """Merges AST + LLM findings for one file, deduplicated per line.

    AST results are deterministic, so they win on (path, line) collisions;
    LLM findings on an AST-covered line are dropped. Among LLM findings on
    the same line, findings about the same category are merged into one,
    keeping the most severe (LLM wording varies between runs). Each finding
    carries `line_text` (when known) so the comment poster can tell whether
    the line's content changed since a previous run's comment.
    """
    merged: List[dict] = list(ast_findings)
    ast_lines = {(d["path"], d["line"]) for d in merged}
    kept: dict = {}  # (path, line, category) -> index into `merged`
    for f in llm_result.findings:
        if (path, f.line) in ast_lines:
            continue  # AST already covers this line deterministically
        entry = {
            "path": path,
            "line": f.line,
            "rule_id": f.rule_id,
            "message": f.message,
            "suggestion": f.suggestion,
            "explanation": f.explanation,
            "severity": f.severity,
            "category": f.category,
            "line_text": (line_texts or {}).get(f.line),
        }
        group = (path, f.line, f.category)
        pos = kept.get(group)
        if pos is None:
            kept[group] = len(merged)
            merged.append(entry)
        elif severity_rank(f.severity) > severity_rank(merged[pos]["severity"]):
            merged[pos] = entry  # same issue re-reported more severely
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
        # Fetch the full changed-file list: skippable files (binary,
        # deleted, lock, minified, generated) are filtered BEFORE the
        # MAX_FILES_PER_PR cap so the cap only spends on analysable files.
        files = client.get_pr_files(repo_full_name, pr_number)
    except GitHubApiError as err:
        logger.error("Failed to fetch changed files for %s#%s: %s",
                     repo_full_name, pr_number, err)
        return []

    # Optional .reviewer.yml at the PR's head SHA. Missing/invalid content
    # logs a warning and yields defaults — it never fails the review. Fetched
    # before any file is selected so ignore_paths applies BEFORE the
    # MAX_FILES_PER_PR cap (same contract as the skippable-file filter).
    config: ReviewConfig = fetch_review_config(client, repo_full_name, head_sha)

    total_changed = len(files)
    analysable = [f for f in files if not _is_skippable_file(f)]
    if config.ignore_paths:
        kept = [f for f in analysable
                if not config.ignores_path(f.get("filename") or "")]
        if len(kept) != len(analysable):
            logger.info(
                "PR %s#%s: .reviewer.yml ignore_paths skipped %d changed "
                "file(s).",
                repo_full_name, pr_number, len(analysable) - len(kept),
            )
        analysable = kept
    if settings.MAX_FILES_PER_PR > 0:
        analysable = analysable[:settings.MAX_FILES_PER_PR]
    skipped = total_changed - len(analysable)
    if skipped > 0:
        skip_reasons = "binary, deleted, lock, minified, generated"
        if config.ignore_paths:
            skip_reasons += ", .reviewer.yml ignore_paths"
        logger.warning(
            "PR %s#%s truncated: analysing %d of %d changed files "
            "(MAX_FILES_PER_PR=%d); %d skipped (%s, or beyond the cap).",
            repo_full_name, pr_number, len(analysable), total_changed,
            settings.MAX_FILES_PER_PR, skipped, skip_reasons,
        )

    parsed = parse_files_payload(
        analysable,
        max_files=settings.MAX_FILES_PER_PR,
        max_lines=settings.GITHUB_MAX_LINES,
        max_lines_per_file=settings.MAX_LINES_PER_FILE,
    )
    if not parsed:
        logger.info("No analysable diff content for %s#%s.",
                    repo_full_name, pr_number)
        return []

    if any(fp.truncated for fp in parsed):
        logger.warning(
            "PR %s#%s truncated: hit GITHUB_MAX_LINES=%d (PR total) or "
            "MAX_LINES_PER_FILE=%d (per file); later lines were not analysed.",
            repo_full_name, pr_number, settings.GITHUB_MAX_LINES,
            settings.MAX_LINES_PER_FILE,
        )

    # custom_instructions (validated + capped at parse time) travels with
    # the reviewer; DiffReviewer sanitizes again before it reaches a prompt.
    reviewer = DiffReviewer(custom_instructions=config.custom_instructions)
    ast_analyzer = ASTAnalyzer()
    results: List[DiffReviewResult] = []
    findings_payload: List[dict] = []
    # PR-level missing-tests rule (rule-based, no LLM): flag the PR when
    # source functions/classes changed but no test file did. Uses the FULL
    # changed-file list (even files beyond the analysis cap).
    missing_tests_rule = MissingTestsRule()
    has_test_change = any(is_test_file(f.get("filename", "")) for f in files)
    first_def: Optional[tuple] = None  # (path, kind, name, line, line_text)

    for fp in parsed:
        try:
            llm_result = reviewer.review_file(fp)
        except Exception as err:  # noqa: BLE001 - one file must not kill the run
            logger.warning("Failed LLM review for %s in %s#%s: %s",
                           fp.path, repo_full_name, pr_number, err)
            llm_result = DiffReviewResult(fp.path)
        # One content fetch reused by the AST rules and the missing-tests
        # rule (source extensions only; others never need it).
        content: Optional[str] = None
        if os.path.splitext(fp.path)[1].lower() in SUPPORTED_AST_EXTENSIONS:
            try:
                content = client.get_file_content(
                    repo_full_name, fp.path, head_sha)
            except Exception as err:  # noqa: BLE001
                logger.warning("Failed to fetch content for %s in %s#%s: %s",
                               fp.path, repo_full_name, pr_number, err)
        try:
            ast_findings = _ast_findings_for_file(
                client, repo_full_name, head_sha, fp, ast_analyzer,
                content=content,
            )
        except Exception as err:  # noqa: BLE001
            logger.warning("Failed AST analysis for %s in %s#%s: %s",
                           fp.path, repo_full_name, pr_number, err)
            ast_findings = []

        if first_def is None and not has_test_change and content \
                and not is_test_file(fp.path):
            definition = missing_tests_rule.first_definition(
                content, {al.line for al in fp.added_lines})
            if definition:
                kind, def_name, def_line = definition
                line_texts = {al.line: al.content for al in fp.added_lines}
                first_def = (fp.path, kind, def_name, def_line,
                             line_texts.get(def_line))

        results.append(llm_result)
        findings_payload.extend(_merge_findings(
            fp.path,
            ast_findings,
            llm_result,
            line_texts={al.line: al.content for al in fp.added_lines},
        ))

    # At most ONE missing-tests finding per PR, anchored on the first
    # changed function/class. Flows into the review, the PR summary and
    # the DB like any other finding.
    if first_def is not None:
        def_path, kind, def_name, def_line, def_text = first_def
        findings_payload.append(missing_tests_rule.finding_dict(
            def_path, kind, def_name, def_line, def_text))
        logger.info(
            "Missing-tests check for %s#%s: no test file changed; flagged "
            "%s `%s` at %s:%d.",
            repo_full_name, pr_number, kind, def_name, def_path, def_line,
        )

    # .reviewer.yml disable_rules: findings from disabled rule IDs never
    # reach the review, the summary comment, or the database.
    if config.disable_rules:
        suppressed = [f for f in findings_payload
                      if f.get("rule_id") in config.disable_rules]
        if suppressed:
            findings_payload = [
                f for f in findings_payload
                if f.get("rule_id") not in config.disable_rules
            ]
            logger.info(
                "PR %s#%s: .reviewer.yml disable_rules suppressed %d "
                "finding(s) for rule(s): %s.",
                repo_full_name, pr_number, len(suppressed),
                ", ".join(sorted({str(f.get("rule_id")) for f in suppressed})),
            )

    logger.info(
        "Review complete for %s#%s: %d file(s), %d merged finding(s).",
        repo_full_name,
        pr_number,
        len(results),
        len(findings_payload),
    )

    # Post ONE consolidated review with inline comments; duplicates from
    # previous pushes/webhook deliveries (same path/line/category-or-
    # severity, unchanged line content) are skipped internally.
    # `insight_source` queries previous findings (same repo, same author
    # when known, excluding this PR's own rows) AFTER the inline review
    # posts, so the summary comment can carry an "Insights" section; the
    # source builds a fresh DatabaseManager per call and is never invoked
    # when nothing is posted.
    def _insight_source() -> List[str]:
        return DatabaseManager().get_pr_insight_lines(
            repo_full_name, pr_number, author, findings_payload
        )

    try:
        ReviewPoster().post_review(
            repo_full_name, pr_number, head_sha, findings_payload,
            files_analysed=len(parsed),
            total_files=total_changed,
            # Repo config overrides the MIN_SEVERITY setting when present
            # (None -> setting); config_applied makes the summary disclose it.
            min_severity=config.min_severity,
            config_applied=config.applied,
            insight_source=_insight_source,
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
