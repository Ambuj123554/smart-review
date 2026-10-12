import logging
import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.analyzer.engine import ASTAnalyzer
from app.github.webhooks import router as github_webhook_router
from app.ai.providers import get_reviewer
from app.ai.reviewer import build_review_prompt, parse_findings_response
from app.database.db import DatabaseManager
from app.schemas.review import (
    ReviewRequest,
    ReviewResponse,
    ReviewSummary,
    ReviewListResponse,
    ReviewHistoryItem,
    Finding
)

logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-Powered Code Review Engine using FastAPI, AST analysis, and the configured LLM provider (Gemini or Groq)."
)

def _cors_settings(origins_raw: str) -> dict:
    """Builds CORSMiddleware kwargs from a comma-separated origins string.

    Never combines allow_origins=["*"] with allow_credentials=True, which
    browsers reject and which would expose cookies to any site.
    """
    origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
    if not origins:
        origins = ["http://localhost:8000"]
    return {
        "allow_origins": origins,
        "allow_credentials": "*" not in origins,
        "allow_methods": ["GET", "POST"],
        "allow_headers": ["Content-Type", "X-Hub-Signature-256", "X-GitHub-Event"],
    }


app.add_middleware(CORSMiddleware, **_cors_settings(settings.ALLOWED_ORIGINS))

# Register GitHub webhook routes
app.include_router(github_webhook_router)




def _calculate_summary(rule_findings, ai_findings) -> ReviewSummary:
    """Aggregates all findings into a severity counts summary."""
    all_findings = rule_findings + ai_findings
    summary = ReviewSummary(
        total_issues=len(all_findings),
        critical=sum(1 for f in all_findings if f.severity.upper() == "CRITICAL"),
        high=sum(1 for f in all_findings if f.severity.upper() == "HIGH"),
        medium=sum(1 for f in all_findings if f.severity.upper() == "MEDIUM"),
        low=sum(1 for f in all_findings if f.severity.upper() == "LOW"),
    )
    return summary


@app.get("/", tags=["Health"])
def root():
    """Root endpoint welcoming users."""
    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "operational",
        "docs_url": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint confirming engine readiness."""
    gemini_configured = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip() not in ("", "your_gemini_api_key_here"))
    return {
        "status": "healthy",
        "ast_engine": "ready",
        "gemini_api_configured": gemini_configured,
        "database_path": settings.DATABASE_PATH
    }


# Initialized lazily so test fixtures can override settings before first use.
ast_analyzer: ASTAnalyzer | None = None
db_manager: DatabaseManager | None = None


def _get_db_manager() -> DatabaseManager:
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def _get_ast_analyzer() -> ASTAnalyzer:
    global ast_analyzer
    if ast_analyzer is None:
        ast_analyzer = ASTAnalyzer()
    return ast_analyzer


def _get_ai_reviewer():
    """Configured LLM reviewer.

    Uses the same provider selection as the webhook path
    (app.ai.providers.get_reviewer: LLM_PROVIDER=gemini|groq). Built per
    request so settings overrides made by tests always apply.
    """
    return get_reviewer()


def _run_ai_review(code: str, rule_findings: list[Finding]) -> tuple[list[Finding], str]:
    """Runs the configured LLM on the code. Never raises.

    Returns (ai_findings, ai_status) where ai_status is "ok" on success or
    "error: <reason>" when the provider is unusable, the call failed, or the
    model returned invalid output. Failures are logged as warnings so a
    broken LLM setup is visible instead of silently producing no findings.
    """
    try:
        reviewer = _get_ai_reviewer()
    except ValueError as err:  # unknown LLM_PROVIDER: fail loudly, not silently
        logger.warning("AI review skipped: %s", err)
        return [], f"error: {err}"

    provider = type(reviewer).__name__

    reason = reviewer.unconfigured_reason()
    if reason:
        logger.warning("AI review skipped (%s): %s", provider, reason)
        return [], f"error: {reason}"

    try:
        raw = reviewer.call(build_review_prompt(code, rule_findings))
    except Exception as err:  # noqa: BLE001 - one bad call must not kill /review
        logger.warning("AI review failed (%s): %s", provider, err)
        return [], "error: " + " ".join(str(err).split())

    if not isinstance(raw, str) or not raw.strip():
        logger.warning("AI review failed (%s): empty response from LLM.", provider)
        return [], "error: empty response from LLM"

    findings = parse_findings_response(raw)
    if findings is None:
        logger.warning(
            "AI review failed (%s): invalid output, expected a JSON array of "
            "findings. Response began with: %.200r",
            provider,
            raw.strip(),
        )
        return [], "error: LLM returned invalid output (expected JSON array)"

    return findings, "ok"


@app.post("/review", response_model=ReviewResponse, tags=["Code Review"])
def analyze_code(request: ReviewRequest):
    """
    Submits Python source code for analysis.
    Performs deterministic AST static analysis and optional AI contextual analysis.
    Stores the result in SQLite database history.
    """
    if not request.code or not request.code.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Source code cannot be empty."
        )

    # 1. Deterministic AST Rule Analysis
    rule_findings = _get_ast_analyzer().analyze(request.code)

    # 2. Contextual AI Analysis via the configured provider (same selection
    #    as the webhook path); failures degrade to ai_status="error: ..."
    #    instead of a 500 or silently empty findings.
    ai_findings, ai_status = _run_ai_review(request.code, rule_findings)

    # 3. Calculate Aggregated Summary
    summary = _calculate_summary(rule_findings, ai_findings)

    # 4. Construct Response Metadata
    review_id = f"rev-{uuid.uuid4().hex[:8]}"
    timestamp = datetime.now(timezone.utc).isoformat()
    identifier = request.identifier or "snippet.py"

    response = ReviewResponse(
        review_id=review_id,
        timestamp=timestamp,
        identifier=identifier,
        summary=summary,
        rule_based_findings=rule_findings,
        ai_findings=ai_findings,
        ai_status=ai_status
    )

    # 5. Persist to SQLite Database
    try:
        _get_db_manager().save_review(response, request.code)
    except Exception as err:
        # DB failure should not prevent returning analysis results to user
        print(f"Warning: Failed to save review to SQLite database: {err}")

    return response


@app.get("/reviews", response_model=ReviewListResponse, tags=["Review History"])
def get_review_history():
    """Retrieves all past code reviews stored in SQLite database."""
    reviews = _get_db_manager().get_all_reviews()
    return ReviewListResponse(
        total_reviews=len(reviews),
        reviews=reviews
    )


@app.get("/stats", tags=["Analytics"])
def get_stats():
    """Aggregated counts of PR findings by severity, category, and repo."""
    return _get_db_manager().get_stats()


@app.get("/reviews/{review_id}", response_model=ReviewHistoryItem, tags=["Review History"])
def get_review_by_id(review_id: str):
    """Retrieves details of a specific past review by ID."""
    review = _get_db_manager().get_review_by_id(review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review record '{review_id}' not found."
        )
    return review
