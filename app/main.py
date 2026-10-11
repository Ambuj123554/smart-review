import uuid
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.analyzer.engine import ASTAnalyzer
from app.ai.reviewer import GeminiReviewer
from app.database.db import DatabaseManager
from app.schemas.review import (
    ReviewRequest,
    ReviewResponse,
    ReviewSummary,
    ReviewListResponse,
    ReviewHistoryItem
)

# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-Powered Code Review Engine using FastAPI, AST analysis, and Google Gemini API."
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
ast_analyzer = ASTAnalyzer()
ai_reviewer = GeminiReviewer()
db_manager = DatabaseManager()


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
    rule_findings = ast_analyzer.analyze(request.code)

    # 2. Contextual AI Analysis via Gemini (with graceful fallback if unconfigured/fails)
    ai_findings = ai_reviewer.analyze(request.code, rule_findings)

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
        ai_findings=ai_findings
    )

    # 5. Persist to SQLite Database
    try:
        db_manager.save_review(response, request.code)
    except Exception as err:
        # DB failure should not prevent returning analysis results to user
        print(f"Warning: Failed to save review to SQLite database: {err}")

    return response


@app.get("/reviews", response_model=ReviewListResponse, tags=["Review History"])
def get_review_history():
    """Retrieves all past code reviews stored in SQLite database."""
    reviews = db_manager.get_all_reviews()
    return ReviewListResponse(
        total_reviews=len(reviews),
        reviews=reviews
    )


@app.get("/reviews/{review_id}", response_model=ReviewHistoryItem, tags=["Review History"])
def get_review_by_id(review_id: str):
    """Retrieves details of a specific past review by ID."""
    review = db_manager.get_review_by_id(review_id)
    if not review:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Review record '{review_id}' not found."
        )
    return review
