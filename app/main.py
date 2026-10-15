import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from html import escape
from typing import List, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import settings
from app.analyzer.engine import ASTAnalyzer
from app.github.webhooks import router as github_webhook_router
from app.ai.providers import active_model, active_provider, get_reviewer
from app.github.comments import RISK_ORDER, effective_min_severity
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Logs the active LLM provider/model at startup (never keys).

    Makes a misconfigured LLM_PROVIDER visible in the logs immediately
    instead of surfacing later as a surprising call to the wrong vendor.
    """
    provider = active_provider()
    model = active_model()
    logger.info(
        "Active LLM provider: %s (model: %s). Set via LLM_PROVIDER.",
        provider,
        model,
    )
    # Validate MIN_SEVERITY at startup so a typo (e.g. MIN_SEVERITY=INFO2)
    # is visible immediately; invalid values warn and fall back to LOW.
    logger.info(
        "Minimum severity for inline comments: %s (MIN_SEVERITY).",
        effective_min_severity(),
    )
    yield


# Initialize FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="AI-Powered Code Review Engine using FastAPI, AST analysis, and the configured LLM provider (Gemini or Groq).",
    lifespan=lifespan,
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


@app.get("/insights", tags=["Analytics"])
def get_insights(repo: str):
    """History-aware insights for one repository (owner/name), computed
    purely from stored PR findings: top categories, top recurring rule IDs
    and findings per author (a null PR author is kept as a null entry).
    Unknown repos return empty lists — counts are never invented."""
    return _get_db_manager().get_repo_insights(repo)


# --------------------------- /dashboard (HTML) -----------------------------

_DASHBOARD_STYLE = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
       Roboto, sans-serif; background: #f4f6fa; color: #1f2937; }
main { max-width: 1080px; margin: 0 auto; padding: 24px 32px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; }
.subtitle { color: #6b7280; margin: 0 0 20px; font-size: 13px; }
.cards { display: flex; gap: 16px; margin-bottom: 20px; }
.card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
        padding: 14px 20px; min-width: 180px; }
.stat-label { color: #6b7280; font-size: 12px; text-transform: uppercase;
              letter-spacing: .04em; }
.stat-value { font-size: 28px; font-weight: 600; margin-top: 2px; }
.grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
        margin-bottom: 20px; }
.panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
         padding: 14px 16px; }
.panel h2 { font-size: 14px; margin: 0 0 10px; }
.bar-row { display: flex; align-items: center; gap: 8px; margin: 6px 0;
           font-size: 13px; }
.bar-label { width: 120px; overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; }
.bar-track { flex: 1; background: #eef2f7; border-radius: 4px; height: 14px; }
.bar { height: 14px; border-radius: 4px; background: #4c7dff; width: 0; }
.bar-count { width: 44px; text-align: right; color: #374151; }
.table-panel { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px;
               padding: 14px 16px; }
.table-panel h2 { font-size: 14px; margin: 0 0 10px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid #eef2f7; }
th { color: #6b7280; font-weight: 600; font-size: 12px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 999px;
         font-size: 11px; font-weight: 600; }
.sev-critical { background: #fee2e2; color: #b91c1c; }
.sev-high { background: #ffedd5; color: #c2410c; }
.sev-medium { background: #fef9c3; color: #a16207; }
.sev-low { background: #dcfce7; color: #15803d; }
.sev-info { background: #e0f2fe; color: #0369a1; }
.muted { color: #9ca3af; }
.empty-state { background: #fff; border: 1px dashed #d1d5db; border-radius: 10px;
               padding: 36px; text-align: center; color: #6b7280;
               margin-bottom: 20px; }
.empty-state h2 { color: #374151; margin: 0 0 6px; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
"""

_DASHBOARD_SCRIPT = """
  // Size the CSS bar charts from their data-value (percent).
  document.querySelectorAll('.bar').forEach(function (bar) {
    bar.style.width = Math.max(2, Number(bar.getAttribute('data-value') || 0)) + '%';
  });
"""


def _dashboard_bars(counts: dict, order: Optional[List[str]] = None) -> str:
    """Renders CSS bar rows for a {label: count} mapping (labels escaped)."""
    if not counts:
        return '<p class="muted">None yet.</p>'
    if order:
        items = ([(k, counts[k]) for k in order if k in counts]
                 + sorted(((k, v) for k, v in counts.items() if k not in order),
                          key=lambda kv: -kv[1]))
    else:
        items = sorted(counts.items(), key=lambda kv: -kv[1])
    top = max(int(v) for v in counts.values()) or 1
    rows = []
    for label, count in items:
        pct = int(round(100 * int(count) / top))
        rows.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{escape(str(label))}</span>'
            '<div class="bar-track">'
            f'<div class="bar" data-value="{pct}"></div>'
            '</div>'
            f'<span class="bar-count">{int(count)}</span>'
            '</div>'
        )
    return "".join(rows)


def _dashboard_recent_rows(recent: List[dict]) -> str:
    """Table rows for the most recent PR reviews; every DB value is escaped."""
    if not recent:
        return '<tr><td colspan="6" class="muted">No reviews yet.</td></tr>'
    rows = []
    for r in recent:
        raw_repo = str(r["repo"])
        repo = escape(raw_repo)
        try:
            pr_number = int(r["pr_number"])
        except (TypeError, ValueError):
            pr_number = 0
        author = escape(str(r["author"])) if r["author"] else "—"
        when = escape(str(r["reviewed_at"]))
        count = int(r["finding_count"] or 0)
        severity = r["highest_severity"]
        if severity:
            sev = escape(str(severity))
            badge = f'<span class="badge sev-{sev.lower()}">{sev}</span>'
        else:
            badge = '<span class="muted">—</span>'
        link = escape(
            f"https://github.com/{raw_repo}/pull/{pr_number}", quote=True)
        rows.append(
            '<tr>'
            f'<td><a href="{link}" target="_blank" '
            f'rel="noopener">{repo}</a></td>'
            f'<td>#{pr_number}</td>'
            f'<td>{author}</td>'
            f'<td>{when}</td>'
            f'<td>{count}</td>'
            f'<td>{badge}</td>'
            '</tr>'
        )
    return "".join(rows)


def _dashboard_html(stats: dict, recent: List[dict]) -> str:
    """Builds the single dashboard page from /stats + recent-review rows.

    Pure server-rendered HTML/CSS with a tiny inline script (bar widths);
    no frontend framework and no external requests.
    """
    total_reviews = int(stats.get("total_reviews") or 0)
    total_findings = int(stats.get("total_findings") or 0)
    by_severity = stats.get("by_severity") or {}
    by_category = stats.get("by_category") or {}
    by_repo = stats.get("by_repo") or {}

    if total_reviews == 0 and total_findings == 0:
        empty_state = (
            '<div class="empty-state">'
            '<h2>No reviews yet</h2>'
            '<p>Open a pull request and the assistant will post its first '
            'review — charts and the review log will appear here.</p>'
            '</div>'
        )
    else:
        empty_state = ""

    return (
        '<!DOCTYPE html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Smart Review Assistant — Dashboard</title>\n'
        f'<style>{_DASHBOARD_STYLE}</style>\n'
        '</head>\n'
        '<body>\n'
        '<main>\n'
        '<h1>Smart Review Assistant — Dashboard</h1>\n'
        '<p class="subtitle">Pull request review activity, from the '
        '<code>/stats</code> data and the review history database.</p>\n'
        f'{empty_state}'
        '<div class="cards">\n'
        '<div class="card"><div class="stat-label">Total reviews</div>'
        f'<div class="stat-value">{total_reviews}</div></div>\n'
        '<div class="card"><div class="stat-label">Total findings</div>'
        f'<div class="stat-value">{total_findings}</div></div>\n'
        '</div>\n'
        '<section class="grid">\n'
        '<div class="panel"><h2>Findings by severity</h2>'
        f'{_dashboard_bars(by_severity, order=RISK_ORDER)}</div>\n'
        '<div class="panel"><h2>Findings by category</h2>'
        f'{_dashboard_bars(by_category)}</div>\n'
        '<div class="panel"><h2>Findings per repository</h2>'
        f'{_dashboard_bars(by_repo)}</div>\n'
        '</section>\n'
        '<section class="table-panel">\n'
        '<h2>Most recent reviews</h2>\n'
        '<table>\n'
        '<thead><tr><th>Repo</th><th>PR</th><th>Author</th><th>Time</th>'
        '<th>Findings</th><th>Highest severity</th></tr></thead>\n'
        f'<tbody>{_dashboard_recent_rows(recent)}</tbody>\n'
        '</table>\n'
        '</section>\n'
        '</main>\n'
        f'<script>{_DASHBOARD_SCRIPT}</script>\n'
        '</body>\n'
        '</html>\n'
    )


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def get_dashboard():
    """Server-rendered dashboard page: totals, severity/category/repo bar
    charts, and the 20 most recent PR reviews linked to GitHub. All values
    read from the database are HTML-escaped before rendering."""
    db = _get_db_manager()
    return HTMLResponse(
        _dashboard_html(db.get_stats(), db.get_recent_pr_reviews(limit=20))
    )


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
