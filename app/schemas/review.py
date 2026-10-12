from typing import List, Optional
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """Structured representation of a single code review finding."""
    rule_id: str = Field(..., description="Unique rule or issue identifier, e.g. SEC-001 or AI-LOGIC-001")
    category: str = Field(..., description="Issue category e.g. Security, Bug Risk, Code Style, Error Handling, Control Flow")
    severity: str = Field(..., description="Severity level: CRITICAL, HIGH, MEDIUM, LOW")
    line: int = Field(..., description="1-indexed line number in the source code where the issue was detected")
    message: str = Field(..., description="Short summary message of the detected issue")
    explanation: str = Field(..., description="Detailed explanation of why this is problematic")
    suggestion: str = Field(..., description="Actionable recommendation or fix suggestion")
    source: str = Field("Rule-Based Analysis", description="Origin of finding: 'Rule-Based Analysis' or 'AI Analysis'")


class ReviewSummary(BaseModel):
    """Aggregated statistics of findings by severity level."""
    total_issues: int = 0
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0


class ReviewRequest(BaseModel):
    """Payload for submitting Python source code for analysis."""
    code: str = Field(..., description="Python source code text to be reviewed")
    identifier: Optional[str] = Field("snippet.py", description="Optional file name or identifier for logging/history")


class ReviewResponse(BaseModel):
    """Structured result returned after reviewing code."""
    review_id: str
    timestamp: str
    identifier: str
    summary: ReviewSummary
    rule_based_findings: List[Finding]
    ai_findings: List[Finding]
    ai_status: str = Field(
        "ok",
        description="AI review outcome: 'ok' or 'error: <reason>'",
    )


class ReviewHistoryItem(BaseModel):
    """Summary item for listing past reviews."""
    review_id: str
    timestamp: str
    identifier: str
    summary: ReviewSummary
    rule_based_findings: List[Finding]
    ai_findings: List[Finding]


class ReviewListResponse(BaseModel):
    """Response containing a list of past code reviews."""
    total_reviews: int
    reviews: List[ReviewHistoryItem]
