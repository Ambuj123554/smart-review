import json
import sqlite3
from datetime import datetime, timezone
from typing import List, Optional
from app.config import settings
from app.schemas.review import ReviewResponse, ReviewHistoryItem, ReviewSummary, Finding


PR_FINDINGS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS pr_reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_full_name TEXT NOT NULL,
        pr_number INTEGER NOT NULL,
        author TEXT,
        head_sha TEXT NOT NULL,
        reviewed_at TEXT NOT NULL,
        UNIQUE (repo_full_name, pr_number, head_sha)
    );
    CREATE INDEX IF NOT EXISTS idx_pr_reviews_repo_time
        ON pr_reviews (repo_full_name, reviewed_at);

    CREATE TABLE IF NOT EXISTS pr_findings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pr_review_id INTEGER NOT NULL REFERENCES pr_reviews(id),
        file_path TEXT NOT NULL,
        line INTEGER NOT NULL,
        severity TEXT NOT NULL,
        category TEXT NOT NULL,
        message TEXT NOT NULL,
        suggestion TEXT,
        explanation TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_pr_findings_sev_cat
        ON pr_findings (severity, category);
    CREATE INDEX IF NOT EXISTS idx_pr_findings_review
        ON pr_findings (pr_review_id);
"""


class DatabaseManager:
    """SQLite Database Manager for persisting and retrieving review history."""

    def __init__(self, db_path: str = None):
        # Read at call time (not import time) so test fixtures can override settings.DATABASE_PATH first
        self.db_path = db_path or settings.DATABASE_PATH
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create the reviews and per-PR tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_id TEXT UNIQUE NOT NULL,
                    timestamp TEXT NOT NULL,
                    identifier TEXT NOT NULL,
                    total_issues INTEGER NOT NULL,
                    critical_count INTEGER NOT NULL,
                    high_count INTEGER NOT NULL,
                    medium_count INTEGER NOT NULL,
                    low_count INTEGER NOT NULL,
                    rule_findings_json TEXT NOT NULL,
                    ai_findings_json TEXT NOT NULL,
                    code_snippet TEXT
                )
            """)
            conn.executescript(PR_FINDINGS_SCHEMA)
            conn.commit()

    def save_pr_review(
        self,
        repo_full_name: str,
        pr_number: int,
        head_sha: str,
        author: Optional[str],
        findings: List[dict],
    ) -> int:
        """Persists one PR review and its findings.

        `findings` is a list of dicts with keys:
        path (file_path is accepted as a legacy alias), line, severity,
        category, message, and optionally suggestion, explanation
        (kept NULL when absent). Returns the pr_reviews row id.

        Raises KeyError when a finding carries neither path nor file_path
        so a schema drift is loud instead of a silent data-loss bug.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO pr_reviews
                    (repo_full_name, pr_number, author, head_sha, reviewed_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (repo_full_name, pr_number, author, head_sha, now),
            )
            cursor.execute(
                """
                SELECT id FROM pr_reviews
                WHERE repo_full_name = ? AND pr_number = ? AND head_sha = ?
                """,
                (repo_full_name, pr_number, head_sha),
            )
            row = cursor.fetchone()
            existing_id = row[0] if row else None
            pr_review_id = existing_id if existing_id is not None else cursor.lastrowid

            # Idempotency: the same (repo, pr, sha) already recorded means a
            # duplicate delivery or a re-review of the same commit; keep the
            # first record rather than double-counting findings.
            if existing_id is not None:
                cursor.execute(
                    "SELECT COUNT(*) FROM pr_findings WHERE pr_review_id = ?",
                    (existing_id,),
                )
                already_saved = cursor.fetchone()[0]
                if already_saved > 0:
                    conn.commit()
                    return existing_id  # duplicate: skip, keep first snapshot
            elif not findings:
                conn.commit()
                return pr_review_id  # nothing to insert after all

            for f in findings:
                # The webhook pipeline (and comment posting) keys findings
                # by "path"; older callers used "file_path". Accept both so
                # the two never drift into a KeyError again.
                path = f.get("path") or f.get("file_path")
                if not path:
                    raise KeyError(
                        "finding is missing 'path' key (legacy 'file_path' "
                        f"also accepted): {sorted(f)}"
                    )
                cursor.execute(
                    """
                    INSERT INTO pr_findings
                        (pr_review_id, file_path, line, severity, category,
                         message, suggestion, explanation, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pr_review_id,
                        path,
                        f["line"],
                        f["severity"],
                        f["category"],
                        f["message"],
                        f.get("suggestion"),
                        f.get("explanation"),
                        now,
                    ),
                )
            conn.commit()
            return pr_review_id

    def save_review(self, response: ReviewResponse, code: str = "") -> None:
        """Stores a completed review into SQLite database."""
        rule_findings_json = json.dumps([f.model_dump() for f in response.rule_based_findings])
        ai_findings_json = json.dumps([f.model_dump() for f in response.ai_findings])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO reviews (
                    review_id, timestamp, identifier, total_issues,
                    critical_count, high_count, medium_count, low_count,
                    rule_findings_json, ai_findings_json, code_snippet
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                response.review_id,
                response.timestamp,
                response.identifier,
                response.summary.total_issues,
                response.summary.critical,
                response.summary.high,
                response.summary.medium,
                response.summary.low,
                rule_findings_json,
                ai_findings_json,
                code[:2000] if code else "" # Store first 2000 chars of code
            ))
            conn.commit()

    def get_all_reviews(self) -> List[ReviewHistoryItem]:
        """Retrieves all past review records from SQLite."""
        reviews: List[ReviewHistoryItem] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reviews ORDER BY id DESC")
            rows = cursor.fetchall()
            for row in rows:
                rule_findings = [Finding(**f) for f in json.loads(row["rule_findings_json"])]
                ai_findings = [Finding(**f) for f in json.loads(row["ai_findings_json"])]
                summary = ReviewSummary(
                    total_issues=row["total_issues"],
                    critical=row["critical_count"],
                    high=row["high_count"],
                    medium=row["medium_count"],
                    low=row["low_count"]
                )
                reviews.append(
                    ReviewHistoryItem(
                        review_id=row["review_id"],
                        timestamp=row["timestamp"],
                        identifier=row["identifier"],
                        summary=summary,
                        rule_based_findings=rule_findings,
                        ai_findings=ai_findings
                    )
                )
        return reviews

    def get_stats(self) -> dict:
        """Aggregate counts for GET /stats: by severity, category, and repo."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            by_severity = dict(cursor.execute(
                "SELECT severity, COUNT(*) FROM pr_findings GROUP BY severity"
            ).fetchall())
            by_category = dict(cursor.execute(
                "SELECT category, COUNT(*) FROM pr_findings GROUP BY category"
            ).fetchall())
            by_repo = dict(cursor.execute(
                """
                SELECT r.repo_full_name, COUNT(*)
                FROM pr_findings f
                JOIN pr_reviews r ON r.id = f.pr_review_id
                GROUP BY r.repo_full_name
                """
            ).fetchall())

            return {
                "total_findings": sum(by_severity.values()),
                "by_severity": by_severity,
                "by_category": by_category,
                "by_repo": by_repo,
            }

    def get_review_by_id(self, review_id: str) -> Optional[ReviewHistoryItem]:
        """Retrieves a single review record by review_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,))
            row = cursor.fetchone()
            if not row:
                return None

            rule_findings = [Finding(**f) for f in json.loads(row["rule_findings_json"])]
            ai_findings = [Finding(**f) for f in json.loads(row["ai_findings_json"])]
            summary = ReviewSummary(
                total_issues=row["total_issues"],
                critical=row["critical_count"],
                high=row["high_count"],
                medium=row["medium_count"],
                low=row["low_count"]
            )
            return ReviewHistoryItem(
                review_id=row["review_id"],
                timestamp=row["timestamp"],
                identifier=row["identifier"],
                summary=summary,
                rule_based_findings=rule_findings,
                ai_findings=ai_findings
            )
