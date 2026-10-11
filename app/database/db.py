import json
import sqlite3
from typing import List, Optional
from app.config import settings
from app.schemas.review import ReviewResponse, ReviewHistoryItem, ReviewSummary, Finding


class DatabaseManager:
    """SQLite Database Manager for persisting and retrieving review history."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DATABASE_PATH
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Create the reviews table if it does not exist."""
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
            conn.commit()

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
