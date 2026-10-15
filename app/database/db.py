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
        rule_id TEXT,
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

# Windows used by the PR-summary insight lines. The numbers are printed in
# the lines' text ("last 10 PRs", "last 3 PRs"), so wording and query must
# stay in sync.
RECURRING_WINDOW_PRS = 10
IMPROVING_WINDOW_PRS = 3
# The summary shows at most this many insight lines.
INSIGHT_LINE_LIMIT = 3


def _ordinal(n: int) -> str:
    """English ordinal: 1 -> 1st, 2 -> 2nd, 3 -> 3rd, 11 -> 11th, ..."""
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _plural(count: int, word: str) -> str:
    """'1 finding' / '6 findings' — counts in insight lines stay exact."""
    return f"{count} {word if count == 1 else word + 's'}"


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
            # pr_findings.rule_id was added for GET /insights; ALTER databases
            # created before that column so the migration works in place.
            # Existing rows keep NULL and are never counted as a rule.
            columns = {
                row["name"]
                for row in cursor.execute("PRAGMA table_info(pr_findings)")
            }
            if "rule_id" not in columns:
                cursor.execute("ALTER TABLE pr_findings ADD COLUMN rule_id TEXT")
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
                         rule_id, message, suggestion, explanation, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        pr_review_id,
                        path,
                        f["line"],
                        f["severity"],
                        f["category"],
                        f.get("rule_id"),
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
        """Aggregate counts for GET /stats: reviews, findings by severity,
        category, and repo."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            total_reviews = cursor.execute(
                "SELECT COUNT(*) FROM pr_reviews"
            ).fetchone()[0]

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
                "total_reviews": total_reviews,
                "total_findings": sum(by_severity.values()),
                "by_severity": by_severity,
                "by_category": by_category,
                "by_repo": by_repo,
            }

    def get_recent_pr_reviews(self, limit: int = 20) -> List[dict]:
        """Most recent PR reviews with finding count and highest severity.

        Powers the /dashboard table (repo, PR number, author, time, finding
        count, highest severity). Severity ordering uses a CASE rank so
        MAX() works; no findings at all yields highest_severity=None.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT r.repo_full_name, r.pr_number, r.author, r.reviewed_at,
                       COUNT(f.id) AS finding_count,
                       MAX(CASE f.severity
                           WHEN 'CRITICAL' THEN 5
                           WHEN 'HIGH' THEN 4
                           WHEN 'MEDIUM' THEN 3
                           WHEN 'LOW' THEN 2
                           ELSE 1 END) AS max_rank
                FROM pr_reviews r
                LEFT JOIN pr_findings f ON f.pr_review_id = r.id
                GROUP BY r.id
                ORDER BY r.reviewed_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rank_to_severity = {
                5: "CRITICAL", 4: "HIGH", 3: "MEDIUM", 2: "LOW",
                1: "INFO", 0: None,
            }
            return [
                {
                    "repo": row["repo_full_name"],
                    "pr_number": row["pr_number"],
                    "author": row["author"],
                    "reviewed_at": row["reviewed_at"],
                    "finding_count": row["finding_count"],
                    "highest_severity": rank_to_severity.get(row["max_rank"]),
                }
                for row in cursor.fetchall()
            ]

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

    # ---------------------- history-aware insights ------------------------

    @staticmethod
    def _recent_pr_numbers(
        cursor: sqlite3.Cursor,
        repo_full_name: str,
        exclude_pr_number: int,
        limit: int,
    ) -> List[int]:
        """Most recently reviewed PR numbers of a repo, newest first.

        `exclude_pr_number` drops the PR currently being reviewed so its own
        rows never count toward its own history. Each PR number appears once
        even when several head SHAs were reviewed (re-pushes).
        """
        rows = cursor.execute(
            """
            SELECT pr_number FROM pr_reviews
            WHERE repo_full_name = ? AND pr_number != ?
            GROUP BY pr_number
            ORDER BY MAX(reviewed_at) DESC, MAX(id) DESC
            LIMIT ?
            """,
            (repo_full_name, exclude_pr_number, limit),
        ).fetchall()
        return [row[0] for row in rows]

    @staticmethod
    def _critical_count(cursor: sqlite3.Cursor, review_ids: List[int]) -> int:
        """CRITICAL findings across the given pr_reviews rows (0 when none)."""
        if not review_ids:
            return 0
        placeholders = ",".join("?" * len(review_ids))
        return cursor.execute(
            f"""
            SELECT COUNT(*) FROM pr_findings
            WHERE UPPER(severity) = 'CRITICAL' AND pr_review_id IN ({placeholders})
            """,
            review_ids,
        ).fetchone()[0]

    def _recurring_line(
        self,
        cursor: sqlite3.Cursor,
        repo_full_name: str,
        pr_number: int,
        author: Optional[str],
        categories: List[str],
    ) -> Optional[str]:
        """Recurring-category line, or None when nothing recurs.

        Counts previous findings whose category is one of THIS PR's
        categories, inside the repo's last RECURRING_WINDOW_PRS PRs and
        excluding the current PR. Scoped to the PR author when the author is
        known (NULL-author history rows never match a named author); with an
        unknown author the count is repo-wide and the line carries no
        @mention. Never emits a count of 0.
        """
        if not categories:
            return None
        window = self._recent_pr_numbers(
            cursor, repo_full_name, pr_number, RECURRING_WINDOW_PRS)
        if not window:
            return None
        review_ph = ",".join("?" * len(window))
        category_ph = ",".join("?" * len(categories))
        sql = (
            "SELECT f.category, COUNT(*) AS n "
            "FROM pr_findings f JOIN pr_reviews r ON r.id = f.pr_review_id "
            f"WHERE r.pr_number IN ({review_ph}) "
            f"AND f.category IN ({category_ph})"
        )
        params: list = list(window) + list(categories)
        if author is not None:
            sql += " AND r.author = ?"
            params.append(author)
        sql += " GROUP BY f.category"
        rows = cursor.execute(sql, params).fetchall()
        if not rows:
            return None
        # Most repeated category; alphabetical tie-break keeps it deterministic.
        category, previous = sorted(
            ((row[0], row[1]) for row in rows),
            key=lambda kv: (-kv[1], kv[0]),
        )[0]
        ordinal = _ordinal(previous + 1)  # this PR's finding is the NEXT one
        if author is not None:
            return (
                f"Recurring: {ordinal} '{category}' finding from @{author} "
                f"in this repo's last {RECURRING_WINDOW_PRS} PRs."
            )
        return (
            f"Recurring: {ordinal} '{category}' finding "
            f"in this repo's last {RECURRING_WINDOW_PRS} PRs."
        )

    @staticmethod
    def _most_common_line(
        cursor: sqlite3.Cursor,
        repo_full_name: str,
        pr_number: int,
    ) -> Optional[str]:
        """Most common category over the repo's previous findings.

        Excludes the current PR (its own findings are not history). None
        when the repo has no previous findings at all.
        """
        row = cursor.execute(
            """
            SELECT f.category, COUNT(*) AS findings,
                   COUNT(DISTINCT r.pr_number) AS prs
            FROM pr_findings f
            JOIN pr_reviews r ON r.id = f.pr_review_id
            WHERE r.repo_full_name = ? AND r.pr_number != ?
            GROUP BY f.category
            ORDER BY findings DESC, prs DESC, f.category ASC
            LIMIT 1
            """,
            (repo_full_name, pr_number),
        ).fetchone()
        if row is None:
            return None
        category, findings, prs = row[0], row[1], row[2]
        return (
            f"Most common issue in this repo: {category} "
            f"({_plural(findings, 'finding')} across {_plural(prs, 'PR')})."
        )

    def _improving_line(
        self,
        cursor: sqlite3.Cursor,
        repo_full_name: str,
        pr_number: int,
    ) -> Optional[str]:
        """Improving line, or None unless it is genuinely true.

        Requires a full window of IMPROVING_WINDOW_PRS previous PRs, zero
        CRITICAL findings in them, AND at least one CRITICAL finding in the
        older history — without an older CRITICAL there was nothing to
        improve from, so the claim would be invented.
        """
        window = self._recent_pr_numbers(
            cursor, repo_full_name, pr_number, IMPROVING_WINDOW_PRS)
        if len(window) < IMPROVING_WINDOW_PRS:
            return None  # cannot claim "the last 3 PRs" with fewer than 3
        window_set = set(window)
        rows = cursor.execute(
            """
            SELECT id, pr_number FROM pr_reviews
            WHERE repo_full_name = ? AND pr_number != ?
            """,
            (repo_full_name, pr_number),
        ).fetchall()
        recent_ids = [row[0] for row in rows if row[1] in window_set]
        older_ids = [row[0] for row in rows if row[1] not in window_set]
        if self._critical_count(cursor, recent_ids) > 0:
            return None
        if self._critical_count(cursor, older_ids) == 0:
            return None  # never had a CRITICAL: nothing improved
        return f"Improving: no CRITICAL findings in the last {IMPROVING_WINDOW_PRS} PRs."

    def get_pr_insight_lines(
        self,
        repo_full_name: str,
        pr_number: int,
        author: Optional[str],
        current_findings: List[dict],
    ) -> List[str]:
        """History-aware insight lines for one PR's summary comment.

        Returns up to INSIGHT_LINE_LIMIT lines, each built only from
        previous findings: this PR's own rows are excluded from every count,
        so a re-review never inflates its own history. `author` may be None
        (unknown PR author -> repo-wide history, no @mention). Returns []
        when the repo has no previous findings — the caller then omits the
        Insights section entirely. Counts are only ever read from the
        database; nothing is estimated.
        """
        categories: List[str] = []
        seen = set()
        for finding in current_findings or []:
            category = str(finding.get("category") or "Code Quality")
            if category not in seen:
                seen.add(category)
                categories.append(category)

        lines: List[str] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            has_history = cursor.execute(
                """
                SELECT 1 FROM pr_findings f
                JOIN pr_reviews r ON r.id = f.pr_review_id
                WHERE r.repo_full_name = ? AND r.pr_number != ?
                LIMIT 1
                """,
                (repo_full_name, pr_number),
            ).fetchone()
            if has_history is None:
                return []  # no history: no section at all

            for line in (
                self._recurring_line(
                    cursor, repo_full_name, pr_number, author, categories),
                self._most_common_line(cursor, repo_full_name, pr_number),
                self._improving_line(cursor, repo_full_name, pr_number),
            ):
                if line:
                    lines.append(line)
        return lines[:INSIGHT_LINE_LIMIT]

    def get_repo_insights(self, repo_full_name: str, top_n: int = 5) -> dict:
        """GET /insights aggregates, computed only from stored rows.

        Top categories, top recurring rule IDs and findings per author for
        one repo. Rows saved before rule_id existed carry NULL and are
        never counted as a rule (inventing a rule id for them would be a
        lie). A NULL author is kept as a `null` entry, not dropped.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            top_categories = [
                {"category": row[0], "findings": row[1], "prs": row[2]}
                for row in cursor.execute(
                    """
                    SELECT f.category, COUNT(*) AS findings,
                           COUNT(DISTINCT r.pr_number) AS prs
                    FROM pr_findings f
                    JOIN pr_reviews r ON r.id = f.pr_review_id
                    WHERE r.repo_full_name = ?
                    GROUP BY f.category
                    ORDER BY findings DESC, prs DESC, f.category ASC
                    LIMIT ?
                    """,
                    (repo_full_name, top_n),
                ).fetchall()
            ]

            top_rule_ids = [
                {"rule_id": row[0], "findings": row[1], "prs": row[2]}
                for row in cursor.execute(
                    """
                    SELECT f.rule_id, COUNT(*) AS findings,
                           COUNT(DISTINCT r.pr_number) AS prs
                    FROM pr_findings f
                    JOIN pr_reviews r ON r.id = f.pr_review_id
                    WHERE r.repo_full_name = ?
                      AND f.rule_id IS NOT NULL AND f.rule_id != ''
                    GROUP BY f.rule_id
                    ORDER BY findings DESC, prs DESC, f.rule_id ASC
                    LIMIT ?
                    """,
                    (repo_full_name, top_n),
                ).fetchall()
            ]

            findings_per_author = [
                {"author": row[0], "findings": row[1], "prs": row[2]}
                for row in cursor.execute(
                    """
                    SELECT r.author, COUNT(*) AS findings,
                           COUNT(DISTINCT r.pr_number) AS prs
                    FROM pr_findings f
                    JOIN pr_reviews r ON r.id = f.pr_review_id
                    WHERE r.repo_full_name = ?
                    GROUP BY r.author
                    ORDER BY findings DESC, r.author IS NULL ASC,
                             r.author ASC
                    """,
                    (repo_full_name,),
                ).fetchall()
            ]

            return {
                "repo": repo_full_name,
                "top_categories": top_categories,
                "top_rule_ids": top_rule_ids,
                "findings_per_author": findings_per_author,
            }
