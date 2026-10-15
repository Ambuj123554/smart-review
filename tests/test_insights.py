"""History-aware insights: PR-summary lines and GET /insights.

Every count below comes from seeded database rows and is asserted exactly,
so no insight line can ever be invented:

- recurring author finding (the current PR's own rows excluded from its count)
- most common category (same exclusion)
- no history -> no lines, so the summary omits the section entirely
- null PR author: repo-wide recurring count without an @mention, plus a
  `null` author entry in /insights instead of a dropped row
"""
import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.config import settings
from app.database.db import DatabaseManager
from app.github.comments import ReviewPoster


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A fresh database file, also wired into the app's cached manager."""
    path = str(tmp_path / "insights.db")
    monkeypatch.setattr(settings, "DATABASE_PATH", path)
    monkeypatch.setattr(main_module, "db_manager", None)
    yield DatabaseManager(db_path=path)


@pytest.fixture
def client(db):
    with TestClient(main_module.app) as c:
        yield c


def _finding(category, severity="HIGH", rule_id=None, line=1):
    finding = {
        "path": "a.py",
        "line": line,
        "severity": severity,
        "category": category,
        "message": f"{category} issue",
    }
    if rule_id is not None:
        finding["rule_id"] = rule_id
    return finding


# --------------------- recurring author finding ----------------------------

def test_recurring_author_finding_line(db):
    """Alice's 3rd 'Security' finding: two previous ones from her, with
    (a) another author's Security finding NOT counted toward her total and
    (b) the current PR's own already-saved rows NOT counted at all."""
    db.save_pr_review("o/r", 1, "sha1", "alice", [_finding("Security")])
    db.save_pr_review("o/r", 2, "sha2", "alice", [_finding("Security")])
    db.save_pr_review("o/r", 3, "sha3", "bob", [_finding("Security")])
    # The current PR (9) is already on disk (redelivery/re-push): its own
    # finding must not turn "3rd" into "4th".
    db.save_pr_review("o/r", 9, "sha9", "alice", [_finding("Security")])

    lines = db.get_pr_insight_lines(
        "o/r", 9, "alice", [_finding("Security")])

    assert lines == [
        "Recurring: 3rd 'Security' finding from @alice "
        "in this repo's last 10 PRs.",
        "Most common issue in this repo: Security (3 findings across 3 PRs).",
    ]


def test_no_recurring_line_for_categories_absent_from_history(db):
    """A category the author never had before does not 'recur'."""
    db.save_pr_review("o/r", 1, "sha1", "alice", [_finding("Security")])

    lines = db.get_pr_insight_lines(
        "o/r", 2, "alice", [_finding("Error Handling")])

    assert not any(line.startswith("Recurring") for line in lines)


# ------------------------ most common category -----------------------------

def test_most_common_category_line(db):
    """Exactly the documented shape: the top previous category with its
    finding and PR counts, current PR's own rows excluded."""
    for pr_number, count in ((1, 2), (2, 2), (3, 1), (4, 1)):
        db.save_pr_review(
            "o/r", pr_number, f"sha{pr_number}", "alice",
            [_finding("Error Handling", line=line)
             for line in range(1, count + 1)],
        )
    db.save_pr_review("o/r", 5, "sha5", "bob",
                      [_finding("Security", line=1),
                       _finding("Security", line=2)])
    # Current PR (6) already on disk: its 2 Error Handling findings and its
    # PR number must not count toward the history.
    db.save_pr_review("o/r", 6, "sha6", "alice",
                      [_finding("Error Handling", line=1),
                       _finding("Error Handling", line=2)])

    lines = db.get_pr_insight_lines(
        "o/r", 6, "alice", [_finding("Testing")])

    assert lines == [
        "Most common issue in this repo: Error Handling "
        "(6 findings across 4 PRs)."
    ]


# ------------------------------ no history ---------------------------------

def test_no_history_returns_no_lines(db):
    """No previous findings for this repo -> no lines at all (the summary
    comment then skips the Insights section entirely)."""
    assert db.get_pr_insight_lines(
        "o/r", 1, "alice", [_finding("Security")]) == []

    # History in ANOTHER repo is not this repo's history.
    db.save_pr_review("other/x", 1, "sha1", "alice", [_finding("Security")])
    assert db.get_pr_insight_lines(
        "o/r", 1, "alice", [_finding("Security")]) == []

    # A PR reviewed with zero findings is not "history" either.
    db.save_pr_review("o/r", 2, "sha2", "alice", [])
    assert db.get_pr_insight_lines(
        "o/r", 3, "alice", [_finding("Security")]) == []


# ------------------------------ null author --------------------------------

def test_null_author_recurring_is_repo_wide_without_mention(db):
    """Unknown PR author: the count is repo-wide (NULL and named authors
    alike) and the line never fabricates an '@None' mention."""
    db.save_pr_review("o/r", 1, "sha1", None, [_finding("Security")])
    db.save_pr_review("o/r", 2, "sha2", "alice", [_finding("Security")])

    lines = db.get_pr_insight_lines("o/r", 3, None, [_finding("Security")])

    assert lines[0] == (
        "Recurring: 3rd 'Security' finding in this repo's last 10 PRs."
    )
    assert "@None" not in "\n".join(lines)
    assert lines[1] == (
        "Most common issue in this repo: Security (2 findings across 2 PRs)."
    )


def test_null_author_history_does_not_recur_for_named_author(db):
    """NULL-author rows can't be attributed to a named author, so a
    named-author review must not pick them up as its own recurrence."""
    db.save_pr_review("o/r", 1, "sha1", None, [_finding("Security")])

    lines = db.get_pr_insight_lines(
        "o/r", 2, "alice", [_finding("Security")])

    assert not any(line.startswith("Recurring") for line in lines)


# --------------------------- improving line ---------------------------------

def test_improving_line_only_when_older_prs_actually_had_criticals(db):
    """'Improving' is only emitted with a full 3-PR window, zero CRITICAL
    findings inside it, and a real CRITICAL to have improved from."""
    db.save_pr_review("o/r", 1, "sha1", "alice",
                      [_finding("Security", "CRITICAL")])
    for pr_number in (2, 3, 4, 5):
        db.save_pr_review("o/r", pr_number, f"sha{pr_number}", "alice",
                          [_finding("Style")])

    lines = db.get_pr_insight_lines("o/r", 6, "alice", [_finding("Logic")])

    assert "Improving: no CRITICAL findings in the last 3 PRs." in lines


def test_improving_line_absent_without_older_critical_history(db):
    """A repo that never had a CRITICAL finding did not 'improve'."""
    for pr_number in (1, 2, 3, 4):
        db.save_pr_review("o/r", pr_number, f"sha{pr_number}", "alice",
                          [_finding("Style")])

    lines = db.get_pr_insight_lines("o/r", 5, "alice", [_finding("Logic")])

    assert not any(line.startswith("Improving") for line in lines)


# ----------------------------- /insights -----------------------------------

def test_insights_endpoint_returns_top_categories_rules_and_authors(db, client):
    for pr_number, count in ((1, 2), (2, 2), (3, 1), (4, 1)):
        db.save_pr_review(
            "o/r", pr_number, f"sha{pr_number}", "alice",
            [_finding("Error Handling", line=line)
             for line in range(1, count + 1)],
        )  # legacy rows without rule_id must never count as a rule
    db.save_pr_review("o/r", 5, "sha5", "alice",
                      [_finding("Security", rule_id="SEC-002", line=1),
                       _finding("Security", rule_id="SEC-002", line=2)])
    db.save_pr_review("o/r", 6, "sha6", "bob",
                      [_finding("Security", rule_id="SEC-002", line=1)])
    db.save_pr_review("other/x", 7, "sha7", "mallory",
                      [_finding("Style", rule_id="AST-004")])  # other repo

    data = client.get("/insights", params={"repo": "o/r"}).json()

    assert data["repo"] == "o/r"
    assert data["top_categories"] == [
        {"category": "Error Handling", "findings": 6, "prs": 4},
        {"category": "Security", "findings": 3, "prs": 2},
    ]
    assert data["top_rule_ids"] == [
        {"rule_id": "SEC-002", "findings": 3, "prs": 2},
    ]
    authors = data["findings_per_author"]
    # alice: PRs 1-4 (6 Error Handling) + PR5 (2 Security) = 8 across 5 PRs.
    assert authors[0] == {"author": "alice", "findings": 8, "prs": 5}
    assert {"author": "bob", "findings": 1, "prs": 1} in authors
    assert not any(entry["author"] == "mallory" for entry in authors)


def test_insights_endpoint_keeps_null_author_entry(db, client):
    db.save_pr_review("o/r", 1, "sha1", None, [_finding("Security")])
    db.save_pr_review("o/r", 2, "sha2", None, [_finding("Security")])
    db.save_pr_review("o/r", 3, "sha3", "alice", [_finding("Style")])

    data = client.get("/insights", params={"repo": "o/r"}).json()

    by_author = {e["author"]: e for e in data["findings_per_author"]}
    assert by_author[None] == {"author": None, "findings": 2, "prs": 2}
    assert by_author["alice"] == {"author": "alice", "findings": 1, "prs": 1}


def test_insights_endpoint_unknown_repo_is_empty_not_invented(db, client):
    data = client.get("/insights", params={"repo": "nope/none"}).json()
    assert data == {
        "repo": "nope/none",
        "top_categories": [],
        "top_rule_ids": [],
        "findings_per_author": [],
    }


def test_insights_endpoint_requires_repo_param(client):
    assert client.get("/insights").status_code == 422


# ------------------- summary comment rendering ------------------------------

def test_summary_body_appends_insights_section():
    body = ReviewPoster()._summary_comment_body(
        [_finding("Security")], files_analysed=1,
        insights=["Recurring: 2nd 'Security' finding from @alice "
                  "in this repo's last 10 PRs."],
    )
    assert "💡 **Insights**" in body
    assert ("- Recurring: 2nd 'Security' finding from @alice "
            "in this repo's last 10 PRs." in body)
    assert body.index("Overall risk") < body.index("💡 **Insights**")


def test_summary_body_omits_section_when_no_lines():
    for insights in (None, []):
        body = ReviewPoster()._summary_comment_body(
            [_finding("Security")], files_analysed=1, insights=insights)
        assert "Insights" not in body


def test_summary_body_caps_insights_at_three_lines():
    lines = [f"line {i}" for i in range(1, 6)]
    body = ReviewPoster()._summary_comment_body(
        [_finding("Security")], files_analysed=1, insights=lines)
    assert "line 3" in body
    assert "line 4" not in body
