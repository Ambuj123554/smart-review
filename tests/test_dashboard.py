"""Tests for the GET /dashboard page (server-rendered HTML)."""
import pytest

from fastapi.testclient import TestClient

import app.main as main_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh DB per test and a cached-manager reset so /dashboard reads it."""
    monkeypatch.setattr(main_module.settings, "DATABASE_PATH",
                        str(tmp_path / "dashboard.db"))
    previous_manager = main_module.db_manager
    monkeypatch.setattr(main_module, "db_manager", None)
    with TestClient(main_module.app) as c:
        yield c
    monkeypatch.setattr(main_module, "db_manager", previous_manager)


def _finding(severity="HIGH", category="Security", message="issue"):
    return [{"path": "app.py", "line": 1, "severity": severity,
             "category": category, "message": message}]


def test_dashboard_returns_200_and_key_labels_without_data(client):
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    body = resp.text
    for label in (
        "Smart Review Assistant — Dashboard",
        "Total reviews",
        "Total findings",
        "Findings by severity",
        "Findings by category",
        "Findings per repository",
        "Most recent reviews",
        "Highest severity",
        "No reviews yet",          # friendly empty state
        '<div class="stat-value">0</div>',
    ):
        assert label in body, f"missing label: {label}"


def test_dashboard_with_data_shows_stats_and_recent_reviews(client):
    db = main_module._get_db_manager()
    db.save_pr_review("acme/widgets", 12, "sha-a", "alice",
                      _finding(severity="HIGH", category="Security"))
    db.save_pr_review("acme/widgets", 13, "sha-b", "bob",
                      _finding(severity="CRITICAL", category="Bug Risk"))
    db.save_pr_review("other/x", 9, "sha-c", "carol",
                      _finding(severity="LOW", category="Style"))

    resp = client.get("/dashboard")
    assert resp.status_code == 200
    body = resp.text

    # Totals in the stat cards.
    assert '<div class="stat-label">Total reviews</div><div class="stat-value">3</div>' in body
    assert '<div class="stat-label">Total findings</div><div class="stat-value">3</div>' in body

    # Breakdown labels with actual data rendered.
    for label in ("Security", "Bug Risk", "Style", "acme/widgets", "other/x"):
        assert label in body, label

    # Recent-reviews table: repo, PR link, author, count, severity.
    assert 'href="https://github.com/acme/widgets/pull/12"' in body
    assert ">acme/widgets</a>" in body
    assert "<td>#12</td>" in body
    assert "<td>alice</td>" in body
    assert "CRITICAL" in body and "HIGH" in body and "LOW" in body

    # No empty state when there IS data.
    assert "<h2>No reviews yet</h2>" not in body


def test_dashboard_escapes_database_values(client):
    """Repo/author/category values from the DB must never reach the page
    unescaped (HTML injection)."""
    db = main_module._get_db_manager()
    evil_repo = 'evil/<script>alert(1)</script>'
    db.save_pr_review(evil_repo, 1, "sha", '<img src=x onerror=alert(2)>',
                      [{"path": "a.py", "line": 1, "severity": "HIGH",
                        "category": "<b>cat</b>", "message": "m"}])

    body = client.get("/dashboard").text

    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "<img src=x" not in body
    assert "&lt;img src=x" in body
    assert "&lt;b&gt;cat&lt;/b&gt;" in body
