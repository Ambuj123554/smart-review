import pytest

from fastapi.testclient import TestClient

from app.database.db import DatabaseManager
from app.main import app


@pytest.fixture
def db():
    import tempfile
    db = DatabaseManager(db_path=tempfile.mktemp(suffix=".db"))
    try:
        yield db
    finally:
        import os
        if os.path.exists(db.db_path):
            os.unlink(db.db_path)


def _findings():
    return [
        {"file_path": "a.py", "line": 1, "severity": "CRITICAL",
         "category": "Security", "message": "eval used", "suggestion": "btw"},
        {"file_path": "a.py", "line": 5, "severity": "LOW",
         "category": "Style", "message": "naming", "explanation": "why"},
        {"file_path": "b.py", "line": 2, "severity": "HIGH",
         "category": "Bug Risk", "message": "off by one"},
    ]


def test_save_and_stats_roundtrip(db):
    db.save_pr_review("o/r", 1, "sha1", "alice", _findings())     # 3 findings
    db.save_pr_review("o/r", 2, "sha2", "bob", [_findings()[0]])   # 1 more (CRITICAL)
    db.save_pr_review("other/x", 9, "sha3", "carol", _findings())  # 3 more

    stats = db.get_stats()
    assert stats["total_findings"] == 7
    assert stats["by_severity"] == {"CRITICAL": 3, "HIGH": 2, "LOW": 2}
    assert stats["by_category"] == {"Security": 3, "Style": 2, "Bug Risk": 2}
    assert stats["by_repo"] == {"o/r": 4, "other/x": 3}


def test_same_sha_saved_once_by_unique_constraint(db):
    db.save_pr_review("o/r", 1, "shaX", "alice", _findings())
    rid = db.save_pr_review("o/r", 1, "shaX", "alice", _findings())  # duplicate push
    stats = db.get_stats()
    assert stats["total_findings"] == 3  # INSERT OR IGNORE: no double rows


def test_stats_empty_db(db):
    stats = db.get_stats()
    assert stats["total_findings"] == 0
    assert stats["by_severity"] == {}
    assert stats["by_category"] == {}
    assert stats["by_repo"] == {}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_get_stats_endpoint_200(client):
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"total_findings", "by_severity", "by_category", "by_repo"}
    assert isinstance(data["total_findings"], int)
