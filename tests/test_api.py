import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ast_engine"] == "ready"


def test_review_endpoint_valid_code():
    payload = {
        "code": "def add(item, items=[]):\n    items.append(item)\n    eval('1+1')",
        "identifier": "test_code.py"
    }
    response = client.post("/review", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "review_id" in data
    assert data["identifier"] == "test_code.py"
    assert data["summary"]["total_issues"] >= 2
    assert data["summary"]["critical"] >= 1 # eval()
    assert data["summary"]["high"] >= 1     # mutable default


def test_review_endpoint_empty_code():
    response = client.post("/review", json={"code": "   "})
    assert response.status_code == 400


def test_get_reviews_history():
    # Perform a review first
    client.post("/review", json={"code": "eval('1+1')", "identifier": "hist_test.py"})
    
    response = client.get("/reviews")
    assert response.status_code == 200
    data = response.json()
    assert "reviews" in data
    assert data["total_reviews"] > 0
