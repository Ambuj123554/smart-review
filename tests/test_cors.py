"""Tests for CORS configuration: wildcard/credentials rule and live headers."""
import pytest
from fastapi.testclient import TestClient

from app.main import app, _cors_settings


# ------------------------- unit: _cors_settings -----------------------------

def test_wildcard_never_enables_credentials():
    """Spec: don't combine '*' with credentials."""
    cfg = _cors_settings("*")
    assert cfg["allow_origins"] == ["*"]
    assert cfg["allow_credentials"] is False


def test_specific_origins_parsed_and_trimmed():
    cfg = _cors_settings(" https://a.example , https://b.example ")
    assert cfg["allow_origins"] == ["https://a.example", "https://b.example"]
    assert cfg["allow_credentials"] is True  # no wildcard present


def test_empty_value_falls_back_to_localhost():
    cfg = _cors_settings("")
    assert cfg["allow_origins"] == ["http://localhost:8000"]
    assert cfg["allow_credentials"] is True


def test_wildcard_among_others_still_disables_credentials():
    cfg = _cors_settings("*,https://a.example")
    assert cfg["allow_credentials"] is False


# ------------------------- integration: live headers ------------------------

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_allowed_origin_gets_cors_header(client):
    resp = client.get("/health", headers={"Origin": "http://localhost:8000"})
    assert resp.status_code == 200
    # Default ALLOWED_ORIGINS in tests includes localhost:8000 -> echo it.
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:8000"


def test_disallowed_origin_gets_no_cors_header(client):
    resp = client.get("/health", headers={"Origin": "https://evil.example"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers
