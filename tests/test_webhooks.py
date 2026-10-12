import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

from app.config import settings

SECRET = "test-webhook-secret"


@pytest.fixture(scope="module", autouse=True)
def set_webhook_secret():
    """Ensures settings.GITHUB_WEBHOOK_SECRET is deterministic for this module.

    conftest.py sets it too; this guards against env drift if tests run alone.
    """
    old = settings.GITHUB_WEBHOOK_SECRET
    settings.GITHUB_WEBHOOK_SECRET = SECRET
    yield
    settings.GITHUB_WEBHOOK_SECRET = old


@pytest.fixture(scope="module")
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _pr_payload(action: str = "opened") -> dict:
    return {
        "action": action,
        "repository": {"full_name": "octocat/hello-world"},
        "pull_request": {
            "number": 42,
            "head": {"sha": "abc123def4567890abc123def4567890abc123de"},
        },
        "installation": {"id": 98765},
    }


def _post_webhook(client: TestClient, event: str, payload: dict, secret: str = SECRET):
    body = json.dumps(payload).encode("utf-8")
    return client.post(
        "/webhook/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": event,
            "X-Hub-Signature-256": _sign(body, secret),
        },
    )


def test_valid_signature_and_trigger(client):
    response = _post_webhook(client, "pull_request", _pr_payload("opened"))
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert data["repo"] == "octocat/hello-world"
    assert data["pr"] == "42"
    assert data["head_sha"] == "abc123def4567890abc123def4567890abc123de"
    assert data["installation_id"] == "98765"


def test_invalid_signature_rejected(client):
    payload = _pr_payload("opened")
    body = json.dumps(payload).encode("utf-8")
    response = client.post(
        "/webhook/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": "sha256=" + "0" * 64,
        },
    )
    assert response.status_code == 401


def test_missing_signature_rejected(client):
    body = json.dumps(_pr_payload("opened")).encode("utf-8")
    response = client.post(
        "/webhook/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
        },
    )
    assert response.status_code == 401


def test_ignored_event_returns_200(client):
    # e.g. issues, push, star — anything that is not pull_request
    response = _post_webhook(client, "issues", {"action": "opened"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ignored"
    assert "issues" in data["reason"]


def test_ignored_action_returns_200(client):
    for action in ("closed", "assigned", "labeled", "edited", "unassigned"):
        response = _post_webhook(client, "pull_request", _pr_payload(action))
        assert response.status_code == 200, f"action={action}"
        data = response.json()
        assert data["status"] == "ignored", f"action={action}"
        assert action in data["reason"], f"action={action}"


def test_valid_signature_no_installation_object(client):
    """Payload without an `installation` key must still return 200.

    Regression test: the response annotation was dict[str, str] but
    installation_id could be None -> ResponseValidationError -> 500.
    """
    payload = {
        "action": "opened",
        "repository": {"full_name": "octocat/hello-world"},
        "pull_request": {
            "number": 7,
            "head": {"sha": "abc123"},
            "user": {"login": "alice"},
        },
    }
    response = _post_webhook(client, "pull_request", payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "accepted"
    assert data["installation_id"] is None


def test_pr_author_extracted_from_payload(client):
    """pull_request.user.login must surface as `author` in the response
    (and is what gets persisted to the DB's author column)."""
    payload = _pr_payload("opened")
    payload["pull_request"]["user"] = {"login": "octocat"}
    response = _post_webhook(client, "pull_request", payload)
    assert response.status_code == 200
    assert response.json()["author"] == "octocat"


def test_missing_pr_author_is_null(client):
    payload = _pr_payload("opened")  # no user key at all
    response = _post_webhook(client, "pull_request", payload)
    assert response.status_code == 200
    assert response.json()["author"] is None


def test_valid_signature_wrong_secret_rejected(client):
    response = _post_webhook(client, "pull_request", _pr_payload("opened"), secret="wrong-secret")
    assert response.status_code == 401
