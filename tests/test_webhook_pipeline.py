"""End-to-end tests of the webhook review pipeline (_process_review).

Regression coverage for two bugs found in a real webhook run:

1. The webhook path constructed the Gemini reviewer directly instead of
   app.ai.providers.get_reviewer(), so LLM_PROVIDER=groq was ignored and the
   run failed against generativelanguage.googleapis.com.
2. The pipeline emits findings keyed by "path" while the DB save expected
   "file_path" -> KeyError 'file_path' and nothing was persisted.
"""
import hashlib
import hmac
import json
import logging

import pytest
import requests
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

import app.main as main_module
from app.ai.reviewer import GeminiReviewer
from app.config import settings

SECRET = "test-webhook-secret"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_HOST = "generativelanguage.googleapis.com"

# One changed Python file; added line 2 is new-side line number 2.
PR_FILES = [{
    "filename": "bads.py",
    "status": "modified",
    "patch": "@@ -1,2 +1,3 @@\n def f(x):\n+    return eval(x)\n def g():\n",
}]

LLM_FINDINGS = [{
    "file": "bads.py",
    "line": 2,
    "severity": "HIGH",
    "category": "Security",
    "message": "eval executes arbitrary code",
    "suggestion": "Use ast.literal_eval",
}]


class FakeGitHubClient:
    """Stands in for GitHubApiClient on the webhook path (files + content)."""

    def __init__(self, *args, **kwargs):
        pass

    def get_pr_files(self, repo_full_name, pr_number, max_files=0):
        return list(PR_FILES)

    def get_file_content(self, repo_full_name, path, ref):
        return None  # skip the AST step; keep the test focused on LLM + DB


def _http_response(status, json_data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status
    resp.headers = headers or {}
    data = json_data if json_data is not None else {}
    resp.json.return_value = data
    resp.text = json.dumps(data) if not isinstance(data, str) else data
    return resp


@pytest.fixture
def pipeline_env(monkeypatch, tmp_path):
    """Groq-configured settings, a fresh DB, mocked HTTP, and Gemini guarded."""
    monkeypatch.setattr(settings, "GITHUB_WEBHOOK_SECRET", SECRET)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test_key")
    monkeypatch.setattr(settings, "GROQ_MODEL", "test-model-70b")

    # Fresh DB per test; reset the cached manager so /stats re-reads the path.
    monkeypatch.setattr(settings, "DATABASE_PATH", str(tmp_path / "pipeline.db"))
    previous_manager = main_module.db_manager
    monkeypatch.setattr(main_module, "db_manager", None)

    # The webhook pipeline builds GitHubApiClient directly: replace it with
    # the fake so no live GitHub request is ever made.
    monkeypatch.setattr("app.github.webhooks.GitHubApiClient", FakeGitHubClient)

    # Record every outgoing HTTP request; serve the canned responses the
    # pipeline needs (Groq completion, GitHub comments list, review post).
    http_calls = []

    def fake_request(self, method, url, **kwargs):
        http_calls.append((method, url, kwargs))
        if url.startswith(GROQ_URL):
            return _http_response(200, {
                "choices": [{"message": {"content": json.dumps(LLM_FINDINGS)}}]
            })
        if url.endswith("/pulls/42/comments"):
            return _http_response(200, [])
        if url.endswith("/pulls/42/reviews"):
            return _http_response(201, {"id": 1})
        if url.endswith("/issues/42/comments"):
            return _http_response(201, {"id": 9})
        raise AssertionError(f"Unexpected HTTP request: {method} {url}")

    monkeypatch.setattr(requests.Session, "request", fake_request)

    # Any call to Gemini at all is a provider-selection bug under
    # LLM_PROVIDER=groq: record it and blow up.
    gemini_calls = []

    def gemini_call(self, prompt):
        gemini_calls.append(prompt)
        raise AssertionError("GeminiReviewer.call must not run when LLM_PROVIDER=groq")

    monkeypatch.setattr(GeminiReviewer, "call", gemini_call)

    yield {"http_calls": http_calls, "gemini_calls": gemini_calls}
    monkeypatch.setattr(main_module, "db_manager", previous_manager)


def _sign(body: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def _post_pr_webhook(client: TestClient):
    payload = {
        "action": "opened",
        "repository": {"full_name": "octocat/hello-world"},
        "pull_request": {
            "number": 42,
            "head": {"sha": "abc123def4567890abc123def4567890abc123de"},
            "user": {"login": "octocat"},
        },
    }
    body = json.dumps(payload).encode("utf-8")
    return client.post(
        "/webhook/github",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": _sign(body),
        },
    )


def test_webhook_pipeline_uses_groq_when_configured(pipeline_env):
    """LLM_PROVIDER=groq: the webhook pipeline must hit the Groq URL and
    never touch Gemini (regression: reviewer was hard-wired to Gemini)."""
    with TestClient(main_module.app) as client:
        response = _post_pr_webhook(client)

    assert response.status_code == 200
    assert response.json()["status"] == "accepted"

    urls = [url for _, url, _ in pipeline_env["http_calls"]]
    assert GROQ_URL in urls, "webhook pipeline never called the Groq endpoint"
    assert not any(GEMINI_HOST in url for url in urls), \
        "webhook pipeline called generativelanguage.googleapis.com"
    assert pipeline_env["gemini_calls"] == []

    # The Groq request itself: bearer key + the configured model.
    groq_call = next(kw for m, url, kw in pipeline_env["http_calls"]
                     if url == GROQ_URL)
    assert groq_call["headers"]["Authorization"] == "Bearer gsk_test_key"
    assert groq_call["json"]["model"] == "test-model-70b"


def test_webhook_pipeline_findings_reach_db_and_stats(pipeline_env):
    """Full pipeline: webhook -> LLM review -> DB save -> /stats.

    Regression: findings are keyed by "path" in the pipeline; the DB save
    expected "file_path" and raised KeyError, so nothing was persisted.
    """
    with TestClient(main_module.app) as client:
        response = _post_pr_webhook(client)
        assert response.status_code == 200

        stats = client.get("/stats").json()

    assert stats["total_findings"] == 1
    assert stats["by_repo"] == {"octocat/hello-world": 1}
    assert stats["by_severity"] == {"HIGH": 1}
    assert stats["by_category"] == {"Security": 1}


def test_startup_logs_active_provider_and_model(pipeline_env, caplog):
    """Startup must log provider + model, never keys."""
    with caplog.at_level(logging.INFO, logger="app.main"):
        with TestClient(main_module.app):
            pass

    text = " ".join(
        record.getMessage() for record in caplog.records if record.name == "app.main"
    )
    assert "groq" in text.lower()
    assert "test-model-70b" in text
    assert "gsk_test_key" not in text  # never log keys
