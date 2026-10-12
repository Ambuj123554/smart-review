"""Tests for /review's AI step: provider selection, ai_status, failure logging.

All Groq HTTP traffic is mocked at requests.Session.post, so these tests
prove the endpoint talks to the webhook path's provider (LLM_PROVIDER=groq
-> GroqReviewer -> api.groq.com) without any network access.
"""
import json
import logging

import pytest
import requests
from fastapi.testclient import TestClient

from app.ai.groq import GROQ_CHAT_URL
from app.config import settings
from app.main import app


@pytest.fixture(scope="module")
def client():
    # conftest.py already pointed settings.DATABASE_PATH at a session temp dir
    # before app.main was imported, so the app's DatabaseManager uses that file.
    with TestClient(app) as c:
        yield c


def _resp(status=200, json_data=None, text=""):
    """A mocked requests response (duck-typed like requests.Response)."""
    r = type("_R", (), {})()
    r.status_code = status
    r.json = lambda: json_data if json_data is not None else {}
    r.headers = {}
    r.text = text or str(json_data or "")
    return r


def _chat(content: str):
    """A well-formed Groq chat-completions body wrapping `content`."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _use_groq(monkeypatch):
    """Point settings at Groq with a usable-looking key."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "gsk_test")


def _mock_post(monkeypatch, response):
    """Replace requests.Session.post; returns the list of captured calls."""
    calls = []

    def fake_post(self, url, *args, **kwargs):
        calls.append({
            "url": url,
            "headers": kwargs.get("headers", {}),
            "payload": kwargs.get("json"),
        })
        return response

    monkeypatch.setattr(requests.Session, "post", fake_post)
    return calls


AI_FINDINGS_JSON = json.dumps([{
    "rule_id": "AI-LOGIC-001",
    "category": "Logic",
    "severity": "HIGH",
    "line": 1,
    "message": "Condition never holds",
    "explanation": "x is compared against None with '=='.",
    "suggestion": "Use 'is None'.",
}])

# Code with a deterministic rule finding so summary stays meaningful even
# when the AI step fails (eval is flagged CRITICAL by the AST rules).
CODE = "def f():\n    return eval('1+1')\n"


def test_review_uses_groq_provider_and_reports_ok(client, monkeypatch):
    _use_groq(monkeypatch)
    calls = _mock_post(monkeypatch, _resp(200, _chat(AI_FINDINGS_JSON)))

    response = client.post("/review", json={"code": CODE, "identifier": "groq.py"})

    assert response.status_code == 200
    data = response.json()
    assert data["ai_status"] == "ok"
    assert len(data["ai_findings"]) == 1
    finding = data["ai_findings"][0]
    assert finding["severity"] == "HIGH"
    assert finding["category"] == "Logic"
    assert finding["source"] == "AI Analysis"

    # The LLM call went to Groq (not Gemini) with the configured key.
    assert len(calls) == 1
    assert calls[0]["url"] == GROQ_CHAT_URL
    assert calls[0]["headers"]["Authorization"] == "Bearer gsk_test"
    assert calls[0]["payload"]["response_format"] == {"type": "json_object"}
    # The prompt carries the code under review.
    assert "eval" in calls[0]["payload"]["messages"][0]["content"]


def test_review_unwraps_json_object_mode_output(client, monkeypatch):
    """Groq JSON mode may wrap the array in an object; /review still works."""
    _use_groq(monkeypatch)
    wrapped = json.dumps({"findings": json.loads(AI_FINDINGS_JSON)})
    _mock_post(monkeypatch, _resp(200, _chat(wrapped)))

    data = client.post("/review", json={"code": CODE}).json()

    assert data["ai_status"] == "ok"
    assert len(data["ai_findings"]) == 1


def test_review_llm_failure_sets_error_status_and_warns(client, monkeypatch, caplog):
    _use_groq(monkeypatch)
    # 401 fails fast (no retries) and yields a GROQ_API_KEY-focused message.
    _mock_post(monkeypatch, _resp(401, {"error": "invalid api key"},
                                  text="Unauthorized"))

    with caplog.at_level(logging.WARNING, logger="app.main"):
        response = client.post("/review", json={"code": CODE})

    assert response.status_code == 200          # rule analysis still returned
    data = response.json()
    assert data["ai_status"].startswith("error:")
    assert "GROQ_API_KEY" in data["ai_status"]
    assert data["ai_findings"] == []
    assert data["summary"]["critical"] >= 1      # eval() still reported
    assert any("AI review failed" in m for m in caplog.messages)


def test_review_invalid_llm_output_sets_error_status_and_warns(
    client, monkeypatch, caplog
):
    _use_groq(monkeypatch)
    _mock_post(monkeypatch, _resp(200, _chat("Sure! Here's my review: [not json")))

    with caplog.at_level(logging.WARNING, logger="app.main"):
        data = client.post("/review", json={"code": CODE}).json()

    assert data["ai_status"].startswith("error:")
    assert "invalid output" in data["ai_status"]
    assert data["ai_findings"] == []
    assert any("invalid output" in m for m in caplog.messages)


def test_review_unconfigured_key_never_calls_network(client, monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "")

    def fail_post(*args, **kwargs):
        raise AssertionError("network call attempted with no API key")

    monkeypatch.setattr(requests.Session, "post", fail_post)

    data = client.post("/review", json={"code": CODE}).json()

    assert data["ai_status"] == (
        "error: GROQ_API_KEY is not configured; set it or use LLM_PROVIDER=gemini."
    )
    assert data["ai_findings"] == []
