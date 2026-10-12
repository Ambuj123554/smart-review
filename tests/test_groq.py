"""Tests for the Groq provider: mocked responses, malformed JSON, 429, config."""
import logging
from unittest.mock import MagicMock

import pytest

from app.ai.diff_reviewer import DiffReviewer
from app.ai.groq import GROQ_CHAT_URL, GroqError, GroqReviewer
from app.ai.providers import get_reviewer
from app.config import settings
from app.github.diff import AddedLine, FilePatch


def _resp(status=200, json_data=None, headers=None, text=""):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_data if json_data is not None else {}
    m.headers = headers or {}
    m.text = text or str(json_data or "")
    return m


def _chat_response(content: str):
    """A well-formed Groq chat-completions body wrapping `content`."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


@pytest.fixture
def groq(monkeypatch):
    monkeypatch.setattr("app.ai.groq.time.sleep", lambda s: None)
    g = GroqReviewer(api_key="gsk_test", model="test-model")
    g.session = MagicMock()
    return g


# ------------------------------- happy path --------------------------------

def test_happy_path_returns_content_and_sends_json_mode(groq):
    groq.session.post.return_value = _resp(
        200, _chat_response('[{"file": "f.py", "line": 1}]')
    )
    out = groq.call("review this")
    assert out == '[{"file": "f.py", "line": 1}]'

    args, kwargs = groq.session.post.call_args
    assert args[0] == GROQ_CHAT_URL
    assert kwargs["headers"]["Authorization"] == "Bearer gsk_test"
    payload = kwargs["json"]
    assert payload["model"] == "test-model"
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["messages"][0]["content"] == "review this"


def test_missing_key_fails_before_any_network_call():
    g = GroqReviewer(api_key="", model="m")
    g.session = MagicMock()
    with pytest.raises(GroqError, match="GROQ_API_KEY"):
        g.call("prompt")
    g.session.post.assert_not_called()


# --------------------------- invalid key / model ----------------------------

def test_invalid_api_key_logs_clear_error(groq, caplog):
    groq.session.post.return_value = _resp(401, {"error": "bad key"}, text="Unauthorized")
    with caplog.at_level(logging.ERROR, logger="app.ai.groq"):
        with pytest.raises(GroqError, match="GROQ_API_KEY"):
            groq.call("prompt")
    assert any("GROQ_API_KEY" in m for m in caplog.messages)


def test_invalid_model_id_logs_clear_error(groq, caplog):
    groq.session.post.return_value = _resp(404, {"error": "model_not_found"},
                                           text="Model does not exist")
    with caplog.at_level(logging.ERROR, logger="app.ai.groq"):
        with pytest.raises(GroqError, match="GROQ_MODEL"):
            groq.call("prompt")
    assert any("test-model" in m for m in caplog.messages)


# ------------------------------- rate limits --------------------------------

def test_429_retries_with_backoff_then_succeeds(groq, monkeypatch):
    sleeps = []
    monkeypatch.setattr("app.ai.groq.time.sleep", sleeps.append)
    groq.session.post.side_effect = [
        _resp(429, {"error": "rate limited"}, text="Too Many Requests"),
        _resp(429, {"error": "rate limited"}, headers={"Retry-After": "5"},
              text="Too Many Requests"),
        _resp(200, _chat_response("[]")),
    ]
    out = groq.call("prompt")
    assert out == "[]"
    assert groq.session.post.call_count == 3
    assert len(sleeps) == 2            # backed off between attempts
    assert sleeps[0] < sleeps[1]       # exponential growth (2.0 -> Retry-After 5)


def test_429_persistent_raises_after_max_attempts(groq):
    groq.session.post.return_value = _resp(
        429, {"error": "rate limited"}, text="Too Many Requests"
    )
    with pytest.raises(GroqError, match="429|attempts"):
        groq.call("prompt")
    assert groq.session.post.call_count == 4  # MAX_ATTEMPTS


# ------------------------- schema/validation unchanged ----------------------

def _fp():
    return FilePatch(path="f.py", added_lines=[AddedLine(line=1, content="x = 1")])


def test_malformed_json_from_groq_is_dropped_by_existing_validation(groq):
    """Malformed JSON content flows through unchanged DiffReviewer validation."""
    groq.session.post.return_value = _resp(
        200, _chat_response("Sure! Here's the review: ``` not json")
    )
    reviewer = DiffReviewer(reviewer=groq)
    result = reviewer.review_file(_fp())
    assert result.dropped_invalid_json is True
    assert result.findings == []


def test_valid_groq_json_goes_through_same_finding_schema(groq):
    content = ('[{"file": "f.py", "line": 1, "severity": "HIGH", '
               '"category": "Security", "message": "unsafe call", '
               '"suggestion": "fix it"}]')
    groq.session.post.return_value = _resp(200, _chat_response(content))
    reviewer = DiffReviewer(reviewer=groq)
    result = reviewer.review_file(_fp())
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.rule_id == "AI-DIFF-001"      # same schema markers as Gemini
    assert f.severity == "HIGH"
    assert f.line == 1
    assert f.source == "AI Analysis"


def test_hallucinated_line_from_groq_also_dropped(groq):
    content = ('[{"file": "f.py", "line": 999, "severity": "LOW", '
               '"category": "Style", "message": "made up"}]')
    groq.session.post.return_value = _resp(200, _chat_response(content))
    reviewer = DiffReviewer(reviewer=groq)
    result = reviewer.review_file(_fp())
    assert result.findings == []
    assert result.dropped_hallucinated_line == 1


# ---------------------------- provider selection ----------------------------

def test_provider_default_is_gemini(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    from app.ai.reviewer import GeminiReviewer
    assert isinstance(get_reviewer(), GeminiReviewer)


def test_provider_groq_selects_groq_reviewer(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    assert isinstance(get_reviewer(), GroqReviewer)


def test_provider_is_case_and_space_insensitive(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", " Groq ")
    assert isinstance(get_reviewer(), GroqReviewer)


def test_unknown_provider_fails_loudly(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "claude")
    with pytest.raises(ValueError, match="LLM_PROVIDER 'claude'"):
        get_reviewer()


def test_diff_reviewer_defaults_to_configured_provider(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    from app.ai.reviewer import GeminiReviewer
    assert isinstance(DiffReviewer().reviewer, GeminiReviewer)
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    assert isinstance(DiffReviewer().reviewer, GroqReviewer)
