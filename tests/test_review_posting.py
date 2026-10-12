import pytest

from unittest.mock import MagicMock

from app.github.comments import ReviewPoster, BOT_MARKER, GitHubPostingError


def _resp(status=200, json_data=None, headers=None):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_data if json_data is not None else {}
    m.headers = headers or {}
    m.text = str(json_data or "")
    return m


def _finding(path="a.py", line=1, message="problem", severity="HIGH",
             category="Security", suggestion=None, explanation=""):
    return {"path": path, "line": line, "message": message,
            "severity": severity, "category": category,
            "suggestion": suggestion, "explanation": explanation}


@pytest.fixture
def client():
    c = MagicMock()
    c.base_url = "https://api.github.com"
    c._headers.return_value = {"Authorization": "Bearer x"}
    return c


@pytest.fixture
def poster(client, monkeypatch):
    # Skip sleeps so tests are fast
    monkeypatch.setattr("app.github.comments.time.sleep", lambda s: None)
    return ReviewPoster(client=client)


def _existing(comment_body, path="a.py", line=1):
    return {"path": path, "line": line, "body": comment_body}


def test_empty_findings_posts_nothing(poster, client):
    assert poster.post_review("o/r", 1, "sha", []) is None
    client.session.request.assert_not_called()


def test_single_batched_review_with_inline_fields(poster, client):
    client.session.request.return_value = _resp(201, {"id": 1})
    result = poster.post_review("o/r", 1, "headsha", [_finding()])

    assert result == {"id": 1}
    sent = client.session.request.call_args.kwargs["json"]
    assert sent["commit_id"] == "headsha"
    assert sent["event"] == "COMMENT"
    assert len(sent["comments"]) == 1
    c = sent["comments"][0]
    assert c["path"] == "a.py"
    assert c["line"] == 1
    assert c["side"] == "RIGHT"
    assert BOT_MARKER in c["body"]
    assert "[HIGH]" in c["body"]


def test_suggestion_becomes_github_suggestion_block(poster, client):
    client.session.request.return_value = _resp(201, {"id": 1})
    poster.post_review("o/r", 1, "sha", [_finding(suggestion="x is None")])
    sent = client.session.request.call_args.kwargs["json"]
    assert "```suggestion\nx is None\n```" in sent["comments"][0]["body"]


def test_dedupes_against_existing_bot_comment(poster, client):
    existing = _existing(BOT_MARKER + "\n**[HIGH] Security:** problem")
    client.session.request.return_value = _resp(200, [existing])
    result = poster.post_review("o/r", 1, "sha", [_finding()])

    assert result is None  # everything duplicated: no review posted
    # Only the GET existing-comments call happened, no POST
    assert client.session.request.call_count == 1
    assert client.session.request.call_args.args[0] == "GET"


def test_does_not_dedupe_against_human_comments(poster, client):
    human = _existing("A human wrote this", )
    client.session.request.side_effect = [
        _resp(200, [human]),    # existing comments
        _resp(201, {"id": 2}),  # review POST
    ]
    result = poster.post_review("o/r", 1, "sha", [_finding()])
    assert result == {"id": 2}


def test_mixed_new_and_duplicate_posts_only_new(poster, client):
    existing = _existing(BOT_MARKER + "\n**[HIGH] Security:** problem", path="a.py", line=1)
    client.session.request.side_effect = [
        _resp(200, [existing]),
        _resp(201, {"id": 3}),
    ]
    findings = [
        _finding(line=1),                                # duplicate
        _finding(line=5, message="another issue"),       # new
    ]
    poster.post_review("o/r", 1, "sha", findings)
    sent = client.session.request.call_args.kwargs["json"]
    assert len(sent["comments"]) == 1
    assert sent["comments"][0]["line"] == 5


def test_retries_on_5xx_then_succeeds(poster, client, monkeypatch):
    client.session.request.side_effect = [
        _resp(200, []),          # existing comments (no dupes)
        _resp(502, {"error": "bad gateway"}),
        _resp(201, {"id": 4}),   # retry succeeds
    ]
    result = poster.post_review("o/r", 1, "sha", [_finding()])
    assert result == {"id": 4}
    assert client.session.request.call_count == 3


def test_rate_limit_uses_retry_after_header(client, monkeypatch):
    monkeypatch.setattr("app.github.comments.time.sleep", lambda s: None)
    poster = ReviewPoster(client=client)
    client.session.request.side_effect = [
        _resp(200, []),
        _resp(403, {"message": "rate limit"}, headers={"Retry-After": "5"}),
        _resp(201, {"id": 5}),
    ]
    result = poster.post_review("o/r", 1, "sha", [_finding()])
    assert result == {"id": 5}


def test_422_unprocessable_is_permanent_no_retry(poster, client):
    client.session.request.side_effect = [
        _resp(200, []),
        _resp(422, {"message": "Validation Failed"}),
    ]
    with pytest.raises(GitHubPostingError, match="422"):
        poster.post_review("o/r", 1, "sha", [_finding()])
    # 1 GET + 1 POST only; no retry on 4xx
    assert client.session.request.call_count == 2


def test_gives_up_after_max_attempts_on_persistent_5xx(poster, client):
    client.session.request.side_effect = (
        [_resp(200, [])] + [_resp(503, {})] * 10
    )
    with pytest.raises(GitHubPostingError, match="Giving up"):
        poster.post_review("o/r", 1, "sha", [_finding()])


def test_dedupe_keys_on_message_not_explanation_or_suggestion(poster, client):
    """Spec: dedupe on file + line + MESSAGE. Suggestion/explanation may vary
    between model runs and must not cause a re-post."""
    existing = _existing(BOT_MARKER + "\n**[HIGH] Security:** problem")
    client.session.request.side_effect = [
        _resp(200, [existing]),
        _resp(201, {"id": 9}),
    ]
    findings = [
        _finding(line=1, suggestion="old suggestion", explanation="old why"),
        _finding(line=2, message="totally different issue",                 # same file+line style but new message
                 suggestion="new suggestion", explanation="new why"),
    ]
    # First finding: same path+line+message as the existing comment -> dupe,
    # even though suggestion/explanation differ from what was stored.
    # Second: new message -> posted.
    poster.post_review("o/r", 1, "sha", findings)
    sent = client.session.request.call_args.kwargs["json"]
    assert len(sent["comments"]) == 1
    assert sent["comments"][0]["line"] == 2


def test_summary_body_contains_severity_counts(poster, client):
    client.session.request.return_value = _resp(201, {"id": 6})
    findings = [
        _finding(severity="CRITICAL", line=1),
        _finding(severity="CRITICAL", line=2),
        _finding(severity="LOW", line=3),
    ]
    poster.post_review("o/r", 1, "sha", findings)
    sent = client.session.request.call_args.kwargs["json"]
    assert "2" in sent["body"] and "CRITICAL" in sent["body"].upper()
