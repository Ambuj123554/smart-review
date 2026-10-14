import pytest

from unittest.mock import MagicMock

from app.github.comments import (
    ReviewPoster,
    BOT_MARKER,
    LINE_HASH_RE,
    GitHubPostingError,
)


def _resp(status=200, json_data=None, headers=None):
    m = MagicMock()
    m.status_code = status
    m.json.return_value = json_data if json_data is not None else {}
    m.headers = headers or {}
    m.text = str(json_data or "")
    return m


def _finding(path="a.py", line=1, message="problem", severity="HIGH",
             category="Security", suggestion=None, explanation="",
             line_text=None):
    return {"path": path, "line": line, "message": message,
            "severity": severity, "category": category,
            "suggestion": suggestion, "explanation": explanation,
            "line_text": line_text}


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


def test_dedupe_ignores_message_suggestion_and_explanation(poster, client):
    """Dedupe on file + line + category/severity tag, never on the message:
    suggestion/explanation/message may all vary between model runs and must
    not cause a re-post."""
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


def test_summary_comment_posted_after_inline_review(poster, client):
    """After the inline review, ONE PR-level summary comment with files,
    severity/category counts and overall risk."""
    client.session.request.side_effect = [
        _resp(200, []),           # existing comments
        _resp(201, {"id": 1}),    # inline review
        _resp(201, {"id": 9}),    # summary comment
    ]
    findings = [
        _finding(severity="CRITICAL", category="Security", line=1),
        _finding(severity="LOW", category="Style", line=2),
    ]
    result = poster.post_review("o/r", 7, "sha", findings, files_analysed=5)
    assert result == {"id": 1}  # returns the inline review result

    assert client.session.request.call_count == 3
    method, url = client.session.request.call_args.args[:2]
    assert method == "POST"
    assert url.endswith("/repos/o/r/issues/7/comments")  # PR-level comment
    body = client.session.request.call_args.kwargs["json"]["body"]
    assert "Files analysed: 5" in body
    assert "Findings: 2 total" in body
    assert "CRITICAL: 1" in body and "LOW: 1" in body       # by severity
    assert "Security: 1" in body and "Style: 1" in body      # by category
    assert "Overall risk: **CRITICAL**" in body               # highest wins


def test_overall_risk_medium_when_no_higher_severities(poster, client):
    client.session.request.side_effect = [
        _resp(200, []),
        _resp(201, {"id": 1}),
        _resp(201, {"id": 9}),
    ]
    poster.post_review("o/r", 7, "sha", [_finding(severity="MEDIUM")],
                       files_analysed=1)
    body = client.session.request.call_args.kwargs["json"]["body"]
    assert "Overall risk: **MEDIUM**" in body


def test_no_summary_when_no_findings(poster, client):
    """Spec: don't post a summary when there are no findings."""
    assert poster.post_review("o/r", 7, "sha", [], files_analysed=5) is None
    client.session.request.assert_not_called()  # no inline, no summary


def test_no_summary_when_all_comments_are_duplicates(poster, client):
    existing = _existing(BOT_MARKER + "\n**[HIGH] Security:** problem")
    client.session.request.return_value = _resp(200, [existing])
    result = poster.post_review("o/r", 7, "sha", [_finding()], files_analysed=3)
    assert result is None
    # Only the GET for existing comments; no review and no summary POST.
    assert client.session.request.call_count == 1
    assert client.session.request.call_args.args[0] == "GET"


def test_summary_failure_does_not_fail_posted_review(poster, client):
    """Inline review already succeeded; a broken summary endpoint must only
    log a warning, not raise away the successful result."""
    client.session.request.side_effect = [
        _resp(200, []),
        _resp(201, {"id": 1}),
        _resp(500, {"error": "boom"}),
    ]
    result = poster.post_review("o/r", 7, "sha", [_finding()], files_analysed=2)
    assert result == {"id": 1}
    assert client.session.request.call_count == 3


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


# --------------------- 403 handling: permission vs rate limit --------------

def _plain_403():
    """A permission 403 with plenty of quota left (X-RateLimit-Remaining ~5000).

    Real example from a webhook run: "Resource not accessible by personal
    access token" with remaining=5000 was retried 4x with backoff as if it
    were a rate limit.
    """
    return _resp(
        403,
        {"message": "Resource not accessible by personal access token"},
        headers={"X-RateLimit-Remaining": "5000", "X-RateLimit-Limit": "5000"},
    )


def test_plain_403_fails_immediately_no_retry(poster, client, monkeypatch):
    """Permission 403 (quota left, no Retry-After) must fail on the spot
    with a message naming the needed GITHUB_TOKEN permission."""
    sleeps = []
    monkeypatch.setattr("app.github.comments.time.sleep", sleeps.append)
    client.session.request.side_effect = [
        _resp(200, []),   # existing comments GET
        _plain_403(),     # review POST -> permission denied
    ]
    with pytest.raises(GitHubPostingError, match="Pull requests: Read and write"):
        poster.post_review("o/r", 1, "sha", [_finding()])
    # Exactly one GET + one POST; no retry, no backoff sleeps.
    assert client.session.request.call_count == 2
    assert sleeps == []


def test_plain_403_message_mentions_permission_not_rate_limit(poster, client):
    client.session.request.side_effect = [_resp(200, []), _plain_403()]
    with pytest.raises(GitHubPostingError) as excinfo:
        poster.post_review("o/r", 1, "sha", [_finding()])
    msg = str(excinfo.value)
    assert "permission error, not a rate limit" in msg
    assert "Resource not accessible" in msg
    assert "GITHUB_TOKEN" in msg


def test_403_with_zero_rate_limit_remaining_is_retried(client, monkeypatch):
    """403 + X-RateLimit-Remaining: 0 IS a rate limit: retry, then succeed."""
    monkeypatch.setattr("app.github.comments.time.sleep", lambda s: None)
    poster = ReviewPoster(client=client)
    client.session.request.side_effect = [
        _resp(200, []),
        _resp(403, {"message": "API rate limit exceeded"},
              headers={"X-RateLimit-Remaining": "0"}),
        _resp(201, {"id": 7}),
    ]
    assert poster.post_review("o/r", 1, "sha", [_finding()]) == {"id": 7}
    assert client.session.request.call_count == 3


def test_403_with_secondary_rate_limit_message_is_retried(client, monkeypatch):
    """403 carrying GitHub's secondary-rate-limit message: retry, then succeed."""
    monkeypatch.setattr("app.github.comments.time.sleep", lambda s: None)
    poster = ReviewPoster(client=client)
    client.session.request.side_effect = [
        _resp(200, []),
        _resp(403, {"message": "You have exceeded a secondary rate limit. "
                               "Please wait a few minutes before you try again."}),
        _resp(201, {"id": 8}),
    ]
    assert poster.post_review("o/r", 1, "sha", [_finding()]) == {"id": 8}
    assert client.session.request.call_count == 3


# -------- regression: duplicate inline comments on re-runs of a PR ---------
#
# Found in a real PR run: a redelivered webhook / new push re-reviewed the
# same lines and the bot posted the same issue again with reworded LLM text
# (e.g. "Potential division by zero" and "Potential division by zero and
# undefined result if exception occurs" both on line 4).

def _posted_inline_comments(client):
    """Builds GET /pulls/{n}/comments entries from the last review POST."""
    payload = client.session.request.call_args.kwargs["json"]
    return [{"path": c["path"], "line": c["line"], "body": c["body"]}
            for c in payload["comments"]]


def _division_finding(message="Potential division by zero", severity="HIGH",
                      line_text="result = total / count"):
    return _finding(line=4, message=message, severity=severity,
                    category="Bug Risk", line_text=line_text)


def test_rerun_on_same_commit_posts_no_duplicate(poster, client):
    """Webhook redelivery / re-review of the SAME commit: nothing new to
    say, so no second review may be posted."""
    finding = _division_finding()

    # First run: PR has no bot comments yet -> the finding is posted.
    client.session.request.side_effect = [_resp(200, []), _resp(201, {"id": 1})]
    assert poster.post_review("o/r", 42, "sha1", [finding]) == {"id": 1}
    posted = _posted_inline_comments(client)
    assert len(posted) == 1

    # Second run, same commit, same finding -> deduped, review not re-posted.
    client.session.request.side_effect = [_resp(200, posted), _resp(201, {})]
    assert poster.post_review("o/r", 42, "sha1", [finding]) is None
    # 2 calls from run 1 + GET-only from run 2: no second review POST.
    assert client.session.request.call_count == 3
    assert client.session.request.call_args.args[0] == "GET"


def test_rerun_with_different_ai_wording_is_not_duplicated(poster, client):
    """Regression for the real bug: the same issue re-reported by the LLM
    with different wording (and even a different severity) on the same
    file+line+category is the SAME issue -> no new comment."""
    first = _division_finding(message="Potential division by zero",
                              severity="MEDIUM")
    client.session.request.side_effect = [_resp(200, []), _resp(201, {"id": 1})]
    poster.post_review("o/r", 42, "sha1", [first])
    posted = _posted_inline_comments(client)

    reworded = _division_finding(
        message="Potential division by zero and undefined result "
                "if exception occurs",
        severity="HIGH",
    )
    client.session.request.side_effect = [_resp(200, posted), _resp(201, {})]
    assert poster.post_review("o/r", 42, "sha1", [reworded]) is None
    assert client.session.request.call_args.args[0] == "GET"  # no POST


def test_rerun_different_category_and_severity_on_same_line_posts(poster, client):
    """A genuinely different issue on an already-commented line (neither
    the category nor the severity tag matches) is NOT a duplicate."""
    existing = _existing(
        BOT_MARKER + "\n**[HIGH] Bug Risk:** Potential division by zero")
    client.session.request.side_effect = [
        _resp(200, [existing]),
        _resp(201, {"id": 2}),
    ]
    poster.post_review("o/r", 42, "sha",
                       [_finding(line=1, severity="LOW", category="Style",
                                 message="rename this variable")])
    sent = client.session.request.call_args.kwargs["json"]
    assert len(sent["comments"]) == 1


def test_new_push_that_changes_the_line_is_reviewed_again(poster, client):
    """A new commit that EDITS the commented line must get a fresh comment
    even though file/line/category/severity are unchanged."""
    first = _division_finding(line_text="result = total / count")
    client.session.request.side_effect = [_resp(200, []), _resp(201, {"id": 1})]
    poster.post_review("o/r", 42, "sha1", [first])
    posted = _posted_inline_comments(client)

    edited = _division_finding(line_text="result = total / divisor")
    client.session.request.side_effect = [_resp(200, posted), _resp(201, {"id": 2})]
    result = poster.post_review("o/r", 42, "sha2", [edited])
    assert result == {"id": 2}
    sent = client.session.request.call_args.kwargs["json"]
    assert sent["commit_id"] == "sha2"
    assert len(sent["comments"]) == 1


def test_existing_bot_comment_without_line_hash_still_dedupes(poster, client):
    """Comments posted before the line-hash marker existed must still be
    recognized as duplicates (missing hash == content treated as unchanged)."""
    existing = _existing(BOT_MARKER + "\n**[HIGH] Bug Risk:** old wording",
                         line=4)
    client.session.request.side_effect = [
        _resp(200, [existing]),
        _resp(201, {"id": 9}),
    ]
    assert poster.post_review(
        "o/r", 42, "sha", [_division_finding(message="new wording")]
    ) is None


def test_same_line_same_category_findings_merge_keeping_most_severe(poster, client):
    """Within a single run, two AI findings on the same line about the same
    category collapse into ONE comment keeping the most severe wording."""
    client.session.request.side_effect = [_resp(200, []), _resp(201, {"id": 1})]
    findings = [
        _division_finding(message="Potential division by zero",
                          severity="MEDIUM"),
        _division_finding(
            message="Potential division by zero and undefined result "
                    "if exception occurs",
            severity="HIGH",
        ),
    ]
    poster.post_review("o/r", 42, "sha", findings)
    sent = client.session.request.call_args.kwargs["json"]
    assert len(sent["comments"]) == 1
    assert "undefined result" in sent["comments"][0]["body"]


def test_comment_body_embeds_line_hash_only_when_line_text_known(poster, client):
    client.session.request.return_value = _resp(201, {"id": 1})
    poster.post_review("o/r", 1, "sha", [_division_finding()])
    body = client.session.request.call_args.kwargs["json"]["comments"][0]["body"]
    assert LINE_HASH_RE.search(body) is not None

    client.session.request.return_value = _resp(201, {"id": 2})
    poster.post_review("o/r", 1, "sha", [_finding()])  # no line_text
    body = client.session.request.call_args.kwargs["json"]["comments"][0]["body"]
    assert LINE_HASH_RE.search(body) is None
