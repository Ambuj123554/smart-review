"""Tests for AST-rule analysis on the webhook path and AST+LLM merging."""
from unittest.mock import MagicMock

from app.ai.diff_reviewer import DiffReviewResult
from app.analyzer.engine import ASTAnalyzer
from app.github.diff import AddedLine, FilePatch
from app.github.webhooks import _ast_findings_for_file, _merge_findings
from app.schemas.review import Finding

# Sample source with two AST-detectable issues:
#   line 1: mutable default (AST-001, HIGH)
#   line 2: eval() call     (SEC-001, CRITICAL)
SOURCE = 'def f(a=[]):\n    eval("1")\n    return a\n'


def _client_with_source(source):
    client = MagicMock()
    client.get_file_content.return_value = source
    return client


def _fp(*lines):
    return FilePatch(path="f.py", added_lines=[AddedLine(line=n, content="x") for n in lines])


def _llm_result(*findings):
    r = DiffReviewResult("f.py")
    r.findings = list(findings)
    return r


def _llm_finding(line, message="llm says hi"):
    return Finding(rule_id="AI-DIFF-001", category="Logic", severity="HIGH",
                   line=line, message=message, explanation="", suggestion="")


def test_ast_analysis_fetches_file_and_keeps_added_lines_only():
    client = _client_with_source(SOURCE)
    # Only line 2 was added -> only the eval() finding survives the filter.
    out = _ast_findings_for_file(client, "o/r", "sha", _fp(2), ASTAnalyzer())
    assert len(out) == 1
    assert out[0]["line"] == 2
    assert out[0]["severity"] == "CRITICAL"
    assert out[0]["path"] == "f.py"
    client.get_file_content.assert_called_once_with("o/r", "f.py", "sha")


def test_ast_analysis_filters_findings_on_unchanged_lines():
    client = _client_with_source(SOURCE)
    # Only line 55 was added; AST issues sit on lines 1-2 -> nothing kept.
    out = _ast_findings_for_file(client, "o/r", "sha", _fp(55), ASTAnalyzer())
    assert out == []


def test_ast_analysis_no_content_returns_empty():
    client = MagicMock()
    client.get_file_content.return_value = None  # unfetchable/binary file
    out = _ast_findings_for_file(client, "o/r", "sha", _fp(1, 2), ASTAnalyzer())
    assert out == []


def test_merge_dedupes_same_line_ast_wins():
    ast_findings = [{"path": "f.py", "line": 2, "message": "Use of dangerous built-in 'eval()'",
                     "suggestion": "Use ast.literal_eval.", "explanation": "",
                     "severity": "CRITICAL", "category": "Security"}]
    llm = _llm_result(_llm_finding(2, "eval can execute arbitrary code"))
    merged = _merge_findings("f.py", ast_findings, llm)
    # Same (path, line): only ONE comment, and it is the deterministic AST one.
    assert len(merged) == 1
    assert merged[0]["message"] == "Use of dangerous built-in 'eval()'"


def test_merge_keeps_llm_finding_on_uncovered_line():
    ast_findings = [{"path": "f.py", "line": 1, "message": "mutable default",
                     "suggestion": "use None", "explanation": "",
                     "severity": "HIGH", "category": "Bug Risk"}]
    llm = _llm_result(_llm_finding(7, "off-by-one in loop bound"))
    merged = _merge_findings("f.py", ast_findings, llm)
    assert [d["line"] for d in merged] == [1, 7]
    assert merged[1]["message"] == "off-by-one in loop bound"


def test_merge_without_ast_findings_is_llm_only():
    llm = _llm_result(_llm_finding(3, "unclear name"), _llm_finding(9, "missing test"))
    merged = _merge_findings("f.py", [], llm)
    assert [d["line"] for d in merged] == [3, 9]


def test_merge_dedupes_within_llm_duplicate_lines():
    llm = _llm_result(_llm_finding(4, "first"), _llm_finding(4, "second"))
    merged = _merge_findings("f.py", [], llm)
    assert len(merged) == 1
    assert merged[0]["message"] == "first"


def test_merge_keeps_most_severe_llm_finding_on_same_line_and_category():
    """Regression: the LLM re-reports the same issue on the same line with
    different wording/severity; the merged run must keep the MOST SEVERE
    one, not whichever came first."""
    mild = Finding(rule_id="AI-DIFF-001", category="Bug Risk", severity="MEDIUM",
                   line=4, message="Potential division by zero",
                   explanation="", suggestion="")
    severe = Finding(
        rule_id="AI-DIFF-001", category="Bug Risk", severity="HIGH", line=4,
        message="Potential division by zero and undefined result if exception occurs",
        explanation="", suggestion="",
    )
    for llm in (_llm_result(mild, severe), _llm_result(severe, mild)):
        merged = _merge_findings("f.py", [], llm)
        assert len(merged) == 1
        assert merged[0]["severity"] == "HIGH"
        assert "undefined result" in merged[0]["message"]


def test_merge_keeps_llm_findings_of_different_categories_on_same_line():
    """Different categories on the same line are different issues: both
    survive the merge (the poster dedupes cross-run, not within a run)."""
    a = Finding(rule_id="AI-DIFF-001", category="Bug Risk", severity="MEDIUM",
                line=4, message="division by zero", explanation="", suggestion="")
    b = Finding(rule_id="AI-DIFF-002", category="Style", severity="LOW",
                line=4, message="unclear name", explanation="", suggestion="")
    merged = _merge_findings("f.py", [], _llm_result(a, b))
    assert [(d["line"], d["category"]) for d in merged] == [
        (4, "Bug Risk"), (4, "Style")]


def test_merge_llm_findings_carry_line_text_when_provided():
    merged = _merge_findings("f.py", [], _llm_result(_llm_finding(3)),
                             line_texts={3: "x = 1"})
    assert merged[0]["line_text"] == "x = 1"


def test_ast_findings_carry_line_text():
    """AST findings carry the commented line's content so the poster can
    detect that a line changed since a previous run's comment."""
    client = _client_with_source(SOURCE)
    fp = FilePatch(path="f.py",
                   added_lines=[AddedLine(line=2, content='    eval("1")')])
    out = _ast_findings_for_file(client, "o/r", "sha", fp, ASTAnalyzer())
    assert out[0]["line_text"] == '    eval("1")'


# ------------------- Task 1: extension skip + truncation -------------------

def test_non_code_extension_skips_content_fetch():
    client = _client_with_source("some text")
    fp = FilePatch(path="docs/README.md",
                   added_lines=[AddedLine(line=3, content="new docs line")])
    out = _ast_findings_for_file(client, "o/r", "sha", fp, ASTAnalyzer())
    assert out == []
    client.get_file_content.assert_not_called()


def test_unsupported_code_extension_skips_content_fetch():
    client = _client_with_source("x = 1")
    fp = FilePatch(path="app.js",
                   added_lines=[AddedLine(line=1, content="var x = 1;")])
    out = _ast_findings_for_file(client, "o/r", "sha", fp, ASTAnalyzer())
    assert out == []
    client.get_file_content.assert_not_called()


def test_supported_python_extension_fetches_content():
    client = _client_with_source(SOURCE)
    fp = FilePatch(path="pkg/module.PY",  # case-insensitive extension check
                   added_lines=[AddedLine(line=2, content="x")])
    out = _ast_findings_for_file(client, "o/r", "sha", fp, ASTAnalyzer())
    assert len(out) == 1  # eval() on added line 2
    client.get_file_content.assert_called_once_with("o/r", "pkg/module.PY", "sha")


def test_max_files_per_pr_default_is_20():
    from app.config import settings
    # conftest never sets MAX_FILES_PER_PR, so the env default applies.
    assert settings.MAX_FILES_PER_PR == 20


def _hermetic_pipeline(monkeypatch, files):
    """Mocks every network/LLM/DB touchpoint of _process_review."""
    from unittest.mock import MagicMock
    import app.github.webhooks as wh
    fake_client = MagicMock()
    fake_client.get_pr_files.return_value = files
    fake_client.get_file_content.return_value = None
    monkeypatch.setattr(wh, "GitHubApiClient", lambda: fake_client)
    monkeypatch.setattr(
        wh, "DiffReviewer",
        # **kwargs: the pipeline passes the repo config's custom_instructions.
        lambda **kwargs: MagicMock(
            review_file=lambda fp: wh.DiffReviewResult(fp.path)),
    )
    monkeypatch.setattr(wh, "ReviewPoster", MagicMock)
    monkeypatch.setattr(wh, "DatabaseManager", MagicMock)
    return wh


def _many_files(n):
    return [
        {"filename": f"f{i}.py", "status": "modified",
         "patch": "@@ -1 +1,2 @@\n-a\n+b\n"}
        for i in range(n)
    ]


def test_truncated_file_count_logs_warning_with_pr_number(monkeypatch, caplog):
    import logging
    wh = _hermetic_pipeline(monkeypatch, _many_files(25))  # cap is 20
    with caplog.at_level(logging.WARNING, logger="app.github.webhooks"):
        wh._process_review("o/r", 4242, "sha", None, "alice")
    warnings = [m for m in caplog.messages if "truncated" in m]
    assert warnings, "expected a truncation warning"
    assert any("o/r#4242" in m for m in warnings), caplog.messages


def test_exactly_at_file_cap_is_not_reported_truncated(monkeypatch, caplog):
    """A PR with exactly MAX_FILES_PER_PR changed files is complete, not
    truncated — the warning must not fire."""
    import logging
    from app.config import settings
    wh = _hermetic_pipeline(monkeypatch, _many_files(settings.MAX_FILES_PER_PR))
    with caplog.at_level(logging.WARNING, logger="app.github.webhooks"):
        wh._process_review("o/r", 7, "sha", None, None)
    assert not [m for m in caplog.messages if "truncated" in m], caplog.messages


def test_line_cap_truncation_logs_warning_with_pr_number(monkeypatch, caplog):
    import logging
    from app.config import settings
    # Each file adds 2 lines; a global budget of 1 forces per-file truncation.
    files = [
        {"filename": f"f{i}.py", "status": "modified",
         "patch": "@@ -1 +1,3 @@\n-a\n+b\n+c\n"}
        for i in range(2)
    ]
    wh = _hermetic_pipeline(monkeypatch, files)
    monkeypatch.setattr(settings, "GITHUB_MAX_LINES", 1)  # force line cap
    with caplog.at_level(logging.WARNING, logger="app.github.webhooks"):
        wh._process_review("o/r", 99, "sha", None, None)
    warnings = [m for m in caplog.messages
                if "GITHUB_MAX_LINES" in m and "o/r#99" in m]
    assert warnings, caplog.messages
