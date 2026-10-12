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
