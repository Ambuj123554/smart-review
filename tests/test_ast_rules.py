import pytest
from app.analyzer.engine import ASTAnalyzer
from app.analyzer.rules.missing_tests import MissingTestsRule, is_test_file


@pytest.fixture
def analyzer():
    return ASTAnalyzer()


# ------------------- missing-tests rule (PR-level, TEST-001) ---------------

CODE = (
    "class Validator:\n"        # line 1
    "    def check(self, x):\n"  # line 2
    "        return x is not None\n"  # line 3
    "\n"                          # line 4
    "def parse(text):\n"          # line 5
    "    return text.strip()\n"   # line 6
)


def test_missing_tests_rule_picks_first_changed_definition():
    rule = MissingTestsRule()
    # Changed lines 5-6 only -> the function, not the class above.
    assert rule.first_definition(CODE, {5, 6}) == ("function", "parse", 5)
    # Changed lines 1-3 -> the class block; the class itself comes first.
    assert rule.first_definition(CODE, {1, 2, 3}) == ("class", "Validator", 1)
    # Only the method's def line changed -> the method.
    assert rule.first_definition(CODE, {2}) == ("function", "check", 2)
    # A body-only edit (line 3) is not "changing the function".
    assert rule.first_definition(CODE, {3}) is None


def test_missing_tests_rule_ignores_no_definition_and_bad_syntax():
    rule = MissingTestsRule()
    assert rule.first_definition("x = 1\ny = 2\n", {1, 2}) is None
    assert rule.first_definition("def broken(:", {1}) is None  # SyntaxError
    assert rule.first_definition("", {1}) is None


def test_missing_tests_finding_shape():
    rule = MissingTestsRule()
    d = rule.finding_dict("app.py", "function", "parse", 5,
                          "def parse(text):")
    assert d["category"] == "Testing"
    assert d["severity"] in ("LOW", "MEDIUM")
    assert (d["path"], d["line"]) == ("app.py", 5)
    assert d["line_text"] == "def parse(text):"
    assert "parse" in d["message"] and "test" in d["message"].lower()
    assert "test" in d["suggestion"].lower()


def test_is_test_file_patterns():
    for p in ("test_app.py", "tests/test_app.py", "app_test.py",
              "pkg/app_test.py", "test/utils.py", "tests/helpers/util.py",
              "src/tests/integration.py"):
        assert is_test_file(p), p
    for p in ("app.py", "contest.py", "latest.py", "src/testing_helper.py",
              "testcase.py", "docs/contributing.md"):
        assert not is_test_file(p), p


def test_missing_tests_rule_not_in_per_file_engine(analyzer):
    """PR-level rule: the per-file engine must not fire it on every file."""
    from app.analyzer.rules.missing_tests import MissingTestsRule as M
    assert M not in analyzer.rule_classes


def test_rule_1_mutable_default(analyzer):
    code = """
def add_item(item, items=[]):
    items.append(item)
"""
    findings = analyzer.analyze(code)
    rule_1_findings = [f for f in findings if f.rule_id == "AST-001"]
    assert len(rule_1_findings) == 1
    assert rule_1_findings[0].severity == "HIGH"
    assert rule_1_findings[0].line == 2


def test_rule_2_bare_except(analyzer):
    code = """
try:
    risky()
except:
    pass
"""
    findings = analyzer.analyze(code)
    rule_2_findings = [f for f in findings if f.rule_id == "AST-002"]
    assert len(rule_2_findings) == 1
    assert rule_2_findings[0].severity == "HIGH"
    assert rule_2_findings[0].line == 4


def test_rule_3_none_bool_comparison(analyzer):
    code = """
if x == None:
    pass
"""
    findings = analyzer.analyze(code)
    rule_3_findings = [f for f in findings if f.rule_id == "AST-003"]
    assert len(rule_3_findings) == 1
    assert rule_3_findings[0].severity == "LOW"


def test_rule_4_unreachable_code(analyzer):
    code = """
def f():
    return 10
    print("unreachable")
"""
    findings = analyzer.analyze(code)
    rule_4_findings = [f for f in findings if f.rule_id == "AST-004"]
    assert len(rule_4_findings) == 1
    assert rule_4_findings[0].severity == "MEDIUM"
    assert rule_4_findings[0].line == 4


def test_rule_5_literal_identity(analyzer):
    code = """
if x is 1000:
    pass
"""
    findings = analyzer.analyze(code)
    rule_5_findings = [f for f in findings if f.rule_id == "AST-005"]
    assert len(rule_5_findings) == 1
    assert rule_5_findings[0].severity == "MEDIUM"


def test_rule_6_eval_exec(analyzer):
    code = """
eval(user_input)
exec(script)
"""
    findings = analyzer.analyze(code)
    sec_1_findings = [f for f in findings if f.rule_id == "SEC-001"]
    assert len(sec_1_findings) == 2
    assert all(f.severity == "CRITICAL" for f in sec_1_findings)


def test_rule_7_hardcoded_secrets(analyzer):
    code = """
api_key = "secret_12345"
password = "my_super_password"
"""
    findings = analyzer.analyze(code)
    sec_2_findings = [f for f in findings if f.rule_id == "SEC-002"]
    assert len(sec_2_findings) == 2
    assert all(f.severity == "CRITICAL" for f in sec_2_findings)


def test_rule_8_sql_injection(analyzer):
    code = """
cursor.execute(f"SELECT * FROM users WHERE id={user_id}")
cursor.execute("DELETE FROM users WHERE email = '" + email + "'")
"""
    findings = analyzer.analyze(code)
    sec_3_findings = [f for f in findings if f.rule_id == "SEC-003"]
    assert len(sec_3_findings) == 2
    assert all(f.severity == "CRITICAL" for f in sec_3_findings)


def test_clean_code(analyzer):
    code = """
def add(a: int, b: int) -> int:
    return a + b
"""
    findings = analyzer.analyze(code)
    assert len(findings) == 0


def test_syntax_error_handling(analyzer):
    code = """
def broken_func(:
    pass
"""
    findings = analyzer.analyze(code)
    assert len(findings) == 1
    assert findings[0].rule_id == "AST-SYNTAX-ERR"
    assert findings[0].severity == "CRITICAL"
