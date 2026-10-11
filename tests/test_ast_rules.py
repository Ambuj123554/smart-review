import pytest
from app.analyzer.engine import ASTAnalyzer


@pytest.fixture
def analyzer():
    return ASTAnalyzer()


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
