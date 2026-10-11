import ast
from typing import List
from app.schemas.review import Finding
from app.analyzer.rules.mutable_default import MutableDefaultRule
from app.analyzer.rules.bare_except import BareExceptRule
from app.analyzer.rules.comparison_rules import NoneBoolEqualityRule, LiteralIdentityRule
from app.analyzer.rules.unreachable_code import UnreachableCodeRule
from app.analyzer.rules.eval_exec import EvalExecRule
from app.analyzer.rules.hardcoded_secrets import HardcodedSecretsRule
from app.analyzer.rules.sql_injection import SQLInjectionRule


class ASTAnalyzer:
    """Orchestrates deterministic AST rule-based analysis on Python source code."""

    def __init__(self):
        # Register all 8 independent rule modules
        self.rule_classes = [
            MutableDefaultRule,      # AST-001
            BareExceptRule,          # AST-002
            NoneBoolEqualityRule,    # AST-003
            UnreachableCodeRule,     # AST-004
            LiteralIdentityRule,     # AST-005
            EvalExecRule,            # SEC-001
            HardcodedSecretsRule,    # SEC-002
            SQLInjectionRule,        # SEC-003
        ]

    def analyze(self, code: str) -> List[Finding]:
        """Parses Python source code and runs all registered AST rules."""
        if not code or not code.strip():
            return []

        code_lines = code.splitlines()

        # Safely parse code into AST
        try:
            tree = ast.parse(code)
        except SyntaxError as err:
            # Handle invalid Python syntax gracefully without crashing
            line_no = err.lineno if err.lineno is not None else 1
            return [
                Finding(
                    rule_id="AST-SYNTAX-ERR",
                    category="Syntax",
                    severity="CRITICAL",
                    line=line_no,
                    message=f"Python SyntaxError: {err.msg}",
                    explanation=f"Invalid Python syntax near line {line_no}. AST analysis could not be completed for remaining nodes.",
                    suggestion="Correct Python syntax error to enable complete AST static analysis.",
                    source="Rule-Based Analysis"
                )
            ]
        except Exception as err:
            return [
                Finding(
                    rule_id="AST-PARSE-ERR",
                    category="Syntax",
                    severity="CRITICAL",
                    line=1,
                    message=f"AST Parse Error: {str(err)}",
                    explanation="Failed to parse Python code structure.",
                    suggestion="Ensure valid Python code input.",
                    source="Rule-Based Analysis"
                )
            ]

        findings: List[Finding] = []

        # Execute each rule independently
        for rule_cls in self.rule_classes:
            rule_instance = rule_cls()
            rule_findings = rule_instance.analyze(tree, code_lines)
            findings.extend(rule_findings)

        # Sort findings chronologically by line number
        findings.sort(key=lambda f: f.line)
        return findings
