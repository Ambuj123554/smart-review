import ast
from abc import ABC, abstractmethod
from typing import List
from app.schemas.review import Finding


class BaseRule(ast.NodeVisitor, ABC):
    """Abstract base class for AST-based static code review rules."""
    
    rule_id: str = "AST-000"
    category: str = "General"
    severity: str = "LOW"
    message: str = "Rule violation"
    explanation: str = "Rule violation detected"
    suggestion: str = "Fix the violation"

    def __init__(self):
        self.findings: List[Finding] = []
        self.code_lines: List[str] = []

    def analyze(self, tree: ast.AST, code_lines: List[str]) -> List[Finding]:
        """Runs the rule inspection on an AST tree."""
        self.findings = []
        self.code_lines = code_lines
        self.visit(tree)
        return self.findings

    def add_finding(
        self,
        line: int,
        message: str = None,
        explanation: str = None,
        suggestion: str = None,
        rule_id: str = None,
        category: str = None,
        severity: str = None
    ) -> None:
        """Helper method to construct and store a Finding object."""
        finding = Finding(
            rule_id=rule_id or self.rule_id,
            category=category or self.category,
            severity=severity or self.severity,
            line=max(1, line),
            message=message or self.message,
            explanation=explanation or self.explanation,
            suggestion=suggestion or self.suggestion,
            source="Rule-Based Analysis"
        )
        self.findings.append(finding)
