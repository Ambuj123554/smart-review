"""PR-level rule: a source change with no test change in the same PR.

Deliberately NOT registered in ASTAnalyzer's per-file rule list (engine.py):
unlike the other rules it needs the whole changed-file list, so the webhook
pipeline invokes it ONCE per PR. It is still a deterministic, rule-based
check — no LLM involved — and reports at most ONE finding, anchored on the
first changed function/class definition.
"""
import ast
import os
from typing import List, Optional, Tuple

from app.analyzer.base import BaseRule


def is_test_file(path: str) -> bool:
    """True for test files: test_*.py, *_test.py, or under tests/ or test/.

    Directory matching is per path component ("src/tests/util.py" counts,
    "contest.py" does not) and case-insensitive.
    """
    normalized = (path or "").replace("\\", "/").lower()
    name = os.path.basename(normalized)
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    parts = normalized.split("/")
    return "tests" in parts or "test" in parts


class MissingTestsRule(BaseRule):
    """Flags a PR that changes source code but adds/changes no test file."""

    rule_id = "TEST-001"
    category = "Testing"
    severity = "MEDIUM"
    message = "Tests appear to be missing for this change"
    explanation = (
        "This PR adds or changes functions/classes but does not add or "
        "change any test file, so the new behaviour is not covered by "
        "automated tests."
    )
    suggestion = (
        "Add tests under tests/ covering this change: the happy path, "
        "edge cases (empty/invalid input), and error handling."
    )

    def first_definition(
        self, code: str, valid_lines
    ) -> Optional[Tuple[str, str, int]]:
        """Locates the first function/class changed by the diff.

        `valid_lines` are the PR's added line numbers for this file. Only
        definitions whose `def`/`class` line itself was added or changed
        count (a body-only edit is not "changing the function"). Returns
        (kind, name, line) with kind in {"function", "class"}, or None.
        """
        if not code or not code.strip():
            return None
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return None
        candidates: List[Tuple[int, str, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                if node.lineno in valid_lines:
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    candidates.append((node.lineno, kind, node.name))
        if not candidates:
            return None
        line, kind, name = min(candidates)
        return kind, name, line

    def finding_dict(
        self,
        path: str,
        kind: str,
        name: str,
        line: int,
        line_text: Optional[str] = None,
    ) -> dict:
        """Pipeline-format finding dict for the one PR-level report."""
        return {
            "path": path,
            "line": line,
            "rule_id": self.rule_id,
            "message": (
                f"Tests appear to be missing: this PR changes the {kind} "
                f"`{name}` without adding or changing any test file"
            ),
            "suggestion": (
                f"Add tests covering `{name}` (happy path, edge cases, "
                "error handling) under tests/"
            ),
            "explanation": self.explanation,
            "severity": self.severity,
            "category": self.category,
            "line_text": line_text,
        }
