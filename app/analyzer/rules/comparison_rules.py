import ast
from app.analyzer.base import BaseRule


class NoneBoolEqualityRule(BaseRule):
    """Rule 3 (AST-003): Comparing None/True/False using == or != instead of is/is not."""
    
    rule_id = "AST-003"
    category = "Code Style"
    severity = "LOW"
    message = "Comparison with None, True, or False using == or != instead of 'is'/'is not'"
    explanation = (
        "Comparing singletons like None, True, or False using equality operators (== or !=) "
        "can trigger custom __eq__ implementations on objects and is less idiomatic than identity comparison."
    )
    suggestion = "Use 'is' or 'is not' for identity comparison with None, True, and False (e.g., 'if x is None:')."

    def _is_singleton(self, expr: ast.expr) -> bool:
        if isinstance(expr, ast.Constant):
            return expr.value in (None, True, False)
        return False

    def visit_Compare(self, node: ast.Compare) -> None:
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Eq, ast.NotEq)):
                if self._is_singleton(left) or self._is_singleton(right):
                    self.add_finding(
                        line=node.lineno,
                        message=self.message,
                        explanation=self.explanation,
                        suggestion=self.suggestion
                    )
            left = right
        self.generic_visit(node)


class LiteralIdentityRule(BaseRule):
    """Rule 5 (AST-005): Using is/is not for value comparison with non-singleton literals."""
    
    rule_id = "AST-005"
    category = "Code Style"
    severity = "MEDIUM"
    message = "Using 'is' or 'is not' for value comparison with a literal"
    explanation = (
        "The 'is' and 'is not' operators check object identity (memory address), not value equality. "
        "Comparing numbers, strings, or other literal values with 'is' may produce unpredictable results "
        "due to CPython memory caching optimizations."
    )
    suggestion = "Use '==' or '!=' for comparing literal values (e.g., 'if x == 1000:')."

    def _is_non_singleton_literal(self, expr: ast.expr) -> bool:
        if isinstance(expr, ast.Constant):
            return expr.value not in (None, True, False)
        if isinstance(expr, (ast.List, ast.Dict, ast.Set, ast.Tuple)):
            return True
        return False

    def visit_Compare(self, node: ast.Compare) -> None:
        left = node.left
        for op, right in zip(node.ops, node.comparators):
            if isinstance(op, (ast.Is, ast.IsNot)):
                if self._is_non_singleton_literal(left) or self._is_non_singleton_literal(right):
                    self.add_finding(
                        line=node.lineno,
                        message=self.message,
                        explanation=self.explanation,
                        suggestion=self.suggestion
                    )
            left = right
        self.generic_visit(node)
