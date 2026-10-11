import ast
from typing import List
from app.analyzer.base import BaseRule


class UnreachableCodeRule(BaseRule):
    """Rule 4 (AST-004): Unreachable code detection after return, raise, break, or continue."""
    
    rule_id = "AST-004"
    category = "Control Flow"
    severity = "MEDIUM"
    message = "Unreachable code detected"
    explanation = (
        "Code placed after a 'return', 'raise', 'break', or 'continue' statement in the same execution "
        "block will never be executed. This often indicates dead code or a logic flaw."
    )
    suggestion = "Remove the unreachable statements or adjust the control flow logic."

    def _check_statement_list(self, stmts: List[ast.stmt]) -> None:
        """Scan a list of statements for dead code after terminating statements."""
        terminator_found = False
        terminator_type = None

        for stmt in stmts:
            if terminator_found:
                self.add_finding(
                    line=stmt.lineno,
                    message=f"Unreachable statement after '{terminator_type}' statement.",
                    explanation=self.explanation,
                    suggestion=self.suggestion
                )
                # Flag only the first unreachable statement in the sequence to avoid spamming
                break

            if isinstance(stmt, ast.Return):
                terminator_found = True
                terminator_type = "return"
            elif isinstance(stmt, ast.Raise):
                terminator_found = True
                terminator_type = "raise"
            elif isinstance(stmt, ast.Break):
                terminator_found = True
                terminator_type = "break"
            elif isinstance(stmt, ast.Continue):
                terminator_found = True
                terminator_type = "continue"

    def generic_visit(self, node: ast.AST) -> None:
        """Inspect all statement bodies in AST nodes."""
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                # If it's a list of AST nodes, check if they are statements
                if value and all(isinstance(item, ast.stmt) for item in value):
                    self._check_statement_list(value)
        super().generic_visit(node)
