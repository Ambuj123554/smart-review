import ast
from app.analyzer.base import BaseRule


class BareExceptRule(BaseRule):
    """Rule 2 (AST-002): Bare except clause detection."""
    
    rule_id = "AST-002"
    category = "Error Handling"
    severity = "HIGH"
    message = "Bare except clause detected"
    explanation = (
        "A bare 'except:' clause catches system-exiting exceptions such as SystemExit, "
        "KeyboardInterrupt, and MemoryError, making it difficult to terminate the program "
        "and hiding unintended bugs or fatal errors."
    )
    suggestion = "Specify explicit exception types (e.g. 'except Exception:' or 'except (ValueError, KeyError):')."

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.add_finding(
                line=node.lineno,
                message=self.message,
                explanation=self.explanation,
                suggestion=self.suggestion
            )
        self.generic_visit(node)
