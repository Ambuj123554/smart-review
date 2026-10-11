import ast
from app.analyzer.base import BaseRule


class EvalExecRule(BaseRule):
    """Rule 6 (SEC-001): eval() and exec() dynamic code execution detection."""
    
    rule_id = "SEC-001"
    category = "Security"
    severity = "CRITICAL"
    message = "Potential dangerous use of eval() or exec() detected"
    explanation = (
        "eval() and exec() dynamically execute Python expressions or statements. "
        "When supplied with untrusted or external input, they create severe Remote Code Execution (RCE) vulnerabilities."
    )
    suggestion = "Avoid eval() and exec(). Use safer alternatives such as ast.literal_eval() for parsing data structures."

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec"):
            self.add_finding(
                line=node.lineno,
                message=f"Use of dangerous built-in '{node.func.id}()' detected.",
                explanation=self.explanation,
                suggestion=self.suggestion
            )
        self.generic_visit(node)
