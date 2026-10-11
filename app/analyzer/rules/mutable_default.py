import ast
from app.analyzer.base import BaseRule


class MutableDefaultRule(BaseRule):
    """Rule 1 (AST-001): Mutable default argument detection."""
    
    rule_id = "AST-001"
    category = "Bug Risk"
    severity = "HIGH"
    message = "Mutable default argument in function definition"
    explanation = (
        "Default arguments are evaluated once when the function is defined, not each time "
        "it is called. Using a mutable object (like a list, dictionary, or set) as a default "
        "argument shares the same object across all function calls, leading to unexpected state mutations."
    )
    suggestion = "Use None as the default argument value and initialize the mutable object inside the function body."

    def _is_mutable(self, default_node: ast.expr) -> bool:
        """Check if an expression represents a mutable object."""
        if default_node is None:
            return False
        
        # Litera list, dict, set
        if isinstance(default_node, (ast.List, ast.Dict, ast.Set)):
            return True
            
        # Call to list(), dict(), set()
        if isinstance(default_node, ast.Call):
            if isinstance(default_node.func, ast.Name) and default_node.func.id in ("list", "dict", "set"):
                return True
                
        return False

    def _check_function(self, node: ast.AST) -> None:
        """Inspect function defaults for mutable objects."""
        defaults = node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
        for default in defaults:
            if self._is_mutable(default):
                self.add_finding(
                    line=default.lineno,
                    message=f"Function '{node.name}' uses a mutable default argument.",
                    explanation=self.explanation,
                    suggestion=self.suggestion
                )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)
        self.generic_visit(node)
