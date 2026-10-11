import ast
from app.analyzer.base import BaseRule


class HardcodedSecretsRule(BaseRule):
    """Rule 7 (SEC-002): Hardcoded secrets and credentials detection."""
    
    rule_id = "SEC-002"
    category = "Security"
    severity = "CRITICAL"
    message = "Potential hardcoded secret or API key detected"
    explanation = (
        "Hardcoding sensitive credentials, passwords, API keys, or private tokens directly into "
        "source code exposes them to version control history leaks and unauthorized access."
    )
    suggestion = "Store credentials in environment variables or secret management services (e.g., os.getenv('API_KEY'))."

    SECRET_KEYWORDS = {
        "password", "pass", "passwd", "secret", "api_key", "apikey",
        "access_key", "access_token", "auth_token", "private_key",
        "jwt_secret", "client_secret"
    }

    def _is_suspicious_var_name(self, var_name: str) -> bool:
        """Check if variable name matches any secret pattern."""
        var_lower = var_name.lower().replace("-", "_")
        return any(keyword in var_lower for keyword in self.SECRET_KEYWORDS)

    def _is_sensitive_string_literal(self, value_node: ast.expr) -> bool:
        """Check if the assigned value is a non-empty string literal."""
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
            val_str = value_node.value.strip()
            # Ignore empty strings or typical environment lookup examples
            if val_str and not val_str.startswith("os.getenv"):
                return True
        return False

    def _check_assignment(self, target_node: ast.AST, value_node: ast.expr, lineno: int) -> None:
        var_name = ""
        if isinstance(target_node, ast.Name):
            var_name = target_node.id
        elif isinstance(target_node, ast.Attribute):
            var_name = target_node.attr

        if var_name and self._is_suspicious_var_name(var_name):
            if self._is_sensitive_string_literal(value_node):
                self.add_finding(
                    line=lineno,
                    message=f"Hardcoded secret/credential assigned to variable '{var_name}'.",
                    explanation=self.explanation,
                    suggestion=self.suggestion
                )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._check_assignment(target, node.value, node.lineno)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._check_assignment(node.target, node.value, node.lineno)
        self.generic_visit(node)
