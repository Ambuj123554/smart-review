import ast
from app.analyzer.base import BaseRule


class SQLInjectionRule(BaseRule):
    """Rule 8 (SEC-003): SQL injection via string formatting or concatenation."""
    
    rule_id = "SEC-003"
    category = "Security"
    severity = "CRITICAL"
    message = "Potential SQL injection vulnerability detected"
    explanation = (
        "Constructing SQL queries using string formatting (f-strings, % formatting, .format()) "
        "or string concatenation (+) allows un-sanitized user input to alter the structure of SQL commands."
    )
    suggestion = "Use parameterized queries (e.g., cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))) instead of string formatting."

    SQL_KEYWORDS = {"select", "insert", "update", "delete", "drop", "alter", "create", "from", "where"}
    QUERY_VAR_NAMES = {"query", "sql", "sql_query", "stmt", "statement", "command", "raw_sql"}

    def _contains_sql(self, text: str) -> bool:
        """Helper to check if string snippet contains common SQL keywords."""
        words = set(text.lower().replace("=", " ").replace("'", " ").replace('"', ' ').replace("+", " ").split())
        return len(words.intersection(self.SQL_KEYWORDS)) > 0

    def _is_vulnerable_query_expression(self, expr: ast.expr) -> bool:
        """Check if an expression represents a dynamic formatted or concatenated query string."""
        if expr is None:
            return False

        # 1. f-string (ast.JoinedStr)
        if isinstance(expr, ast.JoinedStr):
            for value in expr.values:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    if self._contains_sql(value.value):
                        return True

        # 2. String Concatenation (ast.BinOp with +)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            has_sql = False
            for child in ast.walk(expr):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if self._contains_sql(child.value):
                        has_sql = True
                        break
            if has_sql:
                return True

        # 3. .format() call on string literal
        if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "format":
            for child in ast.walk(expr.func.value):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if self._contains_sql(child.value):
                        return True

        # 4. % formatting (ast.BinOp with %)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mod):
            for child in ast.walk(expr.left):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    if self._contains_sql(child.value):
                        return True

        return False

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check for assignments to query variables built with dynamic formatting."""
        for target in node.targets:
            var_name = ""
            if isinstance(target, ast.Name):
                var_name = target.id
            elif isinstance(target, ast.Attribute):
                var_name = target.attr

            if var_name.lower() in self.QUERY_VAR_NAMES or "sql" in var_name.lower() or "query" in var_name.lower():
                if self._is_vulnerable_query_expression(node.value):
                    self.add_finding(
                        line=node.lineno,
                        message=f"SQL injection risk: Dynamic formatted query string assigned to '{var_name}'.",
                        explanation=self.explanation,
                        suggestion=self.suggestion
                    )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check for direct calls to execute() or executemany() with dynamic queries."""
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("execute", "executemany"):
            if node.args:
                query_arg = node.args[0]
                if self._is_vulnerable_query_expression(query_arg):
                    self.add_finding(
                        line=node.lineno,
                        message=f"SQL injection risk in dynamic query passed directly to '{node.func.attr}()'.",
                        explanation=self.explanation,
                        suggestion=self.suggestion
                    )
        self.generic_visit(node)
