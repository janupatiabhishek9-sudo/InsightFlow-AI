"""SQL guardrail based on parsing (sqlglot), not only regex.

Read path: exactly one SELECT/WITH query over allowed tables, no file/network/system functions.
Write path: exactly one UPDATE or DELETE on an allowed table (still needs MODIFY_DATA + approval).
Defense in depth: the DuckDB engine also disables external access after loading data.
"""

from __future__ import annotations

import sqlglot
from pydantic import BaseModel
from sqlglot import exp


def _types(*names: str) -> tuple[type, ...]:
    # sqlglot renames/adds expression classes across versions; resolve defensively.
    return tuple(t for n in names if isinstance(t := getattr(exp, n, None), type))


FORBIDDEN_NODES = _types(
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Command", "TruncateTable",
    "Copy", "Attach", "Detach", "Pragma", "Set", "Use", "Merge", "LoadData", "Transaction", "Commit",
    "Rollback", "Install", "Export",
)
QUERY_NODES = _types("Select", "Union", "Intersect", "Except", "Subquery")
WRITE_NODES = _types("Update", "Delete")

FORBIDDEN_FUNCTION_PREFIXES = (
    "read_", "write_", "glob", "getenv", "current_setting", "sniff_csv", "query", "parquet_",
    "duckdb_", "pragma_", "pg_", "sqlite_", "mysql_", "postgres_", "install", "load", "system",
    "shell", "http", "iceberg_", "delta_", "json_execute", "from_csv", "from_json", "which_secret",
)


class SQLGuardResult(BaseModel):
    allowed: bool
    reason: str = "ok"
    statement_type: str | None = None
    tables: list[str] = []


class SQLGuard:
    def __init__(self, allowed_tables: set[str]):
        self.allowed_tables = {t.lower() for t in allowed_tables}

    # ---- public API -------------------------------------------------------------------------
    def check_read(self, sql: str) -> SQLGuardResult:
        stmt, error = self._parse_single(sql)
        if stmt is None:
            return SQLGuardResult(allowed=False, reason=error)
        if not isinstance(stmt, QUERY_NODES):
            return self._deny(stmt, "only SELECT / WITH queries are allowed on the read path")
        for node in stmt.walk():
            if isinstance(node, FORBIDDEN_NODES):
                return self._deny(stmt, f"forbidden operation {type(node).__name__.upper()} inside query")
        return self._check_references(stmt)

    def check_write(self, sql: str) -> SQLGuardResult:
        stmt, error = self._parse_single(sql)
        if stmt is None:
            return SQLGuardResult(allowed=False, reason=error)
        if not isinstance(stmt, WRITE_NODES):
            return self._deny(stmt, "only UPDATE or DELETE are permitted as data modifications")
        for node in stmt.walk():
            if node is not stmt and isinstance(node, FORBIDDEN_NODES):
                return self._deny(stmt, f"forbidden nested operation {type(node).__name__.upper()}")
        return self._check_references(stmt)

    # ---- helpers ----------------------------------------------------------------------------
    @staticmethod
    def _parse_single(sql: str) -> tuple[exp.Expression | None, str]:
        if not sql or not sql.strip():
            return None, "empty SQL"
        try:
            statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
        except sqlglot.errors.ParseError as e:
            return None, f"SQL could not be parsed: {str(e).splitlines()[0]}"
        if len(statements) != 1:
            return None, f"exactly one statement is allowed, got {len(statements)}"
        return statements[0], ""

    @staticmethod
    def _deny(stmt: exp.Expression, reason: str) -> SQLGuardResult:
        return SQLGuardResult(allowed=False, reason=reason, statement_type=type(stmt).__name__.upper())

    def _check_references(self, stmt: exp.Expression) -> SQLGuardResult:
        for func in stmt.find_all(exp.Func):
            name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
            if name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
                return self._deny(stmt, f"function '{name}' is not allowed (file/network/system access)")

        cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
        tables: list[str] = []
        for table in stmt.find_all(exp.Table):
            if not isinstance(table.this, exp.Identifier):
                return self._deny(stmt, "table functions and file references are not allowed")
            name = table.name.lower()
            if table.args.get("db") or table.args.get("catalog"):
                return self._deny(stmt, f"schema-qualified table '{table.sql()}' is not allowed")
            if any(ch in name for ch in "./\\:"):
                return self._deny(stmt, f"file-like table reference '{table.name}' is not allowed")
            if name not in self.allowed_tables and name not in cte_names:
                return self._deny(stmt, f"table '{table.name}' is not in the allowed tables")
            if name not in cte_names:
                tables.append(name)
        return SQLGuardResult(allowed=True, statement_type=type(stmt).__name__.upper(), tables=sorted(set(tables)))
