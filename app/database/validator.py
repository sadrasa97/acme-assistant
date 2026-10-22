"""Static SQL safety layer. The model's output is parsed (sqlglot, SQLite dialect) and must be a
single read-only SELECT over allow-listed tables. This is defence layer 1; the read-only connection
and authorizer in connection.py are layer 2."""
from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from app.core.errors import SQLValidationError
from app.database.connection import ALLOWED_FUNCTIONS, ALLOWED_TABLES

MAX_SQL_LENGTH = 4000

_FORBIDDEN_NODES = tuple(
    getattr(exp, name)
    for name in (
        "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Command", "Pragma", "Attach",
        "Detach", "Transaction", "Commit", "Rollback", "Merge", "Use", "Set", "Copy", "TruncateTable",
        "LoadData", "Analyze",
    )
    if hasattr(exp, name)
)


def validate_sql(sql: str) -> str:
    """Return the normalised, validated SQL string or raise SQLValidationError."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLValidationError("empty SQL")
    if len(sql) > MAX_SQL_LENGTH:
        raise SQLValidationError("SQL too long")
    if "\x00" in sql:
        raise SQLValidationError("NUL byte in SQL")

    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except (ParseError, TokenError) as exc:
        raise SQLValidationError(f"SQL does not parse: {str(exc)[:200]}") from exc

    if len(statements) != 1:
        raise SQLValidationError(f"exactly one statement allowed, got {len(statements)}")
    tree = statements[0]

    # A `WITH ... SELECT` parses as exp.Select carrying a "with" arg; set operations are exp.Union.
    if not isinstance(tree, (exp.Select, exp.Union)):
        raise SQLValidationError("only SELECT statements are allowed")

    for node in tree.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise SQLValidationError(f"forbidden operation: {type(node).__name__}")
        if isinstance(node, exp.Into):
            raise SQLValidationError("SELECT ... INTO is not allowed")

    cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = (table.name or "").lower()
        if table.args.get("db") or table.args.get("catalog"):
            raise SQLValidationError("schema-qualified tables are not allowed")
        if name in cte_names:
            continue
        if name not in ALLOWED_TABLES:
            raise SQLValidationError(f"table not allowed: {name or '<unnamed>'}")

    for fn in tree.find_all(exp.Anonymous):
        if fn.name.lower() not in ALLOWED_FUNCTIONS:
            raise SQLValidationError(f"function not allowed: {fn.name}")

    return tree.sql(dialect="sqlite")
