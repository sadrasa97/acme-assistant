"""Read-only, sandboxed SQLite execution: RO connection + authorizer + row/time limits."""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import SQLExecutionError

ALLOWED_TABLES = frozenset({"customers", "orders"})

# Scalar / aggregate functions a model-written query may use.
ALLOWED_FUNCTIONS = frozenset({
    "count", "sum", "avg", "min", "max", "round", "abs", "coalesce", "ifnull", "nullif", "length", "lower", "upper",
    "substr", "strftime", "date", "datetime", "julianday", "cast", "total", "iif", "trim", "replace", "typeof",
    "group_concat",
})


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    truncated: bool


def _authorizer(action, arg1, arg2, dbname, source):
    ok, deny = sqlite3.SQLITE_OK, sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_SELECT:
        return ok
    if action == sqlite3.SQLITE_READ:
        return ok if (arg1 or "").lower() in ALLOWED_TABLES else deny
    if action == sqlite3.SQLITE_FUNCTION:
        return ok if (arg2 or "").lower() in ALLOWED_FUNCTIONS else deny
    if action == getattr(sqlite3, "SQLITE_RECURSIVE", 33):
        return ok
    return deny  # everything else: writes, DDL, PRAGMA, ATTACH, transactions...


def open_readonly(db_path: Path) -> sqlite3.Connection:
    if not Path(db_path).exists():
        raise SQLExecutionError(f"database file not found: {db_path}", public_message="The database is not available.")
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, check_same_thread=False)
    conn.set_authorizer(_authorizer)
    return conn


def execute_readonly(db_path: Path, sql: str, *, max_rows: int, timeout_seconds: float) -> QueryResult:
    """Run an already-validated query on a fresh read-only connection with a wall-clock limit."""
    conn = open_readonly(db_path)
    deadline = time.monotonic() + timeout_seconds
    # Called every ~10k VM instructions; returning non-zero aborts the query.
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        columns = [d[0] for d in (cur.description or [])]
        fetched = cur.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise SQLExecutionError("query exceeded time limit", public_message="The query took too long and was stopped.") from exc
        raise SQLExecutionError(f"sqlite operational error: {exc}") from exc
    except sqlite3.DatabaseError as exc:
        raise SQLExecutionError(f"sqlite error: {exc}") from exc
    finally:
        conn.close()
    truncated = len(fetched) > max_rows
    return QueryResult(columns=columns, rows=[list(r) for r in fetched[:max_rows]], truncated=truncated)
