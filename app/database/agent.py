"""DatabaseAgent: natural language -> SQL (LLM, schema-grounded) -> validate -> read-only execute."""
from __future__ import annotations

import sqlite3

from app.agents.contracts import DatabaseResult, Status
from app.core.config import Settings
from app.core.errors import AppError, SQLExecutionError, SQLGenerationError, SQLValidationError
from app.core.logging import get_logger
from app.database.connection import ALLOWED_FUNCTIONS, ALLOWED_TABLES, execute_readonly, open_readonly
from app.database.validator import validate_sql
from app.llm.base import ChatMessage, LLMProvider, strip_code_fence

log = get_logger("database")

SYSTEM_PROMPT = """You translate business questions into ONE read-only SQLite SELECT query.
Rules:
- Output only the SQL, no explanation, no markdown.
- Use only the tables and columns in the schema below. Never invent columns.
- Exactly one statement. SELECT (optionally with a WITH clause) only.
- Dates are ISO text (YYYY-MM-DD); use strftime/date for date logic.
- order status values are exactly: completed, pending, cancelled, refunded.
- Always alias aggregate columns with clear names (e.g. AS total_amount).
- Add LIMIT when listing rows.
- Allowed functions: {functions}.
If the question cannot be answered from this schema, output exactly: CANNOT_ANSWER

Schema:
{schema}"""


def describe_schema(db_path) -> str:
    """Render the live schema (tables, columns, types, PK/FK) from the database itself."""
    conn = open_readonly(db_path)
    conn.set_authorizer(None)  # introspection only; this connection is read-only and never exposed
    try:
        lines = []
        for table in sorted(ALLOWED_TABLES):
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            fks = {r[3]: f"{r[2]}({r[4]})" for r in conn.execute(f"PRAGMA foreign_key_list({table})")}
            lines.append(f"TABLE {table}")
            for _, name, ctype, notnull, _, pk in cols:
                extra = (" PRIMARY KEY" if pk else "") + (" NOT NULL" if notnull and not pk else "")
                extra += f" REFERENCES {fks[name]}" if name in fks else ""
                lines.append(f"  - {name} {ctype}{extra}")
            sample = conn.execute(f"SELECT * FROM {table} LIMIT 2").fetchall()
            lines.append(f"  e.g. rows: {sample}")
        return "\n".join(lines)
    finally:
        conn.close()


class DatabaseAgent:
    def __init__(self, settings: Settings, llm: LLMProvider):
        self._s, self._llm = settings, llm
        self._schema: str | None = None

    @property
    def schema(self) -> str:
        if self._schema is None:
            self._schema = describe_schema(self._s.sqlite_path)
        return self._schema

    def _generate(self, question: str, feedback: str | None = None) -> str:
        system = SYSTEM_PROMPT.format(functions=", ".join(sorted(ALLOWED_FUNCTIONS)), schema=self.schema)
        user = f"Question: {question}"
        if feedback:
            user += f"\n\nYour previous query failed: {feedback}\nReturn a corrected single SELECT."
        raw = self._llm.complete([ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)])
        sql = strip_code_fence(raw).strip()  # the validator rejects multi-statement input itself
        if "CANNOT_ANSWER" in sql:
            raise SQLGenerationError("model reported the question cannot be answered from the schema",
                                     public_message="This question cannot be answered from the available database schema.")
        return sql

    def run(self, question: str) -> DatabaseResult:
        feedback: str | None = None
        attempts = 0
        last_sql: str | None = None
        max_attempts = 1 + max(0, self._s.sql_repair_attempts)
        while attempts < max_attempts:
            attempts += 1
            try:
                sql = validate_sql(self._generate(question, feedback))
                last_sql = sql
                res = execute_readonly(
                    self._s.sqlite_path, sql, max_rows=self._s.sql_max_rows, timeout_seconds=self._s.sql_timeout_seconds
                )
                return DatabaseResult(
                    status=Status.OK, sql=sql, columns=res.columns, rows=res.rows, row_count=len(res.rows),
                    truncated=res.truncated, attempts=attempts,
                )
            except SQLGenerationError as exc:  # model says unanswerable: do not retry
                return DatabaseResult(status=Status.OUT_OF_SCOPE, attempts=attempts, error=exc.public_message)
            except (SQLValidationError, SQLExecutionError) as exc:
                log.warning("sql attempt %d failed: %s | sql=%r", attempts, exc.detail, last_sql)
                feedback = exc.detail
                if attempts >= max_attempts or isinstance(exc, SQLExecutionError) and "time limit" in exc.detail:
                    return DatabaseResult(status=Status.ERROR, sql=last_sql, attempts=attempts, error=exc.public_message)
            except AppError as exc:  # LLM problems
                return DatabaseResult(status=Status.ERROR, sql=last_sql, attempts=attempts, error=exc.public_message)
            except sqlite3.Error:
                log.exception("unexpected sqlite error")
                return DatabaseResult(status=Status.ERROR, sql=last_sql, attempts=attempts, error=SQLExecutionError.public_message)
        return DatabaseResult(status=Status.ERROR, sql=last_sql, attempts=attempts, error=SQLExecutionError.public_message)
