import sqlite3

import pytest

from app.core.errors import SQLExecutionError, SQLValidationError
from app.database.connection import execute_readonly
from app.database.validator import validate_sql


@pytest.mark.parametrize("sql", [
    "SELECT COUNT(*) FROM orders",
    "SELECT c.country, SUM(o.amount) AS t FROM orders o JOIN customers c ON c.id = o.customer_id GROUP BY c.country ORDER BY t DESC LIMIT 5",
    "WITH x AS (SELECT status, SUM(amount) AS s FROM orders GROUP BY status) SELECT * FROM x",
    "SELECT * FROM orders;",
    "SELECT strftime('%Y', order_date) AS y, COUNT(*) FROM orders GROUP BY y",
])
def test_valid_selects_accepted(sql):
    assert validate_sql(sql).upper().startswith(("SELECT", "WITH"))


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "DROP TABLE orders",
    "UPDATE orders SET amount = 0",
    "INSERT INTO customers VALUES (999,'x','y','2024-01-01')",
    "SELECT 1; DROP TABLE orders",
    "SELECT * FROM orders; SELECT * FROM customers",
    "SELECT * FROM sqlite_master",
    "SELECT * FROM users",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x",
    "CREATE TABLE t (a int)",
    "ALTER TABLE orders ADD COLUMN z int",
    "SELECT load_extension('x')",
    "SELECT readfile('/etc/passwd')",
    "SELECT * FROM main.orders",
    "SELEC * FROM orders",
    "",
    "   ",
    "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",
])
def test_dangerous_or_invalid_rejected(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


def test_authorizer_blocks_writes_even_if_validator_bypassed(env):
    """Layer 2: the read-only connection + authorizer stop writes/DDL/PRAGMA/ATTACH on their own."""
    before = sqlite3.connect(env.sqlite_path).execute("SELECT COUNT(*), SUM(amount) FROM orders").fetchone()
    for sql in ["DELETE FROM orders", "DROP TABLE orders", "UPDATE orders SET amount=0",
                "PRAGMA table_info(orders)", "ATTACH DATABASE ':memory:' AS m", "SELECT * FROM sqlite_master"]:
        with pytest.raises(SQLExecutionError):
            execute_readonly(env.sqlite_path, sql, max_rows=10, timeout_seconds=2)
    after = sqlite3.connect(env.sqlite_path).execute("SELECT COUNT(*), SUM(amount) FROM orders").fetchone()
    assert before == after


def test_row_limit_and_truncation(env):
    r = execute_readonly(env.sqlite_path, "SELECT * FROM orders", max_rows=5, timeout_seconds=2)
    assert len(r.rows) == 5 and r.truncated


def test_timeout_aborts_runaway_query(env):
    sql = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c")
    with pytest.raises(SQLExecutionError):
        execute_readonly(env.sqlite_path, validate_sql(sql), max_rows=10, timeout_seconds=0.3)
