import pytest

from app.agents.contracts import Status
from app.database.agent import DatabaseAgent, describe_schema
from app.llm.mock import ScriptedProvider


def test_schema_description_is_grounded_in_real_db(env):
    s = describe_schema(env.sqlite_path)
    assert "TABLE customers" in s and "TABLE orders" in s and "REFERENCES customers(id)" in s


def test_aggregation_with_filter(env, ro_truth, llm_factory):
    q = "What is the total amount of completed orders?"
    agent = DatabaseAgent(env, llm_factory(sql_by_question={
        q: "SELECT ROUND(SUM(amount), 2) AS total_completed FROM orders WHERE status = 'completed'"}))
    r = agent.run(q)
    truth = ro_truth.execute("SELECT ROUND(SUM(amount),2) FROM orders WHERE status='completed'").fetchone()[0]
    assert r.status == Status.OK and r.columns == ["total_completed"] and r.rows[0][0] == truth
    assert r.sql.upper().startswith("SELECT") and r.row_count == 1


def test_join_with_group_by(env, ro_truth, llm_factory):
    q = "How many orders does each country have?"
    sql = ("SELECT c.country, COUNT(o.id) AS order_count FROM orders o JOIN customers c ON c.id = o.customer_id "
           "GROUP BY c.country ORDER BY order_count DESC")
    r = DatabaseAgent(env, llm_factory(sql_by_question={q: sql})).run(q)
    truth = ro_truth.execute(sql).fetchall()
    assert r.status == Status.OK and [tuple(x) for x in r.rows] == truth and r.row_count == len(truth)
    assert sum(row[1] for row in r.rows) == ro_truth.execute("SELECT COUNT(*) FROM orders").fetchone()[0]


def test_dangerous_generated_sql_is_refused_and_db_untouched(env, ro_truth):
    before = ro_truth.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    r = DatabaseAgent(env, ScriptedProvider(replies=["DELETE FROM orders", "DROP TABLE orders"])).run("delete all orders")
    assert r.status == Status.ERROR and r.attempts == 2 and "rejected" in r.error
    assert ro_truth.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == before


def test_repair_attempt_recovers_from_bad_sql(env):
    r = DatabaseAgent(env, ScriptedProvider(replies=[
        "SELECT COUNT(*) FROM orderz", "SELECT COUNT(*) AS n FROM orders"])).run("how many orders")
    assert r.status == Status.OK and r.attempts == 2 and r.rows[0][0] > 0


def test_model_declares_question_unanswerable(env):
    r = DatabaseAgent(env, ScriptedProvider(replies=["CANNOT_ANSWER"])).run("what is the weather")
    assert r.status == Status.OUT_OF_SCOPE and r.sql is None


def test_no_internal_details_leak_in_error(env):
    r = DatabaseAgent(env, ScriptedProvider(replies=["SELECT nope FROM orders"] * 2)).run("x")
    assert r.status == Status.ERROR and "sqlite" not in (r.error or "").lower() and "nope" not in (r.error or "")
