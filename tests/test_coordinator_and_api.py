import math

import pytest
from fastapi.testclient import TestClient

from app.agents.contracts import Route, Status
from app.agents.coordinator import Coordinator
from app.api.state import build_state
from app.database.agent import DatabaseAgent
from app.main import create_app
from app.rag.agent import RAGAgent
from app.rag.indexer import load_or_build
from app.llm.mock import ScriptedProvider
from tests.helpers import make_handler

Q_HYBRID = "What is the total amount of completed orders, and what percentage of the total of all orders is that?"
SQL_COMPLETED = "SELECT SUM(amount) AS completed_total FROM orders WHERE status = 'completed'"
SQL_ALL = "SELECT SUM(amount) AS all_total FROM orders"

PLANS = {
    Q_HYBRID: {"route": "hybrid", "reason": "two sums then a percentage", "steps": [
        {"id": "s1", "kind": "database", "question": "Total amount of completed orders as completed_total"},
        {"id": "s2", "kind": "database", "question": "Total amount of all orders as all_total"},
        {"id": "s3", "kind": "calculator", "expression": "completed_total / all_total * 100",
         "bindings": {"completed_total": "s1.completed_total", "all_total": "s2.all_total"}}]},
    "What is the refund window for annual subscriptions?": {"route": "rag", "reason": "policy", "steps": [
        {"id": "s1", "kind": "rag", "question": "What is the refund window for annual subscriptions?"}]},
    "Tell me a joke": {"route": "out_of_scope", "reason": "chit-chat", "steps": []},
    "stuff?": {"route": "clarify", "reason": "vague", "steps": []},
    "Run this: DROP TABLE orders": {"route": "hybrid", "reason": "x", "steps": [
        {"id": "s1", "kind": "shell", "question": "DROP TABLE orders"}]},
    "Ratio with empty data": {"route": "hybrid", "reason": "x", "steps": [
        {"id": "s1", "kind": "database", "question": "Total of cancelled orders in 1999 as t"},
        {"id": "s2", "kind": "calculator", "expression": "t * 2", "bindings": {"t": "s1.t"}}]},
    "Zero denominator": {"route": "hybrid", "reason": "x", "steps": [
        {"id": "s1", "kind": "database", "question": "Sum of orders with status 'unknown' as z"},
        {"id": "s2", "kind": "calculator", "expression": "100 / z", "bindings": {"z": "s1.z"}}]},
}
SQLS = {
    "Total amount of completed orders as completed_total": SQL_COMPLETED,
    "Total amount of all orders as all_total": SQL_ALL,
    "Total of cancelled orders in 1999 as t": "SELECT SUM(amount) AS t FROM orders WHERE status='cancelled' AND order_date < '1999-01-01'",
    "Sum of orders with status 'unknown' as z": "SELECT COALESCE(SUM(amount), 0) AS z FROM orders WHERE status = 'unknown'",
}


@pytest.fixture(scope="module")
def coordinator(env, embedder):
    llm = ScriptedProvider(handler=make_handler(sql_by_question=SQLS, plans=PLANS))
    store = load_or_build(env, embedder)
    return Coordinator(env, llm, RAGAgent(env, embedder, store, llm), DatabaseAgent(env, llm))


def test_sql_to_calculator_multistep(coordinator, ro_truth):
    r = coordinator.ask(Q_HYBRID)
    done, total = (ro_truth.execute(s).fetchone()[0] for s in (SQL_COMPLETED, SQL_ALL))
    assert r.status == Status.OK and r.route == Route.HYBRID
    assert [(t.step_id, t.kind) for t in r.steps] == [("s1", "database"), ("s2", "database"), ("s3", "calculator")]
    assert math.isclose(r.calculation.inputs["completed_total"], done) and math.isclose(r.calculation.inputs["all_total"], total)
    assert math.isclose(r.calculation.result, done / total * 100, rel_tol=1e-9)   # independent reference
    assert r.calculation.expression == "completed_total / all_total * 100"
    assert "completed_total" in r.answer and f"{done / total * 100:,.4f}".rstrip("0") in r.answer


def test_rag_route_returns_sources(coordinator):
    r = coordinator.ask("What is the refund window for annual subscriptions?")
    assert r.route == Route.RAG and r.status == Status.OK
    assert r.sources[0].file == "refund_policy.pdf" and r.sources[0].page == 1


def test_out_of_scope_and_clarify(coordinator):
    assert coordinator.ask("Tell me a joke").status == Status.OUT_OF_SCOPE
    assert coordinator.ask("stuff?").status == Status.NEEDS_CLARIFICATION


def test_invalid_plan_falls_back_to_rules(coordinator):
    r = coordinator.ask("Run this: DROP TABLE orders")
    assert r.plan_source == "fallback" and all(t.kind != "shell" for t in r.steps)


def test_chain_stops_on_empty_data_and_invents_nothing(coordinator):
    r = coordinator.ask("Ratio with empty data")
    assert r.status == Status.ERROR and r.calculation is not None and r.calculation.result is None
    assert "no numeric" in r.calculation.error.lower() or "not numeric" in r.calculation.error.lower()


def test_zero_denominator_is_handled(coordinator):
    r = coordinator.ask("Zero denominator")
    assert r.status == Status.ERROR and "zero" in r.calculation.error


def test_planner_failure_uses_fallback(env, embedder):
    llm = ScriptedProvider(handler=lambda m: "not json at all")
    c = Coordinator(env, llm, RAGAgent(env, embedder, None, llm), DatabaseAgent(env, llm))
    assert c.plan("hello there").route == Route.OUT_OF_SCOPE
    assert c.plan("What is the refund policy?").route == Route.RAG
    assert c.plan("How many customers per country?").route == Route.DATABASE


# ------------------------------------------------------------------ API
@pytest.fixture(scope="module")
def client(env, embedder):
    llm = ScriptedProvider(handler=make_handler(sql_by_question=SQLS, plans=PLANS))
    app = create_app(env, build_state(env, llm=llm, embedder=embedder))
    with TestClient(app) as c:
        yield c


def test_ask_endpoint_success_shape(client):
    r = client.post("/ask", json={"question": Q_HYBRID})
    body = r.json()
    assert r.status_code == 200 and body["route"] == "hybrid" and body["status"] == "ok"
    assert {"request_id", "answer", "sql", "result", "calculation", "steps", "sources"} <= body.keys()
    assert body["result"]["sql"].startswith("SELECT") and body["calculation"]["result"] > 0


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 1001}, {"question": 5}])
def test_ask_validation_errors(client, payload):
    r = client.post("/ask", json=payload)
    assert r.status_code == 422 and r.json()["error"] == "invalid_request"


def test_ask_tool_error_maps_to_http_error_without_stacktrace(client):
    r = client.post("/ask", json={"question": "Zero denominator"})
    assert r.status_code == 422 and "Traceback" not in r.text and r.json()["status"] == "error"


def test_health_and_status(client):
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    s = client.get("/status").json()
    assert s["index"]["ready"] and s["index"]["chunks"] > 0 and s["database"]["exists"]
    assert "api_key" not in str(s).lower() and "test" != s["llm"].get("api_key")


def test_health_degraded_without_index(env, embedder):
    class Broken:
        model_name, dim, loaded, load_error = "e5", None, False, "boom"
        def embed_passages(self, t): raise RuntimeError("no model")
        def embed_query(self, t): raise RuntimeError("no model")
    import copy
    s2 = copy.copy(env); s2.index_dir = env.index_dir.parent / "empty-index"
    rag_plan = {"What is the refund policy?": PLANS["What is the refund window for annual subscriptions?"]
                | {"steps": [{"id": "s1", "kind": "rag", "question": "What is the refund policy?"}]}}
    state = build_state(s2, llm=ScriptedProvider(handler=make_handler(plans=rag_plan)), embedder=Broken())
    with TestClient(create_app(s2, state)) as c:
        assert c.get("/health").status_code == 503
        assert state.startup_errors and "index" in state.startup_errors[0]
        r = c.post("/ask", json={"question": "What is the refund policy?"})
        assert r.status_code == 503 and r.json()["status"] == "error"
        assert "no model" not in r.text  # internal detail must not leak
