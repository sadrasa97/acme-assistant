"""Reproduce the README example responses with the TEST HARNESS (scripted LLM + hashing test embedder).
Retrieval, chunking, FAISS, SQL validation/execution and the calculator are real; the LLM and the
embedder are test doubles, so this is NOT a measurement of e5-large-v2 or of a live LLM.
Run:  python -m scripts.generate_examples"""
import json
import tempfile
from pathlib import Path

from app.api.state import build_state
from app.core.config import Settings
from app.database.seed import seed_database
from app.llm.mock import ScriptedProvider
from scripts.generate_documents import generate
from tests.helpers import HashingEmbedder, make_handler
from tests.test_coordinator_and_api import PLANS, SQLS

EXTRA_SQL = {
    "Which 3 countries have the highest total completed order amount?":
        "SELECT c.country, ROUND(SUM(o.amount), 2) AS total_amount FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE o.status = 'completed' GROUP BY c.country ORDER BY total_amount DESC LIMIT 3",
    "How many pending orders are there?": "SELECT COUNT(*) AS pending_orders FROM orders WHERE status = 'pending'",
}
EXTRA_PLANS = {
    "What discount do non-profit organizations receive?": {"route": "rag", "reason": "policy", "steps": [
        {"id": "s1", "kind": "rag", "question": "What discount do non-profit organizations receive?"}]},
    "Which 3 countries have the highest total completed order amount?": {"route": "database", "reason": "data", "steps": [
        {"id": "s1", "kind": "database", "question": "Which 3 countries have the highest total completed order amount?"}]},
    "How many pending orders are there?": {"route": "database", "reason": "data", "steps": [
        {"id": "s1", "kind": "database", "question": "How many pending orders are there?"}]},
    "Who won the football world cup in 2018?": {"route": "out_of_scope", "reason": "unrelated", "steps": []},
}
QUESTIONS = [
    "What is the refund window for annual subscriptions?",
    "What discount do non-profit organizations receive?",
    "Which 3 countries have the highest total completed order amount?",
    "How many pending orders are there?",
    "What is the total amount of completed orders, and what percentage of the total of all orders is that?",
    "Who won the football world cup in 2018?",
]
QUESTIONS[4] = next(q for q in PLANS if q.startswith("What is the total amount of completed"))

if __name__ == "__main__":
    from fastapi.testclient import TestClient
    from app.main import create_app
    tmp = Path(tempfile.mkdtemp())
    generate(tmp / "documents"); seed_database(tmp / "acme.db")
    s = Settings(documents_dir=tmp / "documents", index_dir=tmp / "index", sqlite_path=tmp / "acme.db",
                 rag_min_similarity=0.25, llm_api_key="x", _env_file=None)
    llm = ScriptedProvider(handler=make_handler({**SQLS, **EXTRA_SQL}, {**PLANS, **EXTRA_PLANS}))
    with TestClient(create_app(s, build_state(s, llm=llm, embedder=HashingEmbedder()))) as c:
        for q in QUESTIONS:
            body = c.post("/ask", json={"question": q}).json()
            body.pop("request_id", None); body.pop("duration_ms", None)
            for st in body["steps"]: st.pop("duration_ms", None)
            print("### " + q); print(json.dumps(body, indent=2, ensure_ascii=False)); print()
