"""OPTIONAL integration test with the real intfloat/e5-large-v2 model (needs weights; slow on CPU).
Run:  pytest -m integration   (set EMBEDDING_MODEL_PATH to a local copy to avoid the download)"""
import numpy as np
import pytest

from app.core.config import Settings
from app.core.errors import RetrievalError
from app.rag.agent import RAGAgent
from app.rag.embedding import E5Embedder
from app.rag.indexer import build_index

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def e5(env):
    emb = E5Embedder(env)
    try:
        emb.load()
    except RetrievalError as exc:
        pytest.skip(f"e5-large-v2 not available: {exc.detail[:120]}")
    return emb


def test_e5_contract(e5):
    v = e5.embed_passages(["hello world"])
    assert v.shape == (1, 1024) and np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-4)


def test_real_retrieval_and_threshold_calibration(env, e5, llm_factory):
    s = env.model_copy(update={"index_dir": env.index_dir.parent / "e5-index", "rag_min_similarity": 0.0})
    store = build_index(s, e5)
    agent = RAGAgent(s, e5, store, llm_factory())
    r = agent.run("What is the refund window for annual subscriptions?")
    assert r.sources[0].file == "refund_policy.pdf"
    off = e5.embed_query("Who won the football world cup in 2018?")
    print("in-domain top score:", r.top_score, "| off-topic top score:", store.search(off, 1)[0][1])
