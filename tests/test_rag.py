import pytest

from app.agents.contracts import Status
from app.rag.agent import RAGAgent
from app.rag.chunker import chunk_pages
from app.rag.indexer import build_index, is_stale, load_or_build
from app.rag.loader import PageText, load_documents
from app.rag.preprocess import clean_text, preprocess_pages


@pytest.fixture(scope="module")
def store(env, embedder):
    return build_index(env, embedder)


@pytest.fixture
def agent(env, embedder, store, llm_factory):
    return RAGAgent(env, embedder, store, llm_factory())


def test_pdf_text_is_extractable(env):
    pages = load_documents(env.documents_dir)
    assert {p.file for p in pages} == {"refund_policy.pdf", "discount_policy.pdf", "customer_support_handbook.pdf"}
    assert all(p.text.strip() for p in pages)


def test_preprocess_cleaning():
    assert clean_text("subscrip-\ntion  plans\nare   fine\n\n3\n\nNext para") == "subscription plans are fine\n\nNext para"
    pages = [PageText("a.pdf", i, f"Acme Confidential\nBody text number {i} that is long enough to keep.\n{i}") for i in (1, 2, 3)]
    pages.append(PageText("a.pdf", 4, "  \n 4 "))
    out = preprocess_pages(pages)
    assert [p.page for p in out] == [1, 2, 3]
    assert all("Confidential" not in p.text for p in out)


def test_chunking_keeps_page_metadata_and_respects_size(env):
    pages = preprocess_pages(load_documents(env.documents_dir))
    chunks = chunk_pages(pages, 300, 60)
    assert len(chunks) > len(pages)
    assert all(len(c.text) <= 300 and c.page >= 1 and c.file.endswith(".pdf") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.text.rstrip()[-1] in ".!?)" for c in chunks)  # no mid-sentence cuts
    assert any(c.section for c in chunks)


def test_refund_policy_retrieval(agent):
    r = agent.run("What is the refund window for annual subscriptions?")
    assert r.status == Status.OK
    hit = next(s for s in r.sources if s.file == "refund_policy.pdf" and s.page == 1 and "30 calendar days" in s.snippet)
    assert hit.chunk_id and hit.score >= 0.25
    assert "[1]" in r.answer


def test_discount_conditions_retrieval(agent):
    r = agent.run("What discount do non-profit organizations receive?")
    assert r.status == Status.OK
    assert any(s.file == "discount_policy.pdf" and s.page == 2 and "20 percent" in s.snippet for s in r.sources)


def test_support_response_time_retrieval(agent):
    r = agent.run("How fast must Priority 1 tickets be answered?")
    assert r.status == Status.OK
    assert any(s.file == "customer_support_handbook.pdf" and s.page == 2 and "within 1 hour" in s.snippet for s in r.sources)


def test_out_of_scope_question_reports_insufficient_evidence(env, embedder, store, llm_factory):
    llm = llm_factory()
    r = RAGAgent(env, embedder, store, llm).run("Who won the football world cup in 2018?")
    assert r.status == Status.INSUFFICIENT_EVIDENCE and r.sources == []
    assert llm.calls == []  # no LLM call, so nothing can be fabricated


def test_model_can_decline_even_when_chunks_pass_threshold(env, embedder, store):
    from app.llm.mock import ScriptedProvider
    r = RAGAgent(env, embedder, store, ScriptedProvider(replies=["INSUFFICIENT_EVIDENCE"])).run("refund window annual")
    assert r.status == Status.INSUFFICIENT_EVIDENCE and r.sources == []


def test_index_reuse_and_staleness(env, embedder, store):
    assert not is_stale(env)
    calls = {"n": 0}
    orig = embedder.embed_passages
    embedder.embed_passages = lambda t: (calls.__setitem__("n", calls["n"] + 1), orig(t))[1]
    try:
        load_or_build(env, embedder)
        assert calls["n"] == 0  # loaded from disk, not re-embedded
        (env.documents_dir / "refund_policy.pdf").write_bytes(
            (env.documents_dir / "refund_policy.pdf").read_bytes() + b"\n% changed")
        assert is_stale(env)
        load_or_build(env, embedder)
        assert calls["n"] == 1 and not is_stale(env)
    finally:
        embedder.embed_passages = orig
