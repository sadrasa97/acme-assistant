"""RAGAgent: retrieve with the vector index, gate on similarity, answer strictly from the context."""
from __future__ import annotations

from app.agents.contracts import RAGResult, SourceRef, Status
from app.core.config import Settings
from app.core.errors import AppError, RetrievalError
from app.core.logging import get_logger
from app.llm.base import ChatMessage, LLMProvider
from app.rag.embedding import Embedder
from app.rag.vector_store import VectorStore

log = get_logger("rag")

SYSTEM_PROMPT = """You answer questions about Acme Analytics using ONLY the numbered context excerpts provided.
Rules:
- Use only facts stated in the excerpts. Never add outside knowledge or guess.
- If the excerpts do not contain the answer, reply exactly: INSUFFICIENT_EVIDENCE
- Do not present information that is absent from the excerpts as fact.
- Cite the excerpts you used inline like [1], [2]. Do not invent sources, page numbers or quotes.
- Be concise."""

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


class RAGAgent:
    def __init__(self, settings: Settings, embedder: Embedder, store: VectorStore | None, llm: LLMProvider):
        self._s, self._embedder, self.store, self._llm = settings, embedder, store, llm

    def run(self, question: str) -> RAGResult:
        if self.store is None:
            return RAGResult(status=Status.ERROR, answer="", error="document index is not available")
        try:
            hits = self.store.search(self._embedder.embed_query(question), self._s.rag_top_k)
        except AppError as exc:
            return RAGResult(status=Status.ERROR, answer="", error=exc.public_message)
        except Exception as exc:
            log.exception("retrieval failed")
            return RAGResult(status=Status.ERROR, answer="", error=RetrievalError.public_message)

        top = hits[0][1] if hits else None
        relevant = [(c, s) for c, s in hits if s >= self._s.rag_min_similarity]
        if not relevant:
            return RAGResult(
                status=Status.INSUFFICIENT_EVIDENCE, retrieved=len(hits), top_score=top,
                answer="I could not find sufficient evidence in the company documents to answer this question.",
            )

        context = "\n\n".join(
            f"[{i}] (file: {c.file}, page {c.page}{', section: ' + c.section if c.section else ''})\n{c.text}"
            for i, (c, _) in enumerate(relevant, 1)
        )
        reply = self._llm.complete(
            [ChatMessage(role="system", content=SYSTEM_PROMPT),
             ChatMessage(role="user", content=f"Context:\n{context}\n\nQuestion: {question}")]
        ).strip()

        if INSUFFICIENT in reply:
            return RAGResult(
                status=Status.INSUFFICIENT_EVIDENCE, retrieved=len(hits), top_score=top,
                answer="The retrieved documents do not contain enough information to answer this question.",
            )
        sources = [
            SourceRef(file=c.file, page=c.page, chunk_id=c.chunk_id, section=c.section, score=round(s, 4),
                      snippet=c.text[:240])
            for c, s in relevant
        ]
        return RAGResult(status=Status.OK, answer=reply, sources=sources, retrieved=len(hits), top_score=top)
