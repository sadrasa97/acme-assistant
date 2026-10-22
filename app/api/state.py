"""Application container: builds long-lived components once and reports their status."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.agents.coordinator import Coordinator
from app.core.config import Settings
from app.core.logging import get_logger
from app.database.agent import DatabaseAgent
from app.llm import build_provider
from app.llm.base import LLMProvider
from app.rag.agent import RAGAgent
from app.rag.embedding import E5Embedder, Embedder
from app.rag.indexer import is_stale, load_or_build
from app.rag.vector_store import VectorStore

log = get_logger("state")


@dataclass
class AppState:
    settings: Settings
    llm: LLMProvider
    embedder: Embedder
    store: VectorStore | None
    coordinator: Coordinator
    startup_errors: list[str] = field(default_factory=list)

    def status(self) -> dict:
        s = self.settings
        return {
            "llm": {"provider": self.llm.name, "model": self.llm.model_name, "configured": self.llm.is_configured(),
                    "base_url": s.llm_base_url},
            "embedding": {"model": s.embedding_model_name, "local_path": bool(s.embedding_model_path),
                          "loaded": self.embedder.loaded, "load_error": getattr(self.embedder, "load_error", None)},
            "index": {"ready": self.store is not None, "chunks": len(self.store.chunks) if self.store else 0,
                      "stale": is_stale(s) if self.store else None, "dir": str(s.index_dir)},
            "database": {"path": str(s.sqlite_path), "exists": Path(s.sqlite_path).exists()},
            "documents": {"dir": str(s.documents_dir),
                          "pdf_count": len(list(Path(s.documents_dir).glob("*.pdf"))) if Path(s.documents_dir).is_dir() else 0},
            "limits": {"rag_top_k": s.rag_top_k, "rag_min_similarity": s.rag_min_similarity,
                       "sql_max_rows": s.sql_max_rows, "max_plan_steps": s.max_plan_steps},
            "startup_errors": self.startup_errors,
        }

    def ready(self) -> bool:
        return self.store is not None and Path(self.settings.sqlite_path).exists() and self.llm.is_configured()


def build_state(settings: Settings, *, llm: LLMProvider | None = None, embedder: Embedder | None = None) -> AppState:
    llm = llm or build_provider(settings)
    embedder = embedder or E5Embedder(settings)
    errors: list[str] = []
    store: VectorStore | None = None
    try:
        store = load_or_build(settings, embedder)
    except Exception as exc:  # service still starts; /health reports degraded, /ask reports the RAG error
        msg = getattr(exc, "detail", str(exc))
        errors.append(f"index: {msg}")
        log.error("index unavailable at startup: %s", msg)
    rag = RAGAgent(settings, embedder, store, llm)
    db = DatabaseAgent(settings, llm)
    return AppState(settings, llm, embedder, store, Coordinator(settings, llm, rag, db), errors)
