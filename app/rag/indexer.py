"""Index build pipeline: PDFs -> clean pages -> chunks -> e5 embeddings -> FAISS on disk."""
from __future__ import annotations

from app.core.config import Settings
from app.core.logging import get_logger
from app.rag.chunker import chunk_pages
from app.rag.embedding import Embedder
from app.rag.loader import list_pdfs, load_documents
from app.rag.preprocess import preprocess_pages
from app.rag.vector_store import VectorStore, corpus_fingerprint

log = get_logger("indexer")


def current_fingerprint(settings: Settings) -> str:
    return corpus_fingerprint(
        list_pdfs(settings.documents_dir), settings.embedding_model_name, settings.chunk_size, settings.chunk_overlap
    )


def is_stale(settings: Settings) -> bool:
    try:
        return VectorStore.stored_fingerprint(settings.index_dir) != current_fingerprint(settings)
    except Exception:
        return True


def build_index(settings: Settings, embedder: Embedder) -> VectorStore:
    pages = preprocess_pages(load_documents(settings.documents_dir))
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
    log.info("embedding %d chunks from %d pages", len(chunks), len(pages))
    vectors = embedder.embed_passages([c.embed_text for c in chunks])
    store = VectorStore.build(chunks, vectors, {
        "model": settings.embedding_model_name,
        "fingerprint": current_fingerprint(settings),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    })
    store.save(settings.index_dir)
    return store


def load_or_build(settings: Settings, embedder: Embedder) -> VectorStore:
    """Reuse the on-disk index unless the PDFs / chunking / model changed."""
    if not is_stale(settings):
        return VectorStore.load(settings.index_dir)
    log.info("index missing or stale; rebuilding")
    return build_index(settings, embedder)
