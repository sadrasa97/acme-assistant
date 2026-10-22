"""Embedding service for intfloat/e5-large-v2 (lazy, process-wide singleton).

e5 contract: documents are prefixed with "passage: ", queries with "query: ", and vectors are
L2-normalised so inner product == cosine similarity."""
from __future__ import annotations

import threading
from typing import Protocol

import numpy as np

from app.core.config import Settings
from app.core.errors import RetrievalError
from app.core.logging import get_logger

log = get_logger("embedding")

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "


class Embedder(Protocol):
    model_name: str
    dim: int | None

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    @property
    def loaded(self) -> bool: ...


class E5Embedder:
    def __init__(self, settings: Settings):
        self.model_name = settings.embedding_model_name
        self._source = settings.embedding_model_path or settings.embedding_model_name
        self._batch = settings.embedding_batch_size
        self._model = None
        self._lock = threading.Lock()
        self.dim: int | None = None
        self.load_error: str | None = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                log.info("loading embedding model from %s (first run may download weights)", self._source)
                self._model = SentenceTransformer(self._source, device="cpu")
                self.dim = self._model.get_sentence_embedding_dimension()
                self.load_error = None
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {str(exc)[:300]}"
                log.error("embedding model load failed: %s", self.load_error)
                raise RetrievalError(
                    f"embedding model {self._source!r} could not be loaded: {self.load_error}",
                    public_message="The embedding model is not available.",
                ) from exc

    def _encode(self, prefixed: list[str]) -> np.ndarray:
        self.load()
        vecs = self._model.encode(
            prefixed, batch_size=self._batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.ascontiguousarray(vecs, dtype="float32")

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([PASSAGE_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([QUERY_PREFIX + text])[0]
