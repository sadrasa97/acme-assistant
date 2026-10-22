"""FAISS inner-product index (on normalised vectors == cosine) + JSON metadata, persisted to disk.

A manifest stores the model name and a fingerprint of the source PDFs and chunking parameters;
`is_stale()` tells the caller when the index must be rebuilt."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from app.core.errors import RetrievalError
from app.rag.chunker import Chunk

INDEX_FILE, META_FILE, MANIFEST_FILE = "chunks.faiss", "chunks.json", "manifest.json"


def corpus_fingerprint(pdfs: list[Path], model_name: str, chunk_size: int, overlap: int) -> str:
    h = hashlib.sha256(f"{model_name}|{chunk_size}|{overlap}".encode())
    for p in sorted(pdfs):
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


class VectorStore:
    def __init__(self, index: faiss.Index, chunks: list[Chunk], manifest: dict):
        self.index, self.chunks, self.manifest = index, chunks, manifest

    @classmethod
    def build(cls, chunks: list[Chunk], vectors: np.ndarray, manifest: dict) -> "VectorStore":
        if len(chunks) != len(vectors) or not len(chunks):
            raise RetrievalError("cannot build an index from zero chunks or mismatched vectors")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype="float32"))
        return cls(index, chunks, {**manifest, "num_chunks": len(chunks), "dim": int(vectors.shape[1])})

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        k = max(1, min(k, len(self.chunks)))
        scores, ids = self.index.search(np.ascontiguousarray(query_vec[None, :], dtype="float32"), k)
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / INDEX_FILE))
        (directory / META_FILE).write_text(json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=1))
        (directory / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=1))

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        try:
            index = faiss.read_index(str(directory / INDEX_FILE))
            chunks = [Chunk(**c) for c in json.loads((directory / META_FILE).read_text())]
            manifest = json.loads((directory / MANIFEST_FILE).read_text())
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            raise RetrievalError(f"vector index missing or corrupt in {directory}: {exc}") from exc
        if index.ntotal != len(chunks):
            raise RetrievalError("vector index and metadata are out of sync")
        return cls(index, chunks, manifest)

    @staticmethod
    def stored_fingerprint(directory: Path) -> str | None:
        try:
            return json.loads((directory / MANIFEST_FILE).read_text()).get("fingerprint")
        except (OSError, ValueError):
            return None
