"""Test doubles. The embedder is a deterministic hashed bag-of-words model (NOT e5) used so the fast
suite needs no model download; chunking, FAISS search and all SQL/Calculator execution stay real."""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from app.llm.base import ChatMessage

_STOP = {"the", "a", "an", "of", "to", "is", "are", "for", "and", "in", "on", "what", "how", "does", "do", "can", "by", "with", "at", "be", "or", "if", "it"}


class HashingEmbedder:
    model_name = "test-hashing-embedder"
    dim = 512
    loaded = True
    load_error = None

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            if tok in _STOP:
                continue
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def make_handler(sql_by_question: dict[str, str] | None = None, plans: dict[str, dict] | None = None):
    """LLM stand-in that dispatches on which agent's system prompt it receives."""
    sql_by_question = sql_by_question or {}
    plans = plans or {}

    def handler(messages: list[ChatMessage]) -> str:
        system, user = messages[0].content, messages[-1].content
        if "router of an assistant" in system:
            return json.dumps(plans[user])
        if "ONE read-only SQLite SELECT" in system:
            q = user.split("Question: ", 1)[1].split("\n\n", 1)[0]
            return sql_by_question[q]
        if "numbered context excerpts" in system:
            ctx = user.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
            first = ctx.split("\n", 1)[1].split("\n\n[", 1)[0]
            return f"Per the policy documents [1]: {first[:300]}"
        raise AssertionError(f"unexpected prompt: {system[:60]}")

    return handler
