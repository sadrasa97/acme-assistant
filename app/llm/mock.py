"""Scripted LLM provider used ONLY by tests. It never ships as a runtime default."""
from __future__ import annotations

from typing import Callable

from app.llm.base import ChatMessage

Handler = Callable[[list[ChatMessage]], str]


class ScriptedProvider:
    """Replies via a handler function (or a fixed queue) and records every call."""

    name = "scripted-mock"
    model_name = "scripted"

    def __init__(self, handler: Handler | None = None, replies: list[str] | None = None):
        self._handler = handler
        self._replies = list(replies or [])
        self.calls: list[list[ChatMessage]] = []

    def is_configured(self) -> bool:
        return True

    def complete(self, messages, *, json_mode=False, temperature=None, max_tokens=None) -> str:
        self.calls.append(messages)
        if self._handler is not None:
            return self._handler(messages)
        if not self._replies:
            raise AssertionError("ScriptedProvider ran out of replies")
        return self._replies.pop(0)
