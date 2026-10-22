"""Provider-agnostic LLM contract. Agents depend only on this interface."""
from __future__ import annotations

import json
import re
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from app.core.errors import LLMResponseError


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


@runtime_checkable
class LLMProvider(Protocol):
    name: str
    model_name: str

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return the assistant text for a chat. Raises AppError subclasses on failure."""
        ...

    def is_configured(self) -> bool: ...


_FENCE = re.compile(r"^```(?:json|sql)?\s*|\s*```$", re.IGNORECASE)


def strip_code_fence(text: str) -> str:
    return _FENCE.sub("", text.strip()).strip()


def parse_json_object(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences and prose around it."""
    cleaned = strip_code_fence(text)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise LLMResponseError("model output contained no JSON object")
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"model output was not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise LLMResponseError("model output JSON was not an object")
    return value
