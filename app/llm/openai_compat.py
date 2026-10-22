"""Provider for any OpenAI-compatible /chat/completions endpoint (OpenAI, vLLM, Ollama, LM Studio...)."""
from __future__ import annotations

import time

import httpx

from app.core.config import Settings
from app.core.errors import LLMResponseError, LLMTimeoutError, LLMUnavailableError
from app.core.logging import get_logger
from app.llm.base import ChatMessage

log = get_logger("llm")

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatProvider:
    name = "openai-compatible"

    def __init__(self, settings: Settings, client: httpx.Client | None = None):
        self._s = settings
        self.model_name = settings.llm_model_name
        self._client = client or httpx.Client(timeout=settings.llm_timeout_seconds)

    def is_configured(self) -> bool:
        # Local servers often need no key, so only the URL and model are mandatory.
        return bool(self._s.llm_base_url and self._s.llm_model_name)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if not self.is_configured():
            raise LLMUnavailableError("LLM base_url/model_name not configured")

        payload: dict = {
            "model": self._s.llm_model_name,
            "messages": [m.model_dump() for m in messages],
            "temperature": self._s.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self._s.llm_max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        key = self._s.llm_api_key.get_secret_value()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        url = self._s.llm_base_url.rstrip("/") + "/chat/completions"

        attempts = self._s.llm_max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                resp = self._client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                last_error = LLMTimeoutError(f"timeout calling LLM: {exc}")
            except httpx.HTTPError as exc:
                last_error = LLMUnavailableError(f"transport error calling LLM: {type(exc).__name__}")
            else:
                if resp.status_code == 200:
                    return self._extract_text(resp)
                if resp.status_code in _RETRYABLE_STATUS:
                    last_error = LLMUnavailableError(f"LLM returned HTTP {resp.status_code}")
                elif resp.status_code in (401, 403):
                    raise LLMUnavailableError(f"LLM rejected credentials (HTTP {resp.status_code})")
                else:
                    raise LLMResponseError(f"LLM returned HTTP {resp.status_code}")

            if attempt < attempts:
                delay = self._s.llm_retry_backoff_seconds * (2 ** (attempt - 1))
                log.warning("llm call failed (attempt %d/%d): %s; retrying in %.1fs", attempt, attempts, last_error, delay)
                time.sleep(delay)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _extract_text(resp: httpx.Response) -> str:
        try:
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("unexpected LLM response shape") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("LLM returned empty content")
        return text
