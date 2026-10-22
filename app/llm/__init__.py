from app.core.config import Settings
from app.llm.base import ChatMessage, LLMProvider, parse_json_object, strip_code_fence
from app.llm.openai_compat import OpenAICompatProvider


def build_provider(settings: Settings) -> LLMProvider:
    """Single place where a provider is chosen; add local-model providers here."""
    return OpenAICompatProvider(settings)


__all__ = ["ChatMessage", "LLMProvider", "OpenAICompatProvider", "build_provider", "parse_json_object", "strip_code_fence"]
