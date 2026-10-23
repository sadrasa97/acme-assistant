"""Application settings, read from environment variables / a .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider (OpenAI-compatible) ---
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr = SecretStr("")
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_retry_backoff_seconds: float = 1.0

    # --- Embeddings / RAG ---
    embedding_model_name: str = "intfloat/e5-large-v2"
    # Optional local directory holding a pre-downloaded copy of the model.
    embedding_model_path: str = ""
    embedding_batch_size: int = 16
    documents_dir: Path = PROJECT_ROOT / "data" / "documents"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    chunk_size: int = 700
    chunk_overlap: int = 120
    rag_top_k: int = 4
    # Minimum cosine similarity for a chunk to count as evidence. e5 scores are
    # compressed into a narrow high range, so this must be calibrated per corpus.
    rag_min_similarity: float = 0.80

    # --- Database ---
    sqlite_path: Path = PROJECT_ROOT / "data" / "acme.db"
    sql_max_rows: int = 200
    sql_timeout_seconds: float = 5.0
    sql_repair_attempts: int = 1

    # --- Coordinator / API ---
    max_plan_steps: int = 4
    max_tool_calls: int = 4
    request_timeout_seconds: float = 90.0
    max_question_length: int = 1000
    log_level: str = "INFO"

    @field_validator("*", mode="before")
    @classmethod
    def _strip_inline_comment(cls, v):
        """Docker Compose's env_file keeps `VALUE   # comment` as the value; drop such a trailing comment."""
        if isinstance(v, str):
            v = v.split(" #", 1)[0].split("\t#", 1)[0].strip()
            if v.startswith("#"):
                v = ""
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()