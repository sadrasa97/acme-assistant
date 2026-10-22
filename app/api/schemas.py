from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural-language question", examples=["What is the refund window for annual plans?"])

    @field_validator("question")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("question must not be empty")
        if len(v) > 1000:
            raise ValueError("question too long (max 1000 characters)")
        return v


class ErrorBody(BaseModel):
    error: str
    message: str
    request_id: str | None = None
