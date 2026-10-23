"""Typed contracts shared by agents, the coordinator and the API."""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Status(str, Enum):
    OK = "ok"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    OUT_OF_SCOPE = "out_of_scope"
    NEEDS_CLARIFICATION = "needs_clarification"
    ERROR = "error"


class Route(str, Enum):
    RAG = "rag"
    DATABASE = "database"
    CALCULATION = "calculation"
    HYBRID = "hybrid"          # more than one capability chained
    OUT_OF_SCOPE = "out_of_scope"
    CLARIFY = "clarify"


# ---------------- RAG ----------------
class SourceRef(BaseModel):
    file: str
    page: int
    chunk_id: str
    section: str | None = None
    score: float
    snippet: str


class RAGResult(BaseModel):
    status: Status
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    retrieved: int = 0
    top_score: float | None = None
    error: str | None = None


# ---------------- Database ----------------
class DatabaseResult(BaseModel):
    status: Status
    sql: str | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    attempts: int = 0
    error: str | None = None


# ---------------- Calculator ----------------
class CalculatorResult(BaseModel):
    status: Status
    expression: str
    inputs: dict[str, float] = Field(default_factory=dict)
    result: float | None = None
    error: str | None = None


# ---------------- Plan & trace ----------------
class PlanStep(BaseModel):
    """One allow-listed step. Anything not expressible here cannot be executed."""
    id: str = Field(pattern=r"^s[1-9]$")
    kind: Literal["rag", "database", "calculator"]
    question: str | None = None            # rag / database
    expression: str | None = None          # calculator
    # calculator variable -> "<step_id>.<column>" taken from that step's first result row
    bindings: dict[str, str] = Field(default_factory=dict)


class Plan(BaseModel):
    route: Route
    steps: list[PlanStep] = Field(default_factory=list)
    reason: str = ""
    source: Literal["llm", "fallback", "retrieval_override"] = "llm"


class StepTrace(BaseModel):
    step_id: str
    kind: str
    status: Status
    duration_ms: float
    input: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    error: str | None = None
    sql: str | None = None
    output: dict[str, Any] | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    calculation: CalculatorResult | None = None


class AskResponse(BaseModel):
    request_id: str
    status: Status
    route: Route
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    sql: str | None = None
    result: DatabaseResult | None = None
    calculation: CalculatorResult | None = None
    steps: list[StepTrace] = Field(default_factory=list)
    plan_source: str = "llm"
    duration_ms: float = 0.0