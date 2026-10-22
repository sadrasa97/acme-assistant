"""Typed application errors mapped to HTTP responses by the API layer."""
from __future__ import annotations


class AppError(Exception):
    http_status = 500
    code = "internal_error"
    public_message = "An internal error occurred."

    def __init__(self, detail: str = "", *, public_message: str | None = None):
        super().__init__(detail or self.public_message)
        self.detail = detail or self.public_message
        if public_message:
            self.public_message = public_message


class LLMUnavailableError(AppError):
    http_status = 503
    code = "llm_unavailable"
    public_message = "The language model is unavailable or not configured."


class LLMTimeoutError(AppError):
    http_status = 504
    code = "llm_timeout"
    public_message = "The language model did not respond in time."


class LLMResponseError(AppError):
    http_status = 502
    code = "llm_bad_response"
    public_message = "The language model returned an unusable response."


class RetrievalError(AppError):
    http_status = 503
    code = "retrieval_error"
    public_message = "Document retrieval is currently unavailable."


class SQLGenerationError(AppError):
    http_status = 422
    code = "sql_generation_failed"
    public_message = "Could not produce a safe SQL query for this question."


class SQLValidationError(AppError):
    http_status = 422
    code = "sql_rejected"
    public_message = "The generated SQL query was rejected by the safety policy."


class SQLExecutionError(AppError):
    http_status = 422
    code = "sql_execution_failed"
    public_message = "The database query could not be executed."


class ToolError(AppError):
    http_status = 422
    code = "tool_error"
    public_message = "A tool could not complete the requested operation."


class PlanError(AppError):
    http_status = 502
    code = "plan_invalid"
    public_message = "The request could not be routed."


class RequestTimeoutError(AppError):
    http_status = 504
    code = "request_timeout"
    public_message = "The request exceeded its time budget."
