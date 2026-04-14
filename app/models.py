from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


class GenerateRequest(BaseModel):
    """Unified request schema for LLM generation."""

    messages: list[dict[str, str]] | None = None
    prompt: str | None = None
    system_prompt: str | None = None
    provider: str
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024
    fallback_chain: list[str | dict[str, str]] = []
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def check_prompt_or_messages(self) -> GenerateRequest:
        if self.prompt is None and self.messages is None:
            raise ValueError("Either 'prompt' or 'messages' must be provided")
        return self


class UsageResponse(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class GenerateResponse(BaseModel):
    """Unified success response from the proxy."""

    request_id: str
    provider: str
    model: str
    content: str
    finish_reason: str | None
    latency_ms: float
    estimated_cost_usd: float | None
    usage: UsageResponse
    fallback_used: bool
    fallback_provider: str | None
    metadata: dict[str, Any] = {}


class ErrorDetail(BaseModel):
    provider: str
    model: str | None = None
    error: str


class ErrorInfo(BaseModel):
    type: str
    message: str
    details: list[ErrorDetail] | None = None


class ErrorResponse(BaseModel):
    """Structured error response."""

    request_id: str
    error: ErrorInfo
