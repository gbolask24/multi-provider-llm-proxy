from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass
class ProviderResponse:
    content: str
    finish_reason: str | None
    usage: Usage
    raw_response: Any | None = None


class LLMProvider(ABC):
    """Base class for all LLM providers."""

    provider_name: str

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        """Send messages to the provider and return a normalised response."""
        ...
