import pytest
from unittest.mock import AsyncMock

from app.providers.base import LLMProvider, ProviderResponse, Usage
from app.providers import PROVIDER_REGISTRY, register_provider


class MockProvider(LLMProvider):
    """A mock provider for testing proxy service logic."""

    # Satisfy the ABC requirement at the class level; __init__ replaces this
    # instance attribute with an AsyncMock so call tracking works per-instance.
    async def generate(self, messages, model, temperature, max_tokens):  # pragma: no cover
        raise NotImplementedError

    def __init__(self, name: str, response: ProviderResponse | None = None, error: Exception | None = None):
        self.provider_name = name
        self._response = response
        self._error = error
        self.generate = AsyncMock(side_effect=self._generate)

    async def _generate(self, messages, model, temperature, max_tokens):
        if self._error:
            raise self._error
        return self._response


def make_provider_response(
    content="Test response",
    finish_reason="stop",
    input_tokens=10,
    output_tokens=5,
) -> ProviderResponse:
    return ProviderResponse(
        content=content,
        finish_reason=finish_reason,
        usage=Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
    )


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the provider registry before each test."""
    PROVIDER_REGISTRY.clear()
    yield
    PROVIDER_REGISTRY.clear()
