import pytest
from pydantic import ValidationError


def test_request_with_prompt_only():
    from app.models import GenerateRequest

    req = GenerateRequest(prompt="Hello", provider="openai")
    assert req.prompt == "Hello"
    assert req.messages is None


def test_request_with_messages_only():
    from app.models import GenerateRequest

    msgs = [{"role": "user", "content": "Hello"}]
    req = GenerateRequest(messages=msgs, provider="openai")
    assert req.messages == msgs
    assert req.prompt is None


def test_request_missing_prompt_and_messages():
    from app.models import GenerateRequest

    with pytest.raises(ValidationError, match="prompt.*messages"):
        GenerateRequest(provider="openai")


def test_request_messages_takes_precedence_over_prompt():
    from app.models import GenerateRequest

    msgs = [{"role": "user", "content": "From messages"}]
    req = GenerateRequest(prompt="From prompt", messages=msgs, provider="openai")
    # Both are stored, but messages takes precedence (resolved in proxy service)
    assert req.messages == msgs
    assert req.prompt == "From prompt"


def test_request_defaults():
    from app.models import GenerateRequest

    req = GenerateRequest(prompt="Hello", provider="openai")
    assert req.temperature == 0.7
    assert req.max_tokens == 1024
    assert req.fallback_chain == []
    assert req.metadata == {}
    assert req.model is None
    assert req.system_prompt is None


def test_request_with_system_prompt():
    from app.models import GenerateRequest

    req = GenerateRequest(prompt="Hello", provider="openai", system_prompt="Be helpful")
    assert req.system_prompt == "Be helpful"


def test_fallback_chain_string_form():
    from app.models import GenerateRequest

    req = GenerateRequest(
        prompt="Hello",
        provider="openai",
        fallback_chain=["anthropic", "gemini"],
    )
    assert req.fallback_chain == ["anthropic", "gemini"]


def test_fallback_chain_object_form():
    from app.models import GenerateRequest

    req = GenerateRequest(
        prompt="Hello",
        provider="openai",
        fallback_chain=[{"provider": "anthropic", "model": "claude-3-haiku-20240307"}],
    )
    assert req.fallback_chain == [{"provider": "anthropic", "model": "claude-3-haiku-20240307"}]


def test_fallback_chain_mixed_form():
    from app.models import GenerateRequest

    req = GenerateRequest(
        prompt="Hello",
        provider="openai",
        fallback_chain=[
            "gemini",
            {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
        ],
    )
    assert len(req.fallback_chain) == 2


def test_response_model_shape():
    from app.models import GenerateResponse, UsageResponse

    resp = GenerateResponse(
        request_id="abc-123",
        provider="openai",
        model="gpt-4o-mini",
        content="Hello there",
        finish_reason="stop",
        latency_ms=100.5,
        estimated_cost_usd=0.001,
        usage=UsageResponse(input_tokens=10, output_tokens=5, total_tokens=15),
        fallback_used=False,
        fallback_provider=None,
        metadata={},
    )
    assert resp.provider == "openai"
    assert resp.content == "Hello there"
    assert resp.usage.total_tokens == 15


def test_response_nullable_fields():
    from app.models import GenerateResponse, UsageResponse

    resp = GenerateResponse(
        request_id="abc-123",
        provider="openai",
        model="gpt-4o-mini",
        content="Hello",
        finish_reason=None,
        latency_ms=50.0,
        estimated_cost_usd=None,
        usage=UsageResponse(input_tokens=None, output_tokens=None, total_tokens=None),
        fallback_used=False,
        fallback_provider=None,
        metadata={},
    )
    assert resp.finish_reason is None
    assert resp.estimated_cost_usd is None
    assert resp.usage.input_tokens is None


def test_error_response_shape():
    from app.models import ErrorDetail, ErrorInfo, ErrorResponse

    resp = ErrorResponse(
        request_id="abc-123",
        error=ErrorInfo(
            type="all_providers_failed",
            message="All providers in the chain failed",
            details=[
                ErrorDetail(provider="openai", model="gpt-4o-mini", error="timeout"),
            ],
        ),
    )
    assert resp.error.type == "all_providers_failed"
    assert resp.error.details[0].provider == "openai"
    assert resp.error.details[0].model == "gpt-4o-mini"
