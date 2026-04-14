import pytest
from app.exceptions import AllProvidersFailedError, ProviderInvocationError
from app.models import GenerateRequest
from app.providers import register_provider
from tests.conftest import MockProvider, make_provider_response


@pytest.mark.asyncio
async def test_primary_provider_succeeds_no_fallback():
    from app.services.proxy_service import handle_request

    provider = MockProvider("openai", response=make_provider_response(content="Primary response"))
    register_provider("openai", provider)

    request = GenerateRequest(prompt="Hello", provider="openai", model="gpt-4o-mini")
    response = await handle_request(request)

    assert response.provider == "openai"
    assert response.content == "Primary response"
    assert response.fallback_used is False
    assert response.fallback_provider is None


@pytest.mark.asyncio
async def test_fallback_on_primary_failure():
    from app.services.proxy_service import handle_request

    failing_provider = MockProvider(
        "openai", error=ProviderInvocationError("openai", "timeout")
    )
    fallback_provider = MockProvider(
        "anthropic", response=make_provider_response(content="Fallback response")
    )
    register_provider("openai", failing_provider)
    register_provider("anthropic", fallback_provider)

    request = GenerateRequest(
        prompt="Hello",
        provider="openai",
        model="gpt-4o-mini",
        fallback_chain=["anthropic"],
    )
    response = await handle_request(request)

    assert response.provider == "anthropic"
    assert response.content == "Fallback response"
    assert response.fallback_used is True
    assert response.fallback_provider == "anthropic"


@pytest.mark.asyncio
async def test_fallback_with_model_override():
    from app.services.proxy_service import handle_request

    failing_provider = MockProvider(
        "openai", error=ProviderInvocationError("openai", "error")
    )
    fallback_provider = MockProvider(
        "anthropic", response=make_provider_response(content="Fallback")
    )
    register_provider("openai", failing_provider)
    register_provider("anthropic", fallback_provider)

    request = GenerateRequest(
        prompt="Hello",
        provider="openai",
        model="gpt-4o-mini",
        fallback_chain=[{"provider": "anthropic", "model": "claude-3-haiku-20240307"}],
    )
    response = await handle_request(request)

    assert response.provider == "anthropic"
    assert response.model == "claude-3-haiku-20240307"
    assert response.fallback_used is True

    # Verify the fallback provider was called with the overridden model
    fallback_provider.generate.assert_called_once()
    call_args = fallback_provider.generate.call_args
    # Check positional or keyword args for model
    if call_args.kwargs:
        assert call_args.kwargs.get("model") == "claude-3-haiku-20240307"
    else:
        assert call_args[0][1] == "claude-3-haiku-20240307"


@pytest.mark.asyncio
async def test_all_providers_fail():
    from app.services.proxy_service import handle_request

    failing1 = MockProvider("openai", error=ProviderInvocationError("openai", "timeout"))
    failing2 = MockProvider("anthropic", error=ProviderInvocationError("anthropic", "rate_limited"))
    register_provider("openai", failing1)
    register_provider("anthropic", failing2)

    request = GenerateRequest(
        prompt="Hello",
        provider="openai",
        model="gpt-4o-mini",
        fallback_chain=["anthropic"],
    )

    with pytest.raises(AllProvidersFailedError) as exc_info:
        await handle_request(request)

    assert len(exc_info.value.errors) == 2
    assert exc_info.value.errors[0]["provider"] == "openai"
    assert exc_info.value.errors[0]["error"] == "timeout"
    assert exc_info.value.errors[1]["provider"] == "anthropic"
    assert exc_info.value.errors[1]["error"] == "rate_limited"


@pytest.mark.asyncio
async def test_fallback_chain_skips_to_second_fallback():
    from app.services.proxy_service import handle_request

    failing1 = MockProvider("openai", error=ProviderInvocationError("openai", "error"))
    failing2 = MockProvider("anthropic", error=ProviderInvocationError("anthropic", "error"))
    success = MockProvider("gemini", response=make_provider_response(content="Gemini response"))
    register_provider("openai", failing1)
    register_provider("anthropic", failing2)
    register_provider("gemini", success)

    request = GenerateRequest(
        prompt="Hello",
        provider="openai",
        model="gpt-4o-mini",
        fallback_chain=["anthropic", "gemini"],
    )
    response = await handle_request(request)

    assert response.provider == "gemini"
    assert response.content == "Gemini response"
    assert response.fallback_used is True
    assert response.fallback_provider == "gemini"


@pytest.mark.asyncio
async def test_metadata_echoed_in_response():
    from app.services.proxy_service import handle_request

    provider = MockProvider("openai", response=make_provider_response())
    register_provider("openai", provider)

    metadata = {"use_case": "testing", "environment": "staging"}
    request = GenerateRequest(
        prompt="Hello", provider="openai", model="gpt-4o-mini", metadata=metadata
    )
    response = await handle_request(request)

    assert response.metadata == metadata


@pytest.mark.asyncio
async def test_cost_estimated_in_response():
    from app.services.proxy_service import handle_request

    provider = MockProvider(
        "openai",
        response=make_provider_response(input_tokens=1000, output_tokens=500),
    )
    register_provider("openai", provider)

    request = GenerateRequest(prompt="Hello", provider="openai", model="gpt-4o-mini")
    response = await handle_request(request)

    # gpt-4o-mini: (1000 * 0.00000015) + (500 * 0.0000006) = 0.00045
    assert response.estimated_cost_usd is not None
    assert abs(response.estimated_cost_usd - 0.00045) < 1e-10


@pytest.mark.asyncio
async def test_cost_none_for_unknown_model():
    from app.services.proxy_service import handle_request

    provider = MockProvider("openai", response=make_provider_response())
    register_provider("openai", provider)

    request = GenerateRequest(prompt="Hello", provider="openai", model="unknown-model")
    response = await handle_request(request)

    assert response.estimated_cost_usd is None
