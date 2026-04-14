import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.exceptions import ProviderInvocationError
from app.providers.openai_provider import OpenAIProvider


@pytest.fixture
def openai_provider():
    return OpenAIProvider(api_key="test-key")


def _make_openai_response(content="Hello", finish_reason="stop", input_tokens=10, output_tokens=5):
    """Build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    choice.finish_reason = finish_reason

    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    usage.total_tokens = input_tokens + output_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


@pytest.mark.asyncio
async def test_openai_generate_success(openai_provider):
    mock_response = _make_openai_response(content="Test response", input_tokens=20, output_tokens=10)

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await openai_provider.generate(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=100,
        )

    assert result.content == "Test response"
    assert result.finish_reason == "stop"
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 10
    assert result.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_openai_passes_messages_directly(openai_provider):
    mock_response = _make_openai_response()
    messages = [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_create:
        await openai_provider.generate(
            messages=messages, model="gpt-4o-mini", temperature=0.5, max_tokens=200
        )

    mock_create.assert_called_once_with(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.5,
        max_tokens=200,
    )


@pytest.mark.asyncio
async def test_openai_wraps_sdk_error(openai_provider):
    from openai import APIError

    with patch.object(
        openai_provider.client.chat.completions,
        "create",
        new_callable=AsyncMock,
        side_effect=APIError(
            message="Rate limit exceeded",
            request=MagicMock(),
            body=None,
        ),
    ):
        with pytest.raises(ProviderInvocationError) as exc_info:
            await openai_provider.generate(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=100,
            )

    assert exc_info.value.provider == "openai"
    assert "Rate limit" in exc_info.value.message


from app.providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def anthropic_provider():
    return AnthropicProvider(api_key="test-key")


def _make_anthropic_response(content="Hello", stop_reason="end_turn", input_tokens=10, output_tokens=5):
    """Build a mock Anthropic message response."""
    content_block = MagicMock()
    content_block.text = content

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    response = MagicMock()
    response.content = [content_block]
    response.stop_reason = stop_reason
    response.usage = usage
    return response


@pytest.mark.asyncio
async def test_anthropic_generate_success(anthropic_provider):
    mock_response = _make_anthropic_response(content="Anthropic reply", input_tokens=15, output_tokens=8)

    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        result = await anthropic_provider.generate(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-3-haiku-20240307",
            temperature=0.5,
            max_tokens=200,
        )

    assert result.content == "Anthropic reply"
    assert result.finish_reason == "end_turn"
    assert result.usage.input_tokens == 15
    assert result.usage.output_tokens == 8
    assert result.usage.total_tokens == 23


@pytest.mark.asyncio
async def test_anthropic_extracts_system_message(anthropic_provider):
    mock_response = _make_anthropic_response()
    messages = [
        {"role": "system", "content": "Be concise"},
        {"role": "user", "content": "Hello"},
    ]

    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_create:
        await anthropic_provider.generate(
            messages=messages, model="claude-3-haiku-20240307", temperature=0.5, max_tokens=200
        )

    mock_create.assert_called_once_with(
        model="claude-3-haiku-20240307",
        system="Be concise",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.5,
        max_tokens=200,
    )


@pytest.mark.asyncio
async def test_anthropic_no_system_message(anthropic_provider):
    mock_response = _make_anthropic_response()
    messages = [{"role": "user", "content": "Hello"}]

    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
        return_value=mock_response,
    ) as mock_create:
        await anthropic_provider.generate(
            messages=messages, model="claude-3-haiku-20240307", temperature=0.5, max_tokens=200
        )

    # No system kwarg when no system message present
    call_kwargs = mock_create.call_args.kwargs
    assert "system" not in call_kwargs or call_kwargs["system"] is None


@pytest.mark.asyncio
async def test_anthropic_wraps_sdk_error(anthropic_provider):
    from anthropic import APIError

    with patch.object(
        anthropic_provider.client.messages,
        "create",
        new_callable=AsyncMock,
        side_effect=APIError(
            message="Overloaded",
            request=MagicMock(),
            body=None,
        ),
    ):
        with pytest.raises(ProviderInvocationError) as exc_info:
            await anthropic_provider.generate(
                messages=[{"role": "user", "content": "Hello"}],
                model="claude-3-haiku-20240307",
                temperature=0.7,
                max_tokens=100,
            )

    assert exc_info.value.provider == "anthropic"
    assert "Overloaded" in exc_info.value.message
