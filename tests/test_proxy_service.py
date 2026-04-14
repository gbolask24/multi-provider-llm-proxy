import pytest
from app.models import GenerateRequest


def test_resolve_messages_from_prompt():
    from app.services.proxy_service import resolve_messages

    request = GenerateRequest(prompt="Hello world", provider="openai")
    messages = resolve_messages(request)
    assert messages == [{"role": "user", "content": "Hello world"}]


def test_resolve_messages_from_messages_field():
    from app.services.proxy_service import resolve_messages

    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "How are you?"},
    ]
    request = GenerateRequest(messages=msgs, provider="openai")
    messages = resolve_messages(request)
    assert messages == msgs


def test_resolve_messages_prefers_messages_over_prompt():
    from app.services.proxy_service import resolve_messages

    msgs = [{"role": "user", "content": "From messages"}]
    request = GenerateRequest(prompt="From prompt", messages=msgs, provider="openai")
    messages = resolve_messages(request)
    assert messages == [{"role": "user", "content": "From messages"}]


def test_resolve_messages_prepends_system_prompt():
    from app.services.proxy_service import resolve_messages

    request = GenerateRequest(
        prompt="Hello", provider="openai", system_prompt="Be helpful"
    )
    messages = resolve_messages(request)
    assert messages == [
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "Hello"},
    ]


def test_resolve_messages_system_prompt_with_messages():
    from app.services.proxy_service import resolve_messages

    msgs = [{"role": "user", "content": "Hello"}]
    request = GenerateRequest(
        messages=msgs, provider="openai", system_prompt="Be concise"
    )
    messages = resolve_messages(request)
    assert messages[0] == {"role": "system", "content": "Be concise"}
    assert messages[1] == {"role": "user", "content": "Hello"}


def test_parse_fallback_chain_strings():
    from app.services.proxy_service import parse_fallback_chain

    chain = ["anthropic", "gemini"]
    entries = parse_fallback_chain(chain, default_model="gpt-4o-mini")
    assert len(entries) == 2
    assert entries[0].provider == "anthropic"
    assert entries[0].model == "gpt-4o-mini"
    assert entries[1].provider == "gemini"
    assert entries[1].model == "gpt-4o-mini"


def test_parse_fallback_chain_objects():
    from app.services.proxy_service import parse_fallback_chain

    chain = [{"provider": "anthropic", "model": "claude-3-haiku-20240307"}]
    entries = parse_fallback_chain(chain, default_model="gpt-4o-mini")
    assert len(entries) == 1
    assert entries[0].provider == "anthropic"
    assert entries[0].model == "claude-3-haiku-20240307"


def test_parse_fallback_chain_mixed():
    from app.services.proxy_service import parse_fallback_chain

    chain = [
        "gemini",
        {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
    ]
    entries = parse_fallback_chain(chain, default_model="gpt-4o-mini")
    assert len(entries) == 2
    assert entries[0].provider == "gemini"
    assert entries[0].model == "gpt-4o-mini"
    assert entries[1].provider == "anthropic"
    assert entries[1].model == "claude-3-haiku-20240307"


def test_parse_fallback_chain_empty():
    from app.services.proxy_service import parse_fallback_chain

    entries = parse_fallback_chain([], default_model="gpt-4o-mini")
    assert entries == []
