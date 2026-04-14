import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from app.providers import PROVIDER_REGISTRY, register_provider
from app.exceptions import ProviderInvocationError

# Import MockProvider and make_provider_response from conftest
# These are available because conftest.py is in the tests/ directory
from tests.conftest import MockProvider, make_provider_response


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "providers" in data


def test_generate_success(client):
    provider = MockProvider("openai", response=make_provider_response(content="API response"))
    register_provider("openai", provider)

    response = client.post(
        "/v1/generate",
        json={
            "prompt": "Hello",
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["content"] == "API response"
    assert data["provider"] == "openai"
    assert "request_id" in data
    assert "latency_ms" in data
    assert "usage" in data


def test_generate_missing_prompt_and_messages(client):
    response = client.post(
        "/v1/generate",
        json={"provider": "openai", "model": "gpt-4o-mini"},
    )
    assert response.status_code == 422


def test_generate_unsupported_provider(client):
    response = client.post(
        "/v1/generate",
        json={"prompt": "Hello", "provider": "nonexistent", "model": "x"},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["error"]["type"] == "unsupported_provider"


def test_generate_all_providers_fail(client):
    failing = MockProvider("openai", error=ProviderInvocationError("openai", "timeout"))
    register_provider("openai", failing)

    response = client.post(
        "/v1/generate",
        json={"prompt": "Hello", "provider": "openai", "model": "gpt-4o-mini"},
    )
    assert response.status_code == 502
    data = response.json()
    assert data["error"]["type"] == "all_providers_failed"
    assert "request_id" in data


def test_generate_with_fallback_via_api(client):
    failing = MockProvider("openai", error=ProviderInvocationError("openai", "timeout"))
    success = MockProvider("anthropic", response=make_provider_response(content="Fallback via API"))
    register_provider("openai", failing)
    register_provider("anthropic", success)

    response = client.post(
        "/v1/generate",
        json={
            "prompt": "Hello",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "fallback_chain": ["anthropic"],
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["provider"] == "anthropic"
    assert data["fallback_used"] is True


def test_providers_endpoint(client):
    register_provider("openai", MockProvider("openai", response=make_provider_response()))

    response = client.get("/v1/providers")
    assert response.status_code == 200
    data = response.json()
    assert "providers" in data
    names = [p["name"] for p in data["providers"]]
    assert "openai" in names
