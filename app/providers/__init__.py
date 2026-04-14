from app.exceptions import UnsupportedProviderError
from app.providers.base import LLMProvider

PROVIDER_REGISTRY: dict[str, LLMProvider] = {}


def register_provider(name: str, provider: LLMProvider) -> None:
    """Register a provider instance by name."""
    PROVIDER_REGISTRY[name] = provider


def get_provider(name: str) -> LLMProvider:
    """Look up a registered provider. Raises UnsupportedProviderError if not found."""
    if name not in PROVIDER_REGISTRY:
        raise UnsupportedProviderError(name)
    return PROVIDER_REGISTRY[name]


def get_available_providers() -> list[str]:
    """Return names of all registered providers."""
    return list(PROVIDER_REGISTRY.keys())
