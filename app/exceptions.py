class ProviderInvocationError(Exception):
    """Raised when a provider SDK call fails. Normalises provider-specific errors."""

    def __init__(self, provider: str, message: str, retryable: bool = False):
        self.provider = provider
        self.message = message
        self.retryable = retryable
        super().__init__(f"[{provider}] {message}")


class UnsupportedProviderError(Exception):
    """Raised when requesting a provider that is not registered."""

    def __init__(self, provider: str):
        self.provider = provider
        super().__init__(f"Unsupported provider: {provider}")


class AllProvidersFailedError(Exception):
    """Raised when the primary provider and all fallbacks have failed."""

    def __init__(self, request_id: str, errors: list[dict]):
        self.request_id = request_id
        self.errors = errors
        super().__init__(f"All providers failed for request {request_id}")
