from __future__ import annotations

import time
from dataclasses import dataclass
from uuid import uuid4

from loguru import logger

from app.config import settings
from app.exceptions import AllProvidersFailedError, ProviderInvocationError
from app.models import GenerateRequest, GenerateResponse, UsageResponse
from app.pricing import estimate_cost
from app.providers import get_provider
from app.providers.base import ProviderResponse, Usage


@dataclass
class FallbackEntry:
    provider: str
    model: str


def resolve_messages(request: GenerateRequest) -> list[dict[str, str]]:
    """Convert prompt/messages/system_prompt into a canonical message list."""
    if request.messages is not None:
        messages = list(request.messages)
    else:
        messages = [{"role": "user", "content": request.prompt}]

    if request.system_prompt:
        messages.insert(0, {"role": "system", "content": request.system_prompt})

    return messages


def parse_fallback_chain(
    chain: list[str | dict[str, str]], default_model: str
) -> list[FallbackEntry]:
    """Parse the fallback_chain field into a list of FallbackEntry objects."""
    entries = []
    for item in chain:
        if isinstance(item, str):
            entries.append(FallbackEntry(provider=item, model=default_model))
        else:
            entries.append(
                FallbackEntry(
                    provider=item["provider"],
                    model=item.get("model", default_model),
                )
            )
    return entries


async def handle_request(request: GenerateRequest) -> GenerateResponse:
    """Main proxy orchestrator: resolve, route, fallback, normalise."""
    messages = resolve_messages(request)
    request_id = str(uuid4())
    model = request.model or settings.default_model

    providers_to_try = [FallbackEntry(provider=request.provider, model=model)] + (
        parse_fallback_chain(request.fallback_chain, default_model=model)
    )

    errors: list[dict] = []

    for i, entry in enumerate(providers_to_try):
        start = time.perf_counter()
        try:
            provider = get_provider(entry.provider)
            result: ProviderResponse = await provider.generate(
                messages=messages,
                model=entry.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            latency_ms = (time.perf_counter() - start) * 1000

            cost = estimate_cost(entry.model, result.usage)
            fallback_used = i > 0

            logger.info(
                "Request completed",
                request_id=request_id,
                provider=entry.provider,
                model=entry.model,
                latency_ms=round(latency_ms, 2),
                fallback_used=fallback_used,
                estimated_cost_usd=cost,
                status="success",
            )

            return GenerateResponse(
                request_id=request_id,
                provider=entry.provider,
                model=entry.model,
                content=result.content,
                finish_reason=result.finish_reason,
                latency_ms=round(latency_ms, 2),
                estimated_cost_usd=cost,
                usage=UsageResponse(
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                    total_tokens=result.usage.total_tokens,
                ),
                fallback_used=fallback_used,
                fallback_provider=entry.provider if fallback_used else None,
                metadata=request.metadata,
            )

        except ProviderInvocationError as e:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning(
                "Provider attempt failed",
                request_id=request_id,
                provider=entry.provider,
                model=entry.model,
                error=e.message,
                latency_ms=round(latency_ms, 2),
                status="fallback_attempt_failed",
            )
            errors.append(
                {"provider": entry.provider, "model": entry.model, "error": e.message}
            )

    logger.error(
        "All providers failed",
        request_id=request_id,
        status="failed",
        error_type="all_providers_failed",
        attempts=len(errors),
    )
    raise AllProvidersFailedError(request_id=request_id, errors=errors)
