import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.exceptions import AllProvidersFailedError, UnsupportedProviderError
from app.logger import setup_logging
from app.models import ErrorInfo, ErrorResponse
from app.providers import register_provider
from app.router import router

app = FastAPI(title="Multi-Provider LLM Proxy", version="1.0.0")
app.include_router(router)


@app.on_event("startup")
async def startup():
    setup_logging(json_output=not settings.debug)

    if settings.openai_api_key:
        from app.providers.openai_provider import OpenAIProvider
        register_provider("openai", OpenAIProvider(api_key=settings.openai_api_key))
        logger.info("Registered provider: openai")

    if settings.anthropic_api_key:
        from app.providers.anthropic_provider import AnthropicProvider
        register_provider("anthropic", AnthropicProvider(api_key=settings.anthropic_api_key))
        logger.info("Registered provider: anthropic")


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-Duration-Ms"] = str(round(duration_ms, 2))
    return response


@app.exception_handler(UnsupportedProviderError)
async def unsupported_provider_handler(request: Request, exc: UnsupportedProviderError):
    error_resp = ErrorResponse(
        request_id=str(uuid4()),
        error=ErrorInfo(
            type="unsupported_provider",
            message=f"Provider '{exc.provider}' is not available. Check /v1/providers for supported providers.",
        ),
    )
    return JSONResponse(status_code=400, content=error_resp.model_dump())


@app.exception_handler(AllProvidersFailedError)
async def all_providers_failed_handler(request: Request, exc: AllProvidersFailedError):
    from app.models import ErrorDetail

    error_resp = ErrorResponse(
        request_id=exc.request_id,
        error=ErrorInfo(
            type="all_providers_failed",
            message="All providers in the chain failed",
            details=[
                ErrorDetail(provider=e["provider"], model=e.get("model"), error=e["error"])
                for e in exc.errors
            ],
        ),
    )
    return JSONResponse(status_code=502, content=error_resp.model_dump())


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid4())
    logger.exception("Unhandled exception", request_id=request_id)
    error_resp = ErrorResponse(
        request_id=request_id,
        error=ErrorInfo(
            type="internal_error",
            message="An unexpected error occurred",
        ),
    )
    return JSONResponse(status_code=500, content=error_resp.model_dump())
