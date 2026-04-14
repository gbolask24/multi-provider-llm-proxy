from fastapi import APIRouter

from app.exceptions import AllProvidersFailedError, UnsupportedProviderError
from app.models import ErrorDetail, ErrorInfo, ErrorResponse, GenerateRequest, GenerateResponse
from app.providers import get_available_providers
from app.services.proxy_service import handle_request

router = APIRouter()

# Known models per provider (informational only, not authoritative)
KNOWN_MODELS: dict[str, list[str]] = {
    "openai": ["gpt-4o-mini", "gpt-4o"],
    "anthropic": ["claude-3-haiku-20240307", "claude-3-5-sonnet-20241022"],
    "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
}


@router.get("/health")
async def health():
    return {"status": "healthy", "providers": get_available_providers()}


@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    return await handle_request(request)


@router.get("/v1/providers")
async def list_providers():
    available = get_available_providers()
    providers = []
    for name in available:
        providers.append({
            "name": name,
            "known_models": KNOWN_MODELS.get(name, []),
        })
    return {"providers": providers}
