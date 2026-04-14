# Multi-Provider LLM Proxy

A Python API service that sits between your applications and multiple LLM providers. Send one unified request — the proxy handles provider routing, fallback chains, response normalisation, cost estimation, and structured logging.

## Why This Exists

Enterprise AI systems rarely depend on a single LLM provider. Teams need:

- **Provider abstraction** — switch providers without changing application code
- **Fallback resilience** — if one provider is down, automatically try another
- **Normalised responses** — a single response shape regardless of which provider responded
- **Cost visibility** — know what each request costs across providers
- **Observability** — structured logs for every request, including fallback attempts

This proxy provides all of that behind a single REST endpoint.

## Architecture

```
┌──────────────┐      ┌──────────────────┐      ┌──────────┐
│  Application │─────>│  LLM Proxy       │─────>│  OpenAI  │
│              │      │                  │      └──────────┘
│              │      │  - Route         │      ┌──────────┐
│              │<─────│  - Fallback      │─────>│ Anthropic│
│              │      │  - Normalise     │      └──────────┘
│              │      │  - Log + Cost    │      ┌──────────┐
└──────────────┘      └──────────────────┘      │  Gemini  │
                                                └──────────┘
```

## Supported Providers

| Provider | Status | SDK |
|----------|--------|-----|
| OpenAI | Supported | `openai` |
| Anthropic | Supported | `anthropic` |
| Gemini | Planned | `google-genai` |

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd multi-provider-llm-proxy
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run the service

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Send a request

```bash
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what an LLM proxy is in one sentence.",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.3,
    "max_tokens": 100
  }'
```

## API Endpoints

### `POST /v1/generate`

Main proxy endpoint. Accepts a unified request, routes to the specified provider, and returns a normalised response.

**Request:**

```json
{
  "prompt": "Summarise this customer issue in a professional tone",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "temperature": 0.2,
  "max_tokens": 400,
  "fallback_chain": [
    {"provider": "anthropic", "model": "claude-3-haiku-20240307"}
  ],
  "metadata": {
    "use_case": "customer_support",
    "environment": "staging"
  }
}
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `messages` | Either this or `prompt` | — | List of `{role, content}` message objects |
| `prompt` | Either this or `messages` | — | Shorthand for a single user message |
| `system_prompt` | No | — | Prepended as a system message |
| `provider` | Yes | — | Target provider name |
| `model` | No | `gpt-4o-mini` | Model identifier |
| `temperature` | No | `0.7` | Sampling temperature |
| `max_tokens` | No | `1024` | Maximum response tokens |
| `fallback_chain` | No | `[]` | Ordered fallback providers |
| `metadata` | No | `{}` | Echoed back in response for tracing |

**Response (success):**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "provider": "openai",
  "model": "gpt-4o-mini",
  "content": "Here is a professional summary of the customer issue...",
  "finish_reason": "stop",
  "latency_ms": 932,
  "estimated_cost_usd": 0.0018,
  "usage": {
    "input_tokens": 210,
    "output_tokens": 96,
    "total_tokens": 306
  },
  "fallback_used": false,
  "fallback_provider": null,
  "metadata": {
    "use_case": "customer_support",
    "environment": "staging"
  }
}
```

**Response (error):**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "error": {
    "type": "all_providers_failed",
    "message": "All providers in the chain failed",
    "details": [
      {"provider": "openai", "model": "gpt-4o-mini", "error": "timeout"},
      {"provider": "anthropic", "model": "claude-3-haiku-20240307", "error": "rate_limited"}
    ]
  }
}
```

### `GET /health`

Returns service health and registered providers.

### `GET /v1/providers`

Returns available providers with known models.

## Fallback Behaviour

The proxy supports fallback chains. If the primary provider fails (timeout, rate limit, API error), the proxy tries each fallback provider in order.

The fallback chain supports two formats:

```json
// String form: uses the original model
"fallback_chain": ["anthropic"]

// Object form: overrides the model per provider
"fallback_chain": [
  {"provider": "anthropic", "model": "claude-3-haiku-20240307"}
]
```

When fallback succeeds, the response indicates which provider actually served the request:

```json
{
  "provider": "anthropic",
  "fallback_used": true,
  "fallback_provider": "anthropic"
}
```

## Messages-First Contract

The proxy uses a messages-first design:

- `messages` is the primary input (list of `{role, content}` objects)
- `prompt` is shorthand for `[{"role": "user", "content": "..."}]`
- If both are provided, `messages` takes precedence
- `system_prompt` is prepended as a system message

This supports both simple single-turn requests and multi-turn conversations.

## Cost Estimation

Each response includes `estimated_cost_usd` based on token usage and a static pricing map. If the model is not in the pricing map, cost returns as `null` and a warning is logged.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | No | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | No | — | Anthropic API key |
| `GEMINI_API_KEY` | No | — | Google Gemini API key |
| `DEFAULT_PROVIDER` | No | `openai` | Default provider when not specified |
| `DEFAULT_MODEL` | No | `gpt-4o-mini` | Default model when not specified |
| `DEFAULT_TEMPERATURE` | No | `0.7` | Default sampling temperature |
| `DEFAULT_MAX_TOKENS` | No | `1024` | Default max tokens |
| `DEBUG` | No | `false` | Enable debug logging |

At least one provider API key must be configured for the service to be useful.

## Docker

```bash
# Build
docker build -t llm-proxy .

# Run
docker run -p 8000:8000 --env-file .env llm-proxy
```

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fallback.py -v
```

Tests mock all provider SDK calls — no real API keys needed to run them.

## Project Structure

```
app/
  main.py              — FastAPI app, startup, middleware, error handlers
  config.py            — Settings via pydantic-settings
  models.py            — Pydantic request/response schemas
  router.py            — API route handlers
  pricing.py           — Static cost estimation
  logger.py            — Loguru configuration
  exceptions.py        — Custom exception classes
  providers/
    __init__.py        — Provider registry
    base.py            — LLMProvider ABC, ProviderResponse dataclass
    openai_provider.py — OpenAI implementation
    anthropic_provider.py — Anthropic implementation
  services/
    proxy_service.py   — Orchestration, routing, fallback, response building
tests/
  conftest.py          — Shared fixtures and mock providers
  test_models.py       — Request/response validation
  test_pricing.py      — Cost estimation
  test_providers.py    — Provider-specific message mapping
  test_proxy_service.py — Message resolution, fallback parsing
  test_fallback.py     — End-to-end fallback scenarios
  test_api.py          — FastAPI integration tests
```

## Limitations and Future Improvements

- **Streaming** — not supported in this MVP; could be added with SSE
- **Gemini provider** — planned but not yet implemented
- **Retry logic** — fallback is sequential with no per-provider retries
- **Authentication** — no API key auth on the proxy itself
- **Rate limiting** — no built-in rate limiting for downstream callers
- **Persistent logging** — logs go to stderr only; no log aggregation integration
- **Dynamic pricing** — pricing map is static and must be updated manually
