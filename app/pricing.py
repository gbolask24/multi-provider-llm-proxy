from loguru import logger

from app.providers.base import Usage

MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini": {"input": 0.00000015, "output": 0.0000006},
    "gpt-4o": {"input": 0.0000025, "output": 0.00001},
    # Anthropic
    "claude-3-haiku-20240307": {"input": 0.00000025, "output": 0.00000125},
    "claude-3-5-sonnet-20241022": {"input": 0.000003, "output": 0.000015},
}


def estimate_cost(model: str, usage: Usage) -> float | None:
    """Estimate request cost from token usage. Returns None if pricing unavailable."""
    if (
        model not in MODEL_PRICING
        or usage.input_tokens is None
        or usage.output_tokens is None
    ):
        logger.warning(f"Cost estimation unavailable for model: {model}")
        return None
    pricing = MODEL_PRICING[model]
    return (usage.input_tokens * pricing["input"]) + (
        usage.output_tokens * pricing["output"]
    )
