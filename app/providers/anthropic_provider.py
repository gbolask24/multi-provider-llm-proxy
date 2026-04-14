from anthropic import AsyncAnthropic, APIError

from app.exceptions import ProviderInvocationError
from app.providers.base import LLMProvider, ProviderResponse, Usage


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, api_key: str):
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        # Extract system message if present
        system_content = None
        non_system_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                non_system_messages.append(msg)

        kwargs: dict = {
            "model": model,
            "messages": non_system_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_content is not None:
            kwargs["system"] = system_content

        try:
            response = await self.client.messages.create(**kwargs)
        except APIError as e:
            raise ProviderInvocationError(
                provider=self.provider_name,
                message=str(e.message),
                retryable=e.status_code in (429, 500, 502, 503, 529) if hasattr(e, "status_code") else False,
            ) from e
        except Exception as e:
            raise ProviderInvocationError(
                provider=self.provider_name,
                message=str(e),
            ) from e

        content = response.content[0].text if response.content else ""
        usage = response.usage

        return ProviderResponse(
            content=content,
            finish_reason=response.stop_reason,
            usage=Usage(
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                total_tokens=(usage.input_tokens + usage.output_tokens) if usage else None,
            ),
        )
