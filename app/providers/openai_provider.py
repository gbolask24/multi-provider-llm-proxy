from openai import AsyncOpenAI, APIError

from app.exceptions import ProviderInvocationError
from app.providers.base import LLMProvider, ProviderResponse, Usage


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except APIError as e:
            raise ProviderInvocationError(
                provider=self.provider_name,
                message=str(e.message),
                retryable=e.status_code in (429, 500, 502, 503) if hasattr(e, "status_code") else False,
            ) from e
        except Exception as e:
            raise ProviderInvocationError(
                provider=self.provider_name,
                message=str(e),
            ) from e

        choice = response.choices[0]
        usage = response.usage

        return ProviderResponse(
            content=choice.message.content or "",
            finish_reason=choice.finish_reason,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else None,
                output_tokens=usage.completion_tokens if usage else None,
                total_tokens=usage.total_tokens if usage else None,
            ),
        )
