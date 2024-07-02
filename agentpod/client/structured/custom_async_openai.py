from typing import Any, Callable, Iterable, TypeVar, Union

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .dsl.partial import Partial
from .mode import Mode
from .patch import patch
from .utils import Provider

T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]", "Partial[Any]"])


class CustomAsyncOpenAI:
    def __init__(
        self,
        client: AsyncOpenAI,
        mode: Mode = Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
    ):
        self.mode = mode
        self.provider = provider
        self.client = client
        self.create_fn = patch(client.chat.completions.create, mode=mode)

    async def create(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T:
        return await self.create_fn(
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            **kwargs,
        )
