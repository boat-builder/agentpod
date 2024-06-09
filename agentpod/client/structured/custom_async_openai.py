from typing import Any, Callable, Iterable, Self, TypeVar, Union

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .dsl.partial import Partial
from .mode import Mode
from .utils import Provider

T = TypeVar("T", bound=Union[BaseModel, "Iterable[Any]", "Partial[Any]"])


class CustomAsyncOpenAI:
    client: Any | None
    create_fn: Callable[..., Any]
    mode: Mode
    default_model: str | None = None
    provider: Provider

    @property
    def chat(self) -> Self:
        return self

    @property
    def completions(self) -> Self:
        return self

    @property
    def messages(self) -> Self:
        return self

    def __init__(
        self,
        client: Any | None,
        create: Callable[..., Any],
        mode: Mode = Mode.TOOLS,
        provider: Provider = Provider.OPENAI,
        **kwargs: Any,
    ):
        self.client = client
        self.create_fn = create
        self.mode = mode
        self.kwargs = kwargs
        self.provider = provider

    def handle_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        for key, value in self.kwargs.items():
            if key not in kwargs:
                kwargs[key] = value
        return kwargs

    async def create(
        self,
        response_model: type[T],
        messages: list[ChatCompletionMessageParam],
        max_retries: int = 3,
        validation_context: dict[str, Any] | None = None,
        strict: bool = True,
        raw_processor_fn: Callable | None = None,
        **kwargs: Any,
    ) -> T:
        kwargs = self.handle_kwargs(kwargs)
        return await self.create_fn(
            response_model=response_model,
            validation_context=validation_context,
            max_retries=max_retries,
            messages=messages,
            strict=strict,
            raw_processor_fn=raw_processor_fn,
            **kwargs,
        )
